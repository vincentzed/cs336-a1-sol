"""Speed-of-light (SOL) building blocks for the OWT 45-min B200/B300 run.

Design constraints (per project direction):
  * NVIDIA-endorsed kernels only where we leave PyTorch: cuBLAS (matmul),
    cuDNN (attention via SDPA / TransformerEngine), CUDA multi-tensor Adam.
  * No third-party Triton kernel libraries (no Liger, no cut-cross-entropy).
    torch.compile / Inductor is still allowed -- the baseline already uses it.
  * Everything here has a real backward pass (this is training, not serving):
    trtllm_mha / flashinfer decode kernels are deliberately NOT used.

The op-by-op rationale lives in SOL_NOTES.md.
"""
from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lm.modules import (
    Linear,
    MultiHeadSelfAttention,
    PositionWiseFeedForward,
    RelativePositionalEmbedding,
    RMSNorm,
    TransformerLm,
)

logger = logging.getLogger("train.sol")

# TransformerEngine is optional. It is the NVIDIA "Megatron kernel layer":
# cuDNN-fused RMSNorm / attention, FP8 GEMMs. We guard the import so the
# baseline path never depends on it.
try:  # pragma: no cover - availability is environment-specific
    import transformer_engine.pytorch as te  # type: ignore

    _HAS_TE = True
except Exception as _e:  # noqa: BLE001
    te = None  # type: ignore
    _HAS_TE = False
    _TE_IMPORT_ERR = _e

# Gram Newton-Schulz Muon (Dao-AILab, MIT) -- the SOTA orthogonalizer: iterates
# on the symmetric Gram matrix XX^T with Polar Express coefficients, ~2x faster
# than the standard quintic Newton-Schulz, with symmetric CuTeDSL GEMMs (quack).
# "For now, just import it" -- installed via setup_sol_env.sh (nvidia-cutlass-dsl
# 4.6.0 + quack-kernels 0.6.1 + gram-newton-schulz --no-deps). If unavailable
# (env not yet built / GPUs busy) we fall back to the built-in quintic Muon below.
try:  # pragma: no cover
    from gram_newton_schulz import GramNewtonSchulz, POLAR_EXPRESS_COEFFICIENTS

    _HAS_GNS = True
except Exception as _g:  # noqa: BLE001
    GramNewtonSchulz = None  # type: ignore
    POLAR_EXPRESS_COEFFICIENTS = None  # type: ignore
    _HAS_GNS = False
    _GNS_IMPORT_ERR = _g

# quack (Dao-AILab) CuTeDSL kernels: fused linear-CE, RMSNorm, rotary, fused MLP.
# All fwd+bwd, sm100. Each is opt-in via a flag with an eager fallback, so absence
# (or the quack 0.5.x vs 0.6.x API drift) never breaks the working path.
try:  # pragma: no cover
    from quack.linear_cross_entropy import linear_cross_entropy_func as _quack_linear_ce
    _HAS_QUACK_CE = True
except Exception as _qce:  # noqa: BLE001
    _quack_linear_ce = None  # type: ignore
    _HAS_QUACK_CE = False
try:  # pragma: no cover — vendored quack CE with in-kernel softcap (sol:: ops)
    from transformer_lm.quack_ce_softcap import (
        linear_cross_entropy_softcap_func as _quack_linear_ce_softcap,
    )
    _HAS_QUACK_CE_SC = True
except Exception as _qsc:  # noqa: BLE001
    _quack_linear_ce_softcap = None  # type: ignore
    _HAS_QUACK_CE_SC = False
try:  # pragma: no cover
    from quack import rmsnorm as _quack_rmsnorm
    _HAS_QUACK_RMSNORM = True
except Exception as _qrms:  # noqa: BLE001
    _quack_rmsnorm = None  # type: ignore
    _HAS_QUACK_RMSNORM = False
try:  # pragma: no cover
    from quack.rotary import apply_rotary_emb as _quack_apply_rotary
    _HAS_QUACK_ROTARY = True
except Exception as _qro:  # noqa: BLE001
    _quack_apply_rotary = None  # type: ignore
    _HAS_QUACK_ROTARY = False
try:  # pragma: no cover
    from quack.linear import linear_act_func as _quack_linear_act
    _HAS_QUACK_LINEAR_ACT = True
except Exception as _qla:  # noqa: BLE001
    _quack_linear_act = None  # type: ignore
    _HAS_QUACK_LINEAR_ACT = False

# MXFP8 linear, FULL fused MLP, packed QKV rope: wrappers with introspected signatures
# live in transformer_lm.quack_ops (its own guarded imports; no circular import --
# quack_ops only imports transformer_lm inside function bodies).
from transformer_lm.quack_ops import (  # noqa: E402
    HAS_QUACK_MLP as _HAS_QUACK_MLP_FUNC,
    HAS_QUACK_ROTARY_QKV as _HAS_QUACK_ROTARY_QKV,
    quack_mlp_relu2,
    rope_qkv_inplace,
)


# --------------------------------------------------------------------------- #
# 1. Chunked fused linear + cross-entropy (the #1 memory lever).
#
# Baseline materialises logits of shape (B*T, vocab) == (327680, 32000) which is
# ~21 GB in bf16 / ~42 GB in fp32 *before* the softmax, and it is the single
# largest allocation in the step. We never build the full logits tensor: we walk
# the token dimension in chunks, and in the backward we recompute each chunk's
# logits to form grads. Peak logit memory drops to (chunk, vocab). This is the
# same algorithm as Liger's FusedLinearCrossEntropy but implemented in plain
# torch (cuBLAS matmul + torch softmax), no Triton.
# --------------------------------------------------------------------------- #

# Sign-encoded logit softcap (single source of truth; the CuTeDSL kernel in
# quack_ce_softcap.py implements the identical math via the same sign key):
#   softcap > 0:  zc = cap*tanh(z/cap)                       (Gemma-2 form)
#   softcap < 0:  zc = a*sigmoid((z+5)/7.5), a=|softcap|     (modded L1508;
#                 @classiclarryd's evolved cap, b=5 c=7.5 fixed as in modded)
#   softcap == 0: identity.
# The sign threads the form through every existing plumbing site with no extra arg.
_SOFTCAP_B, _SOFTCAP_C = 5.0, 7.5


def apply_softcap(z, softcap: float, *, need_grad: bool = False):
    """Return zc, or (zc, dzc/dz) when need_grad. Matches the fused kernel exactly.
    The tanh branch is arithmetically identical to the original (verified) path."""
    if softcap > 0:
        zt = torch.tanh(z / softcap)
        zc = softcap * zt
        return (zc, 1.0 - zt * zt) if need_grad else zc
    if softcap < 0:
        a = -softcap
        s = torch.sigmoid((z + _SOFTCAP_B) / _SOFTCAP_C)
        zc = a * s
        # dzc/dz = a*s*(1-s)/c = zc*(1-s)/c
        return (zc, zc * (1.0 - s) / _SOFTCAP_C) if need_grad else zc
    return (z, torch.ones_like(z)) if need_grad else z


class _ChunkedLinearCE(torch.autograd.Function):
    # Optional final-logit softcap (cap*tanh(z/cap)) and z-loss (coef*logsumexp^2),
    # both folded in with closed-form grads so the memory-efficient backward holds.
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_size, ignore_index, softcap, z_coef):
        # hidden: (N, d)  weight: (V, d)  targets: (N,)
        N = hidden.shape[0]
        n_valid = (targets != ignore_index).sum().clamp(min=1)
        loss_sum = hidden.new_zeros((), dtype=torch.float32)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            z = (hidden[s:e] @ weight.t()).float()                 # raw logits (c, V)
            zc = apply_softcap(z, softcap)
            t = targets[s:e]
            valid = t != ignore_index
            lse = torch.logsumexp(zc, dim=-1)                      # (c,)
            tgt = zc.gather(-1, t.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            loss_vec = lse - tgt                                   # CE per token
            if z_coef > 0:
                loss_vec = loss_vec + z_coef * lse * lse
            loss_sum += (loss_vec * valid).sum()
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk_size, ctx.ignore_index = chunk_size, ignore_index
        ctx.softcap, ctx.z_coef, ctx.n_valid = softcap, z_coef, n_valid
        return loss_sum / n_valid

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, targets = ctx.saved_tensors
        cs, ii, softcap, z_coef = ctx.chunk_size, ctx.ignore_index, ctx.softcap, ctx.z_coef
        N = hidden.shape[0]
        scale = (grad_out / ctx.n_valid).to(torch.float32)
        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight)
        for s in range(0, N, cs):
            e = min(s + cs, N)
            h = hidden[s:e]
            t = targets[s:e]
            z = (h @ weight.t()).float()                          # (c, V)
            zc, dzc = apply_softcap(z, softcap, need_grad=True)   # dzc/dz chained below
            p = torch.softmax(zc, dim=-1)
            lse = torch.logsumexp(zc, dim=-1, keepdim=True)       # (c, 1)
            g = p.clone()
            rows = torch.arange(e - s, device=h.device)
            g[rows, t.clamp(min=0)] -= 1.0                        # (p - onehot)
            if z_coef > 0:
                g = g + (2.0 * z_coef) * lse * p                  # d(z-loss)/d(zc)
            g = g * dzc                                           # chain through softcap
            valid = (t != ii).unsqueeze(-1)
            g = (g * valid).to(h.dtype) * scale.to(h.dtype)       # (c, V)
            grad_hidden[s:e] = g @ weight
            grad_weight += g.t() @ h
        return grad_hidden, grad_weight, None, None, None, None, None


def chunked_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 32768,
    ignore_index: int = -100,
    softcap: float = 0.0,
    z_coef: float = 0.0,
) -> torch.Tensor:
    """CE over ``(hidden @ weight.T)`` without materialising the full logits.
    Optional final-logit softcap and z-loss. hidden: (..., d), weight: (vocab, d)."""
    hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = targets.reshape(-1)
    return _ChunkedLinearCE.apply(hidden, weight, targets, chunk_size, ignore_index, softcap, z_coef)


def dense_linear_cross_entropy(hidden, weight, targets, ignore_index=-100, softcap=0.0, z_coef=0.0):
    """Materialize logits once (fits at batch<=~320 on a B200) + standard autograd —
    no per-chunk Python loop, no backward recompute. Supports softcap + z-loss."""
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)
    logits = (h @ weight.t()).float()
    logits = apply_softcap(logits, softcap)
    valid = (t != ignore_index).float()
    denom = valid.sum().clamp(min=1)
    lse = torch.logsumexp(logits, dim=-1)
    tgt = logits.gather(-1, t.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    ce = (((lse - tgt) * valid).sum()) / denom
    if z_coef > 0:
        ce = ce + z_coef * ((lse * lse * valid).sum() / denom)
    return ce


def sol_cross_entropy(hidden, weight, targets, *, mode="chunked", chunk_size=8192,
                      ignore_index=-100, softcap=0.0, z_coef=0.0):
    """Dispatch the linear+CE. mode:
      - "quack": quack CuTeDSL fused linear-CE (fast, online-softmax, no recompute).
                 Only valid when softcap==0 and z_coef==0 (kernel has neither); else
                 falls back to chunked with a warning.
      - "dense": materialize logits + standard autograd (supports softcap/z-loss).
      - "chunked": memory-safe custom-backward chunked CE (supports softcap/z-loss).
    """
    if mode == "quack":
        if _HAS_QUACK_CE and softcap == 0 and z_coef == 0:
            h = hidden.reshape(-1, hidden.shape[-1])
            t = targets.reshape(-1)
            # real 0.6.1 sig: linear_cross_entropy_func(x, weight, BIAS, target, ...)
            return _quack_linear_ce(h, weight, None, t, ignore_index=ignore_index,
                                    reduction="mean", inplace_backward=True)
        logger.warning("ce-mode=quack unavailable (has_quack=%s softcap=%.1f z=%.0e) -> chunked",
                       _HAS_QUACK_CE, softcap, z_coef)
        mode = "chunked"
    if mode == "quack-softcap":
        # vendored quack CE with softcap fused in-kernel (transformer_lm/quack_ce_softcap).
        if _HAS_QUACK_CE_SC and softcap != 0 and z_coef == 0:
            return _quack_linear_ce_softcap(hidden.reshape(-1, hidden.shape[-1]), weight,
                                            targets.reshape(-1), softcap,
                                            ignore_index=ignore_index, reduction="mean",
                                            inplace_backward=True)
        logger.warning("ce-mode=quack-softcap unavailable (has=%s softcap=%.1f z=%.0e) -> chunked",
                       _HAS_QUACK_CE_SC, softcap, z_coef)
        mode = "chunked"
    if mode == "dense":
        return dense_linear_cross_entropy(hidden, weight, targets, ignore_index, softcap, z_coef)
    return chunked_linear_cross_entropy(hidden, weight, targets, chunk_size, ignore_index, softcap, z_coef)


# --------------------------------------------------------------------------- #
# 2. Muon optimizer for 2-D matmul params (+ fused CUDA AdamW for the rest).
#
# Every top-5 leaderboard entry used Muon/NorMuon; it buys ~1.5-2x steps-to-loss.
# The Newton-Schulz orthogonalisation is pure cuBLAS matmuls in bf16 -- no Triton.
# Embeddings, the LM head, and all 1-D params (RMSNorm gains) go to fused AdamW.
# --------------------------------------------------------------------------- #
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration -> approx orthogonalisation of G (2-D)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.t()
    return X


def _muon_update(grad, buf, beta, ns_steps, nesterov):
    buf.lerp_(grad, 1 - beta)                       # momentum EMA
    upd = grad.lerp_(buf, beta) if nesterov else buf
    upd = zeropower_via_newtonschulz5(upd, steps=ns_steps)
    upd = upd * math.sqrt(max(1.0, grad.size(-2) / grad.size(-1)))  # RMS-match scale
    return upd


class FallbackMuon(torch.optim.Optimizer):
    """Built-in quintic Newton-Schulz Muon. Used only if gram-newton-schulz is
    not importable; the GNS Muon (Gram iteration + Polar Express) is preferred."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p)
                upd = _muon_update(p.grad, st["momentum_buffer"],
                                   group["momentum"], group["ns_steps"], group["nesterov"])
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(upd.reshape(p.shape).to(p.dtype), alpha=-group["lr"])
        return loss


class NorMuonGNS(torch.optim.Optimizer):
    """NorMuon on top of the Gram-Newton-Schulz orthogonalizer.

    = Muon (speed) + NorMuon per-row 2nd-moment normalization (loss-per-step).
    Orthogonalization uses GNS (Gram iteration + Polar Express); if GNS is
    unavailable we fall back to the built-in quintic NS. The GNS orthogonalizer
    is built with compile_kwargs=None to avoid the reduce-overhead/CUDA-graph
    path that GNS's README says breaks on Blackwell.
    """

    def __init__(self, params, lr=2e-3, momentum=0.95, beta2=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0, cautious_wd=False, bf16_master=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, beta2=beta2,
                                      nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))
        # cautious WD: decay only entries whose update ALSO shrinks |p| (update o is
        # subtracted, so o*p > 0 means the step agrees the weight should shrink).
        self.cautious_wd = cautious_wd
        # bf16 mantissa tracking: keep a bf16 remainder so updates land in effective
        # ~fp32 precision without a full fp32 master copy (modded-nanogpt trick).
        self.bf16_master = bf16_master
        self._ortho = None
        if _HAS_GNS:
            try:
                # compile OFF: dodges the Blackwell reduce-overhead breakage.
                self._ortho = GramNewtonSchulz(
                    ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
                    gram_newton_schulz_reset_iterations=[2],
                    compile_kwargs=None,
                )
                logger.info("NorMuonGNS: using gram-newton-schulz orthogonalizer (compile off)")
            except Exception as e:  # noqa: BLE001 - e.g. quack not installed yet
                logger.warning("GNS orthogonalizer init failed (%r) -> quintic NS", e)
        if self._ortho is None:
            logger.warning("NorMuonGNS: falling back to built-in quintic Newton-Schulz")

    def _orthogonalize_batch(self, U, ns_steps):
        """U: (S, r, c) -> (S, r, c). One GNS call for the whole shape-group."""
        if self._ortho is not None:
            return self._ortho(U)  # GramNewtonSchulz normalizes each matrix independently
        return torch.stack([zeropower_via_newtonschulz5(U[i], steps=ns_steps) for i in range(U.shape[0])])

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        from collections import defaultdict
        for group in self.param_groups:
            beta, beta2 = group["momentum"], group["beta2"]
            ns_steps, nesterov = group["ns_steps"], group["nesterov"]
            lr, wd = group["lr"], group["weight_decay"]
            by_shape = defaultdict(list)  # batch same-shape matrices -> one GNS call each
            for p in group["params"]:
                if p.grad is not None:
                    by_shape[tuple(p.shape)].append(p)
            for shape, params in by_shape.items():
                for p in params:
                    st = self.state[p]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(p)
                        st["v_row"] = torch.zeros_like(p[..., 0:1], dtype=torch.float32)
                bufs = torch.stack([self.state[p]["momentum_buffer"] for p in params])  # (S,r,c)
                grads = torch.stack([p.grad for p in params])                            # (S,r,c)
                v_rows = torch.stack([self.state[p]["v_row"] for p in params])           # (S,r,1)
                bufs.lerp_(grads, 1 - beta)
                u = grads.lerp(bufs, beta) if nesterov else bufs
                o = self._orthogonalize_batch(u, ns_steps).float()                       # (S,r,c)
                # --- NorMuon: per-row 2nd-moment normalize, preserving each matrix's norm ---
                vnorm = o.norm(dim=(-2, -1), keepdim=True)                               # (S,1,1)
                v_rows.lerp_(o.mul(o).mean(dim=-1, keepdim=True), 1 - beta2)             # (S,r,1)
                o = o * v_rows.sqrt().add(1e-10).reciprocal()
                o = o * (vnorm / o.norm(dim=(-2, -1), keepdim=True).add(1e-10))
                o = o * math.sqrt(max(1.0, shape[-2] / shape[-1]))
                for i, p in enumerate(params):
                    st = self.state[p]
                    st["momentum_buffer"].copy_(bufs[i])
                    st["v_row"].copy_(v_rows[i])
                    mt = self.bf16_master and p.dtype == torch.bfloat16
                    if mt:
                        if "mant_rem" not in st:
                            st["mant_rem"] = torch.zeros_like(p)
                        # effective-fp32 weight = bf16 param + bf16 remainder
                        w = p.float().add_(st["mant_rem"].float())
                    else:
                        w = p
                    if wd:
                        if self.cautious_wd:
                            # decay only where the step already shrinks |p| (o*p > 0)
                            mask = (o[i] * w) > 0
                            w.mul_(1 - lr * wd * mask.to(w.dtype))
                        else:
                            w.mul_(1 - lr * wd)
                    w.add_(o[i].to(w.dtype), alpha=-lr)
                    if mt:
                        p.copy_(w)                          # rounds to bf16
                        st["mant_rem"].copy_(w.sub_(p.float()))  # keep the lost mantissa
        return loss


def build_optimizers(model: nn.Module, *, muon_lr: float, adam_lr: float, embed_lr: float,
                     weight_decay: float, betas=(0.9, 0.95), eps=1e-8,
                     cautious_wd: bool = False, bf16_master: bool = False,
                     ve_lr: float | None = None):
    """Param-group hygiene (Fable audit item #6):
      - 2-D transformer matmul weights -> NorMuonGNS  (weight decay applied)
      - embeddings / lm_head           -> fused AdamW, lr=embed_lr, weight_decay=0
      - norms / 1-D params             -> fused AdamW, lr=adam_lr,  weight_decay=0

    Each param group stashes `base_lr` so the scheduler scales groups by their own
    base (embeddings get a higher LR; norms carry no weight decay).
    """
    inner = getattr(model, "_orig_mod", model)
    muon_params, embed_params, norm_params, ve_params = [], [], [], []
    seen = set()
    # Arch-extra scalar banks / tiny gates: AdamW no-wd group, NEVER Muon (GNS
    # orthogonalization is meaningless/broken for (L,2) banks and 12->1 gates).
    _adam_keys = ("resid_lambdas", "post_lambdas", "x0_lambdas", "_xsa_alpha",
                  "smear", "skip_gate", "_skip_lambda", "_backout", "_second_gate",
                  "_ve_gate")  # matches _ve_gate (scalar) AND _ve_gate_w (per-head, 2-D)
    for name, p in inner.named_parameters():  # dedups tied params by default
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if "value_embeds" in name:
            # dedicated group: modded gives value_embeds adam betas (0.75, 0.95) and a
            # high LR (train_gpt.py L1855 lr_mul=75 over their base) -- fast-moving
            # per-token features want a short first-moment memory.
            ve_params.append(p)
        elif ("token_embeddings" in name) or ("lm_head" in name) or ("second_embeds" in name):
            # vocab-indexed tables belong with the embeddings (Muon would orthogonalize
            # the (vocab, d) value table -- wrong geometry for an embedding).
            embed_params.append(p)
        elif any(k in name for k in _adam_keys):
            norm_params.append(p)
        elif p.ndim >= 2:
            muon_params.append(p)
        else:
            norm_params.append(p)

    groups = []
    if embed_params:
        groups.append({"params": embed_params, "lr": embed_lr, "weight_decay": 0.0})
    if ve_params:
        groups.append({"params": ve_params, "lr": ve_lr if ve_lr is not None else embed_lr,
                       "weight_decay": 0.0, "betas": (0.75, 0.95)})  # modded L1855
    if norm_params:
        groups.append({"params": norm_params, "lr": adam_lr, "weight_decay": 0.0})
    adamw = torch.optim.AdamW(groups, betas=betas, eps=eps, fused=True)  # CUDA multi-tensor
    muon = NorMuonGNS(muon_params, lr=muon_lr, weight_decay=weight_decay,
                      cautious_wd=cautious_wd, bf16_master=bf16_master)

    for opt in (muon, adamw):  # stash base LR for the scheduler
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
    logger.info("optimizers: NorMuonGNS(%d matrices, lr=%.2g) | AdamW embed(%d, lr=%.2g, wd=0) "
                "+ norm/1D(%d, lr=%.2g, wd=0) + ve(%d, lr=%.2g, betas=0.75/0.95)",
                len(muon_params), muon_lr, len(embed_params), embed_lr,
                len(norm_params), adam_lr, len(ve_params),
                (ve_lr if ve_lr is not None else embed_lr))
    return muon, adamw


# --------------------------------------------------------------------------- #
# 3. TransformerEngine RMSNorm surgery (cuDNN/CUDA fused norm, no Triton).
#    Opt-in via --rmsnorm te. Falls back with a warning if TE is absent.
# --------------------------------------------------------------------------- #
class TERMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        if not _HAS_TE:
            raise RuntimeError(f"TransformerEngine unavailable: {_TE_IMPORT_ERR!r}")
        self.norm = te.RMSNorm(d_model, eps=eps, params_dtype=dtype)

    def forward(self, x):
        return self.norm(x)


# --------------------------------------------------------------------------- #
# 6. SOL model: identical architecture, but forward() stops at the final norm
#    and returns hidden states. The LM-head projection is fused into the
#    chunked cross-entropy so the (B*T, vocab) logits are never materialised.
# --------------------------------------------------------------------------- #
class SOLTransformerLm(TransformerLm):
    def __init__(self, *args, tie_embeddings: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        # token_embeddings.embeddings and lm_head.weights are both (vocab, d_model).
        if tie_embeddings:
            self.lm_head.weights = self.token_embeddings.embeddings  # share the Parameter
            # The baseline inits the embedding at std=1 (fine for input lookup, but as a
            # tied OUTPUT projection that makes logits std ~sqrt(d_model) -> softcap
            # saturates -> dead grads). Re-init to std=1/sqrt(d_model) so init logits ~unit.
            d = self.token_embeddings.embeddings.shape[1]
            std = d ** -0.5
            with torch.no_grad():
                nn.init.trunc_normal_(self.token_embeddings.embeddings, 0.0, std, -3 * std, 3 * std)
        self.value_embeds = None  # set by enable_value_embeds()
        self.value_embeds_k = None  # (k, vocab, d) multi-table variant; enable_value_embeds_k()
        self._ve_assign = None      # per-layer table index or None (modded ".01...234" map)
        # Arch extras (Thomas-Li / modded-nanogpt stack) -- all None/off by default;
        # populated by enable_arch_extras(). Attrs must exist for the compiled forward.
        self.second_embeds = None   # extra untied input table, zero-init gate
        self.smear_gate = None      # prev-token mix (modded L1245/1404)
        self.x0_lambdas = None      # per-layer x0 re-injection, zeros (modded L1259/1413)
        self.resid_lambdas = None   # (L, 2) residual scales, sqrt(1.1) (modded L1264/1474)
        self.post_lambdas = None    # (L, 2) sublayer-output scales, ones (modded L1256)
        self.skip_gate = None       # u-net skip gate (modded L1415/1432)
        self._backout = None        # negative re-add of the mid-layer cache

    def forward(self, x):  # (B, T) -> (B, T, d_model) hidden
        h = self.token_embeddings(x)
        if self.second_embeds is not None:  # second input embedding (gate zero-init)
            h = h + self._second_gate * self.second_embeds(x)
        if self.smear_gate is not None:  # smear: mix sigmoid-gated PREVIOUS token embed
            g = self._smear_lambda * torch.sigmoid(self.smear_gate(h[:, 1:, :12]))
            h = torch.cat([h[:, :1], h[:, 1:] + g * h[:, :-1]], dim=1)
        if self.value_embeds_k is not None:  # k tables, modded layer map (overrides 1-table)
            vek = self.value_embeds_k[:, x]  # (k, B, T, d) -- one gather, all tables
            for li, lyr in enumerate(self.layers):
                j = self._ve_assign[li]
                lyr.attention._cur_ve = vek[j] if j is not None else None
        elif self.value_embeds is not None:  # per-token value injection (stashed for attention)
            ve = self.value_embeds(x)
            for lyr in self.layers:
                lyr.attention._cur_ve = ve
        extras = (self.resid_lambdas is not None or self.x0_lambdas is not None
                  or self.skip_gate is not None)
        if not extras:  # exact pre-existing path
            for layer in self.layers:
                h = layer(h)
            return self.final_norm(h)
        L = len(self.layers)
        x0 = _rms_last(h) if self.x0_lambdas is not None else None
        x0l = self.x0_lambdas.unbind(0) if x0 is not None else None
        ra = rf = pa = pf = None
        if self.resid_lambdas is not None:  # unbind avoids per-step select_backward kernels
            ra, rf = self.resid_lambdas[:, 0].unbind(0), self.resid_lambdas[:, 1].unbind(0)
            pa, pf = self.post_lambdas[:, 0].unbind(0), self.post_lambdas[:, 1].unbind(0)
        cache_a = cache_b = None
        for i, layer in enumerate(self.layers):
            if self.skip_gate is not None and i == (3 * L) // 4 and cache_a is not None:
                # zero-init scale * sig(gate): exactly identity at init (modded's
                # sig(-1.5)*2*sig(gate) form leaks ~0.18); grads flow via sig(0)=0.5.
                sg = self._skip_lambda * torch.sigmoid(self.skip_gate(x0[..., :12] if x0 is not None else h[..., :12]))
                h = h + sg * cache_a
            a = layer.attention(layer.norm1(h))
            h = (ra[i] * h + pa[i] * a) if ra is not None else (h + a)
            if x0l is not None:
                h = h + x0l[i] * x0
            f = layer.ffn(layer.norm2(h))
            h = (rf[i] * h + pf[i] * f) if rf is not None else (h + f)
            if self.skip_gate is not None:
                if i == L // 4:
                    cache_a = h
                if i == L // 2:
                    cache_b = h
        if self._backout is not None and cache_b is not None:
            h = h - self._backout * cache_b
        return self.final_norm(h)

    def lm_weight(self) -> torch.Tensor:
        return self.lm_head.weights


def enable_value_embeds(model: nn.Module, device=None, dtype=None) -> None:
    """Value embeddings (arXiv 2410.17897): a per-token table added into attention V,
    gated by a zero-init learnable scalar per layer. NB: the per-forward attribute stash
    graph-breaks under --fullgraph (use default compile). Flag: --value-embeds."""
    inner = getattr(model, "_orig_mod", model)
    vocab, d = inner.token_embeddings.embeddings.shape
    inner.value_embeds = nn.Embedding(vocab, d, device=device, dtype=dtype)
    nn.init.normal_(inner.value_embeds.weight, std=0.01)
    for m in inner.modules():
        if isinstance(m, MultiHeadSelfAttention):
            m._ve_gate = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
    logger.info("value embeddings enabled (shared vocab x d table, per-layer gated)")


def enable_value_embeds_k(model: nn.Module, k: int, device=None, dtype=None,
                          gate_mode: str = "scalar") -> None:
    """Modded-faithful multi-table value embeddings (train_gpt.py L1194/L1398).

    modded (num_layers=12, k=5): ve = [None, t0, t1, *[None]*(L-1-k), t2, t3, t4]
    -- the "@photomz .01...234 structure": no VE at layer 0, unique tables at the
    two EARLY layers (1, 2), none in the middle, unique tables at the LAST three.
    Generalization to any L, k: early = (k-1)//2 tables at layers 1..early,
    late = k - early tables at the last `late` layers, None elsewhere
    (reproduces modded exactly at L=12, k=5; requires L-1-k >= 0).

    Storage: one (k, vocab, d) parameter, 0.01*randn (spherical-gaussian init) --
    name contains "value_embeds" so build_optimizers routes it to the embed AdamW
    group.

    gate_mode:
      "scalar"   -- one zero-init scalar gate per assigned layer:  v += g * ve.
      "per-head" -- modded's data-dependent per-head gate (train_gpt.py L1451):
                    g = 2*sigmoid(Linear_{h,12}([x_normed[:6], ve[:6]])), a (num_heads,)
                    gate broadcast over head_dim. The Linear weight is zero-init.
                    We RETAIN the zero-init scalar as an outer multiplier so init
                    parity is still exact (modded's bare 2*sigmoid(0)=1 would make ve
                    active at init); the scalar ramps the whole per-head gate in from 0.
    """
    inner = getattr(model, "_orig_mod", model)
    L = len(inner.layers)
    if L - 1 - k < 0:
        raise ValueError(f"value-embeds-k: need num_layers-1 >= k (got L={L}, k={k})")
    vocab, d = inner.token_embeddings.embeddings.shape
    inner.value_embeds_k = nn.Parameter(
        0.01 * torch.randn(k, vocab, d, device=device, dtype=dtype))
    early = (k - 1) // 2
    late = k - early
    assign = [None] + list(range(early)) + [None] * (L - 1 - k) + list(range(early, k))
    assert len(assign) == L
    inner._ve_assign = assign
    for li, a in enumerate(assign):
        if a is not None:
            attn = inner.layers[li].attention
            attn._ve_gate = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
            if gate_mode == "per-head":
                # (num_heads, 12) zero-init Linear weight (no bias), modded L1451.
                attn._ve_gate_w = nn.Parameter(
                    torch.zeros(attn.h, 12, device=device, dtype=dtype))
    logger.info("value embeddings k=%d enabled: assign=%s (early=%d late=%d) gate=%s",
                k, assign, early, late, gate_mode)


def enable_arch_extras(model: nn.Module, *, x0_lambdas=False, resid_lambdas=False,
                       unet_skips=False, smear=False, second_embed=False,
                       device=None, dtype=None) -> None:
    """Thomas-Li / modded-nanogpt arch stack (each init-neutral: zero/near-zero gates).
    Exact formulations from scratchpad/train_gpt.py (line refs in the forward)."""
    inner = getattr(model, "_orig_mod", model)
    L = len(inner.layers)
    vocab, d = inner.token_embeddings.embeddings.shape
    if x0_lambdas:
        inner.x0_lambdas = nn.Parameter(torch.zeros(L, device=device, dtype=dtype))
    if resid_lambdas:  # modded inits: resid sqrt(1.1), post ones (NOT init-neutral; their choice)
        inner.resid_lambdas = nn.Parameter(torch.full((L, 2), 1.1 ** 0.5, device=device, dtype=dtype))
        inner.post_lambdas = nn.Parameter(torch.ones(L, 2, device=device, dtype=dtype))
    if unet_skips:
        inner.skip_gate = Linear(12, 1, device=device, dtype=dtype)
        nn.init.zeros_(inner.skip_gate.weights)
        # zero-init scale (see forward): exact init-parity; learned upward from 0.
        inner._skip_lambda = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        inner._backout = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
    if smear:
        inner.smear_gate = Linear(12, 1, device=device, dtype=dtype)
        nn.init.zeros_(inner.smear_gate.weights)
        inner._smear_lambda = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
    if second_embed:
        inner.second_embeds = nn.Embedding(vocab, d, device=device, dtype=dtype)
        nn.init.normal_(inner.second_embeds.weight, std=0.01)
        inner._second_gate = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
    logger.info("arch extras: x0=%s resid=%s unet=%s smear=%s second_embed=%s",
                x0_lambdas, resid_lambdas, unet_skips, smear, second_embed)


def apply_init_tweaks(model: nn.Module) -> None:
    """modded-nanogpt init hygiene: zero-init all residual-writing projections (attention
    wo, FFN w2) so every block starts as identity, AND (required, not optional) an
    UNTIED lm_head at std=0.005. The two halves are coupled: with zeroed blocks the
    residual stream is the raw embedding, and a TIED head then echoes the input token
    (logits = norm(e_t)·E^T, self-similarity spike) -> init CE ~31 instead of ~10.8
    (measured, GPU bisect 2026-07-11). Near-zero untied head gives ~uniform init logits."""
    inner = getattr(model, "_orig_mod", model)
    n = 0
    with torch.no_grad():
        for m in inner.modules():
            if isinstance(m, MultiHeadSelfAttention):
                nn.init.zeros_(m.wo.weights); n += 1
            if isinstance(m, ReLU2FFN):
                nn.init.zeros_(m.w2.weights); n += 1
            elif isinstance(m, PositionWiseFeedForward) and hasattr(m, "w2"):
                nn.init.zeros_(m.w2.weights); n += 1
        if inner.lm_head.weights is inner.token_embeddings.embeddings:  # tied -> untie
            w = torch.empty_like(inner.token_embeddings.embeddings)
            nn.init.trunc_normal_(w, 0.0, 0.005, -0.015, 0.015)
            inner.lm_head.weights = nn.Parameter(w)  # routed to embed group by name
            logger.info("init tweaks: UNTIED lm_head (std=0.005) — required with zero-init blocks")
    logger.info("init tweaks: zero-initialized %d output projections (wo/w2)", n)


def mtp_cross_entropy(hidden, weight, targets, weights, *, mode="quack", chunk_size=8192,
                      ignore_index=-100, softcap=0.0, z_coef=0.0):
    """Multi-token prediction (modded-nanogpt mtp_weights): weighted SUM of CEs at
    offsets 0..k-1. Offset i predicts targets[:, i:] from hidden[:, :T-i] (targets are
    already next-token-shifted, so offset i = predicting token t+1+i). Weights beyond
    index 0 anneal to 0 over training (plumbed by the train loop); w<=0 offsets skipped."""
    total = None
    for i, w in enumerate(weights):
        if w <= 0:
            continue
        h = hidden if i == 0 else hidden[:, :-i].contiguous()
        t = targets if i == 0 else targets[:, i:].contiguous()
        li = sol_cross_entropy(h, weight, t, mode=mode, chunk_size=chunk_size,
                               ignore_index=ignore_index, softcap=softcap, z_coef=z_coef)
        total = li * w if total is None else total + li * w
    return total


def swap_rmsnorm_te(model: nn.Module) -> int:
    """Replace every baseline RMSNorm with a TE fused RMSNorm. Returns count."""
    if not _HAS_TE:
        logger.warning("swap_rmsnorm_te requested but TE unavailable -- keeping torch RMSNorm")
        return 0
    n = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, RMSNorm):
                d = child.gamma.shape[0]
                repl = TERMSNorm(d, eps=child.eps, dtype=child.gamma.dtype,
                                 device=child.gamma.device)
                setattr(module, child_name, repl)
                n += 1
    logger.info("swapped %d RMSNorm -> TE fused RMSNorm", n)
    return n


class QuackRMSNorm(nn.Module):
    """quack CuTeDSL RMSNorm (fwd+bwd). Normalizes over the last dim, gain `weight`."""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        if not _HAS_QUACK_RMSNORM:
            raise RuntimeError("quack rmsnorm unavailable")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        return _quack_rmsnorm(x, self.weight, eps=self.eps)


def swap_rmsnorm_quack(model: nn.Module) -> int:
    """Replace every baseline RMSNorm with quack CuTeDSL RMSNorm (copies the gain)."""
    if not _HAS_QUACK_RMSNORM:
        logger.warning("swap_rmsnorm_quack requested but quack rmsnorm unavailable -- keeping torch")
        return 0
    n = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, RMSNorm):
                d = child.gamma.shape[0]
                repl = QuackRMSNorm(d, eps=child.eps, dtype=child.gamma.dtype, device=child.gamma.device)
                with torch.no_grad():
                    repl.weight.copy_(child.gamma)
                setattr(module, child_name, repl)
                n += 1
    logger.info("swapped %d RMSNorm -> quack CuTeDSL RMSNorm", n)
    return n


# --------------------------------------------------------------------------- #
# 4. Async pinned-memory prefetcher (kills the synchronous H2D stall in get_batch).
# --------------------------------------------------------------------------- #
class CudaPrefetcher:
    """Fully-async random-window sampler.

    v1 was only HALF async: the H2D copy rode a side stream but the CPU prep
    (gather + astype(int64) + a FRESH pin_memory() alloc per batch) ran inline in
    next() — measured ~5-35ms/step of main-thread stall (the "22ms gap" between
    wall 236ms/step and the profiler's 214ms of GPU sections). v2:
      * persistent pinned int32 double-buffers (pin alloc happens once),
      * a background thread fills the next buffer during GPU compute,
      * int32 over the wire (4x less than int64), .long() upcast on-GPU,
      * per-buffer CUDA events so the filler never races an in-flight H2D.
    """

    def __init__(self, data: np.ndarray, batch_size: int, context_length: int, device: str):
        from concurrent.futures import ThreadPoolExecutor

        self.data = data
        self.bs = batch_size
        self.ctx = context_length
        self.device = device
        self.cuda = torch.device(device).type == "cuda"
        self.stream = torch.cuda.Stream() if self.cuda else None
        self._next = None
        if self.cuda:
            self._pool = ThreadPoolExecutor(max_workers=1)
            self._alloc_buffers()
            self._future = self._pool.submit(self._fill, 0)
            self.preload()

    def _alloc_buffers(self):
        self._pin = [(torch.empty((self.bs, self.ctx), dtype=torch.int32).pin_memory(),
                      torch.empty((self.bs, self.ctx), dtype=torch.int32).pin_memory())
                     for _ in range(2)]
        self._evt = [torch.cuda.Event(), torch.cuda.Event()]
        for e in self._evt:  # mark "safe to write" initially
            e.record()

    def _fill(self, i):
        self._evt[i].synchronize()  # don't overwrite a buffer with an in-flight H2D
        starts = self._draw_starts()
        idx = starts[:, None] + np.arange(self.ctx)[None, :]
        np.copyto(self._pin[i][0].numpy(), self.data[idx].astype(np.int32, copy=False))
        np.copyto(self._pin[i][1].numpy(), self.data[idx + 1].astype(np.int32, copy=False))
        return i

    def _draw_starts(self):
        if not getattr(self, "shuffled", False):  # default: iid windows WITH replacement
            return np.random.randint(0, self.data.shape[0] - self.ctx - 1, size=self.bs)
        # shuffled epoch: permuted NON-OVERLAPPING window starts -> every consumed token
        # unique until the (never-reached) epoch boundary. Same marginal distribution as
        # random windows (val protocol unchanged); kills the ~22% with-replacement repeats.
        if getattr(self, "_queue", None) is None or self._qpos + self.bs > len(self._queue):
            grid = np.arange(0, self.data.shape[0] - self.ctx - 1, self.ctx, dtype=np.int64)
            off = np.random.randint(0, self.ctx)  # random phase so epochs differ
            self._queue = np.random.permutation(np.minimum(grid + off, self.data.shape[0] - self.ctx - 1))
            self._qpos = 0
            logger.info("shuffled sampler: new epoch of %d unique windows (phase %d)", len(self._queue), off)
        s = self._queue[self._qpos:self._qpos + self.bs]
        self._qpos += self.bs
        return s

    def preload(self):
        i = self._future.result()
        xp, yp = self._pin[i]
        with torch.cuda.stream(self.stream):
            x = xp.to(self.device, non_blocking=True).long()
            y = yp.to(self.device, non_blocking=True).long()
            self._evt[i].record(self.stream)  # buffer i reusable once H2D drains
            self._next = (x, y)
        self._future = self._pool.submit(self._fill, 1 - i)

    def set_batch_size(self, bs: int):
        """Batch-size rampup support: takes effect immediately (drops the in-flight batch)."""
        if bs == self.bs:
            return
        self.bs = bs
        if self.cuda:
            self._future.result()  # drain the in-flight fill for the old shape
            torch.cuda.synchronize()
            self._alloc_buffers()
            self._future = self._pool.submit(self._fill, 0)
            self.preload()

    def next(self):
        if not self.cuda:
            starts = np.random.randint(0, self.data.shape[0] - self.ctx - 1, size=self.bs)
            idx = starts[:, None] + np.arange(self.ctx)[None, :]
            x = torch.from_numpy(self.data[idx].astype(np.int64)).to(self.device)
            y = torch.from_numpy(self.data[idx + 1].astype(np.int64)).to(self.device)
            return x, y
        torch.cuda.current_stream().wait_stream(self.stream)
        x, y = self._next
        x.record_stream(torch.cuda.current_stream())
        y.record_stream(torch.cuda.current_stream())
        self.preload()
        return x, y


# --------------------------------------------------------------------------- #
# 5. Blackwell SDPA fwd/bwd via CuTe-DSL (FlashAttention-4) -- primary attention.
#
# FA4 is a CuTeDSL (CUTLASS Python DSL, *not* Triton) implementation of Flash
# Attention with dedicated sm100 forward AND backward kernels -- exactly the
# Blackwell SDPA fwd/bwd path we want for training. `flash_attn_func` is backed
# by a torch.autograd.Function, so gradients flow. Alternatives: cuDNN (via the
# latest cudnn-frontend, selected through PyTorch's CUDNN_ATTENTION SDPA
# backend) and the portable default SDPA. All three run on the same q/k/v.
#
# FA4 layout is (B, S, H, D); the baseline model works in (B, H, S, D), so we
# transpose in/out. head_dim=64 is supported by the sm100 kernel.
# --------------------------------------------------------------------------- #
import types  # noqa: E402

import einops  # noqa: E402

ATTN_BACKENDS = ("fa4", "fa4op", "cudnn", "torch")

# FA4 wrapped as a proper torch custom op ("sol::fa4") so Dynamo treats it as ONE
# opaque in-graph node (fake impl does shape-prop; register_autograd calls the FA4
# bwd kernel) -> torch.compile(fullgraph=True) works without tracing the CuTeDSL
# internals (which crashed under allow_in_graph). Mirrors FlashAttnFunc fwd/bwd.
# Enable via --attn-backend fa4op --fullgraph. RISKS to verify on Modal: (1) the lse
# shape in register_fake is ASSUMED (B,H,S) fp32 — if FA4's lse differs, compile
# shape-prop errors; (2) grads must match the graph-break `fa4` path numerically.
_FA4_OP = None
try:  # pragma: no cover
    @torch.library.custom_op("sol::fa4", mutates_args=())
    def _fa4_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                softmax_scale: float, softcap: float,
                window_left: int) -> tuple[torch.Tensor, torch.Tensor]:
        from flash_attn.cute.interface import _flash_attn_fwd
        out, lse, _, _ = _flash_attn_fwd(q.contiguous(), k.contiguous(), v.contiguous(),
                                         softmax_scale=softmax_scale,
                                         causal=True, softcap=softcap,
                                         window_size_left=None if window_left < 0 else window_left,
                                         return_lse=True)
        # contiguous-out contract: FA4 allocates out via empty_like(q) (inherits caller's
        # view strides); the fake promises contiguous, so enforce it here (copy is ~free).
        return out.contiguous(), lse.contiguous()

    @_fa4_op.register_fake
    def _(q, k, v, softmax_scale, softcap, window_left):
        # Real kernel allocates CONTIGUOUS (b,s,h,d) out + (b,h,s) fp32 lse (verified on
        # GPU); empty_like would wrongly inherit the caller's transposed-view strides.
        b, s, h, d = q.shape
        return q.new_empty((b, s, h, d)), q.new_empty((b, h, s), dtype=torch.float32)

    # The backward MUST itself be a custom op: register_autograd's callable is traced
    # by AOTAutograd under torch.compile, and a raw CuTeDSL call inside it is an
    # untraceable side effect that gets silently DROPPED -> compiled bwd returned
    # empty dq/dk/dv (dq~0, dk/dv uninitialized garbage; root-caused via repro).
    @torch.library.custom_op("sol::fa4_bwd", mutates_args=())
    def _fa4_bwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor, dout: torch.Tensor, lse: torch.Tensor,
                    softmax_scale: float, softcap: float,
                    window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from flash_attn.cute.interface import _flash_attn_bwd
        dq, dk, dv = _flash_attn_bwd(q.contiguous(), k.contiguous(), v.contiguous(),
                                     out.contiguous(), dout.contiguous(), lse.contiguous(),
                                     softmax_scale, True, softcap,
                                     None if window_left < 0 else window_left)
        return dq.contiguous(), dk.contiguous(), dv.contiguous()

    @_fa4_bwd_op.register_fake
    def _(q, k, v, out, dout, lse, softmax_scale, softcap, window_left):
        return (q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape))

    def _fa4_setup(ctx, inputs, output):
        q, k, v, softmax_scale, softcap, window_left = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale, ctx.softcap, ctx.window_left = softmax_scale, softcap, window_left

    def _fa4_bwd(ctx, grad_out, grad_lse):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _fa4_bwd_op(q, k, v, out, grad_out, lse,
                                 ctx.softmax_scale, ctx.softcap, ctx.window_left)
        return dq, dk, dv, None, None, None

    torch.library.register_autograd("sol::fa4", _fa4_bwd, setup_context=_fa4_setup)
    _FA4_OP = _fa4_op
except Exception as _fa4e:  # noqa: BLE001
    _FA4_OP = None


@torch._dynamo.disable
def _fa4_eager(qs, ks, vs, scale, window):
    """Eager FA4, fenced from Dynamo: without this fence Dynamo resume-traces INTO
    flash_attn/CuTeDSL internals after the pybind graph break (recompile_limit noise on
    cutlass arith.const + wasted warmup). The break itself is unavoidable and ~free."""
    from flash_attn.cute import flash_attn_func  # CuTe-DSL sm100 fwd/bwd
    ws = (int(window), 0) if window and window > 0 else (None, None)
    o = flash_attn_func(qs, ks, vs, causal=True, softmax_scale=scale, window_size=ws)
    return o[0] if isinstance(o, tuple) else o


@torch._dynamo.disable
def sol_attention_varlen(q, k, v, cu_seqlens, max_seqlen, scale=None, window=0):
    """Doc-aware (BOS-packed) attention: q/k/v (total_tokens, H, D), cu_seqlens int32
    (n_docs+1,). No attention across document boundaries. For the packing loader."""
    from flash_attn.cute import flash_attn_varlen_func
    ws = (int(window), 0) if window and window > 0 else (None, None)
    o = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                               max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                               causal=True, softmax_scale=scale, window_size=ws)
    return o[0] if isinstance(o, tuple) else o


@torch._dynamo.disable
def _sol_attn_varlen_fenced(mod, q, k, v, scale, window):
    """Varlen path, fully fenced: reads mod._cu_seqlens INSIDE the fence so the
    varying-shape tensor never enters the compiled graph (no shape guards / no
    per-batch recompiles under dynamic=False). q/k/v (B,H,N,D) -> same shape out.
    NB: the fence graph-breaks -> --data-packing varlen is incompatible with --fullgraph."""
    b, h, n, d = q.shape
    fl = lambda t: t.transpose(1, 2).reshape(b * n, h, d)  # row-major flatten matches loader's cu_seqlens
    o = sol_attention_varlen(fl(q), fl(k), fl(v), mod._cu_seqlens, mod._max_seqlen,
                             scale=scale, window=window)
    return o.view(b, n, h, d).transpose(1, 2)


def set_varlen_batch(model, cu_seqlens, max_seqlen=0):
    """Stash this batch's cu_seqlens on each attention module (call OUTSIDE the
    compiled forward, once per step). None disables the varlen path (dense attention);
    the compiled graph only guards on the varlen_active bool (2 graphs total)."""
    inner = getattr(model, "_orig_mod", model)
    for m in inner.modules():
        if isinstance(m, MultiHeadSelfAttention) and hasattr(m, "_sol"):
            m._cu_seqlens = cu_seqlens
            m._max_seqlen = int(max_seqlen)
            m._sol.varlen_active = cu_seqlens is not None


def sol_attention(q, k, v, backend="fa4", scale=None, window=0):
    """q, k, v: (B, H, S, D). Causal self-attention. scale=None -> 1/sqrt(D).
    window>0 = sliding-window (left) attention -- FA4 backends only."""
    if backend in ("fa4", "fa4op"):
        qs, ks, vs = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # -> (B,S,H,D)
        if backend == "fa4op" and _FA4_OP is not None:
            s = float(scale) if scale is not None else float(q.shape[-1] ** -0.5)
            # contiguous at the op boundary: setup_context saves these exact tensors for
            # the FA4 bwd kernel, which needs consistent (b,s,h,d) layouts (strided saved
            # views -> non-finite grads, observed on GPU).
            out, _lse = _FA4_OP(qs.contiguous(), ks.contiguous(), vs.contiguous(),
                                s, 0.0, int(window) if window else -1)
            return out.transpose(1, 2)
        return _fa4_eager(qs, ks, vs, scale, window).transpose(1, 2)
    if window:
        logger.warning("attn window=%d ignored on backend=%s (FA4 only)", window, backend)
    if backend == "cudnn":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def _rms_last(t, eps=1e-6):
    """Param-free RMS-norm over the last dim (head_dim) -- QK-norm."""
    return t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + eps)


def _sol_attn_forward(self, x):
    """MultiHeadSelfAttention.forward + SOL SDPA + QK-norm + (optional) partial RoPE,
    fused QKV (+in-place packed rope), learnable attn scale, and attention output gate.
    Config read from self._sol / attrs."""
    cfg = self._sol
    n = x.shape[-2]
    roped = False
    if getattr(self, "_wqkv", None) is not None:
        # ONE (3d, d) GEMM instead of three; optional ONE in-place rope kernel for q&k.
        qkv = (x @ self._wqkv.mT).view(x.shape[0], n, 3, self.h, -1)  # (b, n, 3, h, d)
        if getattr(self, "_fused_rope", False):
            qkv = rope_qkv_inplace(qkv, self._rope_cos[:n], self._rope_sin[:n])
            roped = True
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))  # each (b, h, n, d)
    else:
        Q, K, V = self.wq(x), self.wk(x), self.wv(x)
        q = einops.rearrange(Q, "b n (h dk) -> b h n dk", h=self.h)
        k = einops.rearrange(K, "b n (h dk) -> b h n dk", h=self.h)
        v = einops.rearrange(V, "b n (h dv) -> b h n dv", h=self.h)
        if getattr(self, "_fused_rope", False):
            # quack CuTeDSL rotary on (b, s, h, d); cos/sin sized (s, rotary_dim/2) so it
            # rotates the first rotary_dim dims and passes the rest -> partial RoPE for free.
            cos, sin = self._rope_cos[:n], self._rope_sin[:n]
            q = _quack_apply_rotary(q.transpose(1, 2), cos, sin).transpose(1, 2)
            k = _quack_apply_rotary(k.transpose(1, 2), cos, sin).transpose(1, 2)
            roped = True
    if getattr(self, "_cur_ve", None) is not None and getattr(self, "_ve_gate", None) is not None:
        ve = einops.rearrange(self._cur_ve, "b n (h d) -> b h n d", h=self.h)  # value embedding into V
        if getattr(self, "_ve_gate_w", None) is not None:
            # modded L1451: per-head data-dependent gate g = 2*sigmoid(W @ [x[:6], ve[:6]]).
            # x is the normed attention input; self._cur_ve is (b, n, h*d).
            gate_in = torch.cat([x[..., :6], self._cur_ve[..., :6]], dim=-1)  # (b, n, 12)
            g = 2.0 * torch.sigmoid(F.linear(gate_in, self._ve_gate_w))       # (b, n, h)
            g = g.transpose(1, 2).unsqueeze(-1)                               # (b, h, n, 1)
            v = v + self._ve_gate * g * ve   # scalar (zero-init) keeps init parity exact
        else:
            v = v + self._ve_gate * ve
    if not roped and self.rope is not None:
        pos = torch.arange(n, device=x.device)
        rd = getattr(self, "_rotary_dim", None)  # partial (half-truncate) RoPE
        if rd is None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        else:  # rotate only the first rd dims; leave the rest stationary
            q = torch.cat([self.rope(q[..., :rd], pos), q[..., rd:]], dim=-1)
            k = torch.cat([self.rope(k[..., :rd], pos), k[..., rd:]], dim=-1)
    if cfg.qk_norm:
        q, k = _rms_last(q), _rms_last(k)
    scale = None
    if getattr(self, "_attn_scale", None) is not None:  # learnable scale replaces 1/sqrt(d)
        q = q * self._attn_scale
        scale = 1.0
    if getattr(cfg, "varlen_active", False):
        o = _sol_attn_varlen_fenced(self, q, k, v, scale, getattr(cfg, "window", 0))
    else:
        o = sol_attention(q, k, v, cfg.backend, scale=scale, window=getattr(cfg, "window", 0))
    if getattr(self, "_xsa_alpha", None) is not None:
        # Gated XSA (arXiv:2603.09078, modded L1127-30): subtract per-head tanh(alpha)
        # fraction of the v-hat-aligned component of the attention output. zeros init.
        vn = F.normalize(v, dim=-1, eps=1e-4)
        proj = (o * vn).sum(-1, keepdim=True)
        o = o - torch.tanh(self._xsa_alpha).view(1, -1, 1, 1).to(o.dtype) * proj * vn
    if getattr(self, "_attn_gate", None) is not None:  # context-based no-op (sink replacement)
        g = torch.sigmoid(self._attn_gate(x))            # (b, n, h)
        o = o * einops.rearrange(g, "b n h -> b h n").unsqueeze(-1)
    return self.wo(einops.rearrange(o, "b h n dv -> b n (h dv)", h=self.h))


def _rope_cos_sin(rotary_dim, max_seq_len, theta, device, dtype):
    """Standard (non-interleaved) RoPE cos/sin tables, shape (max_seq_len, rotary_dim/2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def setup_sol_attention(model: nn.Module, *, backend: str = "fa4", qk_norm: bool = True,
                        partial_rope_frac: float = 1.0, attn_gate: bool = False,
                        learnable_scale: bool = False, fused_rope: bool = False,
                        fused_qkv: bool = False, attn_window: int = 0, xsa: bool = False,
                        rope_theta: float = 10000.0,
                        max_seq_len: int | None = None, device=None) -> int:
    """Configure + rebind each attention module's forward. Call BEFORE torch.compile.
    fused_qkv: one (3d,d) QKV GEMM (+ one in-place packed-rope kernel when fused_rope).
    NB Muon then orthogonalizes the stacked QKV as ONE matrix (ablate; modded-nanogpt
    banks q/k/v separately). attn_window>0: sliding-window attention (FA4 only)."""
    if backend not in ATTN_BACKENDS:
        raise ValueError(f"backend must be one of {ATTN_BACKENDS}, got {backend!r}")
    if fused_rope and not _HAS_QUACK_ROTARY:
        logger.warning("fused_rope requested but quack.rotary unavailable -- eager RoPE")
        fused_rope = False
    if fused_qkv and fused_rope and not _HAS_QUACK_ROTARY_QKV:
        logger.warning("fused_qkv packed rope unavailable -- falling back to eager rope")
        fused_rope = False
    # NB: FA4 is a CuTeDSL/quack kernel (tvm_ffi + DLPack) that CANNOT be traced by
    # Inductor -> it graph-breaks (and allow_in_graph makes it worse: Inductor tries to
    # trace the kernel and crashes on DLPack null-ptr). For a FULL graph, use native
    # SDPA (backend="torch"/"cudnn"), which is what baseline uses and why baseline is fast.
    n = 0
    for m in model.modules():
        if not isinstance(m, MultiHeadSelfAttention):
            continue
        d_model = m.wq.weights.shape[1]
        d_k = d_model // m.h
        dtype = m.wq.weights.dtype
        if partial_rope_frac < 1.0 and m.rope is not None:
            rd = max(2, (int(d_k * partial_rope_frac) // 2) * 2)  # even, <= d_k
            m.rope = RelativePositionalEmbedding(rope_theta, rd, max_seq_len, device=device)
            m._rotary_dim = rd
        if fused_rope and m.rope is not None:
            rotary_dim = getattr(m, "_rotary_dim", d_k)
            cos, sin = _rope_cos_sin(rotary_dim, max_seq_len, rope_theta, device, dtype)
            m.register_buffer("_rope_cos", cos, persistent=False)
            m.register_buffer("_rope_sin", sin, persistent=False)
            m._fused_rope = True
        if attn_gate:
            gate = Linear(d_model, m.h, device=device, dtype=dtype)
            nn.init.zeros_(gate.weights)  # sigmoid(0)=0.5 to start
            m._attn_gate = gate
        if learnable_scale:
            m._attn_scale = nn.Parameter(torch.tensor(0.1, device=device, dtype=dtype))
        if xsa:  # per-head zeros -> tanh(0)=0 disables XSA at init
            m._xsa_alpha = nn.Parameter(torch.zeros(m.h, device=device, dtype=dtype))
        if fused_qkv:
            # one (3d, d) GEMM; init preserved by stacking the existing weights.
            w = torch.cat([m.wq.weights.data, m.wk.weights.data, m.wv.weights.data], dim=0)
            m._wqkv = nn.Parameter(w)
            del m.wq, m.wk, m.wv  # remove the per-projection Linears (params must not double-register)
        m._sol = types.SimpleNamespace(backend=backend, qk_norm=qk_norm, window=attn_window)
        m.forward = types.MethodType(_sol_attn_forward, m)
        n += 1
    logger.info("SOL attention on %d modules: backend=%s qk_norm=%s partial_rope=%.2f gate=%s "
                "scale=%s fused_rope=%s fused_qkv=%s window=%d", n, backend, qk_norm,
                partial_rope_frac, attn_gate, learnable_scale, fused_rope, fused_qkv, attn_window)
    return n


# --------------------------------------------------------------------------- #
# 7. ReLU^2 FFN (drops the SwiGLU gate matrix -> fewer params, bigger batch).
#    Maps to Megatron `fused_weighted_squared_relu`; the elementwise square is
#    fused by Inductor. Swapped in post-hoc so the baseline stays pristine.
# --------------------------------------------------------------------------- #
class ReLU2FFN(nn.Module):
    """ReLU^2 FFN. mode:
      "eager" -- x@w1.T, square, x@w2.T (Inductor fuses the square)
      "fc1"   -- relu_sq fused into the fc1 GEMM epilogue (quack linear_act_func)
      "mlp"   -- FULL fused MLP fc1+relu_sq+fc2 via quack mlp_func (fwd+bwd, recompute:
                 saves only x, re-does the fc1 GEMM in bwd -> big activation-memory win)."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None, fused: bool = False,
                 mode: str | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.mode = mode if mode is not None else ("fc1" if fused else "eager")
        if self.mode == "mlp" and not _HAS_QUACK_MLP_FUNC:
            self.mode = "fc1" if _HAS_QUACK_LINEAR_ACT else "eager"
        if self.mode == "fc1" and not _HAS_QUACK_LINEAR_ACT:
            self.mode = "eager"

    def forward(self, x):
        if self.mode == "mlp":
            return quack_mlp_relu2(x, self.w1.weights, self.w2.weights)
        if self.mode == "fc1":
            # relu_sq(x @ w1.T) fused in the GEMM epilogue; w2 is a plain linear.
            # linear_act_func returns (preact, postact) -- preact is saved for the fused
            # bwd (dact) when grads are needed; feeding the raw tuple onward was a bug.
            _preact, h = _quack_linear_act(x, self.w1.weights, "relu_sq",
                                           store_preact=torch.is_grad_enabled())
            return self.w2(h)
        return self.w2(torch.relu(self.w1(x)).square())


def swap_ffn_relu2(model: nn.Module, fused: bool = False, mode: str | None = None) -> int:
    """Replace every SwiGLU/SiLU FFN with a ReLU^2 FFN (fresh init). Returns count.
    mode: eager|fc1|mlp (see ReLU2FFN); legacy fused=True == mode='fc1'."""
    mode = mode if mode is not None else ("fc1" if fused else "eager")
    if mode == "mlp" and not _HAS_QUACK_MLP_FUNC:
        logger.warning("full fused MLP requested but quack.mlp unavailable -- trying fc1")
        mode = "fc1"
    if mode == "fc1" and not _HAS_QUACK_LINEAR_ACT:
        logger.warning("fused ReLU^2 requested but quack.linear unavailable -- eager ReLU^2")
        mode = "eager"
    n = 0
    for m in model.modules():
        for cn, child in list(m.named_children()):
            if isinstance(child, PositionWiseFeedForward):
                d_ff, d_model = child.w1.weights.shape  # Linear weights are (out, in)
                setattr(m, cn, ReLU2FFN(d_model, d_ff, device=child.w1.weights.device,
                                        dtype=child.w1.weights.dtype, mode=mode))
                n += 1
    logger.info("swapped %d FFN -> ReLU^2 (mode=%s)", n, mode)
    return n


# --------------------------------------------------------------------------- #
# 8. EMA (Polyak) weight averaging for EVAL ONLY.
#
# Keep a slow fp32 running average of the weights; swap it in for the validation
# forward pass, then restore the training weights -- the averaged weights are
# NEVER written back, so the training trajectory is bit-identical. Averages away
# iterate noise (esp. in the WSD cooldown tail) -> lower reported val loss, free.
# fp32 shadow matters: with decay=0.999 the (1-decay)=1e-3 update underflows in bf16.
# --------------------------------------------------------------------------- #
class EmaWeights:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        inner = getattr(model, "_orig_mod", model)
        self.params = [p for _, p in inner.named_parameters()]
        self.shadow = [p.detach().float().clone() for p in self.params]  # fp32 master average

    @torch.no_grad()
    def update(self):
        # fast path: one fused kernel-group instead of ~350 lerp_ launches + a full
        # fp32 copy of every param per step (p.detach().float() was ~1GB/step of churn)
        try:
            torch._foreach_lerp_(self.shadow, self.params, 1 - self.decay)
            return
        except Exception:  # noqa: BLE001 -- mixed-dtype foreach unsupported -> slow path
            pass
        return self._update_slow()

    @torch.no_grad()
    def _update_slow(self):
        for p, s in zip(self.params, self.shadow):
            s.lerp_(p.detach().float(), 1 - self.decay)

    @contextmanager
    def averaged(self):
        """Temporarily load the EMA weights (for eval), then restore training weights."""
        backup = [p.detach().clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.data.copy_(s.to(p.dtype))
        try:
            yield
        finally:
            for p, b in zip(self.params, backup):
                p.data.copy_(b)
