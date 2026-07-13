"""quack CuTeDSL training-op wrappers (all signatures INTROSPECTED, not guessed).

Verified against quack 0.5.3 (local venv) — guards keep 0.6.1 (Modal) working:
  mxfp8_gemm_quantize(A=(M,K), B_as_NK=(N,K)) -> A @ B.mT   [K-dim block-32 scales]
  mlp_func(x, w1, w2, activation, ..., recompute=bool)      [full fused MLP fwd+bwd]
  linear_func(x, weight, bias=None, fuse_grad_accum=False)  [fused dgrad/wgrad sched]
  apply_rotary_emb_qkv_(qkv5d, cos, sin)                    [in-place q&k rope, one kernel;
                                                             cos/sin (s, rd/2) -> partial rope,
                                                             rotary_dim % 8 == 0]

Assessment of the rest of quack for our training (introspected 2026-07-11):
  cross_entropy  — standalone CE on materialized logits; we never materialize (linear-CE). No.
  softmax        — our only softmaxes live inside FA4/linear-CE kernels already. No.
  topk           — sampling/MoE-routing; not in the training path. No.
  hadamard_transform — rotation-based quant/outlier control (QuIP-style); relevant only if
                   we push past MXFP8 to FP4 or add rotation tricks. Not now.
  gemm_sq_reduce — GEMM + fused per-column squared-reduce; could fuse NorMuon's row-norms
                   into the GNS GEMMs, but the optimizer is 1% of the step. Not worth it.
  gemm_norm_act  — raw dispatch only (NO autograd wrapper in quack); the fused-MLP path
                   (mlp_func) covers the GEMM+act fusion, Inductor covers the norm. Skipped.
"""
from __future__ import annotations

import logging
import types

import torch
import torch.nn as nn

logger = logging.getLogger("train.sol.quack")

try:  # pragma: no cover
    from quack.gemm_blockscaled_interface import (
        mxfp8_gemm as _mxfp8_gemm,
        mxfp8_gemm_quantize as _mxfp8_gemm_quantize,
        mxfp8_quantize as _mxfp8_quantize,
    )
    from quack.mx_utils import to_mx_compiled as _to_mx_compiled
    HAS_MXFP8 = True
except Exception:  # noqa: BLE001
    _mxfp8_gemm = _mxfp8_gemm_quantize = _mxfp8_quantize = _to_mx_compiled = None  # type: ignore
    HAS_MXFP8 = False
try:  # pragma: no cover
    from quack.mlp import mlp_func as _quack_mlp_func
    HAS_QUACK_MLP = True
except Exception:  # noqa: BLE001
    _quack_mlp_func = None  # type: ignore
    HAS_QUACK_MLP = False
try:  # pragma: no cover
    from quack.linear import linear_func as _quack_linear_func
    HAS_QUACK_LINEAR = True
except Exception:  # noqa: BLE001
    _quack_linear_func = None  # type: ignore
    HAS_QUACK_LINEAR = False
try:  # pragma: no cover
    from quack.rotary import apply_rotary_emb_qkv_ as _quack_rotary_qkv_
    HAS_QUACK_ROTARY_QKV = True
except Exception:  # noqa: BLE001
    _quack_rotary_qkv_ = None  # type: ignore
    HAS_QUACK_ROTARY_QKV = False


# --------------------------------------------------------------------------- #
# MXFP8 linear (the ~2x GEMM lever). fwd: y = x @ W.mT with both operands
# quantized to MXFP8 (block-32 scales along K) inside mxfp8_gemm_quantize.
# bwd "all": dgrad dx = dout @ W (contraction over N -> quantize along N) and
# wgrad dW = dout.mT @ x (contraction over M -> quantize along M), both via the
# same A @ B.mT convention with zero-copy .mT views made contiguous.
# bwd "fwd": bf16 backward (conservative first config).
# lm_head/CE stay bf16 by construction (we only swap QKVO + FFN linears).
# --------------------------------------------------------------------------- #
class _MXFP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bwd_mxfp8):
        batch_shape = x.shape[:-1]
        xf = x.reshape(-1, x.shape[-1])
        out = _mxfp8_gemm_quantize(xf, weight, out_dtype=x.dtype)  # (M, N)
        ctx.save_for_backward(xf, weight)
        ctx.bwd_mxfp8 = bwd_mxfp8
        return out.reshape(*batch_shape, weight.shape[0])

    @staticmethod
    def backward(ctx, dout):
        xf, w = ctx.saved_tensors
        batch_shape = dout.shape[:-1]
        do = dout.reshape(-1, dout.shape[-1]).contiguous()  # (M, N)
        if ctx.bwd_mxfp8 and do.shape[0] % 32 == 0:
            # dx = do @ W : A=do (M,N), B_as_NK=W.mT (K,N) -> quantize along N (%32 ok)
            dx = _mxfp8_gemm_quantize(do, w.mT.contiguous(), out_dtype=do.dtype)
            # dW = do.mT @ x : A=do.mT (N,M), B_as_NK=xf.mT (K,M) -> quantize along M
            dw = _mxfp8_gemm_quantize(do.mT.contiguous(), xf.mT.contiguous(), out_dtype=w.dtype)
        else:
            dx = do @ w
            dw = do.mT @ xf
        return dx.reshape(*batch_shape, w.shape[1]), dw, None


@torch._dynamo.disable
def mxfp8_linear(x: torch.Tensor, weight: torch.Tensor, bwd_mxfp8: bool = False) -> torch.Tensor:
    # Fenced from Dynamo: the quack blockscaled launcher does cute from_dlpack on the
    # fp8-e8m0 scale tensors, which hard-errors under fake-tensor tracing (observed).
    return _MXFP8Linear.apply(x, weight, bwd_mxfp8)


def _mxfp8_fwd(self, x, _bwd):
    return mxfp8_linear(x, self.weights, _bwd)


# --------------------------------------------------------------------------- #
# MXFP8 v2: CACHED weight quantization (fixes v1's 624ms: v1 re-quantized W and
# W.mT in eager to_mx on EVERY GEMM call). v2 quantizes weights once per
# optimizer step (both orientations, via the compiled quantizer) and keeps the
# per-use activation/grad quants compiled as well.
#   mode "cached":     fp8 fwd + fp8 dgrad (cached W quants), bf16 wgrad
#                      (wgrad is the accuracy-critical GEMM and needs 2 extra
#                       transposed per-use quants -- worst cost/benefit).
#   mode "cached-all": + fp8 wgrad (both operands per-use-quantized).
# Weight quant uses FLOOR scaling (quack 0.5.3 has no RCEIL) -- note for parity.
# --------------------------------------------------------------------------- #
class _MXCache:
    __slots__ = ("wq", "ws", "wtq", "wts", "dirty")

    def __init__(self):
        self.wq = self.ws = self.wtq = self.wts = None
        self.dirty = True


@torch.no_grad()
def _refresh_cache(weight: torch.Tensor, cache: _MXCache) -> None:
    # fwd:  y = x @ W.mT      -> B_as_NK = W      (N,K), scales along K
    # dgrad: dx = dO @ W      -> B_as_NK = W.mT   (K,N), scales along N
    cache.wq, cache.ws = _to_mx_compiled(weight.contiguous(), 32)
    cache.wtq, cache.wts = _to_mx_compiled(weight.mT.contiguous(), 32)
    cache.dirty = False


class _MXFP8LinearCached(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, cache, wgrad_fp8):
        if cache.dirty:
            _refresh_cache(weight, cache)
        batch_shape = x.shape[:-1]
        xf = x.reshape(-1, x.shape[-1])
        xq, xs = _to_mx_compiled(xf.contiguous(), 32)
        out = _mxfp8_gemm(xq, cache.wq.mT, xs, cache.ws.mT, out_dtype=x.dtype)
        ctx.save_for_backward(xf, weight)
        ctx.cache, ctx.wgrad_fp8 = cache, wgrad_fp8
        return out.reshape(*batch_shape, weight.shape[0])

    @staticmethod
    def backward(ctx, dout):
        xf, w = ctx.saved_tensors
        cache = ctx.cache
        batch_shape = dout.shape[:-1]
        do = dout.reshape(-1, dout.shape[-1]).contiguous()  # (M, N)
        # dgrad: dx = dO @ W  (contraction over N; cached W.mT quant, scales along N)
        dq, ds = _to_mx_compiled(do, 32)
        dx = _mxfp8_gemm(dq, cache.wtq.mT, ds, cache.wts.mT, out_dtype=do.dtype)
        if ctx.wgrad_fp8 and do.shape[0] % 32 == 0:
            # wgrad: dW = dO.mT @ x (contraction over M; both per-use, transposed quants)
            dtq, dts = _to_mx_compiled(do.mT.contiguous(), 32)
            xtq, xts = _to_mx_compiled(xf.mT.contiguous(), 32)
            dw = _mxfp8_gemm(dtq, xtq.mT, dts, xts.mT, out_dtype=w.dtype)
        else:
            dw = do.mT @ xf  # bf16 wgrad (accuracy-critical; cuBLAS)
        return dx.reshape(*batch_shape, w.shape[1]), dw, None, None


@torch._dynamo.disable
def mxfp8_linear_cached(x, weight, cache, wgrad_fp8=False):
    return _MXFP8LinearCached.apply(x, weight, cache, wgrad_fp8)


def _mxfp8_cached_fwd(self, x, _wgrad):
    return mxfp8_linear_cached(x, self.weights, self._mx_cache, _wgrad)


def mark_mxfp8_dirty(model: nn.Module) -> None:
    """Call after optimizer.step(): weights changed, cached quants must refresh."""
    for m in model.modules():
        c = getattr(m, "_mx_cache", None)
        if c is not None:
            c.dirty = True


def _quack_linear_fwd(self, x):
    return _quack_linear_func(x, self.weights)


def swap_linears_mxfp8(model: nn.Module, mode: str = "fwd") -> int:
    """Route attention QKVO + FFN w1/w2 Linears through MXFP8 GEMMs.
    mode: "fwd" (bf16 backward) | "all" (mxfp8 dgrad+wgrad too). lm_head untouched.
    Requires in/out features % 32 (blockscale); skips (with a warning) otherwise."""
    if not HAS_MXFP8:
        logger.warning("--mxfp8 requested but quack blockscaled interface unavailable -- skipped")
        return 0
    from transformer_lm.modules import MultiHeadSelfAttention
    from transformer_lm.sol_modules import ReLU2FFN
    cached = mode.startswith("cached")
    bwd = mode == "all"
    wgrad_fp8 = mode == "cached-all"
    n = 0
    for m in model.modules():
        names = ("wq", "wk", "wv", "wo") if isinstance(m, MultiHeadSelfAttention) else (
            ("w1", "w2") if isinstance(m, ReLU2FFN) else ())
        for name in names:
            lin = getattr(m, name, None)
            if lin is None or not hasattr(lin, "weights"):
                continue
            N, K = lin.weights.shape
            if N % 32 or K % 32:
                logger.warning("mxfp8 skip %s (%d,%d): dims must be %%32", name, N, K)
                continue
            if cached:
                lin._mx_cache = _MXCache()
                lin.forward = types.MethodType(
                    lambda self, x, _w=wgrad_fp8: _mxfp8_cached_fwd(self, x, _w), lin)
            else:
                lin.forward = types.MethodType(lambda self, x, _b=bwd: _mxfp8_fwd(self, x, _b), lin)
            n += 1
    logger.info("MXFP8 linears: %d swapped (mode=%s; lm_head/CE stay bf16)", n, mode)
    return n


def swap_linears_quack(model: nn.Module) -> int:
    """Route wo/w2 (the post-attention/post-act projections) through quack linear_func
    (tuned tiles + fused dgrad/wgrad scheduling). fuse_grad_accum stays OFF: quack marks
    it incompatible with torch.compile."""
    if not HAS_QUACK_LINEAR:
        logger.warning("--quack-linear requested but quack.linear unavailable -- skipped")
        return 0
    from transformer_lm.modules import MultiHeadSelfAttention
    from transformer_lm.sol_modules import ReLU2FFN
    n = 0
    for m in model.modules():
        names = ("wo",) if isinstance(m, MultiHeadSelfAttention) else (
            ("w2",) if isinstance(m, ReLU2FFN) else ())
        for name in names:
            lin = getattr(m, name, None)
            if lin is not None and hasattr(lin, "weights"):
                lin.forward = types.MethodType(_quack_linear_fwd, lin)
                n += 1
    logger.info("quack linear_func on %d projections (wo/w2)", n)
    return n


def quack_mlp_relu2(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                    recompute: bool = False) -> torch.Tensor:
    """Full fused MLP: relu_sq(x @ w1.mT) @ w2.mT via quack mlp_func (fwd+bwd).
    recompute=False keeps the preact (faster bwd; +~16GB acts at batch320/d_ff2048x12,
    fine on B200/B300). recompute=True re-does the fc1 GEMM in bwd (measured +11ms/step
    on B300 -- only worth it when memory-bound)."""
    return _quack_mlp_func(x, w1, w2, "relu_sq", recompute=recompute)


def rope_qkv_inplace(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """In-place rope on packed qkv (b, s, 3, h, d) -- ONE kernel rotates q and k.
    cos/sin are (s, rotary_dim/2); rotates the first rotary_dim dims (partial rope),
    rotary_dim % 8 == 0. Autograd handled by quack's ApplyRotaryEmbQKV_."""
    return _quack_rotary_qkv_(qkv, cos, sin)
