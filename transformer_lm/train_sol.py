"""SOL training loop for the OWT 45-min B200/B300 run.

Differences vs transformer_lm/train_script.py (the baseline), op-by-op in
SOL_NOTES.md. Everything below is cuBLAS / cuDNN / CUDA / CuTe-DSL -- no
third-party Triton kernels. torch.compile (Inductor) stays on for the memory
-bound elementwise ops (RMSNorm/RoPE/SwiGLU), same as baseline.

Key changes:
  * model.forward returns hidden states; the LM head is fused into a chunked
    linear cross-entropy so the (B*T, vocab) logits are never materialised.
  * Muon on 2-D matmul params + fused CUDA AdamW on embeddings/head/1-D params.
  * Attention SDPA -> FlashAttention-4 CuTe-DSL sm100 (fwd+bwd) by default,
    with cuDNN / torch alternates.
  * Async pinned-memory prefetcher (no synchronous H2D stall).
  * grad-norm clip is foreach and does not .item()-sync every step.
  * optional TransformerEngine fused RMSNorm (--rmsnorm te).

W&B: reads WANDB_API_KEY from the environment. The key is never stored in the
repo. Set it with `export WANDB_API_KEY=...` or `wandb login` before running.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import wandb

from transformer_lm.modules import get_batch, get_lr_cosine_schedule, save_checkpoint
from transformer_lm.sol_modules import (
    CudaPrefetcher,
    EmaWeights,
    SOLTransformerLm,
    build_optimizers,
    enable_value_embeds,
    mtp_cross_entropy,
    set_varlen_batch,
    setup_sol_attention,
    sol_cross_entropy,
    swap_ffn_relu2,
    swap_rmsnorm_quack,
    swap_rmsnorm_te,
)
from transformer_lm.train_script import load_tokens, log_metrics, setup_logging, setup_wandb

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOL Transformer LM training (B200/B300).")
    d = p.add_argument_group("data")
    d.add_argument("--train-tokens", type=str, required=True)
    d.add_argument("--val-tokens", type=str, default=None)
    d.add_argument("--tokens-dtype", type=str, default="uint16", choices=["uint16", "int32", "int64"])
    d.add_argument("--data-packing", type=str, default="random", choices=["random", "bos", "varlen"],
                   help="bos: windows start at doc starts (BosAlign); varlen: + cu_seqlens for "
                        "FA4 varlen attention (no cross-doc attention)")
    d.add_argument("--data-sampling", type=str, default="random", choices=["random", "shuffled"],
                   help="shuffled = permuted non-overlapping windows (no with-replacement repeats)")
    d.add_argument("--vocab-pkl", type=str, default="tokenizers/owt/vocab.pkl",
                   help="vocab pickle used to locate <|endoftext|> for --data-packing")

    m = p.add_argument_group("model")
    m.add_argument("--vocab-size", type=int, required=True)
    m.add_argument("--d-model", type=int, default=768)
    m.add_argument("--num-layers", type=int, default=12)
    m.add_argument("--num-heads", type=int, default=12)
    m.add_argument("--d-ff", type=int, default=2048)
    m.add_argument("--max-seq-len", type=int, default=1024)
    m.add_argument("--rope-theta", type=float, default=10000.0)
    m.add_argument("--ffn", type=str, default="relu2", choices=["relu2", "swiglu", "silu"])

    o = p.add_argument_group("optimization")
    o.add_argument("--batch-size", type=int, default=512)
    o.add_argument("--context-length", type=int, default=1024)
    o.add_argument("--total-iters", type=int, default=20000)
    o.add_argument("--max-wall-sec", type=float, default=None)
    o.add_argument("--schedule-by-wall", action="store_true",
                   help="drive the LR schedule by elapsed/max-wall-sec instead of step/total-iters "
                        "(robust to node-speed variance; decay tail always completes)")
    # NorMuonGNS base lr (~1e-3..2e-3). embed gets its own (higher) LR; norms carry no wd.
    o.add_argument("--muon-lr", type=float, default=2e-3)
    o.add_argument("--adam-lr", type=float, default=3e-3, help="LR for norms / 1-D params")
    o.add_argument("--embed-lr", type=float, default=6e-3, help="LR for embeddings / lm_head")
    o.add_argument("--lr-schedule", type=str, default="wsd", choices=["wsd", "cosine"],
                   help="wsd = warmup-stable-decay (fixed-budget SOTA); ablate against cosine")
    o.add_argument("--wsd-decay-frac", type=float, default=0.2, help="fraction of steps for WSD decay tail")
    o.add_argument("--lr-min-ratio", type=float, default=0.1, help="final LR as a fraction of max")
    o.add_argument("--warmup-iters", type=int, default=500)
    o.add_argument("--cosine-cycle-iters", type=int, default=None)
    o.add_argument("--weight-decay", type=float, default=0.1)
    o.add_argument("--beta1", type=float, default=0.9)
    o.add_argument("--beta2", type=float, default=0.95)
    o.add_argument("--grad-clip", type=float, default=1.0)
    o.add_argument("--ce-chunk", type=int, default=32768, help="token-rows per CE chunk")
    o.add_argument("--adam-every", type=int, default=1, help=">1: step AdamW params every Nth iter (speed)")
    o.add_argument("--muon-momentum-warmup", type=int, default=0,
                   help="steps to ramp Muon momentum 0.85->0.95 (0 = off / fixed 0.95)")
    o.add_argument("--muon-momentum-decay-last", type=int, default=0,
                   help="decay Muon momentum 0.95->0.85 over the last N steps (modded-nanogpt)")
    o.add_argument("--cautious-wd", action="store_true",
                   help="Muon weight decay only where the update also shrinks |p| (modded-nanogpt)")
    o.add_argument("--bf16-mt", action="store_true",
                   help="bf16 mantissa-tracking master weights for Muon params (effective ~fp32 updates)")
    o.add_argument("--bs-ramp", type=str, default=None,
                   help="'start:end:frac' discrete batch-size rampup over the first frac of total_iters; "
                        "group LRs scale by (bs/end)**0.5 during the ramp (2 extra compiles)")

    e = p.add_argument_group("ema-for-eval")
    e.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True,
                   help="Polyak weight averaging used for the val forward pass only")
    e.add_argument("--ema-decay", type=float, default=0.999)
    e.add_argument("--ema-start", type=int, default=-1, help="start EMA at this step; -1 = warmup_iters")

    a = p.add_argument_group("arch")
    a.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    a.add_argument("--tie-embeddings", action=argparse.BooleanOptionalAction, default=True)
    a.add_argument("--logit-softcap", type=float, default=30.0, help="0 disables final-logit softcap")
    a.add_argument("--softcap-form", type=str, default="tanh", choices=["tanh", "sigmoid"],
                   help="tanh: cap*tanh(z/cap) (Gemma-2). sigmoid: cap*sigmoid((z+5)/7.5) "
                        "(modded L1508, @classiclarryd). Passed to the CE as a sign-encoded softcap.")
    a.add_argument("--z-loss", type=float, default=1e-4, help="0 disables z-loss")
    # nanoGPT-speedrun-validated; individually toggleable so we can ablate each
    a.add_argument("--partial-rope", type=float, default=0.5,
                   help="fraction of head_dim that rotates (1.0 = full RoPE; 0.5 = half-truncate)")
    a.add_argument("--attn-gate", action=argparse.BooleanOptionalAction, default=True,
                   help="attention output gate (context no-op / attention-sink replacement)")
    a.add_argument("--attn-scale", action=argparse.BooleanOptionalAction, default=True,
                   help="learnable attention scale replacing 1/sqrt(d) (pairs with QK-norm)")
    a.add_argument("--no-norm-gammas", action="store_true",
                   help="freeze all RMSNorm gains at 1.0 (Thomas: 'no learnable norms')")
    a.add_argument("--value-embeds", action="store_true",
                   help="per-token value embedding injected into attention V (arXiv 2410.17897); breaks --fullgraph")
    a.add_argument("--ve-lr", type=float, default=None,
                   help="LR for the value-embeds AdamW group (betas 0.75/0.95); default = --embed-lr")
    a.add_argument("--value-embeds-k", type=int, default=0,
                   help="k separate VE tables with the modded layer map (None,t0,t1,...,t2,t3,t4); "
                        "5 = modded-faithful; overrides --value-embeds")
    a.add_argument("--ve-gates", type=str, default="scalar", choices=["scalar", "per-head"],
                   help="scalar: one zero-init gate/layer. per-head: modded's data-dependent "
                        "g=2*sigmoid(W[x[:6],ve[:6]]) per head (train_gpt.py L1451).")
    # Thomas-Li / modded-nanogpt arch stack (all default off; zero/near-zero-init gates)
    a.add_argument("--mtp", type=str, default="",
                   help="multi-token prediction weights, e.g. '1,0.5,0.25'; extra weights "
                        "linearly anneal to 0 over total_iters (modded-nanogpt style)")
    a.add_argument("--x0-lambdas", action="store_true",
                   help="per-layer learnable re-injection of the normed input embedding (zeros)")
    a.add_argument("--resid-lambdas", action="store_true",
                   help="learnable per-sublayer residual/output scales (init sqrt(1.1)/1.0, modded)")
    a.add_argument("--init-tweaks", action="store_true",
                   help="zero-init all residual-writing projections (wo/w2); lm_head-std skipped when tied")
    a.add_argument("--unet-skips", action="store_true",
                   help="gated u-net skip (L/4 cache re-added at 3L/4) + zero-init backout of the L/2 cache")
    a.add_argument("--smear", action="store_true",
                   help="sigmoid-gated previous-token embedding mix (modded smear)")
    a.add_argument("--second-embed", action="store_true",
                   help="second untied input embedding table, zero-init gate")
    a.add_argument("--xsa", action="store_true",
                   help="gated XSA: per-head tanh-gated subtraction of the v-aligned attn component")

    s = p.add_argument_group("sol")
    s.add_argument("--attn-backend", type=str, default="fa4", choices=["fa4", "fa4op", "cudnn", "torch"])
    s.add_argument("--rmsnorm", type=str, default="torch", choices=["torch", "te", "quack"])
    s.add_argument("--ce-mode", type=str, default="chunked", choices=["chunked", "dense", "quack", "quack-softcap"],
                   help="quack = CuTeDSL fused linear-CE (fast; needs --logit-softcap 0 --z-loss 0)")
    s.add_argument("--fused-rope", action="store_true", help="quack CuTeDSL fused RoPE")
    s.add_argument("--fused-ffn", action="store_true", help="quack epilogue-fused ReLU^2 fc1")
    s.add_argument("--fused-mlp", action="store_true",
                   help="FULL fused MLP fc1+relu_sq+fc2 via quack mlp_func (supersedes --fused-ffn)")
    s.add_argument("--fused-qkv", action="store_true",
                   help="one (3d,d) QKV GEMM + in-place packed QKV rope (with --fused-rope); "
                        "NB Muon orthogonalizes the stacked QKV as one matrix")
    s.add_argument("--quack-linear", action="store_true",
                   help="wo/w2 projections via quack linear_func (tuned tiles)")
    s.add_argument("--mxfp8", type=str, default="off", choices=["off", "fwd", "all", "cached", "cached-all"],
                   help="MXFP8 GEMMs for QKVO+FFN linears (fwd only, or +dgrad/wgrad); lm_head stays bf16")
    s.add_argument("--attn-window", type=int, default=0,
                   help=">0: sliding-window attention (left window, FA4 backends only)")
    s.add_argument("--eval-ctx", type=int, default=0,
                   help="val context length (0 = --context-length; set 512 for leaderboard comparability)")
    s.add_argument("--no-prefetch", action="store_true")

    c = p.add_argument_group("checkpointing")
    c.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    c.add_argument("--checkpoint-interval", type=int, default=2000)

    lg = p.add_argument_group("logging")
    lg.add_argument("--log-interval", type=int, default=10)
    lg.add_argument("--val-interval", type=int, default=500)
    lg.add_argument("--val-batches", type=int, default=20)
    lg.add_argument("--log-level", type=str, default="INFO")
    lg.add_argument("--log-file", type=str, default=None)
    lg.add_argument("--wandb-project", type=str, default=None)
    lg.add_argument("--wandb-entity", type=str, default=None)
    lg.add_argument("--wandb-run-name", type=str, default=None)
    lg.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    rt = p.add_argument_group("runtime")
    rt.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    rt.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    rt.add_argument("--seed", type=int, default=0)
    rt.add_argument("--compile", action="store_true")
    rt.add_argument("--fullgraph", action="store_true",
                    help="torch.compile(fullgraph=True) — needs FA4 allow_in_graph (auto-registered)")
    rt.add_argument("--compile-mode", type=str, default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    rt.add_argument("--compile-cache", type=str, default=None,
                    help="mega-cache path: load Dynamo/Inductor artifacts at start (skips the "
                         "~60-90s compile warmup, ~3%% of a 45-min budget), save at end")

    args = p.parse_args()
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.total_iters
    if args.ema_start < 0:
        args.ema_start = args.warmup_iters  # skip the high-LR warmup chaos
    if args.eval_ctx == 0:
        args.eval_ctx = args.context_length
    if args.bs_ramp:
        s, e, f = args.bs_ramp.split(":")
        args.bs_ramp = (int(s), int(e), float(f))
        if args.bs_ramp[1] != args.batch_size:
            logger.warning("--bs-ramp end=%d overrides --batch-size %d (steady-state batch)",
                           args.bs_ramp[1], args.batch_size)
            args.batch_size = args.bs_ramp[1]
    args.mtp_weights = [float(w) for w in args.mtp.split(",")] if args.mtp else None
    if args.mtp_weights is not None and (len(args.mtp_weights) < 2 or args.mtp_weights[0] <= 0):
        raise SystemExit("--mtp needs >=2 comma-separated weights with weights[0] > 0")
    return args


def build_model(args) -> SOLTransformerLm:
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    # Build with a baseline-valid activation, then swap to ReLU^2 post-hoc if requested.
    base_act = "swiglu" if args.ffn == "relu2" else args.ffn
    model = SOLTransformerLm(
        vocab_size=args.vocab_size, d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, d_ff=args.d_ff, max_seq_len=args.max_seq_len,
        theta=args.rope_theta, use_rmsnorm=True, use_rope=True, activation=base_act,
        device=args.device, dtype=dtype, tie_embeddings=args.tie_embeddings,
    )
    if args.ffn == "relu2":
        swap_ffn_relu2(model, mode="mlp" if args.fused_mlp else ("fc1" if args.fused_ffn else "eager"))
    if args.rmsnorm == "te":
        swap_rmsnorm_te(model)
    elif args.rmsnorm == "quack":
        swap_rmsnorm_quack(model)
    setup_sol_attention(  # BEFORE compile
        model, backend=args.attn_backend, qk_norm=args.qk_norm,
        partial_rope_frac=args.partial_rope, attn_gate=args.attn_gate,
        learnable_scale=args.attn_scale, fused_rope=args.fused_rope, fused_qkv=args.fused_qkv,
        attn_window=args.attn_window, xsa=args.xsa, rope_theta=args.rope_theta,
        max_seq_len=args.max_seq_len, device=args.device,
    )
    if args.value_embeds_k > 0:
        from transformer_lm.sol_modules import enable_value_embeds_k
        enable_value_embeds_k(model, args.value_embeds_k, device=args.device, dtype=dtype,
                              gate_mode=args.ve_gates)
        if args.value_embeds:
            logger.info("--value-embeds-k overrides --value-embeds (1-table path disabled)")
    elif args.value_embeds:
        enable_value_embeds(model, device=args.device, dtype=dtype)
    if args.x0_lambdas or args.resid_lambdas or args.unet_skips or args.smear or args.second_embed:
        from transformer_lm.sol_modules import enable_arch_extras
        enable_arch_extras(model, x0_lambdas=args.x0_lambdas, resid_lambdas=args.resid_lambdas,
                           unet_skips=args.unet_skips, smear=args.smear,
                           second_embed=args.second_embed, device=args.device, dtype=dtype)
    if args.init_tweaks:  # LAST: must zero the FINAL wo/w2 (after any FFN/attention swaps)
        from transformer_lm.sol_modules import apply_init_tweaks
        apply_init_tweaks(model)
        if args.tie_embeddings:
            logger.info("init-tweaks: lm_head std=0.005 SKIPPED (tied embeddings)")
    if args.mxfp8 != "off":  # AFTER the FFN/attention swaps so it sees the final Linears
        from transformer_lm.quack_ops import swap_linears_mxfp8
        swap_linears_mxfp8(model, mode=args.mxfp8)
    if args.quack_linear:
        from transformer_lm.quack_ops import swap_linears_quack
        swap_linears_quack(model)
    if args.no_norm_gammas:
        frozen = 0
        for name, p_ in model.named_parameters():
            if ("gamma" in name or "norm" in name.lower()) and p_.dim() == 1:
                with torch.no_grad():
                    p_.fill_(1.0)
                p_.requires_grad_(False)
                frozen += 1
        logger.info("no-norm-gammas: froze %d norm gains at 1.0", frozen)
    n = sum(p.numel() for p in model.parameters())
    logger.info("SOL model: %.2fM params | ffn=%s qk_norm=%s tie=%s softcap=%.0f z_loss=%.0e | "
                "attn=%s partial_rope=%.2f gate=%s learn_scale=%s | rmsnorm=%s dtype=%s",
                n / 1e6, args.ffn, args.qk_norm, args.tie_embeddings, args.logit_softcap, args.z_loss,
                args.attn_backend, args.partial_rope, args.attn_gate, args.attn_scale,
                args.rmsnorm, args.dtype)
    return model


@torch.no_grad()
def evaluate(model, lm_weight, val_tokens, args) -> float:
    """FIXED eval protocol: the val windows are drawn from a dedicated RNG (seed 1234),
    identical for every run/seed/config. Previously they came off the GLOBAL RNG (seeded
    by args.seed + advanced by training history), so each run scored a different ~8%
    val subsample (sigma ~0.003-0.005) -- and best-of-N seed selection was partially
    selecting lucky val subsets rather than better models. Never couple eval sampling
    to run randomness."""
    set_varlen_batch(model, None)  # val batches are dense random windows
    model.eval()
    losses = []
    rng = np.random.default_rng(1234)
    n = val_tokens.shape[0]
    for _ in range(args.val_batches):
        starts = rng.integers(0, n - args.eval_ctx - 1, size=args.batch_size)
        idx = starts[:, None] + np.arange(args.eval_ctx)[None, :]
        x = torch.from_numpy(val_tokens[idx].astype(np.int64)).to(args.device, non_blocking=True)
        y = torch.from_numpy(val_tokens[idx + 1].astype(np.int64)).to(args.device, non_blocking=True)
        hidden = model(x)
        # Reported val loss = plain CE (apply softcap if the model trains with it; NO z-loss).
        loss = sol_cross_entropy(hidden, lm_weight, y, mode=args.ce_mode, chunk_size=args.ce_chunk,
                                 softcap=args.logit_softcap, z_coef=0.0)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def eval_maybe_ema(model, lm_weight, val_tokens, args, ema, step) -> float:
    """Run validation under the EMA-averaged weights when EMA is active."""
    if ema is not None and step >= args.ema_start:
        with ema.averaged():
            return evaluate(model, lm_weight, val_tokens, args)
    return evaluate(model, lm_weight, val_tokens, args)


def lr_frac(it, args) -> float:
    """Schedule multiplier in [lr_min_ratio, 1]. WSD (fixed-budget SOTA) or cosine."""
    w = args.warmup_iters
    if it < w:
        return (it + 1) / max(1, w)
    if args.lr_schedule == "cosine":
        return get_lr_cosine_schedule(it, 1.0, args.lr_min_ratio, w, args.cosine_cycle_iters)
    # WSD: stable at 1.0, then linear decay over the last wsd_decay_frac of steps.
    decay_start = args.total_iters * (1.0 - args.wsd_decay_frac)
    if it < decay_start:
        return 1.0
    prog = (it - decay_start) / max(1.0, args.total_iters - decay_start)
    return 1.0 + prog * (args.lr_min_ratio - 1.0)


def set_lr(optimizers, frac):
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * frac


def bs_for_iter(it, args) -> tuple[int, float]:
    """Discrete batch-size rampup (start -> mid -> end over the first frac of steps).

    Returns (batch_size, lr_scale) with lr_scale = (bs/end)**0.5, multiplied into the
    LR schedule (never replacing it). Discrete stages keep torch.compile(dynamic=False)
    happy: 3 shapes total = 2 extra compiles (amortized by --compile-cache).
    """
    if not args.bs_ramp:
        return args.batch_size, 1.0
    s, e, f = args.bs_ramp
    window = f * args.total_iters
    if window <= 0 or it >= window:
        return e, 1.0
    mid = max(8, int(round((s + e) / 2 / 8)) * 8)
    bs = s if it < window / 2 else mid
    return bs, (bs / e) ** 0.5


def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    wandb_active = setup_wandb(args)

    train_tokens = load_tokens(args.train_tokens, args.tokens_dtype)
    val_tokens = load_tokens(args.val_tokens, args.tokens_dtype) if args.val_tokens else None

    model = build_model(args)
    lm_weight = model.lm_weight()  # grabbed before compile-wrap; shares storage
    if args.compile:
        if args.compile_cache and os.path.exists(args.compile_cache):
            try:  # mega-cache: pre-populate Dynamo/Inductor caches from a prior run
                with open(args.compile_cache, "rb") as f:
                    torch.compiler.load_cache_artifacts(f.read())
                logger.info("compile cache loaded from %s", args.compile_cache)
            except Exception as e:  # noqa: BLE001 -- stale/incompatible cache must never kill a run
                logger.warning("compile cache load failed (%r) -- cold compile", e)
        logger.info("torch.compile(fullgraph=%s, mode=%s, dynamic=False)", args.fullgraph, args.compile_mode)
        model = torch.compile(model, fullgraph=args.fullgraph, mode=args.compile_mode, dynamic=False)

    muon, adamw = build_optimizers(
        model, muon_lr=args.muon_lr, adam_lr=args.adam_lr, embed_lr=args.embed_lr,
        weight_decay=args.weight_decay, betas=(args.beta1, args.beta2),
        cautious_wd=args.cautious_wd, bf16_master=args.bf16_mt,
        ve_lr=args.ve_lr,
    )
    optimizers = (muon, adamw)

    ema = EmaWeights(model, args.ema_decay) if args.ema else None
    if ema is not None:
        logger.info("EMA-for-eval: decay=%.4f start=%d (val forward uses averaged weights)",
                    args.ema_decay, args.ema_start)

    logger.info("naive loss floor log(vocab)=%.4f", math.log(args.vocab_size))
    params = [p for p in model.parameters() if p.requires_grad]

    prefetch = not args.no_prefetch
    if args.data_packing != "random" and not prefetch:
        raise SystemExit("--data-packing bos/varlen requires the prefetcher (drop --no-prefetch)")
    loader = None
    if prefetch:
        from transformer_lm.data_packing import make_loader
        loader = make_loader(args.data_packing, train_tokens, args.batch_size, args.context_length,
                             args.device, vocab_pkl=args.vocab_pkl, tokens_path=args.train_tokens)
        if getattr(args, "data_sampling", "random") == "shuffled":
            # permuted non-overlapping windows: no with-replacement repeats (+~22% fresh
            # tokens at our 0.53-epoch consumption), same marginal distribution as random.
            # NB: make_loader("random") returns a _RandomAdapter WRAPPER -- the sampler
            # (and _draw_starts) live on .inner. Setting the flag on the wrapper was a
            # silent no-op (r10-r13 "shuffled" runs actually ran plain random sampling).
            target = getattr(loader, "inner", loader)
            if not hasattr(target, "_draw_starts"):
                raise SystemExit("--data-sampling shuffled: sampler object has no _draw_starts")
            target.shuffled = True
            logger.info("data sampling: shuffled epoch (unique windows)")

    model.train()
    t_start = time.time()
    recent: deque[tuple[int, float]] = deque(maxlen=20)
    recent.append((0, t_start))
    grad_norm_t = torch.zeros((), device=args.device)
    stop_reason, last_it = "total_iters", 0

    for it in range(args.total_iters):
        last_it = it
        cur_bs, bs_lr_scale = bs_for_iter(it, args)
        cu_seqlens = None
        if prefetch:
            loader.set_batch_size(cur_bs)
            inputs, targets, cu_seqlens = loader.next()
        else:
            inputs, targets = get_batch(train_tokens, cur_bs, args.context_length, args.device)

        sched_it = it
        if args.schedule_by_wall and args.max_wall_sec:
            # drive the LR schedule by ELAPSED TIME, not step index: on a slow/contended
            # node the decay tail then completes regardless of achieved throughput
            # (a step-indexed schedule wall-stops mid-decay and ruins the run).
            # ANCHOR AT THE FIRST STEP, not process start: compile/autotune (~3-5 min)
            # otherwise consumes the LR warmup before step 0 -> full LR immediately ->
            # divergence (observed on B200 attempt-3: val exploded to 13).
            if it <= 1 or "_wall_anchor" not in locals():
                _wall_anchor = time.time()
            budget = max(1.0, args.max_wall_sec - (_wall_anchor - t_start))
            sched_it = min(args.total_iters,
                           max(0.0, time.time() - _wall_anchor) / budget * args.total_iters)
        frac = lr_frac(sched_it, args) * bs_lr_scale  # rampup LR scaling composes multiplicatively
        set_lr(optimizers, frac)
        if args.muon_momentum_warmup > 0:  # ramp Muon momentum 0.85 -> 0.95
            mom = 0.85 + 0.10 * min(1.0, it / args.muon_momentum_warmup)
            if args.muon_momentum_decay_last > 0:  # Thomas/modded: decay back in the last N steps
                tail = args.total_iters - args.muon_momentum_decay_last
                if sched_it >= tail:
                    mom = 0.95 - 0.10 * min(1.0, (sched_it - tail) / args.muon_momentum_decay_last)
            for g in muon.param_groups:
                g["momentum"] = mom

        # varlen contract: cu_seqlens is STASHED on the attention modules (side channel,
        # read inside the dynamo fence) so its varying shape never enters the compiled
        # graph -> no per-batch recompiles under dynamic=False. None -> dense attention.
        set_varlen_batch(model, cu_seqlens, args.context_length)
        hidden = model(inputs)
        if args.mtp_weights is not None:
            # extra offsets anneal linearly to 0 over the run (modded [1,.5,.25] -> [1])
            anneal = max(0.0, 1.0 - it / max(1, args.total_iters))
            w = [args.mtp_weights[0]] + [b * anneal for b in args.mtp_weights[1:]]
            loss = mtp_cross_entropy(hidden, lm_weight, targets, w, mode=args.ce_mode,
                                     chunk_size=args.ce_chunk, softcap=args.logit_softcap,
                                     z_coef=args.z_loss)
        else:
            loss = sol_cross_entropy(hidden, lm_weight, targets, mode=args.ce_mode, chunk_size=args.ce_chunk,
                                     softcap=args.logit_softcap, z_coef=args.z_loss)

        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        loss.backward()

        # foreach clip; the returned norm tensor is only .item()-synced at log time.
        if args.grad_clip > 0:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(params, args.grad_clip, foreach=True)
        muon.step()
        if it % args.adam_every == 0:  # Adam-on-odd-steps: update embed/norm params every Nth iter
            adamw.step()
        if args.mxfp8.startswith("cached"):  # weights changed -> cached fp8 quants stale
            from transformer_lm.quack_ops import mark_mxfp8_dirty
            mark_mxfp8_dirty(model)
        if ema is not None and it >= args.ema_start:
            ema.update()

        elapsed = time.time() - t_start
        if args.max_wall_sec is not None and elapsed >= args.max_wall_sec:
            stop_reason = "wall_clock"
            logger.info("wall-clock stop: %.1fs >= %.1fs at iter %d", elapsed, args.max_wall_sec, it)
            break

        if it % args.log_interval == 0:
            now = time.time()
            recent.append((it, now))
            fi, ft = recent[0]
            ips = (it - fi) / max(now - ft, 1e-9) if it > fi else (it + 1) / max(now - t_start, 1e-9)
            log_metrics(
                {
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm_t.item(),
                    "lr_muon": args.muon_lr * frac,
                    "lr_adam": args.adam_lr * frac,
                    "lr_embed": args.embed_lr * frac,
                    "ips_window": ips,
                    "tokens_per_sec": ips * cur_bs * args.context_length,
                    "elapsed_sec": now - t_start,
                },
                step=it, wandb_active=wandb_active,
            )

        if val_tokens is not None and it > 0 and it % args.val_interval == 0:
            log_metrics({"val/loss": eval_maybe_ema(model, lm_weight, val_tokens, args, ema, it)},
                        step=it, wandb_active=wandb_active)

        if it > 0 and it % args.checkpoint_interval == 0:
            save_checkpoint(model, muon, it, os.path.join(args.checkpoint_dir, f"iter_{it:07d}.pt"))

    final_iter = last_it + 1
    save_checkpoint(model, muon, final_iter, os.path.join(args.checkpoint_dir, f"iter_{final_iter:07d}_final.pt"))
    if ema is not None:
        # save the EMA shadow too: checkpoints hold RAW weights (save is outside
        # ema.averaged()), but the REPORTED final val uses EMA — without this file the
        # reported model was unrescorable offline (canon scored a different model).
        inner_ = getattr(model, "_orig_mod", model)
        torch.save({n: s for (n, _), s in zip(inner_.named_parameters(), ema.shadow)},
                   os.path.join(args.checkpoint_dir, "ema_final.pt"))
        logger.info("EMA shadow saved: ema_final.pt")
    total = time.time() - t_start
    logger.info("done: stop=%s elapsed=%.1fs iters=%d", stop_reason, total, final_iter)
    if args.compile and args.compile_cache:
        try:  # save AFTER the run so the next run warm-starts (safe: post-wall-clock)
            artifacts = torch.compiler.save_cache_artifacts()
            if artifacts is not None:
                with open(args.compile_cache, "wb") as f:
                    f.write(artifacts[0])
                logger.info("compile cache saved to %s (%.1f MB)",
                            args.compile_cache, len(artifacts[0]) / 1e6)
        except Exception as e:  # noqa: BLE001
            logger.warning("compile cache save failed: %r", e)
    if val_tokens is not None:
        final_val = eval_maybe_ema(model, lm_weight, val_tokens, args, ema, final_iter)
        log_metrics({"val/loss": final_val, "elapsed_sec": total},
                    step=final_iter, wandb_active=wandb_active)


def main() -> None:
    args = parse_args()
    # Sign-encode the softcap form: sigmoid form is passed to the CE as a NEGATIVE
    # softcap (apply_softcap / the fused kernel read the sign as the form selector).
    # Every CE call site uses args.logit_softcap, so mutate it once here -> train and
    # in-loop eval stay in lockstep (canonical eval mirrors this via --softcap-form).
    if args.softcap_form == "sigmoid" and args.logit_softcap > 0:
        args.logit_softcap = -abs(args.logit_softcap)
    setup_logging(args.log_level, args.log_file)
    logger.info("args: %s", vars(args))
    try:
        train(args)
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
