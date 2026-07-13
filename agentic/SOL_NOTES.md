# SOL notes — OWT 45-min run on B200/B300 (sm100, CUDA 13.3)

Goal: minimise OWT validation loss inside a 45-min single-GPU budget by pushing
each op toward its **speed-of-light** roofline — `max(FLOPs/peak_FLOPs,
bytes/peak_HBM_BW)`. Compute-bound ops (GEMM, attention) are already near SOL via
vendor kernels; the wins are in the memory-bound ops and the optimizer.

## Ground rules (project direction)

- **NVIDIA-endorsed kernels only** where we leave eager PyTorch: cuBLAS (matmul),
  cuDNN (attention), CuTe-DSL / CUTLASS (FlashAttention-4), CUDA multi-tensor Adam.
- **No third-party Triton kernel libraries** — no Liger, no cut-cross-entropy.
  `torch.compile`/Inductor stays on (baseline already uses it) for the
  elementwise/reduction fusions; that is the sanctioned way to get fused
  RMSNorm/RoPE/SwiGLU without hand-writing or vendoring Triton.
- **This is training, not serving.** Every kernel must have a backward.
  `trtllm_mha` / flashinfer decode kernels are **inference-only** (KV-cache, page
  tables, no autograd) and are deliberately excluded — they are exactly what the
  serving benchmark on this box uses, and they produce no gradients.

## Op-by-op

| Op (baseline loc) | Baseline | Bound | At SOL? | SOL action (this branch) | NVIDIA/CuTe reference |
|---|---|---|---|---|---|
| **Cross-entropy** (`modules.py:440`, called outside `compile` at `train_script.py:278`) | eager softmax over a materialised `(B·T, 32000)` logits tensor ≈ **21 GB bf16 / 42 GB fp32** — largest alloc in the step | memory | ❌ far | **chunked fused linear+CE** (`sol_modules._ChunkedLinearCE`): walk token rows in chunks, recompute logits in backward; peak logits = `(chunk, vocab)`. Pure torch (cuBLAS + softmax). Unlocks larger batch. | Megatron `fused_cross_entropy` is **vocab-parallel and still materialises logits** (TP, not this trick) → we roll our own |
| **Optimizer** (`modules.py:453` hand-rolled AdamW python loop) | per-param python loop, eager, not compiled | latency + memory | ❌ | **Muon** on 2-D matmul params (Newton-Schulz = cuBLAS bf16 matmuls) + **fused CUDA AdamW** (`torch.optim.AdamW(fused=True)`) on embeddings/head/1-D | modded-nanogpt Muon; CUDA multi-tensor Adam |
| **Attention** (`modules.py:347` `F.sdpa(is_causal=True)`) | fused SDPA (backend auto) | compute | ~ | **FlashAttention-4 CuTe-DSL sm100 fwd+bwd** (`flash_attn_func`, default); alternates: cuDNN (`SDPBackend.CUDNN_ATTENTION`, latest cudnn-frontend) and torch SDPA | `flash_attn/cute/flash_{fwd,bwd}_sm100.py`; cudnn-frontend v1.26/1.27 `sdpa/{fwd,bwd}` |
| **grad clip** (`train_script.py:285`) | python loop + `.item()` **every step** → host sync bubble | latency | ❌ | `clip_grad_norm_(foreach=True)`, `.item()` only at log interval | torch foreach |
| **Dataloading** (`modules.py:528` `get_batch`) | synchronous `.to(device)`, no pin/prefetch → GPU stalls each step | H2D | ❌ | `CudaPrefetcher`: pinned memory + side-stream copy, double-buffered | standard NVIDIA prefetch pattern |
| **RMSNorm** (`modules.py:43`) | eager fp32 upcast, under `torch.compile` | memory | ~ | Inductor already fuses it; optional **TE fused RMSNorm** (`--rmsnorm te`) for the hand-tuned CUDA kernel | TE `RMSNorm`; Megatron `FusedLayerNorm` |
| **RoPE** (`modules.py:157`) | `stack`+`flatten` materialisation each step, under `compile` | memory | ~ | left to Inductor for now; phase-2 = fused RoPE kernel | TE `apply_rotary_pos_emb(fused=True)`; Megatron `fused_apply_rotary_pos_emb` (NOT `fused_mla_yarn_rope_apply` — that is MLA-only) |
| **SwiGLU FFN** (`modules.py:102`) | 3 cuBLAS GEMMs + eager `silu*gate`, under `compile` | GEMM compute-bound; act memory-bound | ~ | GEMMs already cuBLAS; act fused by Inductor. Phase-2 = FP8 GEMMs (TE) | Megatron `fused_bias_swiglu`, `fused_weighted_squared_relu`; TE `LayerNormMLP` |
| **Linear / QKVO GEMMs** (`modules.py:11`) | `x@W.T` → cuBLAS | compute | ✅ | leave on cuBLAS in bf16; only FP8 (TE) beats it | TE `Linear` FP8 |

## Priority order (largest SOL lever first)

1. **Chunked linear-CE** — removes the 21–42 GB logits tensor; the report's own
   post-mortem estimated this is worth ~0.2–0.3 val-loss via bigger batch.
2. **Muon + fused AdamW** — ~1.5–2× steps-to-loss (every top-5 entry used Muon)
   *and* kills the python optimizer loop.
3. **Async prefetch + foreach clip** — removes per-step H2D stall and host syncs.
4. **FA4 sm100 attention** — the CuTe-DSL Blackwell SDPA fwd/bwd; at ctx=1024
   this is a smaller lever than 1–3 (FFN/CE dominate FLOPs) but it is the correct
   training attention kernel and matters at longer ctx.
5. **TE RMSNorm** — modest over Inductor; free.
6. **(phase 2) FP8 GEMMs via TransformerEngine** — the only way past cuBLAS on
   the compute-bound ops (~2× matmul on Blackwell); needs scaling/stability care
   inside a 45-min budget, so it is deliberately gated off by default.

## Why not just "use Megatron"?

Megatron-Core's `megatron/core/fusions/` is the right *interface* reference
(`FusedLayerNorm`, `fused_softmax`, `fused_bias_swiglu`,
`fused_weighted_squared_relu`, `fused_mla_yarn_rope_apply`, `fused_cross_entropy`)
and its kernels are all CUDA/apex/TE — no Triton. But: its cross-entropy is TP
vocab-parallel (still materialises logits), and pulling the full Megatron/TE
stack into a single-GPU 45-min run is heavy integration for little marginal gain
over `Inductor + FA4 + Muon + fused-Adam`. We borrow the *interfaces* (TE
RMSNorm, the fused-op taxonomy) rather than the whole framework.

## Is each op actually SOTA? (honest audit)

| Op | This branch | SOTA? | The actual frontier |
|---|---|---|---|
| Muon orthogonalization | **gram-newton-schulz** (Gram iter + Polar Express coeffs, quack symmetric GEMMs) | **yes, frontier** | Dao-AILab GNS (Jul 2026) is the current fastest; ~2× standard NS. Plain quintic NS (our fallback) is a notch below. |
| Attention | **FA4 CuTe-DSL sm100** fwd/bwd | **yes** | FA4 / cuDNN are co-SOTA for Blackwell training attn; benchmark both for hd64. |
| Cross-entropy | chunked linear-CE (pure torch) | **best under the no-Triton rule** | Unconstrained SOTA is cut-cross-entropy / Liger FLCE, but those are Triton (excluded). Our chunking recomputes logits in bwd (extra FLOPs) — the price of no-Triton. |
| AdamW | fused CUDA multi-tensor | **yes** | Apex FusedAdam is equivalent; nothing meaningfully faster. |
| RMSNorm | Inductor (opt. TE) | near-SOTA | TE fused RMSNorm is the hand-tuned frontier; Inductor is ~90% of it. |
| RoPE | Inductor | **not frontier** | general fused RoPE (TE `apply_rotary_pos_emb(fused=True)`; Megatron `fused_apply_rotary_pos_emb`) left on the table. NB: `fused_mla_yarn_rope_apply` is MLA-only — not applicable to standard MHA RoPE. |
| FFN GEMMs | bf16 cuBLAS | SOTA for bf16 | **FP8 (TE)** is the real frontier — deferred to phase 2 (stability). |

Net: Muon and attention are now genuinely frontier; CE is frontier *within the
no-Triton constraint*; RoPE and FP8-GEMM are the two knowingly-left levers.

## torch bwd/optimizer interfaces worth using (introspected)

Confirmed present and stable (2.8dev → 2.12), highly relevant to SOL:
- **`Tensor.register_post_accumulate_grad_hook`** — run the optimizer step
  per-param *during* backward, then free that grad. "Optimizer-in-backward":
  removes the peak of holding all grads at once. Clean win for the AdamW-side
  params (Muon needs the whole grad matrix, so keep Muon post-backward).
- `AdamW(fused=True, capturable=True)` — fused multi-tensor + CUDA-graph-capturable.
- 88 `torch._foreach_*` primitives (what Muon's batched per-matrix ops build on).
- optimizer step hooks; `torch.func` (grad/jvp/vjp/functional_call) for functional autograd.

## Version matrix (the "latest is critical" pins)

| Component | Pin | Notes |
|---|---|---|
| torch | **2.12** (cu13) | box is CUDA 13.3 / driver 610; install from the cu13 wheel index |
| TransformerEngine | **≥2.16** (v2.16.1 / v2.17) | requires only `torch>=2.1` → **2.12 supported**; already needs cudnn-frontend≥1.25 |
| nvidia-cutlass-dsl | **4.6.0** `[cu13]` | Python≥3.10 ✓; pulls cu13 libs |
| quack-kernels | **0.6.1** | requires cutlass-dsl==4.6.0 (satisfied) |
| gram-newton-schulz | **0.1.6 `--no-deps`** | its `==4.5.2`/quack`==0.5.0` pins are bypassed so 4.6.0 wins |
| flash-attn-4 | wheel `[cu13]`, else **from source** | CuTe-DSL sm100 fwd/bwd |
| nvidia-cudnn-cu13 / -frontend | ≥9.13 / ≥1.26 | latest cuDNN kernels for the CUDNN_ATTENTION SDPA path |

Build order + conflict handling live in `scripts/setup_sol_env.sh` — **run it only
when a GPU is free** (it compiles CuTeDSL/FA4 kernels and verifies on-device).
The Python code just `import`s gram-newton-schulz (falls back to the built-in
quintic Muon until the env is built).
