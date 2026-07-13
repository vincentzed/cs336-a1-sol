# SOL checklist (from the Fable 5 audit)

Status: [x] done · [~] in progress / this branch · [ ] todo · [A] to ablate · [L8r] deferred

## Adopting now (confirmed)
- [~] **NorMuon on GNS** — GNS gives fast orthogonalization (speed); NorMuon adds
  per-row 2nd-moment normalize (loss-per-step). Compose via our own Muon loop
  calling `GramNewtonSchulz` (not GNS's `Muon` class).
- [~] **GNS+compile fix** — construct `GramNewtonSchulz(compile_kwargs=None)`;
  its default `mode="reduce-overhead"` (CUDA graphs) breaks on Blackwell.
- [~] **Param-group hygiene** — separate (higher) LR for embeddings; **no weight
  decay on norms / 1-D params**; Muon on 2-D matmul weights only.
- [~] **ReLU² FFN** — replaces SwiGLU; better loss/step + drops gate matrix
  (→ smaller model, bigger batch); maps to Megatron `fused_weighted_squared_relu`.
- [~] **QK-norm** — param-free RMS over head_dim on q,k before SDPA; lets LR go higher.
- [~] **Logit softcap + z-loss** — final-logit `cap*tanh(logits/cap)` + z-loss,
  both folded into the chunked CE (cheap, kernel-agnostic).
- [~] **Tied embeddings** — share token_embeddings ↔ lm_head; frees ~25M params.
- [A]  **LR schedule: WSD/trapezoidal** — add alongside cosine and **ablate** (only
  this one gets an A/B; the rest we adopt outright).
- [~] **cuDNN pin → 9.24.0.43** (latest; ABI-compatible over torch's bundled 9.20).

## Biggest lever — its own pass (not this commit)
- [L8r] **MXFP8 training** — bf16→MXFP8 GEMMs on sm100 tcgen05 (~2× GEMM, ~20-30%
  e2e → more tokens → lower loss). All our GEMM dims 128-aligned → eligible.
  Two routes: **(a) quack `GemmSm100` blockscaled + `quack/blockscaled/quantize.py`
  (CuTeDSL, uses NVIDIA RCEIL recipe arXiv:2506.08027 — round-UP scales for bf16
  parity; torchao's FLOOR default hurts grads)**, or (b) TE (cuDNN/CUDA, NOT
  CuTeDSL). Kernels exist; must-build = autograd wiring + fwd/bwd quant scaling.
  Do after the bf16 SOL run is A/B'd. Highest value, highest effort.

## CuTeDSL kernel opportunities (from the CuTeDSL study — quack is the trove)
- [L8r] **Fused GEMM+ReLU² via quack `gemm_act`/`gemm_dact`** (`ActActivation.relu_sq`)
  — fuses activation into the GEMM epilogue + fused backward (`gemm_norm_act` also
  folds the pre-norm). Net-new, kernels exist. Medium wall-clock (beats Inductor
  esp. on bwd). Lower effort than MXFP8.
- [ ] **quack `linear_cross_entropy`** (CuTeDSL online-softmax) — faster CE inner
  than our torch chunked-CE. CAVEAT: no softcap/z-loss → benchmark vs ours first
  (the x@Wᵀ matmul likely dominates, so delta may be small); adopt only if it wins
  AND we can fork in softcap/z-loss.
- [skip] quack rmsnorm (PyTorch already vendors `torch/_vendor/quack/rmsnorm.py` →
  Inductor may already route RMSNorm to CuTeDSL on Blackwell), softmax/layernorm
  (Inductor, memory-bound), cudnn-frontend CuTeDSL ops (attention/MoE-sparse only —
  relevant ONLY if we adopt sliding-window/block-sparse attn: `bsa_bwd_sm100.py`).
  TE has zero CuTeDSL. No upstream NVIDIA CuTeDSL linear-CE (quack's is canonical).

## Lower priority / infra
- [ ] **Optimizer-in-backward** (`register_post_accumulate_grad_hook`) for the
  AdamW-side params → frees grad memory in bwd (bigger batch). Keep Muon post-bwd.
- [ ] **fp32 master weights or stochastic rounding** for bf16 params (precision leak).
- [ ] **Attention backend**: benchmark FA4 (beta 4.0.0b21) vs cuDNN on-device;
  likely default **cuDNN** for the scored run (more battle-tested at hd64).
- [ ] **Tune CE `chunk_size`** (currently 32768, untuned).
- [ ] **Cross-doc attention masking** on concatenated OWT (small leak).
- [ ] **Fast BPE prep** — use HF `tokenizers`/tiktoken for the one-time (off-clock)
  tokenization; the CS336 pure-python BPE is slow.
- [ ] **Version watch**: torch 2.13.0 is out (cudnn pin 9.20); TE 2.17; FA4 beta only.

## auto-nanogpt audit — decision (2026-07-11)
- [x] **EMA-for-eval** (Polyak, fp32 shadow, decay 0.999, val-only swap, restore
  bit-identical) — ADOPTED. The one useful item.
- [rej] Aurora, Contra-Muon, Muon-VS, NorMuon-lite, Soft-Muon, radial damping,
  LACV, per-role LR recalibration, momentum warmup/cooldown, SOAP/KL-Shampoo —
  DECLINED (user: not useful; Aurora explicitly rejected). Not implementing.

## Arch loss-per-step levers (flagged by audit, NOT scoped — invasive)
- [L8r] u-net skips, backout, smear, XSA, second embedding, attention gates,
  value embeddings. High cumulative loss impact; revisit after the above land.

## Next iteration — user directives (2026-07-11), after the parallel bench
1. [ ] **Replace all RMSNorm with quack's CuTeDSL rmsnorm** (fwd+bwd). Test it
   (earlier marked skip because Inductor handles it — user wants it explicitly).
2. [ ] **Triton ban LIFTED for nanoGPT kernels only** — pull modded-nanogpt
   `triton_kernels.py` fusions WHERE the fusion is clearly useful:
   FusedLinearReLUSquare, FusedSoftcappedCrossEntropy (has softcap built-in →
   compare vs our chunked CE), XXT/XTX/ba+cAA (Muon/GNS polar helpers). Only
   nanoGPT-origin Triton is allowed; no other third-party Triton libs.
3. [ ] **MXFP8: implement via BOTH quack AND TE, benchmark both. Do NOT use the
   v0/initial impl of either** — use the mature path (quack blockscaled GEMM w/
   RCEIL recipe; TE current-scaling MXFP8). (TE needs a working build — its ep.cpp
   vs box NCCL headers blocked it; revisit / disable EP for the MXFP8 build.)
4. [x] **Use parallel GPUs** — bench now runs baseline+SOL-wsd+SOL-cos concurrently
   on GPU 7/6/5 (scripts/bench_sol_parallel.sh). Fan future ablations across GPUs
   (<=4 at a time per shared-box etiquette unless told otherwise).

## Measured verdicts (2026-07-11, scripts/local_profile.py on B300, batch 320)
- chunked CE: fwd=68 ce=126 bwd=478 opt=7.5 total=681ms (~481K tok/s)
- quack CE:   fwd=68 ce=13.5 bwd=188 opt=6.4 total=277ms (~1.18M tok/s) ← the fix
- [DEAD] fullgraph/native-SDPA: ~0 gain. [DEAD] batched-Muon: ~0 gain (opt was 1% of step).
- [DEAD] allow_in_graph(FA4) + fa4op custom-op: DLPack crash (CuTeDSL untraceable).
- [BUG-FIXED] quack 0.6.1 CE sig has bias positional: (x, weight, None, target, ...).
- [BUG-FIXED] all runs pre-r2 used total-iters=100000 + wall stop → LR never decayed.
