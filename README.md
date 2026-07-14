# A Speed-of-Light Study on CS336 Assignment 1

*(what forty-five minutes of Blackwell actually buys you, measured rather than imagined)*

This repository is a systems study built on the CS336 Spring 2026 leaderboard task: train a
language model from scratch on OpenWebText, one GPU, forty-five minutes on the wall clock,
lowest validation loss wins. We took the strongest public baseline, replaced every op with
the best kernel NVIDIA's ecosystem offers, wired the modern optimizer literature in
correctly, adopted the architecture tricks the speedrun community has already paid for —
and measured every single step, including the many that failed.

## Results

All numbers are validation loss at context 512, 45:00 wall-clock, one GPU, full OWT corpus,
scored by the **canonical protocol** (deterministic full sweep of every non-overlapping
window of the validation set — `scripts/eval_canon.py` — on the EMA weights the run would
actually submit; see "Measurement" below for why this pedantry exists).

| Model / venue | Val loss | Notes |
|---|---|---|
| **This work — 1× B300** | **2.9497** | L20 + 5-table value embeds + softcap-23 |
| **This work — 1× B200** (leaderboard hardware) | **2.9571** | L14 + d_ff 5632 + 5-table value embeds + per-head gates + softcap-20 |
| Best leaderboard entry ever (Thomas Li, B200) | 3.0354 | private code, public writeup |
| #2 (Nick Rui, B200) | 3.1003 | |
| Public baseline we forked (reproduced) | 3.2500 | their claim: 3.2508 — it reproduces |

Run-to-run noise is ±0.0003 on B200 (four-seed replicates), ±0.004-0.007 on B300. The
margin over the best-ever leaderboard entry on its own hardware is ~0.08 — over two hundred
noise units. The live frontier (every run, every config) is public on
[W&B](https://wandb.ai/vincentzed-university-of-waterloo/cs336-a1-sol).

## What this is not

Read this before quoting the table. The leaderboard's rules are explicit: your code must be
your own, external implementations are off-limits, AI assistance disqualifies. This work
**fails all three tests, on purpose** — it imports FlashAttention-4, quack, and
gram-newton-schulz; it was engineered with heavy AI involvement; it stands on a public
baseline. It is a *measurement of the ceiling*, not an *entry*. If you fork this to put
numbers in that table, the dishonesty is yours. A leaderboard number certifies the work
behind it; keep it that way.

## The recipe (current best)

**Model** — d_model 1024, 14-20 layers (14 is the measured optimum on B200; the
compute-optimal depth moved twice as throughput improved), 8 heads (head_dim 128), d_ff
5632 (up from the inherited 4096 — the wider FFN costs only ~12% throughput on B200 and
bought the final −0.009; the optimum is not monotone, 4864 loses), ReLU² FFN, ctx 512,
32k vocab, tied embeddings. QK-norm, learnable attention scale,
attention output gate, partial RoPE (0.5). **Five value-embedding tables** (0.01·randn,
gated into attention V at the first/last layers, modded-nanogpt mapping) — the single
largest architecture lever we measured (−0.025); per-head data-dependent gates add a
further −0.004. x0 re-injection, U-net skips with backout, smear, second input embedding,
XSA — each flag-gated, each ablated, each individually retained on evidence. Logit softcap
`20·tanh(z/20)` fused into the CE kernel (−0.01; the cap's presence matters, its exact
value and shape do not — we tested 20/23/26/30 and the evolved sigmoid form).

**Optimizer** — NorMuon on a Gram-Newton-Schulz orthogonalizer (Polar Express coefficients,
batched by shape, quack symmetric CuTeDSL GEMMs) for 2-D weights at lr 8e-3; fused AdamW
elsewhere (embeddings 2.4e-2; VE tables get their own group with betas (0.75, 0.95));
cautious weight decay; bf16 mantissa-tracking master weights; momentum warmup 0.85→0.95;
**no gradient clipping**. WSD schedule with **80% of the run in linear decay** to 6.7%.
A 60-trial Optuna study (schedule-complete miniatures, TPE) confirmed this point is a broad
plateau — the search validated the config rather than improving it, which is itself worth
knowing.

**Systems** — quack CuTeDSL fused linear-cross-entropy (the decisive kernel: 681→277 ms/step
against our first honest implementation), with softcap added to the vendored kernel (+0.5 ms);
FlashAttention-4 wrapped as a proper `torch.library.custom_op` (fwd *and* bwd registered —
AOTAutograd silently drops untraceable side-effects in custom-op backwards, a bug that cost
us a day) so the whole model compiles as **one graph**; fused QKV projection + in-place
packed CuTeDSL rope; quack RMSNorm; a fully-async loader (persistent pinned double-buffers,
background sampling thread, int32 wire format — host idle measured at 0.9%); foreach EMA.
`torch.compile(dynamic=False)`; batch 256×512 (measured optimum — the throughput knee is
flat above it and steps beat tokens below it); shuffled-without-replacement window sampling;
wall-clock-anchored LR schedule for node-speed robustness. Measured MFU ≈ 26% of B300's
nominal bf16 peak, which is roughly half of what our shape mix can theoretically reach —
the remainder is small-GEMM physics, not sloppiness.

## What actually mattered (ranked by measured effect)

1. **Model size and shape** (~0.3): the 91M baseline skeleton was pinned above 3.3 no matter
   how fast we made it. One configuration line outweighed the entire kernel campaign, and we
   learned it from reading the top entries' writeups — *after* a day of polishing the wrong
   model. The optimum then kept moving (16→20→14-16 layers) as our throughput improved:
   compute-optimal allocation is a moving target that tracks your own engineering.
2. **Letting the schedule finish** (~0.05): a step-indexed schedule that wall-stops mid-decay
   silently costs more than almost any optimization buys. Calibrate iterations to measured
   throughput, or anchor the schedule to the clock.
3. **The optimizer stack** (~0.05 cumulative): NorMuon+GNS, hygiene, momentum shape, no-clip.
4. **Architecture** (~0.04): value embeddings dominate; the cheap-neutral five add up;
   most of the famous tricks individually do little (their originators say the same).
5. **Kernels** (~0.02-0.03): real, measured, and the smallest entry on this list — the
   scaling law compresses throughput into loss logarithmically. The kernels' true value is
   *enabling* the bigger model at the same wall-clock, not the direct loss delta.

## The graveyard (causes of death recorded)

Fullgraph and CUDA-graph capture (~1%: the step is kernel-bound; graph topology taxes
microseconds against 100 ms kernels). Batched-Muon-as-speedup (the optimizer was 1% of the
step; we imported another project's bottleneck). MXFP8 via quack 0.5.3 (the fp8 GEMM itself
is 1.32× cuBLAS; the pure-torch quantizer feeding it runs at 6% of HBM bandwidth — net
0.16×). MXFP8 via TransformerEngine (built and benched on B200 after five attempts: 1.25×
on the GEMM mix ≈ ~9% end-to-end — real, but unadopted at this margin in our setup).
Doc-aligned (BOS) and doc-packed (varlen) data (+0.08/+0.15: the eval samples random dense
windows; train/eval distribution match beats aesthetics — the packing that helps
modded-nanogpt helps because *their* val is packed too). Multi-token prediction. Batch
ramps, batch 128, batch 512. Deeper (24L) and wider (1152) at the old config; deeper again
(L18+) at the new one. 16 heads. RoPE theta 2048. Frozen norm gains. Sigmoid softcap form.
EMA horizons other than 0.999 (0.9995 is a disaster, 0.9985 a wash). Decaying the LR tail
all the way to zero. Muon momentum decay-back over the last 2k steps.
Seven value-embedding tables (a throughput bug made it moot; five beats three). A "better"
tokenizer (measured: the legal direction *worsens* the leaderboard number, the gaming
direction improves it only by shrinking the yardstick — bits/byte gets worse; documented in
`agentic/TOKENIZER_STUDY.md` and left alone).

## Measurement (the part we'd emphasize to anyone attempting this)

Upstream defines no evaluation protocol beyond "context length 512" — the baseline (and,
initially, we) sampled val batches off the *run-seeded* RNG, which lets seed selection
silently select easy validation subsets. Our reported numbers went through three eras:
per-run-random subsample → fixed-window subsample → **deterministic full sweep** (every
non-overlapping ctx-512 window, token-weighted, nothing to sample or select), applied to the
**EMA weights** (which the run actually reports; they beat raw weights by 0.002-0.005 and
were, for a while, not even saved — the reported model didn't exist on disk). Softcap is
part of the model's forward and must be applied at eval (+0.08 mis-score otherwise). Every
comparison in this repo pins weights × protocol × head × kernel path; we verified the score
is bit-stable across attention backends. The canonical record book lives in
[`results/CANON.md`](results/CANON.md).

## Reproduction

```bash
bash scripts/setup_sol_env.sh      # torch 2.12 cu130 + cuDNN 9.24 + quack + GNS + FA4
python scripts/tokenize_owt.py     # full OWT -> uint16 .npy (~2.7B tokens)
python -m transformer_lm.train_sol \
  --train-tokens data/owt_train_full.npy --val-tokens data/owt_valid.npy --vocab-size 32000 \
  --d-model 1024 --num-layers 14 --num-heads 8 --d-ff 5632 \
  --max-seq-len 512 --context-length 512 --batch-size 256 \
  --total-iters 10400 --max-wall-sec 2700 --warmup-iters 500 --schedule-by-wall \
  --lr-schedule wsd --wsd-decay-frac 0.8 --lr-min-ratio 0.067 \
  --muon-lr 8e-3 --adam-lr 1.2e-2 --embed-lr 2.4e-2 --weight-decay 0.1 \
  --ce-mode quack-softcap --logit-softcap 20 --z-loss 0 \
  --attn-backend fa4op --rmsnorm quack --fused-rope --fused-qkv \
  --muon-momentum-warmup 300 --grad-clip 0 --cautious-wd --bf16-mt \
  --value-embeds-k 5 --ve-gates per-head \
  --x0-lambdas --unet-skips --smear --second-embed --xsa \
  --data-sampling shuffled --eval-ctx 512 --compile \
  --val-interval 1000000 --checkpoint-interval 1000000
# score what you'd submit:
python scripts/eval_canon.py CKPT --softcap 20 --ema ckpt/.../ema_final.pt \
  --num-layers 14 --d-ff 5632 --value-embeds-k 5 --ve-gates per-head \
  --x0-lambdas --unet-skips --smear --second-embed --xsa
```

Calibrate `--total-iters` to your silicon with a 150-step probe first; `--schedule-by-wall`
protects the decay tail against node variance either way. `modal_bench.py` fans the same
experiments across cloud B200s (up to ten at a time in our runs) if you'd rather rent your
mistakes than queue for them. `agentic/SOL_NOTES.md` holds the op-by-op kernel reasoning;
`agentic/SOL_TODO.md` holds the full ledger of verdicts.

## Debts

**StuffByLiang's** [transformer-lm-from-scratch](https://github.com/StuffByLiang/transformer-lm-from-scratch)
(the baseline, whose published number survived our attempt to disprove it — the model,
tokenizer, and training-loop skeleton in `transformer_lm/` are theirs, see their repo for
the original README); **KellerJordan's modded-nanogpt** and its community, the closest
thing this niche has to a peer-reviewed literature; **Thomas Li and Nick Rui**, whose
leaderboard writeups redirected this entire effort with two paragraphs of candor;
**Dao-AILab's** quack, flash-attention, and gram-newton-schulz. The failures above are
ours; several of the successes are theirs, rented. A leaderboard is a conversation — this
document tries to pay it forward.
