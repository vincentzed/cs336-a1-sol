#!/usr/bin/env bash
# TinyStories training run, tuned for Apple Silicon (MPS).
# Token budget: 32 * 5000 * 256 = ~41 M tokens. Val loss at step 5000 ≈ 1.80
# on an M4 Max after ~36 min. Good smoke test before running the bigger OWT config.
#
# Usage:
#   ./scripts/run_train.sh                   # local, no wandb
#   ./scripts/run_train.sh --wandb           # also logs to W&B project transformer-lm
set -euo pipefail

WANDB_ARGS=""
if [[ "${1:-}" == "--wandb" ]]; then
  WANDB_ARGS="--wandb-project transformer-lm --wandb-run-name ts-$(date +%Y%m%d-%H%M%S)"
fi

uv run python -m transformer_lm.train_script \
  --train-tokens data/ts_train.npy \
  --val-tokens   data/ts_val.npy \
  --vocab-size   10000 \
  --d-model      512 \
  --num-layers   4 \
  --num-heads    16 \
  --d-ff         1344 \
  --max-seq-len  256 \
  --rope-theta   10000 \
  --context-length 256 \
  --batch-size   32 \
  --total-iters  5000 \
  --lr-max       1e-3 \
  --lr-min       1e-4 \
  --warmup-iters 200 \
  --weight-decay 0.1 \
  --beta1        0.9 \
  --beta2        0.95 \
  --eps          1e-8 \
  --grad-clip    1.0 \
  --log-interval      20 \
  --val-interval      500 \
  --val-batches       20 \
  --checkpoint-interval 1000 \
  --checkpoint-dir    checkpoints/tinystories \
  --device       mps \
  --dtype        float32 \
  --compile \
  --seed         0 \
  $WANDB_ARGS
