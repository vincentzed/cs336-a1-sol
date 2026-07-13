#!/usr/bin/env bash
# Parallel A/B: baseline + SOL-wsd + SOL-cosine concurrently on 3 GPUs (~45 min
# total instead of ~135 sequential). Each run pinned to its own GPU + log file.
# Requires the tokenized data + the SOL env (setup_sol_env.sh). Reads WANDB_API_KEY.
set -uo pipefail

WALL="${WALL:-2700}"; VOCAB="${VOCAB:-32000}"; STAMP="${STAMP:-r1}"
OWT_TRAIN="${OWT_TRAIN:-data/owt_train.npy}"; OWT_VAL="${OWT_VAL:-data/owt_valid.npy}"
WP="${WANDB_PROJECT:-cs336-a1-sol}"
WB=online; [[ -z "${WANDB_API_KEY:-}" ]] && WB=disabled
# GPUs to use (<=4 per shared-box etiquette): baseline, sol-wsd, sol-cosine
G_BASE="${G_BASE:-7}"; G_WSD="${G_WSD:-6}"; G_COS="${G_COS:-5}"

COMMON="--train-tokens $OWT_TRAIN --val-tokens $OWT_VAL --vocab-size $VOCAB \
  --d-model 768 --num-layers 12 --num-heads 12 --d-ff 2048 --max-seq-len 1024 \
  --context-length 1024 --total-iters 100000 --max-wall-sec $WALL \
  --warmup-iters 500 --weight-decay 0.1 --dtype bfloat16 --compile \
  --wandb-project $WP --wandb-mode $WB"

echo "=== parallel bench: baseline(GPU$G_BASE) sol_wsd(GPU$G_WSD) sol_cos(GPU$G_COS) | wall ${WALL}s wandb=$WB ==="

CUDA_VISIBLE_DEVICES=$G_BASE uv run --no-sync python -m transformer_lm.train_script $COMMON \
  --batch-size 320 --lr-max 2.5e-3 --lr-min 2.5e-4 \
  --checkpoint-dir "checkpoints/baseline_${STAMP}" --wandb-run-name "baseline_${STAMP}" \
  > "logs/baseline_${STAMP}.log" 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES=$G_WSD uv run --no-sync python -m transformer_lm.train_sol $COMMON \
  --batch-size 512 --muon-lr 2e-3 --adam-lr 3e-3 --embed-lr 6e-3 --attn-backend fa4 --lr-schedule wsd \
  --checkpoint-dir "checkpoints/sol_wsd_${STAMP}" --wandb-run-name "sol_wsd_${STAMP}" \
  > "logs/sol_wsd_${STAMP}.log" 2>&1 &
P2=$!

CUDA_VISIBLE_DEVICES=$G_COS uv run --no-sync python -m transformer_lm.train_sol $COMMON \
  --batch-size 512 --muon-lr 2e-3 --adam-lr 3e-3 --embed-lr 6e-3 --attn-backend fa4 --lr-schedule cosine \
  --checkpoint-dir "checkpoints/sol_cos_${STAMP}" --wandb-run-name "sol_cos_${STAMP}" \
  > "logs/sol_cos_${STAMP}.log" 2>&1 &
P3=$!

wait $P1; echo "baseline exit=$?"
wait $P2; echo "sol_wsd exit=$?"
wait $P3; echo "sol_cos exit=$?"
echo "ALL_RUNS_DONE — final val/loss:"
for r in baseline sol_wsd sol_cos; do
  echo "  $r: $(grep -oE 'val/loss=[0-9.]+' "logs/${r}_${STAMP}.log" | tail -1)"
done
