#!/usr/bin/env bash
# A/B benchmark: BASELINE first (control), then SOL, on the same GPU/box/commit.
#
# Both run from THIS checkout. The baseline path (transformer_lm.train_script) is
# byte-identical to main HEAD / upstream StuffByLiang -- no git revert needed to
# get a clean control, but you can `git checkout main -- transformer_lm/train_script.py`
# to be paranoid. SOL path is transformer_lm.train_sol.
#
# Prereqs:
#   * OWT tokenized as uint16 .npy (train + val) using tokenizers/owt/.
#     e.g. uv run scripts/train_bpe_owt.py  (then your tokenize step)
#   * export WANDB_API_KEY=...   (NEVER commit it; read from env only)
#   * uv sync --extra sol        (flash-attn-4[cu13], cudnn-frontend, etc.)
set -euo pipefail

GPU="${GPU:-7}"                                   # you asked for device 7
WALL="${WALL:-2700}"                              # 45 min = 2700 s
VOCAB="${VOCAB:-32000}"
OWT_TRAIN="${OWT_TRAIN:-data/owt_train.npy}"
OWT_VAL="${OWT_VAL:-data/owt_valid.npy}"
WANDB_PROJECT="${WANDB_PROJECT:-cs336-a1-sol}"
STAMP="${STAMP:-run}"                             # pass a fixed stamp for reproducible names

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARN: WANDB_API_KEY not set -> W&B logging will be disabled." >&2
  WBMODE="disabled"
else
  WBMODE="online"
fi
for f in "$OWT_TRAIN" "$OWT_VAL"; do
  [[ -f "$f" ]] || { echo "ERROR: missing tokenized data '$f' (see prereqs)"; exit 1; }
done

echo "=== GPU $GPU | wall ${WALL}s | vocab $VOCAB | wandb=$WBMODE ==="

# ---------------------------------------------------------------- 1) BASELINE
echo "### [1/3] BASELINE (train_script) ###"
CUDA_VISIBLE_DEVICES="$GPU" uv run --no-sync python -m transformer_lm.train_script \
  --train-tokens "$OWT_TRAIN" --val-tokens "$OWT_VAL" --vocab-size "$VOCAB" \
  --d-model 768 --num-layers 12 --num-heads 12 --d-ff 2048 --max-seq-len 1024 \
  --batch-size 320 --context-length 1024 --total-iters 100000 --max-wall-sec "$WALL" \
  --lr-max 2.5e-3 --lr-min 2.5e-4 --warmup-iters 500 --weight-decay 0.1 \
  --dtype bfloat16 --compile \
  --checkpoint-dir "checkpoints/baseline_${STAMP}" \
  --wandb-project "$WANDB_PROJECT" --wandb-mode "$WBMODE" --wandb-run-name "baseline_${STAMP}"

# SOL defaults ON: ReLU^2 FFN, QK-norm, tied embeddings, logit-softcap 30, z-loss 1e-4,
# NorMuon-on-GNS + fused-AdamW (embed higher LR, no-wd norms), chunked-CE, FA4 sm100.
SOL_COMMON=(--train-tokens "$OWT_TRAIN" --val-tokens "$OWT_VAL" --vocab-size "$VOCAB"
  --d-model 768 --num-layers 12 --num-heads 12 --d-ff 2048 --max-seq-len 1024
  --batch-size 512 --context-length 1024 --total-iters 100000 --max-wall-sec "$WALL"
  --muon-lr 2e-3 --adam-lr 3e-3 --embed-lr 6e-3 --warmup-iters 500 --weight-decay 0.1
  --attn-backend fa4 --rmsnorm torch --ce-chunk 32768 --dtype bfloat16 --compile
  --wandb-project "$WANDB_PROJECT" --wandb-mode "$WBMODE")

# ------------------------------------------------------------- 2) SOL (WSD LR)
echo "### [2/3] SOL — WSD schedule ###"
CUDA_VISIBLE_DEVICES="$GPU" uv run --no-sync python -m transformer_lm.train_sol "${SOL_COMMON[@]}" \
  --lr-schedule wsd --checkpoint-dir "checkpoints/sol_wsd_${STAMP}" --wandb-run-name "sol_wsd_${STAMP}"

# --------------------------------------------------- 3) SOL (cosine LR ablation)
echo "### [3/3] SOL — cosine schedule (LR ablation vs WSD) ###"
CUDA_VISIBLE_DEVICES="$GPU" uv run --no-sync python -m transformer_lm.train_sol "${SOL_COMMON[@]}" \
  --lr-schedule cosine --checkpoint-dir "checkpoints/sol_cos_${STAMP}" --wandb-run-name "sol_cos_${STAMP}"

echo "=== done: compare val/loss @ 45min: baseline_${STAMP} vs sol_wsd_${STAMP} vs sol_cos_${STAMP} ==="
