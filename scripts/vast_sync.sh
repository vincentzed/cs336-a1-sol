#!/usr/bin/env bash
# Sync code/data to a Vast.ai (or any ssh-reachable) pod, and pull checkpoints back.
#
# Setup (once per pod, after clicking RENT on vast.ai):
#   export POD_HOST=ssh9.vast.ai POD_PORT=12345 POD_USER=root
#   export POD_PATH=/root/assignment1-basics      # optional; default below
#
# Usage:
#   ./vast_sync.sh push-code          # rsync source + configs
#   ./vast_sync.sh push-data-ts       # rsync ts_{train,val}.npy (~1 GB)
#   ./vast_sync.sh push-data-owt      # rsync owt_{train,val}.npy (~5.5 GB)
#   ./vast_sync.sh pull-ckpts         # rsync checkpoints back to ./checkpoints/sweep/
#   ./vast_sync.sh ssh                # ssh into the pod
#   ./vast_sync.sh setup              # run `pip install -e .` on the pod
#   ./vast_sync.sh wandb-login        # propagate local wandb key to the pod
#   ./vast_sync.sh nvidia-smi         # quick sanity check
#   ./vast_sync.sh sweep              # full 5000-step LR sweep (wandb: transformer-lm-lr-sweep)
#   ./vast_sync.sh sweep-divergence   # high LRs, no grad clip — probe the divergence cliff
#   ./vast_sync.sh sweep-const-lr     # constant LR, no clip, dense val+samples (transformer-lm-lr-cliff)
#   ./vast_sync.sh all                # push-code + push-data-ts + setup

set -euo pipefail

: "${POD_HOST:?set POD_HOST (from vast.ai ssh command)}"
: "${POD_PORT:?set POD_PORT}"
: "${POD_USER:=root}"
: "${POD_PATH:=/root/transformer-lm-from-scratch}"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH="ssh -p $POD_PORT $POD_USER@$POD_HOST"
RSYNC_SSH="ssh -p $POD_PORT"

cmd="${1:-}"

case "$cmd" in
  push-code)
    echo ">> pushing code to $POD_USER@$POD_HOST:$POD_PATH"
    rsync -avz --progress \
      -e "$RSYNC_SSH" \
      --exclude='.venv/' \
      --exclude='.git/' \
      --exclude='.claude/' \
      --exclude='__pycache__/' \
      --exclude='.DS_Store' \
      --exclude='checkpoints/' \
      --exclude='notebooks/' \
      --exclude='data/' \
      --exclude='data2/' \
      --exclude='*.ipynb' \
      "$REPO_DIR/" "$POD_USER@$POD_HOST:$POD_PATH/"
    ;;

  push-data-ts)
    echo ">> pushing TinyStories tokenized data"
    $SSH "mkdir -p $POD_PATH/data"
    rsync -avz --progress \
      -e "$RSYNC_SSH" \
      "$REPO_DIR/data/ts_train.npy" "$REPO_DIR/data/ts_val.npy" \
      "$POD_USER@$POD_HOST:$POD_PATH/data/"
    ;;

  push-data-owt)
    echo ">> pushing OpenWebText tokenized data (~5.5 GB, be patient)"
    $SSH "mkdir -p $POD_PATH/data"
    rsync -avz --progress \
      -e "$RSYNC_SSH" \
      "$REPO_DIR/data/owt_train.npy" "$REPO_DIR/data/owt_val.npy" \
      "$POD_USER@$POD_HOST:$POD_PATH/data/"
    ;;

  pull-ckpts)
    echo ">> pulling checkpoints back to ./checkpoints/sweep/"
    mkdir -p "$REPO_DIR/checkpoints/sweep"
    rsync -avz --progress \
      -e "$RSYNC_SSH" \
      "$POD_USER@$POD_HOST:$POD_PATH/ckpt_lr_*/" \
      "$REPO_DIR/checkpoints/sweep/"
    ;;

  ssh)
    exec $SSH
    ;;

  setup)
    echo ">> installing deps on pod (into /venv/main)"
    $SSH "source /venv/main/bin/activate && cd $POD_PATH && pip install -e . && python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'"
    ;;

  wandb-login)
    KEY=$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])" 2>/dev/null)
    if [ -z "$KEY" ]; then
      echo "no wandb key found in ~/.netrc — run 'wandb login' locally first" >&2
      exit 1
    fi
    echo ">> propagating wandb key to pod"
    $SSH "source /venv/main/bin/activate && wandb login --relogin $KEY"
    ;;

  nvidia-smi)
    $SSH "nvidia-smi"
    ;;

  sweep)
    echo ">> launching 8-GPU LR sweep on pod"
    $SSH "cd $POD_PATH && bash -s" <<'REMOTE'
set -euo pipefail
source /venv/main/bin/activate
mkdir -p logs
lrs=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2)
TOTAL=5000
VOCAB=tokenizers/tinystories/vocab.pkl
MERGES=tokenizers/tinystories/merges.pkl
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -m transformer_lm.train_script \
    --train-tokens data/ts_train.npy \
    --val-tokens   data/ts_val.npy \
    --vocab-size 10000 \
    --lr-max ${lrs[$i]} \
    --total-iters $TOTAL \
    --checkpoint-interval $TOTAL \
    --compile \
    --checkpoint-dir ckpt_lr_${lrs[$i]} \
    --sample-interval 1000 \
    --sample-vocab $VOCAB \
    --sample-merges $MERGES \
    --wandb-project transformer-lm-lr-sweep \
    --wandb-run-name lr_${lrs[$i]} \
    > logs/lr_${lrs[$i]}.txt 2>&1 &
done
wait
echo "sweep done"
REMOTE
    ;;

  sweep-divergence)
    # Part (b) sweep: high LRs, grad clip disabled, short runs to see the cliff.
    echo ">> launching 8-GPU divergence sweep (grad clip OFF, high LRs)"
    $SSH "cd $POD_PATH && bash -s" <<'REMOTE'
set -euo pipefail
source /venv/main/bin/activate
mkdir -p logs
# High LRs that probe the stability edge of AdamW without grad clip.
lrs=(1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1.0 3.0)
TOTAL=1000         # divergence shows in first few hundred steps; no need for full training
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -m transformer_lm.train_script \
    --train-tokens data/ts_train.npy \
    --val-tokens   data/ts_val.npy \
    --vocab-size 10000 \
    --lr-max ${lrs[$i]} \
    --total-iters $TOTAL \
    --checkpoint-interval $TOTAL \
    --grad-clip 0 \
    --compile \
    --checkpoint-dir ckpt_div_${lrs[$i]} \
    --wandb-project transformer-lm-lr-sweep \
    --wandb-run-name div_${lrs[$i]} \
    > logs/div_${lrs[$i]}.txt 2>&1 &
done
wait
echo "divergence sweep done"
REMOTE
    ;;

  sweep-const-lr)
    # Clean part (b): constant LR (no warmup, no cosine decay), no grad clip,
    # short runs to see the cliff precisely. Dense val + sample logging for writeup.
    echo ">> launching 8-GPU const-LR cliff sweep (constant LR, no clip, dense val+samples)"
    $SSH "cd $POD_PATH && bash -s" <<'REMOTE'
set -euo pipefail
source /venv/main/bin/activate
mkdir -p logs
# LRs focused on the cliff (based on sweep-divergence observation: cliff ~1e-1 to 3e-1).
lrs=(1e-2 3e-2 1e-1 2e-1 3e-1 5e-1 1.0 3.0)
TOTAL=500
VOCAB=tokenizers/tinystories/vocab.pkl
MERGES=tokenizers/tinystories/merges.pkl
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -m transformer_lm.train_script \
    --train-tokens data/ts_train.npy \
    --val-tokens   data/ts_val.npy \
    --vocab-size 10000 \
    --lr-max ${lrs[$i]} \
    --lr-min ${lrs[$i]} \
    --warmup-iters 0 \
    --cosine-cycle-iters 1000000000 \
    --total-iters $TOTAL \
    --checkpoint-interval $TOTAL \
    --grad-clip 0 \
    --val-interval 50 \
    --val-batches 20 \
    --log-interval 10 \
    --sample-interval 100 \
    --sample-vocab $VOCAB \
    --sample-merges $MERGES \
    --compile \
    --checkpoint-dir ckpt_const_${lrs[$i]} \
    --wandb-project transformer-lm-lr-cliff \
    --wandb-run-name const_${lrs[$i]} \
    > logs/const_${lrs[$i]}.txt 2>&1 &
done
wait
echo "const-lr cliff sweep done"
REMOTE
    ;;

  all)
    "$0" push-code
    "$0" push-data-ts
    "$0" setup
    ;;

  *)
    grep -E '^#' "$0" | sed 's/^# \{0,1\}//' | head -n 20
    exit 1
    ;;
esac
