"""Canonical eval: deterministic FULL SWEEP of the validation set at ctx 512.

Upstream defines no protocol beyond "validate at context length 512" — the
leaderboard numbers are self-reported subsample estimates (the public baseline
samples 20 random batches off the run-seeded global RNG; so did we, until
c06a0ef). This script removes every degree of freedom: all non-overlapping
ctx-512 windows of the val set, in order, token-weighted mean CE, bf16, raw or
EMA weights as stored. Nothing to sample, nothing to seed, nothing to select.

Usage (from repo root, venv python):
  CUDA_VISIBLE_DEVICES=N python scripts/eval_canon.py CKPT_PATH \
      --num-layers 20 [--d-model 1024 --num-heads 8 --d-ff 4096] [--value-embeds ...]
Extra model flags are forwarded to train_sol's parser (arch flags must match the ckpt).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

E = "/tmp/claude-1012/-home-brayden/49c069ec-6b97-4caf-8b5f-c5a81a8dd40f/scratchpad/sol_exp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--val-tokens", default=f"{E}/data/owt_valid.npy")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ema", default=None,
                    help="path to ema_final.pt — score the EMA model (what the run reported)")
    ap.add_argument("--softcap", type=float, default=0.0,
                    help="apply the logit softcap — REQUIRED for softcap-trained models "
                         "(the cap is part of the model forward; plain CE mis-scores them)")
    ap.add_argument("--softcap-form", type=str, default="tanh", choices=["tanh", "sigmoid"],
                    help="must match the value the model was TRAINED with (eval-vs-inference).")
    args, model_flags = ap.parse_known_args()

    from transformer_lm.train_sol import build_model, parse_args
    argv = ["e", "--train-tokens", "x", "--vocab-size", "32000", "--batch-size", "1",
            "--d-model", "1024", "--num-heads", "8", "--d-ff", "4096",
            "--max-seq-len", str(args.ctx), "--logit-softcap", "0", "--z-loss", "0",
            "--rmsnorm", "quack", "--fused-rope", "--fused-qkv", "--wandb-mode", "disabled",
            *model_flags]
    old, sys.argv = sys.argv, argv
    margs = parse_args(); sys.argv = old
    model = build_model(margs)
    sd = torch.load(args.ckpt, map_location="cuda", weights_only=False)["model"]
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()})
    if args.ema:
        shadow = torch.load(args.ema, map_location="cuda", weights_only=False)
        with torch.no_grad():
            for n, p_ in model.named_parameters():
                if n in shadow:
                    p_.copy_(shadow[n].to(p_.dtype))
        print(f"loaded EMA shadow over {sum(1 for n,_ in model.named_parameters() if n in shadow)} params")
    model.eval()
    w = model.lm_weight()

    val = np.load(args.val_tokens, mmap_mode="r")
    n_win = (len(val) - 1) // args.ctx          # non-overlapping, full coverage
    starts_all = np.arange(n_win, dtype=np.int64) * args.ctx
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for b in range(0, n_win, args.batch):
            s = starts_all[b:b + args.batch]
            idx = s[:, None] + np.arange(args.ctx)[None, :]
            x = torch.from_numpy(val[idx].astype(np.int64)).cuda()
            y = torch.from_numpy(val[idx + 1].astype(np.int64)).cuda()
            logits = model(x) @ w.t()
            if args.softcap > 0:
                # Sign-encode the form for the shared helper (matches training exactly).
                from transformer_lm.sol_modules import apply_softcap
                cap = -args.softcap if args.softcap_form == "sigmoid" else args.softcap
                logits = apply_softcap(logits, cap)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                   y.reshape(-1), reduction="sum")
            tot_loss += loss.item()
            tot_tok += y.numel()
    print(f"CANON[softcap={args.softcap}:{args.softcap_form}] ckpt={os.path.basename(os.path.dirname(args.ckpt))} "
          f"windows={n_win} tokens={tot_tok:,} val_loss={tot_loss / tot_tok:.5f}")


if __name__ == "__main__":
    main()
