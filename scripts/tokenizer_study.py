"""Tokenizer-lever study: quantify how the self-trained 32k BPE changes the
leaderboard metric's UNITS (CE per token), without any GPU training.

Phases (argparse subcommands so bash can orchestrate/checkpoint):
  train   --slice-mb N --out tokenizers/<tag>       train a 32k BPE on the first N MB
  analyze --tok tokenizers/<tag> --tag NAME         bytes/token, vocab use, unigram CE, bpb
  npy     --tok tokenizers/<tag> --suffix _weak     tokenize train+val -> drop-in .npy

Unigram CE estimator: fit token unigram probs (add-0.5 smoothing) on a fixed
500MB train slice (offset 2GB, shared across variants, disjoint from any
variant's 50MB training slice), then CE = -mean log p over tokenized owt_valid.
To first order a trained model's CE shifts between tokenizers like the unigram
CE does (the units move; the modeling problem is ~the same in bits/byte).
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

E = "/tmp/claude-1012/-home-brayden/49c069ec-6b97-4caf-8b5f-c5a81a8dd40f/scratchpad/sol_exp"
TRAIN_TXT = f"{E}/data/owt_train.txt"
VALID_TXT = f"{E}/data/owt_valid.txt"
UNIGRAM_SLICE = (2 << 30, 500 << 20)  # offset 2GB, 500MB — shared fit slice


def _slice_file(src, out, offset, length):
    if os.path.exists(out) and os.path.getsize(out) == length:
        return out
    with open(src, "rb") as f, open(out, "wb") as o:
        f.seek(offset)
        remaining = length
        while remaining > 0:
            b = f.read(min(1 << 24, remaining))
            if not b:
                break
            o.write(b)
            remaining -= len(b)
    return out


def cmd_train(args):
    from transformer_lm.tokenizer import run_train_bpe
    os.makedirs(args.out, exist_ok=True)
    if args.slice_mb > 0:
        src = _slice_file(TRAIN_TXT, f"/tmp/bpe_slice_{args.slice_mb}mb.txt", 0,
                          args.slice_mb << 20)
    else:
        src = TRAIN_TXT
    t0 = time.time()
    vocab, merges = run_train_bpe(src, 32000, ["<|endoftext|>"], num_workers=args.workers)
    dt = time.time() - t0
    with open(f"{args.out}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{args.out}/merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    print(f"TRAIN_DONE out={args.out} slice_mb={args.slice_mb} vocab={len(vocab)} "
          f"merges={len(merges)} secs={dt:.0f}", flush=True)


def _tok(tokdir):
    from transformer_lm.tokenizer import Tokenizer
    return Tokenizer.from_files(f"{tokdir}/vocab.pkl", f"{tokdir}/merges.pkl",
                                special_tokens=["<|endoftext|>"])


def cmd_analyze(args):
    tok = _tok(args.tok)
    val_bytes = os.path.getsize(VALID_TXT)
    val_ids = tok.encode_file(VALID_TXT, show_progress=False, num_workers=args.workers)
    n_val = len(val_ids)
    uniq = len(np.unique(val_ids))
    bpt = val_bytes / n_val

    fit_src = _slice_file(TRAIN_TXT, "/tmp/unigram_fit_slice.txt", *UNIGRAM_SLICE)
    fit_ids = tok.encode_file(fit_src, show_progress=False, num_workers=args.workers)
    counts = np.bincount(fit_ids, minlength=32000).astype(np.float64)
    probs = (counts + 0.5) / (counts.sum() + 0.5 * 32000)
    ce = float(-np.log(probs[val_ids]).mean())            # nats/token
    bpb = ce / np.log(2) / bpt                             # bits per BYTE (unit-free)
    print(f"ANALYZE tag={args.tag} tokens_val={n_val:,} bytes/token={bpt:.4f} "
          f"vocab_used={uniq}/32000 ({100*uniq/32000:.1f}%) unigram_CE={ce:.4f} "
          f"unigram_bits_per_byte={bpb:.4f}", flush=True)


def cmd_npy(args):
    tok = _tok(args.tok)
    for src, out in [(VALID_TXT, f"{E}/data/owt_valid{args.suffix}.npy"),
                     (TRAIN_TXT, f"{E}/data/owt_train_full{args.suffix}.npy")]:
        t0 = time.time()
        ids = tok.encode_file(src, show_progress=False, num_workers=args.workers)
        assert ids.max() < 65536
        np.save(out, ids.astype(np.uint16))
        print(f"NPY_SAVED {out} tokens={len(ids):,} secs={time.time()-t0:.0f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--slice-mb", type=int, required=True, help="0 = full file")
    t.add_argument("--out", required=True)
    t.add_argument("--workers", type=int, default=96)
    a = sub.add_parser("analyze")
    a.add_argument("--tok", required=True)
    a.add_argument("--tag", required=True)
    a.add_argument("--workers", type=int, default=96)
    n = sub.add_parser("npy")
    n.add_argument("--tok", required=True)
    n.add_argument("--suffix", required=True)
    n.add_argument("--workers", type=int, default=96)
    args = p.parse_args()
    {"train": cmd_train, "analyze": cmd_analyze, "npy": cmd_npy}[args.cmd](args)
