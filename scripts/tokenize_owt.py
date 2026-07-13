"""Tokenize OWT text -> uint16 .npy with the repo's 32k BPE (parallel).

Usage: uv run --no-sync python scripts/tokenize_owt.py [num_workers]
Produces data/owt_valid.npy and data/owt_train.npy (bench_sol.sh defaults).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np

from transformer_lm.tokenizer import Tokenizer

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 128
tok = Tokenizer.from_files(
    "tokenizers/owt/vocab.pkl", "tokenizers/owt/merges.pkl", special_tokens=["<|endoftext|>"]
)
for src, out in [("data/owt_valid.txt", "data/owt_valid.npy"),
                 ("data/owt_train_sub.txt", "data/owt_train.npy")]:
    arr = tok.encode_file(src, show_progress=True, num_workers=NW)
    np.save(out, arr)
    print(f"SAVED {out}: {arr.shape[0]:,} tokens dtype={arr.dtype}", flush=True)
print("TOKENIZE_DONE", flush=True)
