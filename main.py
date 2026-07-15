"""Reproduce the SOL leaderboard result on a single B200: `uv run main.py`.

Does three things, resumable at file granularity (delete an artifact to redo a phase):
  1. data:  download OpenWebText (stanford-cs336/owt-sample), tokenize with the repo's
            32k BPE (tokenizers/owt — the SAME vocab every reported number used) ->
            data/owt_train_full.npy (~2.73B tokens), data/owt_valid.npy (~66.4M).
  2. train: the record configuration — L14, d_ff 5632, 5-table value embeds with
            per-head gates, softcap-20 — 45:00 wall-clock, schedule anchored to the
            clock, checkpoints in out/.
  3. eval:  the canonical protocol (scripts/eval_canon.py — deterministic full sweep
            of every non-overlapping ctx-512 validation window, EMA weights, softcap
            in the forward). Prints the number the README reports.

Expected: val loss 2.957 +- 0.001 on B200 (four-seed replicate noise was +-0.0003;
node-to-node throughput variance moves it a little more). ~30-60 min of CPU
tokenization on first run, then 45 min of training, then a few minutes of eval.

`uv run main.py --smoke` runs the same three phases in miniature (a slice of the
corpus, 150 training steps) to validate the environment end-to-end in ~15 min.
"""
import argparse
import gzip
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
OWT = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main"
TRAIN_NPY = os.path.join(ROOT, "data", "owt_train_full.npy")
VALID_NPY = os.path.join(ROOT, "data", "owt_valid.npy")


def fetch(split: str, max_bytes: int | None = None) -> str:
    txt = os.path.join(ROOT, "data", f"owt_{split}.txt")
    if os.path.exists(txt):
        print(f"[data] {txt} present, skipping download", flush=True)
        return txt
    gz = txt + ".gz"
    print(f"[data] downloading owt_{split}.txt.gz ...", flush=True)
    urllib.request.urlretrieve(f"{OWT}/owt_{split}.txt.gz", gz)
    n = 0
    with gzip.open(gz, "rb") as fi, open(txt + ".part", "wb") as fo:
        while max_bytes is None or n < max_bytes:
            b = fi.read(1 << 24)
            if not b:
                break
            fo.write(b)
            n += len(b)
    os.rename(txt + ".part", txt)
    os.remove(gz)
    print(f"[data] {split}: {n:,} bytes of text", flush=True)
    return txt


def tokenize(src: str, out: str) -> None:
    if os.path.exists(out):
        print(f"[data] {out} present, skipping tokenize", flush=True)
        return
    import numpy as np

    from transformer_lm.tokenizer import Tokenizer

    # tokenizers/owt is the vocab of record: every number in the README and in
    # results/CANON.md is in its token units (tokenizers/owt_full is a study
    # artifact and was never used for a reported run).
    tok = Tokenizer.from_files(
        os.path.join(ROOT, "tokenizers/owt/vocab.pkl"),
        os.path.join(ROOT, "tokenizers/owt/merges.pkl"),
        special_tokens=["<|endoftext|>"],
    )
    print(f"[data] tokenizing {src} with {os.cpu_count()} workers ...", flush=True)
    ids = tok.encode_file(src, show_progress=True, num_workers=os.cpu_count())
    assert ids.max() < 65536
    np.save(out, ids.astype(np.uint16))
    print(f"[data] SAVED {out}: {len(ids):,} tokens", flush=True)


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="miniature end-to-end validation (~15 min): data slice, 150 steps")
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    ckpt_dir = os.path.join(ROOT, "out", "smoke" if args.smoke else "record")

    # -- 1. data -------------------------------------------------------------
    tokenize(fetch("valid"), VALID_NPY)
    tokenize(fetch("train", max_bytes=200 * 10**6 if args.smoke else None), TRAIN_NPY)

    # -- 2. train (the record configuration) ---------------------------------
    steps = ["--total-iters", "150", "--max-wall-sec", "600"] if args.smoke else \
            ["--total-iters", "10400", "--max-wall-sec", "2700"]
    run([sys.executable, "-m", "transformer_lm.train_sol",
         "--train-tokens", TRAIN_NPY, "--val-tokens", VALID_NPY, "--vocab-size", "32000",
         "--d-model", "1024", "--num-layers", "14", "--num-heads", "8", "--d-ff", "5632",
         "--max-seq-len", "512", "--context-length", "512", "--batch-size", "256",
         *steps, "--warmup-iters", "500", "--schedule-by-wall",
         "--lr-schedule", "wsd", "--wsd-decay-frac", "0.8", "--lr-min-ratio", "0.067",
         "--muon-lr", "8e-3", "--adam-lr", "1.2e-2", "--embed-lr", "2.4e-2",
         "--weight-decay", "0.1", "--ema-decay", "0.999", "--muon-momentum-warmup", "300",
         "--ce-mode", "quack-softcap", "--logit-softcap", "20", "--z-loss", "0",
         "--attn-backend", "fa4op", "--rmsnorm", "quack", "--fused-rope", "--fused-qkv",
         "--grad-clip", "0", "--cautious-wd", "--bf16-mt",
         "--value-embeds-k", "5", "--ve-gates", "per-head",
         "--x0-lambdas", "--unet-skips", "--smear", "--second-embed", "--xsa",
         "--data-sampling", "shuffled", "--eval-ctx", "512", "--compile",
         "--val-interval", "1000000", "--checkpoint-interval", "1000000",
         "--checkpoint-dir", ckpt_dir, "--wandb-mode", "disabled"])

    # -- 3. canonical eval ----------------------------------------------------
    import glob
    cks = sorted(glob.glob(os.path.join(ckpt_dir, "iter_*_final.pt")))
    assert cks, f"no final checkpoint in {ckpt_dir}"
    run([sys.executable, "scripts/eval_canon.py", cks[-1],
         "--val-tokens", VALID_NPY, "--softcap", "20", "--softcap-form", "tanh",
         "--ema", os.path.join(ckpt_dir, "ema_final.pt"),
         "--num-layers", "14", "--d-ff", "5632", "--value-embeds-k", "5",
         "--ve-gates", "per-head",
         "--x0-lambdas", "--unet-skips", "--smear", "--second-embed", "--xsa"])
    print("[done] the CANON[...] line above (EMA weights) is the reported number.", flush=True)


if __name__ == "__main__":
    main()
