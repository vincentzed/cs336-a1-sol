import time
import tracemalloc
import pickle
import argparse
import tempfile
import os
from pathlib import Path

from transformer_lm.tokenizer import run_train_bpe
from transformer_lm._chunking import find_chunk_boundaries

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "tokenizers" / "tinystories"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=int, default=1,
                        help="Use 1/fraction of the corpus (e.g. 100 for 1/100)")
    args = parser.parse_args()

    input_path = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"

    if args.fraction > 1:
        # Write a sub-chunk to a temp file
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, args.fraction, b"<|endoftext|>")
            f.seek(boundaries[0])
            chunk = f.read(boundaries[1] - boundaries[0])
        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False)
        tmp.write(chunk)
        tmp.close()
        actual_path = tmp.name
        print(f"Using 1/{args.fraction} of corpus: {len(chunk)} bytes")
    else:
        actual_path = str(input_path)
        print(f"Using full corpus")

    tracemalloc.start()
    start = time.time()

    vocab, merges = run_train_bpe(
        input_path=actual_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        use_cache=True,
    )

    elapsed = time.time() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if args.fraction > 1:
        os.unlink(actual_path)

    print(f"Time: {elapsed:.1f}s")
    print(f"Peak memory: {peak_memory / 1e9:.2f} GB")
    print(f"Vocab size: {len(vocab)}")
    if args.fraction > 1:
        print(f"Estimated full time: {elapsed * args.fraction:.0f}s")

    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token} ({len(longest_token)} bytes)")

    if args.fraction == 1:
        with open(OUTPUT_DIR / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        with open(OUTPUT_DIR / "merges.pkl", "wb") as f:
            pickle.dump(merges, f)
        print(f"Saved vocab and merges to {OUTPUT_DIR}")
