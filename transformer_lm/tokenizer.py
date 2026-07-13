import hashlib
import multiprocessing
import os
import pickle
import time
import regex as re
from collections import Counter, defaultdict
from tqdm import tqdm
from transformer_lm._chunking import find_chunk_boundaries

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_chunk(input_path, start, end, special_tokens):
    """Read a file chunk, split on special tokens, apply GPT2 regex, return word counts."""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    if special_tokens:
        pattern = "|".join(re.escape(st) for st in sorted(special_tokens, key=len, reverse=True))
        parts = re.split(pattern, chunk)
    else:
        parts = [chunk]
    counts = Counter()
    for part in parts:
        counts += Counter(m.group() for m in re.finditer(GPT2_PAT, part))
    return counts


def _pretokenize_chunk_star(args):
    return pretokenize_chunk(*args)


def _load_or_compute_cache(cache_path, compute_fn, label):
    """Load from pickle cache if exists, otherwise compute and optionally save."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached {label} from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    result = compute_fn()
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved {label} cache to {cache_path}")
    return result


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_workers=None,
    num_chunks=64,
    use_cache=False,
    end_of_text_token=b"<|endoftext|>",
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # --- Initial vocab: 256 bytes + special tokens ---
    vocab = {i: bytes([i]) for i in range(256)}
    for i, st in enumerate(special_tokens):
        vocab[256 + i] = st.encode("utf-8")

    # --- Cache key based on input path ---
    if use_cache:
        cache_key = hashlib.md5(str(input_path).encode()).hexdigest()[:12]
        pretok_cache = f"/tmp/pretok_cache_{cache_key}.pkl"
        pairs_cache = f"/tmp/paircounts_cache_{cache_key}.pkl"
    else:
        pretok_cache = pairs_cache = None

    # --- Step 1: Pre-tokenize (parallel, cached) ---
    t0 = time.time()

    def do_pretokenize():
        workers = num_workers or os.cpu_count() or 8
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, end_of_text_token)
        args = [(input_path, s, e, special_tokens)
                for s, e in zip(boundaries[:-1], boundaries[1:])]
        merged = Counter()
        with multiprocessing.Pool(workers) as pool:
            for counts in tqdm(pool.imap_unordered(_pretokenize_chunk_star, args),
                               total=len(args), desc="Pre-tokenizing"):
                merged += counts
        return merged

    merged_counts = _load_or_compute_cache(pretok_cache, do_pretokenize, "pre-tokenization")

    # Convert words to int lists (byte values 0-255). Ints hash faster than bytes.
    id_to_word = {}
    word_freq = {}
    for wid, (word_str, freq) in enumerate(merged_counts.items()):
        id_to_word[wid] = list(word_str.encode("utf-8"))
        word_freq[wid] = freq
    print(f"Pre-tokenization: {time.time() - t0:.2f}s ({len(id_to_word)} unique words)")

    # --- Step 2: Count adjacent pairs (cached) ---
    t0 = time.time()

    def do_count_pairs():
        freq = defaultdict(int)
        index = defaultdict(set)
        for word_id, word in tqdm(id_to_word.items(), desc="Counting pairs", total=len(id_to_word)):
            word_count = word_freq[word_id]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                freq[pair] += word_count
                index[pair].add(word_id)
        return dict(freq), dict(index)

    raw_freq, raw_index = _load_or_compute_cache(pairs_cache, do_count_pairs, "pair counts")
    pair_freq = defaultdict(int, raw_freq)
    pair_index = defaultdict(set, raw_index)
    print(f"Initial pair counting: {time.time() - t0:.2f}s ({len(pair_freq)} unique pairs)")

    # --- Step 3: Merge loop with frequency buckets ---
    num_merges = vocab_size - len(vocab)
    assert num_merges >= 0

    # Token ID -> bytes mapping (for lexicographic tie-breaking)
    tid_to_bytes = {i: bytes([i]) for i in range(256)}
    for i, st in enumerate(special_tokens):
        tid_to_bytes[256 + i] = st.encode("utf-8")
    next_tid = len(vocab)

    # Sort key cache: pair -> (bytes_a, bytes_b) for tie-breaking
    sort_key = {p: (tid_to_bytes[p[0]], tid_to_bytes[p[1]]) for p in pair_freq}

    # Frequency buckets: freq -> {pairs with that freq}
    buckets = defaultdict(set)
    for pair, f in pair_freq.items():
        buckets[f].add(pair)

    merges_int = []
    t0 = time.time()
    for _ in tqdm(range(num_merges), desc="Merging"):
        if not buckets:
            break

        # Find best pair: highest freq, lexicographic tie-break
        max_freq = max(buckets)
        best_pair = max(buckets[max_freq], key=lambda p: sort_key[p])
        buckets[max_freq].discard(best_pair)
        if not buckets[max_freq]:
            del buckets[max_freq]

        # Record merge
        merges_int.append(best_pair)
        merged_id = next_tid
        tid_to_bytes[merged_id] = tid_to_bytes[best_pair[0]] + tid_to_bytes[best_pair[1]]
        next_tid += 1

        # Apply merge to every word containing this pair
        for word_id in list(pair_index[best_pair]):
            word = id_to_word[word_id]
            word_count = word_freq[word_id]
            i = 0
            while i < len(word) - 1:
                if word[i] != best_pair[0] or word[i + 1] != best_pair[1]:
                    i += 1
                    continue

                # Decrement old neighbor pairs
                if i > 0 and (word[i - 1], word[i]) != best_pair:
                    _move(buckets, pair_freq, (word[i - 1], word[i]), -word_count)
                if i + 2 < len(word) and (word[i + 1], word[i + 2]) != best_pair:
                    _move(buckets, pair_freq, (word[i + 1], word[i + 2]), -word_count)

                # Replace pair with merged token
                word[i] = merged_id
                del word[i + 1]

                # Increment new neighbor pairs
                if i > 0:
                    new_pair = (word[i - 1], word[i])
                    _move(buckets, pair_freq, new_pair, word_count)
                    pair_index[new_pair].add(word_id)
                    if new_pair not in sort_key:
                        sort_key[new_pair] = (tid_to_bytes[new_pair[0]], tid_to_bytes[new_pair[1]])
                if i + 1 < len(word):
                    new_pair = (word[i], word[i + 1])
                    _move(buckets, pair_freq, new_pair, word_count)
                    pair_index[new_pair].add(word_id)
                    if new_pair not in sort_key:
                        sort_key[new_pair] = (tid_to_bytes[new_pair[0]], tid_to_bytes[new_pair[1]])

        del pair_freq[best_pair]
        del pair_index[best_pair]

    print(f"Merging done: {time.time() - t0:.2f}s total")

    # --- Build output vocab and merges in bytes ---
    final_vocab = {i: bytes([i]) for i in range(256)}
    for i, st in enumerate(special_tokens):
        final_vocab[256 + i] = st.encode("utf-8")
    for mi, (a, b) in enumerate(merges_int):
        final_vocab[256 + len(special_tokens) + mi] = tid_to_bytes[a] + tid_to_bytes[b]

    merges = [(tid_to_bytes[a], tid_to_bytes[b]) for a, b in merges_int]
    return final_vocab, merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab  # int -> bytes
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        # Add special tokens to vocab if missing
        for st in self.special_tokens:
            b = st.encode("utf-8")
            if b not in self.bytes_to_id:
                tid = len(self.vocab)
                self.vocab[tid] = b
                self.bytes_to_id[b] = tid
        # Pre-compute special token -> id for fast lookup
        self._special_ids = {st: self.bytes_to_id[st.encode("utf-8")] for st in self.special_tokens}
        # Pre-compile patterns
        if self.special_tokens:
            escaped = [re.escape(st) for st in sorted(self.special_tokens, key=len, reverse=True)]
            self._special_pat = re.compile("(" + "|".join(escaped) + ")")
        else:
            self._special_pat = None
        self._gpt2_pat_bytes = re.compile(GPT2_PAT.encode())
        self._byte_tokens = [bytes([i]) for i in range(256)]
        self._word_cache = {}  # word_bytes -> list[int]

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids = []
        # Split into segments: alternating [text, special, text, special, ...]
        # If no special tokens, just one segment.
        parts = self._special_pat.split(text) if self._special_pat else [text]
        for part in parts:
            special_id = self._special_ids.get(part)
            if special_id is not None:
                ids.append(special_id)
            elif part:
                for match in self._gpt2_pat_bytes.finditer(part.encode("utf-8")):
                    word_bytes = match.group()
                    cached = self._word_cache.get(word_bytes)
                    ids.extend(cached if cached is not None else self._encode_word(word_bytes))
        return ids

    def _encode_word(self, word_bytes: bytes) -> list[int]:
        """Apply BPE merges to a single pre-token (bytes) and return token IDs."""
        if not word_bytes:
            return []
        tokens = [self._byte_tokens[b] for b in word_bytes]
        while len(tokens) > 1:
            # Find the pair with the lowest merge rank
            best_pair = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                rank = self.merge_rank.get((tokens[i], tokens[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = (tokens[i], tokens[i + 1])
            if best_pair is None:
                break
            # Merge all occurrences of best_pair left-to-right
            merged = best_pair[0] + best_pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        result = [self.bytes_to_id[t] for t in tokens]
        self._word_cache[word_bytes] = result
        return result

    def encode_iterable(self, iterable):
        """Lazily encode an iterable of strings, yielding token IDs."""
        for text in iterable:
            yield from self.encode(text)

    def encode_file(self, path, show_progress=False, num_workers=None):
        """Encode a file into token IDs. Uses multiprocessing for large files."""
        import numpy as np
        if num_workers is None:
            num_workers = os.cpu_count() or 1
        num_chunks = max(num_workers * 4, 64)
        with open(path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        merges = list(self.merge_rank.keys())
        args = [(i, self.vocab, merges, self.special_tokens, path, start, end)
                for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
        results = [None] * len(args)
        with multiprocessing.Pool(num_workers) as pool:
            for idx, chunk_arr in tqdm(
                    pool.imap_unordered(_encode_chunk_worker, args),
                    total=len(args), desc="Encoding", disable=not show_progress):
                results[idx] = chunk_arr
        return np.concatenate(results)

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


def _encode_chunk_worker(args):
    """Worker: build a Tokenizer, encode a file chunk, return (index, uint16 numpy array)."""
    import numpy as np
    idx, vocab, merges, special_tokens, path, start, end = args
    tok = Tokenizer(vocab, merges, special_tokens)
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return idx, np.array(tok.encode(chunk), dtype=np.uint16)


def _move(buckets, pair_freq, pair, delta):
    """Adjust a pair's frequency by delta and move it between buckets."""
    old = pair_freq[pair]
    new = old + delta
    pair_freq[pair] = new
    if old > 0:
        buckets[old].discard(pair)
        if not buckets[old]:
            del buckets[old]
    if new > 0:
        buckets[new].add(pair)


if __name__ == "__main__":
    import tempfile
    from pprint import pprint
    corpus = "low low low low low lower lower lower widest widest widest " \
             "newest newest newest newest newest newest<|endoftext|>"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(corpus)
        tmp = f.name
    vocab, merges = run_train_bpe(tmp, 256 + 10, ["<|endoftext|>"])
    pprint(list(vocab.items())[256:])
    pprint(merges)
    os.unlink(tmp)
