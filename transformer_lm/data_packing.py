"""Doc-aware batch sampling: BOS-aligned windows and varlen packing.

Modes (--data-packing):
  random: baseline behavior -- windows start anywhere, sequences start mid-doc and
          attend across document boundaries (leaks cross-doc context).
  bos:    BosAlign (modded-nanogpt 2025-07-12 record) -- window STARTS are sampled
          only at document starts. Fixed shapes, compile-safe, nothing else changes.
  varlen: rows are contiguous doc-runs; emits int32 cu_seqlens over the flattened
          (bs*ctx) stream (standard flash-attn convention) so FA4 varlen attention
          can stop attention at doc boundaries. Loader contract:
          next() -> (x, y, cu_seqlens);  cu_seqlens is None for random/bos.

The document-start index (positions right after each <|endoftext|>, plus 0) is
precomputed once per token file and cached next to it as <file>.docidx.npy.
"""
from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import torch

logger = logging.getLogger("train.data")


def find_eot_id(vocab_pkl: str, special: bytes = b"<|endoftext|>") -> int:
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    for i, b in vocab.items():
        if b == special:
            return i
    raise ValueError(f"{special!r} not found in {vocab_pkl}")


def doc_starts_index(tokens: np.ndarray, eot_id: int, cache_path: str | None = None) -> np.ndarray:
    """int64 sorted positions where documents start (0 + every position after an EOT)."""
    if cache_path and os.path.exists(cache_path):
        idx = np.load(cache_path)
        logger.info("doc index: loaded %s (%d docs)", cache_path, len(idx))
        return idx
    parts = [np.array([0], dtype=np.int64)]
    chunk = 1 << 24  # memmap-friendly scan
    for s in range(0, len(tokens), chunk):
        hits = np.flatnonzero(np.asarray(tokens[s:s + chunk]) == eot_id).astype(np.int64) + s + 1
        parts.append(hits)
    idx = np.unique(np.concatenate(parts))
    idx = idx[idx < len(tokens) - 1]  # a doc "starting" at the last token is useless
    if cache_path:
        np.save(cache_path, idx)
    logger.info("doc index: %d docs (mean len %.0f tok)%s", len(idx), len(tokens) / max(1, len(idx)),
                f" -> cached {cache_path}" if cache_path else "")
    return idx


class PackedPrefetcher:
    """CudaPrefetcher twin for bos/varlen modes (pinned memory + side-stream copy).

    next() -> (x, y, cu_seqlens) with cu_seqlens None in bos mode. In varlen mode
    every row start is a boundary and internal doc starts add more; cu_seqlens is
    int32 on-device, [0 ... bs*ctx], over the flattened x.
    """

    def __init__(self, data: np.ndarray, doc_starts: np.ndarray, batch_size: int,
                 context_length: int, device: str, mode: str = "bos"):
        assert mode in ("bos", "varlen")
        self.data, self.mode = data, mode
        self.bs, self.ctx = batch_size, context_length
        self.device = device
        self.cuda = torch.device(device).type == "cuda"
        # only doc starts that leave room for a full window + next-token target
        self.starts = doc_starts[doc_starts < data.shape[0] - context_length - 1]
        if len(self.starts) < 1:
            raise ValueError("no eligible document starts for this context length")
        self.stream = torch.cuda.Stream() if self.cuda else None
        self._next = None
        if self.cuda:
            self.preload()

    def set_batch_size(self, bs: int):
        if bs == self.bs:
            return
        self.bs = bs
        if self.cuda:
            self.preload()

    def _sample_cpu(self):
        starts = np.random.choice(self.starts, size=self.bs)  # every row begins at a doc start
        idx = starts[:, None] + np.arange(self.ctx)[None, :]
        x = torch.from_numpy(self.data[idx].astype(np.int64)).pin_memory()
        y = torch.from_numpy(self.data[idx + 1].astype(np.int64)).pin_memory()
        cu = None
        if self.mode == "varlen":
            bounds = [np.array([0], dtype=np.int64)]
            for r, s in enumerate(starts):
                lo = np.searchsorted(self.starts, s + 1)          # internal doc starts in (s, s+ctx)
                hi = np.searchsorted(self.starts, s + self.ctx)
                bounds.append(self.starts[lo:hi] - s + r * self.ctx)
                bounds.append(np.array([(r + 1) * self.ctx], dtype=np.int64))  # row end/start boundary
            cu = torch.from_numpy(np.unique(np.concatenate(bounds)).astype(np.int32)).pin_memory()
        return x, y, cu

    def preload(self):
        x, y, cu = self._sample_cpu()
        with torch.cuda.stream(self.stream):
            self._next = (x.to(self.device, non_blocking=True),
                          y.to(self.device, non_blocking=True),
                          cu.to(self.device, non_blocking=True) if cu is not None else None)

    def next(self):
        if not self.cuda:
            x, y, cu = self._sample_cpu()
            return (x.to(self.device), y.to(self.device),
                    cu.to(self.device) if cu is not None else None)
        torch.cuda.current_stream().wait_stream(self.stream)
        x, y, cu = self._next
        for t in (x, y) + ((cu,) if cu is not None else ()):
            t.record_stream(torch.cuda.current_stream())
        self.preload()
        return x, y, cu


class _RandomAdapter:
    """Wraps CudaPrefetcher to the unified (x, y, None) contract."""

    def __init__(self, inner):
        self.inner = inner

    def set_batch_size(self, bs):
        self.inner.set_batch_size(bs)

    def next(self):
        x, y = self.inner.next()
        return x, y, None


def make_loader(mode: str, data: np.ndarray, batch_size: int, context_length: int,
                device: str, *, vocab_pkl: str | None = None, tokens_path: str | None = None):
    """Unified loader factory. All loaders yield (x, y, cu_seqlens-or-None)."""
    if mode == "random":
        from transformer_lm.sol_modules import CudaPrefetcher
        return _RandomAdapter(CudaPrefetcher(data, batch_size, context_length, device))
    eot = find_eot_id(vocab_pkl)
    cache = (tokens_path + ".docidx.npy") if tokens_path else None
    ds = doc_starts_index(data, eot, cache)
    return PackedPrefetcher(data, ds, batch_size, context_length, device, mode=mode)
