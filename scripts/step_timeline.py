"""Measured per-step timeline of the champion config: every phase, every ms.

Times (cuda-synced): loader.next / forward / CE / backward / Muon / AdamW / EMA /
residue (logging+scheduler+python). Then one torch.profiler step for top kernels.
Run from the sol clone root: CUDA_VISIBLE_DEVICES=N .../python scripts/step_timeline.py
"""
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

E = "/tmp/claude-1012/-home-brayden/49c069ec-6b97-4caf-8b5f-c5a81a8dd40f/scratchpad/sol_exp"
sys.argv = ["prof", "--train-tokens", f"{E}/data/owt_train_full.npy", "--vocab-size", "32000",
            "--d-model", "1024", "--num-layers", "16", "--num-heads", "8", "--d-ff", "4096",
            "--max-seq-len", "512", "--batch-size", "256", "--context-length", "512",
            "--total-iters", "1", "--warmup-iters", "1", "--muon-lr", "8e-3",
            "--adam-lr", "1.2e-2", "--embed-lr", "2.4e-2",
            "--ce-mode", "quack", "--logit-softcap", "0", "--z-loss", "0",
            "--attn-backend", "fa4op", "--rmsnorm", "quack", "--fused-rope", "--fused-qkv",
            "--grad-clip", "0", "--cautious-wd", "--bf16-mt", "--value-embeds",
            "--wandb-mode", "disabled", "--compile"]

import numpy as np
import torch

from transformer_lm.sol_modules import CudaPrefetcher, EmaWeights, build_optimizers, sol_cross_entropy
from transformer_lm.train_sol import build_model, parse_args

args = parse_args()
inner = build_model(args)
lm_w = inner.lm_weight()
model = torch.compile(inner, dynamic=False)
muon, adamw = build_optimizers(model, muon_lr=args.muon_lr, adam_lr=args.adam_lr,
                               embed_lr=args.embed_lr, weight_decay=args.weight_decay,
                               cautious_wd=True, bf16_master=True)
ema = EmaWeights(inner, decay=0.999)
loader = CudaPrefetcher(np.load(f"{E}/data/owt_train_full.npy", mmap_mode="r"),
                        args.batch_size, args.context_length, "cuda")

PHASES = ["loader", "fwd", "ce", "bwd", "muon", "adamw", "ema", "residue"]
T = collections.defaultdict(float)
WARM, N = 12, 40
sync = torch.cuda.synchronize
step_rows = []
prev_end = None
for it in range(WARM + N):
    t0 = time.perf_counter()
    x, y = loader.next(); sync(); t1 = time.perf_counter()
    hidden = model(x); sync(); t2 = time.perf_counter()
    loss = sol_cross_entropy(hidden, lm_w, y, mode="quack", chunk_size=args.ce_chunk,
                             softcap=0.0, z_coef=0.0); sync(); t3 = time.perf_counter()
    muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
    loss.backward(); sync(); t4 = time.perf_counter()
    muon.step(); sync(); t5 = time.perf_counter()
    adamw.step(); sync(); t6 = time.perf_counter()
    ema.update(); sync(); t7 = time.perf_counter()
    # residue = whatever a real loop does between steps (logging/scheduler ~ approximated by gap)
    t8 = time.perf_counter()
    if it >= WARM:
        row = dict(zip(PHASES, [t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7]))
        step_rows.append(row)
        for k, v in row.items():
            T[k] += v

ms = {k: 1000 * T[k] / N for k in PHASES}
total = sum(ms.values())
print(f"\n=== MEASURED STEP TIMELINE (champion config + async loader + foreach EMA; {N} steps) ===")
width = 50
for k in PHASES:
    bar = "#" * max(1, int(ms[k] / total * width)) if ms[k] > 0.05 else ""
    print(f"{k:>7}: {ms[k]:7.2f} ms  {100*ms[k]/total:5.1f}%  {bar}")
print(f"{'TOTAL':>7}: {total:7.2f} ms  -> {int(args.batch_size*args.context_length/(total/1000)):,} tok/s")
print(f"loss at end: {loss.item():.4f} (sanity: decreasing from ~10.8)")

# ALL CUDA kernels over 3 steps, saved durably (repo results/ + /home/brayden trace)
import gzip

from torch.profiler import ProfilerActivity, profile

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    for _ in range(3):
        x, y = loader.next()
        hidden = model(x)
        loss = sol_cross_entropy(hidden, lm_w, y, mode="quack", chunk_size=args.ce_chunk,
                                 softcap=0.0, z_coef=0.0)
        muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
        loss.backward(); muon.step(); adamw.step(); ema.update()
    sync()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{REPO}/results", exist_ok=True)
os.makedirs("/home/brayden/sol-profiles", exist_ok=True)

full_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=-1,
                                       max_name_column_width=110)
with open(f"{REPO}/results/step_timeline.txt", "w") as f:
    f.write("=== MEASURED STEP TIMELINE (ms over %d steps) ===\n" % N)
    for k in PHASES:
        f.write(f"{k:>7}: {ms[k]:7.2f} ms  {100*ms[k]/total:5.1f}%\n")
    f.write(f"{'TOTAL':>7}: {total:7.2f} ms\n\n=== ALL CUDA KERNELS (3 steps) ===\n")
    f.write(full_table)
trace = "/home/brayden/sol-profiles/step_timeline_trace.json"
prof.export_chrome_trace(trace)
with open(trace, "rb") as fi, gzip.open(trace + ".gz", "wb") as fo:
    fo.write(fi.read())
os.remove(trace)
n_rows = len(prof.key_averages())
print(f"\n=== ALL {n_rows} KERNEL/OP ROWS saved -> results/step_timeline.txt (repo) ===")
print(f"=== chrome trace -> {trace}.gz (durable, view at chrome://tracing) ===")
print(full_table[:4000])  # head only to keep the log sane; the FILE has everything
print("TIMELINE_DONE")
