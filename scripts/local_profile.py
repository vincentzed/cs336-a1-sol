"""Per-section step profiler (fwd / CE / bwd / optimizer). Run from repo root:
  CUDA_VISIBLE_DEVICES=N PYTHONPATH=. .venv/bin/python scripts/local_profile.py --attn-backend fa4 --ce-mode chunked
Extra argv is passed straight to train_sol's parser."""
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.argv = ["prof", "--train-tokens", "data/owt_train.npy", "--vocab-size", "32000",
            "--batch-size", "320", "--context-length", "1024", "--max-seq-len", "1024",
            "--total-iters", "1", "--warmup-iters", "1", "--wandb-mode", "disabled",
            "--no-ema", "--compile"] + sys.argv[1:]

import numpy as np
import torch

from transformer_lm.sol_modules import CudaPrefetcher, build_optimizers, sol_cross_entropy
from transformer_lm.train_sol import build_model, parse_args

args = parse_args()
inner = build_model(args)
lm_w = inner.lm_weight()
model = torch.compile(inner, dynamic=False, fullgraph=args.fullgraph)
muon, adamw = build_optimizers(model, muon_lr=args.muon_lr, adam_lr=args.adam_lr,
                               embed_lr=args.embed_lr, weight_decay=args.weight_decay,
                               betas=(args.beta1, args.beta2))
params = [p for p in model.parameters() if p.requires_grad]
loader = CudaPrefetcher(np.load("data/owt_train.npy", mmap_mode="r"),
                        args.batch_size, args.context_length, "cuda")
T = collections.defaultdict(float)
WARM, N = 12, 40
for it in range(WARM + N):
    x, y = loader.next()
    torch.cuda.synchronize(); a = time.perf_counter()
    hidden = model(x); torch.cuda.synchronize(); b = time.perf_counter()
    loss = sol_cross_entropy(hidden, lm_w, y, mode=args.ce_mode, chunk_size=args.ce_chunk,
                             softcap=args.logit_softcap, z_coef=args.z_loss)
    torch.cuda.synchronize(); c = time.perf_counter()
    muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
    loss.backward(); torch.cuda.synchronize(); d = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
    muon.step(); adamw.step(); torch.cuda.synchronize(); e = time.perf_counter()
    if it >= WARM:
        T["fwd"] += b - a; T["ce"] += c - b; T["bwd"] += d - c; T["opt"] += e - d
ms = {k: 1000 * v / N for k, v in T.items()}
tot = sum(ms.values())
print(f"LOCALPROF ms/step: fwd={ms['fwd']:.1f} ce={ms['ce']:.1f} bwd={ms['bwd']:.1f} "
      f"opt={ms['opt']:.1f} total={tot:.1f} tok/s~={int(args.batch_size * args.context_length / (tot / 1000))}",
      flush=True)
