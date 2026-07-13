"""TE MXFP8-vs-bf16 GEMM-mix bench for the LOCAL B300 (sm_103), run inside the
NGC pytorch container (TE prebuilt). Mirrors modal_te.py::bench().

  docker run --rm --gpus '"device=N"' -v <sol>:/w nvcr.io/nvidia/pytorch:26.06-py3 \
      python /w/scripts/te_bench_local.py
"""
import time

import torch

torch.manual_seed(0)
dev = "cuda"
M = 256 * 512                       # batch*ctx rows — our exact GEMM mix
SHAPES = [(1024, 3072), (1024, 1024), (1024, 4096), (4096, 1024)]  # qkv, wo, fc1, fc2
LAYERS = 16
WARM, N = 10, 30


def bench(make_linear, ctx_factory, label):
    blocks = [[make_linear(k, n) for (k, n) in SHAPES] for _ in range(LAYERS)]
    xs = {k: torch.randn(M, k, device=dev, dtype=torch.bfloat16, requires_grad=True)
          for (k, _) in SHAPES}
    def step():
        with ctx_factory():
            loss = 0.0
            for blk in blocks:
                for (k, _), lin in zip(SHAPES, blk):
                    loss = loss + lin(xs[k]).float().square().mean()
        loss.backward()
        for x in xs.values():
            x.grad = None
    for _ in range(WARM):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        step()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / N * 1000
    print(f"{label}: {ms:.2f} ms/iter", flush=True)
    return ms


import contextlib

r = {}
r["torch_bf16"] = bench(
    lambda k, n: torch.nn.Linear(k, n, bias=False, device=dev, dtype=torch.bfloat16),
    contextlib.nullcontext, "torch_bf16")

import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe

print("TE version:", getattr(te, "__version__", "?"), flush=True)
r["te_bf16"] = bench(
    lambda k, n: te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device=dev),
    contextlib.nullcontext, "te_bf16")

rec = None
for name in ("MXFP8BlockScaling", "Float8CurrentScaling", "DelayedScaling"):
    if hasattr(te_recipe, name):
        rec = getattr(te_recipe, name)()
        print("recipe:", name, flush=True)
        break
r["te_mxfp8"] = bench(
    lambda k, n: te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device=dev),
    lambda: te.fp8_autocast(enabled=True, fp8_recipe=rec), "te_mxfp8")

print("RESULTS:", r, "| mxfp8 speedup vs torch bf16:",
      round(r["torch_bf16"] / r["te_mxfp8"], 3), flush=True)
