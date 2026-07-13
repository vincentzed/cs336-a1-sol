"""MXFP8 v2 kernel-level gate: is quack's mxfp8_gemm actually faster than cuBLAS
bf16 at OUR shapes (M=131072), and what do the per-use quantizes cost?

Run from repo root on a free GPU:
  CUDA_VISIBLE_DEVICES=N .venv-python scripts/mxfp8_bench.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from quack.gemm_blockscaled_interface import mxfp8_gemm, mxfp8_quantize
from quack.mx_utils import to_mx, to_mx_compiled

DEV = "cuda"
M = 256 * 512
SHAPES = [("qkv", 3072, 1024), ("wo", 1024, 1024), ("fc1", 4096, 1024), ("fc2", 1024, 4096)]


def timeit(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    return ts[len(ts) // 2]


print(f"M={M}; median-of-30 ms", flush=True)
tot = {"cublas": 0.0, "fp8gemm": 0.0, "fp8+xq": 0.0}
for name, N, K in SHAPES:
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
    t_cublas = timeit(lambda: x @ w.mT)
    xq, xs = mxfp8_quantize(x)
    wq, ws = mxfp8_quantize(w)
    t_fp8 = timeit(lambda: mxfp8_gemm(xq, wq.mT, xs, ws.mT, out_dtype=torch.bfloat16))
    t_xq_eager = timeit(lambda: to_mx(x, 32))
    t_xq_comp = timeit(lambda: to_mx_compiled(x, 32))
    t_fp8_use = t_fp8 + t_xq_comp  # realistic per-use fwd cost w/ cached weight quant
    tot["cublas"] += t_cublas; tot["fp8gemm"] += t_fp8; tot["fp8+xq"] += t_fp8_use
    print(f"{name:4s} (N={N},K={K}): cublas={t_cublas:.3f}  fp8gemm={t_fp8:.3f} "
          f"(x{t_cublas/t_fp8:.2f})  xq_eager={t_xq_eager:.3f} xq_comp={t_xq_comp:.3f} "
          f"fp8+xq={t_fp8_use:.3f} (x{t_cublas/t_fp8_use:.2f})", flush=True)
print(f"MIX: cublas={tot['cublas']:.2f}ms  fp8gemm={tot['fp8gemm']:.2f}ms "
      f"(x{tot['cublas']/tot['fp8gemm']:.2f})  fp8+per-use-quant={tot['fp8+xq']:.2f}ms "
      f"(x{tot['cublas']/tot['fp8+xq']:.2f})", flush=True)
print("GATE:", "PROMISING -> e2e worth it" if tot["cublas"] / tot["fp8+xq"] > 1.1 else
      "DEAD at kernel level (fp8+quant not >1.1x cuBLAS)", flush=True)
