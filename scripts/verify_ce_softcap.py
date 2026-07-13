"""Verify vendored softcap-CE kernel vs the validated chunked reference.

Checks loss + dhidden + dweight at (131072,1024)x(32000,1024) bf16, softcap 30 & 23,
plus ignore_index handling and timing vs plain quack CE.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from transformer_lm.sol_modules import chunked_linear_cross_entropy, sol_cross_entropy

torch.manual_seed(0)
dev = "cuda"
M, D, V = 131072, 1024, 32000
h0 = (torch.randn(M, D, device=dev, dtype=torch.bfloat16) * 0.5)
w0 = torch.randn(V, D, device=dev, dtype=torch.bfloat16) * 0.02
tgt = torch.randint(0, V, (M,), device=dev)
tgt[::97] = -100  # exercise ignore_index

def run(mode, softcap):
    h = h0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    loss = sol_cross_entropy(h, w, tgt, mode=mode, softcap=softcap, z_coef=0.0)
    loss.backward()
    return loss.detach().float(), h.grad.detach(), w.grad.detach()

def rel(a, b):
    return ((a - b).float().norm() / b.float().norm().clamp_min(1e-12)).item()

ok = True
for cap in (30.0, 23.0):
    l_ref, dh_ref, dw_ref = run("chunked", cap)
    l_new, dh_new, dw_new = run("quack-softcap", cap)
    r = (abs(l_new.item() - l_ref.item()) / abs(l_ref.item()), rel(dh_new, dh_ref), rel(dw_new, dw_ref))
    print(f"softcap={cap}: loss ref={l_ref.item():.6f} new={l_new.item():.6f} "
          f"| rel: loss={r[0]:.2e} dh={r[1]:.2e} dw={r[2]:.2e}", flush=True)
    ok &= r[0] < 2e-3 and r[1] < 3e-2 and r[2] < 3e-2  # bf16 grad tolerance

# sanity: softcap-CE(cap→huge) ≈ plain quack CE
l_plain, _, _ = run("quack", 0.0)
l_huge, _, _ = run("quack-softcap", 1e4)
print(f"cap=1e4 vs plain quack: {l_huge.item():.6f} vs {l_plain.item():.6f} "
      f"(rel {abs(l_huge.item()-l_plain.item())/l_plain.item():.2e})", flush=True)

def bench(mode, softcap, n=30):
    h = h0.clone().requires_grad_(True); w = w0.clone().requires_grad_(True)
    for _ in range(5):
        sol_cross_entropy(h, w, tgt, mode=mode, softcap=softcap, z_coef=0.0).backward()
        h.grad = None; w.grad = None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        sol_cross_entropy(h, w, tgt, mode=mode, softcap=softcap, z_coef=0.0).backward()
        h.grad = None; w.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

t_plain = bench("quack", 0.0)
t_sc = bench("quack-softcap", 30.0)
t_chunk = bench("chunked", 30.0)
print(f"TIMING ms (fwd+bwd incl. linear): quack={t_plain:.2f} quack-softcap={t_sc:.2f} "
      f"chunked-softcap={t_chunk:.2f}", flush=True)
print("VERIFY_PASS" if ok else "VERIFY_FAIL", flush=True)
