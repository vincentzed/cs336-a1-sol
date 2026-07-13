"""Measure direct_copy churn in muon.step(): batched (current) vs un-batched per-param.

Profiles ONLY the optimizer step (model fwd/bwd once to populate grads, then N
optimizer steps under torch.profiler). Reports per-variant: wall ms/step,
direct_copy kernel count + ms, GemmSymmetric ms. Saves to results/.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = "/tmp/claude-1012/-home-brayden/49c069ec-6b97-4caf-8b5f-c5a81a8dd40f/scratchpad/sol_exp"
sys.argv = ["x", "--train-tokens", "x", "--vocab-size", "32000", "--batch-size", "1",
            "--d-model", "1024", "--num-layers", "16", "--num-heads", "8", "--d-ff", "4096",
            "--max-seq-len", "512", "--logit-softcap", "0", "--z-loss", "0",
            "--rmsnorm", "quack", "--fused-rope", "--fused-qkv", "--value-embeds",
            "--wandb-mode", "disabled"]

import numpy as np
import torch

from transformer_lm.sol_modules import NorMuonGNS, build_optimizers, sol_cross_entropy
from transformer_lm.train_sol import build_model, parse_args

args = parse_args()
model = build_model(args)          # no compile: optimizer-only focus
lm_w = model.lm_weight()
muon, adamw = build_optimizers(model, muon_lr=8e-3, adam_lr=1.2e-2, embed_lr=2.4e-2,
                               weight_decay=0.1, cautious_wd=True, bf16_master=True)
x = torch.randint(0, 32000, (32, 512), device="cuda")
y = torch.randint(0, 32000, (32, 512), device="cuda")


def populate_grads():
    muon.zero_grad(set_to_none=True)
    loss = sol_cross_entropy(model(x), lm_w, y, mode="quack", softcap=0.0, z_coef=0.0)
    loss.backward()


def bench(label, step_fn, n=20):
    from torch.profiler import ProfilerActivity, profile
    populate_grads()
    for _ in range(5):
        step_fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            step_fn()
        torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / n * 1000
    copies_ms = copies_n = symm_ms = 0.0
    for k in prof.key_averages():
        if "direct_copy" in k.key:
            copies_ms += k.self_device_time_total / 1e3 / n
            copies_n += k.count / n
        if "GemmSymmetric" in k.key:
            symm_ms += k.self_device_time_total / 1e3 / n
    line = (f"{label:>22}: wall={wall:6.2f} ms/step | direct_copy {copies_n:6.1f} calls "
            f"{copies_ms:5.2f} ms | symmGEMM {symm_ms:5.2f} ms")
    print(line, flush=True)
    return line


lines = ["muon.step() copy churn (grads fixed; 234M config; B300)"]
lines.append(bench("batched (current)", muon.step))

# un-batched: force group-size-1 batching by monkeypatching the batch builder
orth = NorMuonGNS._orthogonalize_batch
def per_param_step():
    # temporarily shrink shape groups to singletons via the internal path:
    muon.step(force_unbatched=True) if "force_unbatched" in muon.step.__code__.co_varnames \
        else muon.step()
try:
    # introspect: does NorMuonGNS expose an unbatched path?
    import inspect
    src_has = "force_unbatched" in inspect.getsource(NorMuonGNS.step)
except Exception:
    src_has = False
if src_has:
    lines.append(bench("un-batched", per_param_step))
else:
    # fallback: monkeypatch _orthogonalize_batch to loop per-matrix (same math,
    # per-param GNS calls) so the surrounding stack/unbind cost stays visible.
    def loop_orth(self, u, ns):
        return torch.stack([orth(self, u[i:i+1], ns)[0] for i in range(u.shape[0])])
    NorMuonGNS._orthogonalize_batch = loop_orth
    lines.append(bench("per-matrix GNS calls", muon.step))
    NorMuonGNS._orthogonalize_batch = orth

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{REPO}/results", exist_ok=True)
with open(f"{REPO}/results/muon_copy_bench.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("MUON_BENCH_DONE")
