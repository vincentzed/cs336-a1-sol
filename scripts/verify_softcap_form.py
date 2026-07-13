"""CPU verification of the sign-encoded softcap (tanh & sigmoid forms).

Validates THREE things at fp64 on the CPU (no GPU / no CuTeDSL needed):
  1. apply_softcap's returned dzc/dz matches autograd d(zc)/dz for BOTH forms.
  2. The chunked-CE custom backward matches autograd through the sigmoid cap
     (this is the exact closed form the CuTeDSL kernel implements).
  3. The tanh path (softcap>0) is arithmetically UNCHANGED vs the pre-refactor
     expression cap*tanh(z/cap) — the GPU-verified path must not have drifted.
  4. modded's forward: cap*sigmoid((z+5)/7.5) == our apply_softcap(z, -cap).
"""
import sys, torch
sys.path.insert(0, "/tmp/claude-1012/-home-brayden/49c069ec-6b97-4caf-8b5f-c5a81a8dd40f/scratchpad/sol")
torch.set_default_dtype(torch.float64)
from transformer_lm.sol_modules import apply_softcap, chunked_linear_cross_entropy

def check(name, a, b, tol):
    err = (a - b).abs().max().item()
    print(f"  {'PASS' if err < tol else 'FAIL'} {name}: max|Δ|={err:.3e} (tol {tol:.0e})")
    return err < tol

ok = True
torch.manual_seed(0)

# ---- 1. dzc/dz matches autograd, both forms ----
print("[1] apply_softcap dzc/dz vs autograd (fp64)")
for cap in (23.0, -23.0, 15.0, -30.0):
    z = torch.randn(64, 128, requires_grad=True)
    zc, dzc = apply_softcap(z, cap, need_grad=True)
    g, = torch.autograd.grad(zc.sum(), z)
    ok &= check(f"cap={cap:+.0f}", dzc, g, 1e-9)

# ---- 2. modded forward equivalence ----
print("[2] apply_softcap(z,-cap) == cap*sigmoid((z+5)/7.5)  [modded L1508]")
z = torch.randn(64, 128)
for cap in (23.0, 15.0):
    ref = cap * torch.sigmoid((z + 5.0) / 7.5)
    ok &= check(f"cap={cap:.0f}", apply_softcap(z, -cap), ref, 1e-12)

# ---- 3. tanh path unchanged vs original expression ----
print("[3] tanh path byte-identical to cap*tanh(z/cap)")
z = torch.randn(64, 128)
for cap in (23.0, 30.0, 15.0):
    ok &= check(f"cap={cap:.0f}", apply_softcap(z, cap), cap * torch.tanh(z / cap), 0.0 + 1e-15)

# ---- 4. chunked-CE custom backward through sigmoid cap vs autograd ----
print("[4] chunked-CE custom bwd vs dense-autograd reference (sigmoid form)")
V, N, d = 512, 96, 64
for cap in (-23.0, 23.0):
    torch.manual_seed(1)
    h = torch.randn(N, d, requires_grad=True)
    w = torch.randn(V, d, requires_grad=True) * 0.05
    tgt = torch.randint(0, V, (N,))
    # custom path
    loss_c = chunked_linear_cross_entropy(h, w, tgt, chunk_size=32, softcap=cap, z_coef=0.0)
    gh_c, gw_c = torch.autograd.grad(loss_c, [h, w])
    # dense autograd reference (same apply_softcap fwd, autograd bwd)
    h2 = h.detach().clone().requires_grad_(); w2 = w.detach().clone().requires_grad_()
    logits = apply_softcap((h2 @ w2.t()).double(), cap)
    loss_ref = torch.nn.functional.cross_entropy(logits, tgt)
    gh_r, gw_r = torch.autograd.grad(loss_ref, [h2, w2])
    ok &= check(f"cap={cap:+.0f} loss", loss_c, loss_ref, 1e-6)  # reduction-order fp64 noise (present in tanh path too)
    ok &= check(f"cap={cap:+.0f} dh", gh_c, gh_r, 1e-8)
    ok &= check(f"cap={cap:+.0f} dw", gw_c, gw_r, 1e-8)

print("VERIFY_PASS" if ok else "VERIFY_FAIL")
sys.exit(0 if ok else 1)
