"""CPU verification of per-head VE gates (Feature 1).

  1. Init parity: with gates at init (scalar zero), k=5 per-head loss == k=0 loss, EXACT.
  2. Init parity: k=5 per-head == k=5 scalar at init (both zero -> ve inactive).
  3. Optimizer routing: _ve_gate_w (num_heads,12) lands in AdamW no-wd, NEVER Muon;
     value_embeds_k in the dedicated ve group (betas 0.75/0.95).
  4. Gate becomes active once the scalar is nonzero (sanity: not a dead branch).
"""
import sys, torch
sys.path.insert(0, "/tmp/claude-1012/-home-brayden/49c069ec-6b97-4caf-8b5f-c5a81a8dd40f/scratchpad/sol")
torch.manual_seed(0)
from transformer_lm.train_sol import build_model, parse_args
from transformer_lm.sol_modules import (enable_value_embeds_k, build_optimizers,
                                        sol_cross_entropy, NorMuonGNS)

def make(argv_extra):
    argv = ["e", "--train-tokens", "x", "--vocab-size", "256", "--batch-size", "1",
            "--d-model", "128", "--num-layers", "6", "--num-heads", "4", "--d-ff", "256",
            "--max-seq-len", "64", "--context-length", "64", "--logit-softcap", "0",
            "--z-loss", "0", "--wandb-mode", "disabled", "--attn-backend", "torch",
            "--rmsnorm", "torch"] + argv_extra
    sys.argv = argv
    args = parse_args()
    args.device = "cpu"
    args.dtype = "float32"
    torch.manual_seed(0)
    m = build_model(args)
    return m, args

ok = True
V, T = 256, 64
x = torch.randint(0, V, (2, T))
y = torch.randint(0, V, (2, T))

def loss_of(m):
    w = m.token_embeddings.embeddings if m.lm_head.weights is m.token_embeddings.embeddings else m.lm_head.weights
    with torch.no_grad():
        h = m(x)
        return sol_cross_entropy(h, w, y, mode="chunked", softcap=0.0).item()

# baseline k=0
m0, a0 = make([])
l0 = loss_of(m0)
# k=5 per-head at init (rebuild identically, then enable)
m1, a1 = make([])
enable_value_embeds_k(m1, 5, gate_mode="per-head")
l1 = loss_of(m1)
# k=5 scalar at init
m2, a2 = make([])
enable_value_embeds_k(m2, 5, gate_mode="scalar")
l2 = loss_of(m2)

print(f"[1] init parity per-head vs k=0:  {l1:.10f} vs {l0:.10f}  Δ={abs(l1-l0):.2e}", 
      "PASS" if abs(l1-l0) < 1e-9 else "FAIL"); ok &= abs(l1-l0) < 1e-9
print(f"[2] init parity per-head vs scalar: Δ={abs(l1-l2):.2e}", 
      "PASS" if abs(l1-l2) < 1e-9 else "FAIL"); ok &= abs(l1-l2) < 1e-9

# [3] routing
muon, adamw = build_optimizers(m1, muon_lr=1e-3, adam_lr=3e-3, embed_lr=6e-3,
                               weight_decay=0.1, ve_lr=None)
muon_ids = {id(p) for g in muon.param_groups for p in g["params"]}
gate_w = [p for n, p in m1.named_parameters() if "_ve_gate_w" in n]
vet = [p for n, p in m1.named_parameters() if "value_embeds_k" in n]
in_muon = any(id(p) in muon_ids for p in gate_w)
print(f"[3a] {len(gate_w)} per-head gate weights, in Muon: {in_muon}",
      "PASS" if (gate_w and not in_muon) else "FAIL"); ok &= bool(gate_w) and not in_muon
# ve group betas
ve_grp = [g for g in adamw.param_groups if any(id(p) in {id(q) for q in vet} for p in g["params"])]
betas_ok = ve_grp and ve_grp[0]["betas"] == (0.75, 0.95)
print(f"[3b] value_embeds_k betas {ve_grp[0]['betas'] if ve_grp else None}",
      "PASS" if betas_ok else "FAIL"); ok &= bool(betas_ok)
# gate_w in an AdamW group, no weight decay
gate_grp = [g for g in adamw.param_groups if any(id(p) in {id(q) for q in gate_w} for p in g["params"])]
gwd = gate_grp and gate_grp[0]["weight_decay"] == 0.0
print(f"[3c] gate weights in AdamW wd={gate_grp[0]['weight_decay'] if gate_grp else None}",
      "PASS" if gwd else "FAIL"); ok &= bool(gwd)

# [4] gate activates when scalar != 0
with torch.no_grad():
    for n, p in m1.named_parameters():
        if n.endswith("_ve_gate"):  # scalar
            p.fill_(0.5)
l_active = loss_of(m1)
print(f"[4] loss changes when scalar activated: Δ={abs(l_active-l1):.2e}",
      "PASS" if abs(l_active-l1) > 1e-4 else "FAIL"); ok &= abs(l_active-l1) > 1e-4

print("VERIFY_PASS" if ok else "VERIFY_FAIL")
sys.exit(0 if ok else 1)
