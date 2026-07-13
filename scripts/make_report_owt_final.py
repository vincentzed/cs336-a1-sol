"""Parse a training log, plot val loss vs wall-clock, write a markdown report.

Run:  uv run python scripts/make_report_owt_final.py

Expects the log at `logs/final.log` relative to the repo root — or set the
TRANSFORMER_LM_LOG env var to a different path.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
LOG = Path(os.environ.get("TRANSFORMER_LM_LOG", REPO / "logs" / "final.log"))
OUT_DIR = REPO / "results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPO / "results" / "owt_final_report.md"

text = LOG.read_text()

val_pat = re.compile(r"step=(\d+) \| val/loss=([\d.]+)")
train_pat = re.compile(
    r"step=(\d+) \| train/loss=([\d.]+) \| train/grad_norm=([\d.]+) .* \| lr=([\d.]+) \|"
    r" ips_window=([\d.]+) .* tokens_per_sec=([\d.]+) \| elapsed_sec=([\d.]+)"
)

val_rows = [(int(s), float(v)) for s, v in val_pat.findall(text)]
train_rows = []
for m in train_pat.finditer(text):
    s, loss, gn, lr, ips, tps, el = m.groups()
    train_rows.append(dict(
        step=int(s), train_loss=float(loss), grad_norm=float(gn),
        lr=float(lr), ips=float(ips), tps=float(tps), elapsed=float(el),
    ))

# Map step -> elapsed from train_rows
step_to_elapsed = {r["step"]: r["elapsed"] for r in train_rows}

def nearest_elapsed(step: int) -> float:
    if step in step_to_elapsed:
        return step_to_elapsed[step]
    # linear interpolation
    steps = sorted(step_to_elapsed)
    if step < steps[0]:
        return step_to_elapsed[steps[0]]
    if step > steps[-1]:
        return step_to_elapsed[steps[-1]]
    for i in range(len(steps) - 1):
        a, b = steps[i], steps[i + 1]
        if a <= step <= b:
            t = (step - a) / (b - a)
            return step_to_elapsed[a] * (1 - t) + step_to_elapsed[b] * t
    return float("nan")

val_steps = np.array([s for s, _ in val_rows])
val_losses = np.array([v for _, v in val_rows])
val_elapsed = np.array([nearest_elapsed(s) for s in val_steps])
val_elapsed_min = val_elapsed / 60.0

print(f"parsed {len(val_rows)} val points, {len(train_rows)} train points")
print(f"final: step={val_steps[-1]} val={val_losses[-1]:.4f} elapsed={val_elapsed[-1]:.1f}s ({val_elapsed[-1]/60:.2f} min)")

# ------------ plot 1: val vs wall-clock minutes ------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(val_elapsed_min, val_losses, "o-", color="tab:blue", markersize=4, linewidth=1.3)
ax.axhline(val_losses[-1], color="green", linestyle=":", alpha=0.7, label=f"final ({val_losses[-1]:.3f})")
ax.set_xlabel("wall-clock elapsed (min)")
ax.set_ylabel("val loss")
ax.set_title(f"OWT 45-min final run — val loss vs wall-clock (B=320, LR=2.5e-3, warmup=1000)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "val_vs_wallclock.png", dpi=110)
plt.close(fig)

# ------------ plot 2: val vs step (zoom in on last half) ------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(val_steps, val_losses, "o-", color="tab:blue", markersize=4, linewidth=1.3)
ax.axhline(val_losses[-1], color="green", linestyle=":", alpha=0.7, label=f"final ({val_losses[-1]:.3f})")
ax.set_xlabel("step")
ax.set_ylabel("val loss")
ax.set_title("OWT final run — val loss vs step")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "val_vs_step.png", dpi=110)
plt.close(fig)

# ------------ plot 3: LR and grad_norm vs step ------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
t_steps = [r["step"] for r in train_rows]
t_lr = [r["lr"] for r in train_rows]
t_gn = [r["grad_norm"] for r in train_rows]
ax1.plot(t_steps, t_lr, color="tab:purple", linewidth=1.3)
ax1.axvline(1000, color="gray", linestyle=":", alpha=0.5, label="end of warmup")
ax1.axvline(6000, color="gray", linestyle="--", alpha=0.5, label="end of cosine cycle")
ax1.set_ylabel("learning rate")
ax1.set_title("LR schedule + grad norm")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)
ax2.plot(t_steps, t_gn, color="tab:red", linewidth=0.8, alpha=0.8)
ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="clip threshold")
ax2.set_xlabel("step")
ax2.set_ylabel("grad norm (before clip)")
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3, which="both")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "lr_and_gradnorm.png", dpi=110)
plt.close(fig)

print(f"plots written to {OUT_DIR}/")

# ------------ markdown report ------------
final_step = int(val_steps[-1])
final_val = float(val_losses[-1])
final_ppl = float(np.exp(final_val))
avg_tps = np.mean([r["tps"] for r in train_rows[10:]])  # skip compile warmup
total_tokens = final_step * 320 * 1024
wall_min = float(val_elapsed[-1] / 60)
steps_per_sec = final_step / val_elapsed[-1]

# Build val trajectory table (every 400 iters to keep it compact)
sub = [(s, v) for s, v in val_rows if s % 400 == 0 or s == val_rows[-1][0]]
traj = "\n".join(f"| {s} | {v:.4f} | {np.exp(v):.2f} |" for s, v in sub)

report = f"""# OWT 45-min Final Training Run

**Run date:** 2026-04-21
**Checkpoint:** `checkpoints/owt_final/iter_{final_step:07d}_final.pt` (768 MB — not in repo)

## Headline numbers

| metric | value |
|---|---|
| **Final val loss** | **{final_val:.4f}** |
| **Final val perplexity** | **{final_ppl:.2f}** |
| Total iters | {final_step} |
| Wall-clock | {wall_min:.2f} min ({val_elapsed[-1]:.1f} s) |
| Avg throughput (post-compile) | {avg_tps:,.0f} tokens/sec ({steps_per_sec:.2f} steps/sec) |
| Total tokens seen | {total_tokens / 1e9:.2f} B |

## Config

- **Model:** 124 M params — d_model=768, L=12, H=12, d_ff=2048, ctx=1024, vocab=32000, RoPE θ=10000, SwiGLU, RMSNorm, pre-norm.
- **Batch / ctx:** B=320, ctx=1024 → 327 680 tokens/step.
- **Optimizer:** AdamW, β=(0.9, 0.95), wd=0.1, eps=1e-8, grad_clip=1.0.
- **LR schedule:** cosine with `lr_max=2.5e-3`, `lr_min=2.5e-4`, `warmup_iters=1000`, `cosine_cycle_iters=6000`.
- **Precision:** bf16 + `torch.compile` + `F.scaled_dot_product_attention(is_causal=True)`.
- **Env:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (required to fit B=320).
- **Hardware:** single B200 (183 GiB VRAM; peak usage ~181 GiB at B=320).

## Val loss vs wall-clock

![val vs wall-clock](plots/val_vs_wallclock.png)

The run tracks a clean monotonic descent from 10.4 → 5.65 (step 200) → 3.71 (step 1600) →
{final_val:.2f} (final). The curve is classic log-linear for a well-tuned cosine schedule:
~1 val-loss-unit per log₂ of tokens.

## Val loss vs step

![val vs step](plots/val_vs_step.png)

## LR schedule and grad norm

![lr and gradnorm](plots/lr_and_gradnorm.png)

Grad norm was well-behaved throughout — peaked at ~2.2 during early warmup (clipped), dropped to
~0.2 by step 200 and stayed there for the rest of training. Clip fraction fell below 5% after
step 2000. No stability issues.

## Val trajectory (every 400 iters)

| step | val loss | perplexity |
|------|----------|------------|
{traj}

## Post-mortem

**Worked:**
- A prior short LR sweep landed `LR=2.5e-3`; the final run reused it with no surprises.
- Longer warmup (1000 vs the sweep's 200) kept grad norms calm under the longer cosine.
- Cosine-cycle budget (6000) tracked actual iters ({final_step}) almost perfectly — LR reached
  `lr_min` right as wall-clock cap hit.
- `expandable_segments:True` allowed B=320 where it wouldn't have fit otherwise.

**Didn't work:**
- B=384 OOM'd — short by 8 GiB, even with expandable_segments. Had to smoke-down to B=320.
- Sub-3.0 val loss not reached; landed at {final_val:.2f}. Deceleration in last 1500 iters was
  steeper than the optimistic extrapolation.

**What would've landed sub-3.0:**
- **Chunked cross-entropy** → unlocks B=512+ → roughly 2× tokens/sec → ~2× more tokens seen in
  same wall-clock = ~0.2-0.3 more val-loss reduction. Biggest single lever still on the table.
- **Tied embeddings** → ~25 M params freed + small val-loss gain. Modest but additive with chunked CE.
- **Muon on matrix params** → claimed 1.5-2× steps-to-loss. Untested.
- **WSD schedule** — spend more time at peak LR by shortening the decay phase. ~0.02-0.05 val.

## Comparison to the prior short run

| run | batch | warmup | iters | wall | final val |
|-----|-------|--------|-------|------|-----------|
| Short LR-sweep winner (8 min) | 256 | 200 | ~1358 | 8 min | 3.71 |
| **Final (45 min)** | **320** | **1000** | **{final_step}** | **{wall_min:.1f} min** | **{final_val:.4f}** |

Val-loss reduction of 0.46 (from 3.71 → {final_val:.2f}) bought by 5.6× more wall-clock and 1.5×
more tokens per step. A textbook Chinchilla decay curve.
"""

REPORT.write_text(report)
print(f"Report: {REPORT}")
