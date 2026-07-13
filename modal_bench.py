"""Run the CS336 SOL bench on Modal B200s (sm100). Self-contained: tokenizes OWT
on Modal into a Volume, then fans baseline + SOL configs across B200s in parallel.

After `modal token new`:
  modal secret create wandb WANDB_API_KEY=<key>
  modal run modal_bench.py          # prep_data (once) -> train all configs

Modal B200 is real sm100 -> exact match for FA4's sm100 kernels.
"""
import modal

app = modal.App("cs336-sol-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.12.1", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("einops", "jaxtyping", "numpy", "psutil", "regex", "tqdm", "wandb")
    .pip_install("nvidia-cudnn-cu13==9.24.0.43", "nvidia-cudnn-frontend>=1.26")
    .pip_install("quack-kernels")                       # FA4 resolves 0.5.3 — SAME as the verified local env
    .pip_install("gram-newton-schulz==0.1.6", extra_options="--no-deps")
    .pip_install("flash-attn-4[cu13]", extra_options="--pre --no-build-isolation")  # pip uses --pre (not uv's --prerelease)
    # NB: do NOT re-pin quack 0.6.1 here — 0.6.1-on-B200 is a hang suspect; 0.5.3 verified locally
    # (both versions have the bias positional in linear_cross_entropy_func).
    .add_local_dir("transformer_lm", "/root/transformer_lm")
    .add_local_dir("tokenizers", "/root/tokenizers")
    .add_local_dir("scripts", "/root/scripts")
)
vol = modal.Volume.from_name("cs336-owt-data", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb")
OWT = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main"


@app.function(image=image, cpu=32.0, timeout=3600, volumes={"/data": vol})
def prep_data(train_gb: int = 4):
    """Download OWT + tokenize (32k BPE) -> /data/owt_{train,valid}.npy. Idempotent.
    Pure-Python download/decompress (no wget/gunzip in the minimal image)."""
    import gzip
    import os
    import sys
    import urllib.request
    if os.path.exists("/data/owt_train.npy") and os.path.exists("/data/owt_valid.npy"):
        print("data already present; skipping", flush=True); return

    def fetch(split, max_bytes=None):
        gzp, txt = f"/tmp/owt_{split}.txt.gz", f"/tmp/owt_{split}.txt"
        print(f"downloading {split}...", flush=True)
        urllib.request.urlretrieve(f"{OWT}/owt_{split}.txt.gz", gzp)
        n = 0
        with gzip.open(gzp, "rb") as fi, open(txt, "wb") as fo:
            while max_bytes is None or n < max_bytes:
                b = fi.read(1 << 24)
                if not b:
                    break
                fo.write(b); n += len(b)
        print(f"{split}: wrote {n:,} bytes", flush=True); return txt

    valid_txt = fetch("valid")
    train_txt = fetch("train", max_bytes=train_gb * 10**9)
    sys.path.insert(0, "/root")
    import numpy as np
    from transformer_lm.tokenizer import Tokenizer
    tok = Tokenizer.from_files("/root/tokenizers/owt/vocab.pkl", "/root/tokenizers/owt/merges.pkl",
                               special_tokens=["<|endoftext|>"])
    for src, out in [(valid_txt, "/data/owt_valid.npy"), (train_txt, "/data/owt_train.npy")]:
        arr = tok.encode_file(src, show_progress=True, num_workers=32)
        np.save(out, arr); print(f"SAVED {out}: {arr.shape[0]:,} tokens", flush=True)
    vol.commit()


COMMON = [
    "--train-tokens", "/data/owt_train.npy", "--val-tokens", "/data/owt_valid.npy",
    "--vocab-size", "32000", "--d-model", "768", "--num-layers", "12", "--num-heads", "12",
    "--d-ff", "2048", "--max-seq-len", "1024", "--context-length", "1024",
    "--total-iters", "100000", "--max-wall-sec", "2700", "--warmup-iters", "500",
    "--weight-decay", "0.1", "--dtype", "bfloat16", "--compile",
    "--wandb-project", "cs336-a1-sol", "--wandb-mode", "online",
]
CONFIGS = {
    # B200 = 178 GiB. batch 320 for all (faithful to the leaderboard B200 hw) -> clean A/B.
    "baseline": ("transformer_lm.train_script", ["--batch-size", "320", "--lr-max", "2.5e-3", "--lr-min", "2.5e-4"]),
    "sol_wsd":  ("transformer_lm.train_sol", ["--batch-size", "320", "--ce-chunk", "8192", "--muon-lr", "2e-3",
                                              "--adam-lr", "3e-3", "--embed-lr", "6e-3", "--attn-backend", "fa4",
                                              "--lr-schedule", "wsd"]),
    "sol_cos":  ("transformer_lm.train_sol", ["--batch-size", "320", "--ce-chunk", "8192", "--muon-lr", "2e-3",
                                              "--adam-lr", "3e-3", "--embed-lr", "6e-3", "--attn-backend", "fa4",
                                              "--lr-schedule", "cosine"]),
}


@app.function(image=image, gpu="B200", timeout=5400, volumes={"/data": vol},
              secrets=[wandb_secret], retries=0)
def train(name: str, module: str, extra: list[str]):
    import os
    import subprocess
    import sys
    cmd = [sys.executable, "-m", module, *COMMON, *extra,
           "--checkpoint-dir", f"/data/ckpt/{name}", "--wandb-run-name", name]
    # quack cache: NEVER point QUACK_HOME at the Volume (FileLock on FUSE can hang forever).
    # Instead: copy a persisted cache in from the volume, run on local disk, copy back after.
    import shutil
    if os.path.isdir("/data/quack_cache"):
        shutil.copytree("/data/quack_cache", "/root/.quack", dirs_exist_ok=True)
        print("quack cache warmed from volume", flush=True)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # reclaim fragmentation
    print(f"[{name}] launching", flush=True)
    rc = subprocess.run(cmd, cwd="/root", env=env).returncode
    if os.path.isdir("/root/.quack"):  # persist autotune/JIT cache for the next container
        shutil.copytree("/root/.quack", "/data/quack_cache", dirs_exist_ok=True)
        vol.commit()
        print("quack cache persisted to volume", flush=True)
    return name, rc


@app.local_entrypoint()
def main():
    prep_data.remote()  # tokenize into the volume (idempotent) before training
    handles = [train.spawn(n, m, e) for n, (m, e) in CONFIGS.items()]
    for h in handles:
        name, rc = h.get()
        print(f"DONE {name}: exit={rc}")


# ---- throughput probe: measure steady-state tok/s for fullgraph variants ----
PROBE_COMMON = [
    "--train-tokens", "/data/owt_train.npy", "--vocab-size", "32000",
    "--d-model", "768", "--num-layers", "12", "--num-heads", "12", "--d-ff", "2048",
    "--max-seq-len", "1024", "--context-length", "1024", "--batch-size", "320",
    "--ce-chunk", "8192", "--muon-lr", "2e-3", "--adam-lr", "3e-3", "--embed-lr", "6e-3",
    "--lr-schedule", "wsd", "--warmup-iters", "50",
    "--total-iters", "150", "--val-interval", "100000", "--dtype", "bfloat16",
    "--compile", "--wandb-mode", "disabled", "--no-ema",
]


@app.function(image=image, gpu="B200", timeout=1800, volumes={"/data": vol})
def probe(label: str, extra: list[str]):
    import os
    import re
    import subprocess
    import sys
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    r = subprocess.run([sys.executable, "-m", "transformer_lm.train_sol", *PROBE_COMMON, *extra],
                       cwd="/root", env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    tps = re.findall(r"tokens_per_sec=([0-9.]+)", out)
    if r.returncode != 0 or not tps:
        print(f"[{label}] rc={r.returncode} — output tail:\n{out[-3000:]}", flush=True)
    vals = [float(x) for x in tps][-5:]
    return label, (sorted(vals)[len(vals) // 2] if vals else None), r.returncode


R4BASE = [
    "--train-tokens", "/data/owt_train_full.npy",
    "--d-model", "1024", "--num-layers", "16", "--num-heads", "8", "--d-ff", "4096",
    "--max-seq-len", "512", "--context-length", "512", "--batch-size", "256",
    "--total-iters", "6800", "--warmup-iters", "500",
    "--lr-schedule", "wsd", "--wsd-decay-frac", "0.8", "--lr-min-ratio", "0.067",
    "--muon-lr", "8e-3", "--adam-lr", "1.2e-2", "--embed-lr", "2.4e-2",
    "--ce-mode", "quack", "--logit-softcap", "0", "--z-loss", "0",
    "--attn-backend", "fa4", "--rmsnorm", "quack", "--fused-rope", "--fused-qkv",
    "--muon-momentum-warmup", "300", "--grad-clip", "0", "--cautious-wd", "--bf16-mt",
    "--eval-ctx", "512",
]


@app.local_entrypoint()
def hp_main():
    """B200-internal HP micro-grid on the r4_nobos-class config (5 parallel B200s).
    Self-contained comparison — valid within-B200, independent of local B300 waves."""
    grid = {
        "hp_control":  [],
        "hp_wd005":    ["--weight-decay", "0.05"],
        "hp_warm300":  ["--warmup-iters", "300"],
        "hp_lrmin003": ["--lr-min-ratio", "0.03"],
        "hp_adamr":    ["--adam-lr", "8e-3", "--embed-lr", "1.6e-2"],
    }
    hs = [train.spawn(f"b200_{n}", "transformer_lm.train_sol", R4BASE + extra)
          for n, extra in grid.items()]
    for h in hs:
        name, rc = h.get()
        print(f"DONE {name}: exit={rc}")


@app.local_entrypoint()
def repro_main():
    """Faithful repro of StuffByLiang's published 3.25 recipe on B200: their EXACT
    flags (warmup 1000, cosine-cycle 6000, total 10000, wall-stop) + FULL data
    (owt_train_full.npy on the volume). Later args override COMMON (argparse last-wins)."""
    name, rc = train.remote("baseline_repro_full", "transformer_lm.train_script", [
        "--train-tokens", "/data/owt_train_full.npy",
        "--batch-size", "320", "--total-iters", "10000",
        "--lr-max", "2.5e-3", "--lr-min", "2.5e-4",
        "--warmup-iters", "1000", "--cosine-cycle-iters", "6000",
        "--val-interval", "200",
    ])
    print(f"DONE {name}: exit={rc}")


@app.local_entrypoint()
def quack_main():
    """Headline SOL run with fixed quack CE on B200 (leaderboard hw). total-iters set to
    achievable steps so WSD actually decays (later args override COMMON's 100000)."""
    prep_data.remote()
    name, rc = train.remote("sol_quack_b200", "transformer_lm.train_sol", [
        "--batch-size", "320", "--ce-mode", "quack", "--logit-softcap", "0", "--z-loss", "0",
        "--attn-backend", "fa4", "--lr-schedule", "wsd", "--muon-lr", "2e-3",
        "--adam-lr", "3e-3", "--embed-lr", "6e-3", "--eval-ctx", "512",
        "--total-iters", "4300",
    ])
    print(f"DONE {name}: exit={rc}")


@app.local_entrypoint()
def probe_main():
    # B200 smoke: does quack-CE (0.5.3) train at all on B200? 150 iters under the probe's 1800s timeout.
    variants = {
        "quack_b200_smoke": ["--attn-backend", "fa4", "--ce-mode", "quack",
                             "--logit-softcap", "0", "--z-loss", "0"],
    }
    hs = {k: probe.spawn(k, v) for k, v in variants.items()}
    for k, h in hs.items():
        label, tps, rc = h.get()
        print(f"PROBE {label}: steady_tok/s={tps} exit={rc}")


@app.local_entrypoint()
def l20_main():
    """B200 delta measurement at the CURRENT champion config (r11_L20).
    Iters = 9139 (B300) x 0.62 measured B200/B300 throughput ~ 5700."""
    name, rc = train.remote("b200_L20_champion", "transformer_lm.train_sol", [
        "--train-tokens", "/data/owt_train_full.npy",
        "--d-model", "1024", "--num-layers", "20", "--num-heads", "8", "--d-ff", "4096",
        "--max-seq-len", "512", "--context-length", "512", "--batch-size", "256",
        "--total-iters", "7800", "--warmup-iters", "500", "--schedule-by-wall",
        "--lr-schedule", "wsd", "--wsd-decay-frac", "0.8", "--lr-min-ratio", "0.067",
        "--muon-lr", "8e-3", "--adam-lr", "1.2e-2", "--embed-lr", "2.4e-2",
        "--ce-mode", "quack", "--logit-softcap", "0", "--z-loss", "0",
        "--attn-backend", "fa4op", "--rmsnorm", "quack", "--fused-rope", "--fused-qkv",
        "--muon-momentum-warmup", "300", "--grad-clip", "0", "--cautious-wd", "--bf16-mt",
        "--value-embeds", "--x0-lambdas", "--unet-skips", "--smear", "--second-embed", "--xsa",
        "--eval-ctx", "512",
    ])
    print(f"DONE {name}: exit={rc}")


@app.local_entrypoint()
def ve5_main():
    """B200 confirmation of the ve5+cap20 record config (canon-EMA est ~2.946 on B300).
    Wall-scheduled so the decay completes on any node speed."""
    name, rc = train.remote("b200_ve5cap20", "transformer_lm.train_sol", [
        "--train-tokens", "/data/owt_train_full.npy",
        "--d-model", "1024", "--num-layers", "20", "--num-heads", "8", "--d-ff", "4096",
        "--max-seq-len", "512", "--context-length", "512", "--batch-size", "256",
        "--total-iters", "7750", "--warmup-iters", "500", "--schedule-by-wall",
        "--lr-schedule", "wsd", "--wsd-decay-frac", "0.8", "--lr-min-ratio", "0.067",
        "--muon-lr", "8e-3", "--adam-lr", "1.2e-2", "--embed-lr", "2.4e-2",
        "--weight-decay", "0.1", "--ema-decay", "0.999", "--muon-momentum-warmup", "300",
        "--ce-mode", "quack-softcap", "--logit-softcap", "20", "--z-loss", "0",
        "--attn-backend", "fa4op", "--rmsnorm", "quack", "--fused-rope", "--fused-qkv",
        "--muon-momentum-warmup", "300", "--grad-clip", "0", "--cautious-wd", "--bf16-mt",
        "--value-embeds-k", "5",
        "--x0-lambdas", "--unet-skips", "--smear", "--second-embed", "--xsa",
        "--data-sampling", "shuffled", "--eval-ctx", "512",
    ])
    print(f"DONE {name}: exit={rc}")


VE5CAP20 = [
    "--train-tokens", "/data/owt_train_full.npy",
    "--d-model", "1024", "--num-layers", "20", "--num-heads", "8", "--d-ff", "4096",
    "--max-seq-len", "512", "--context-length", "512", "--batch-size", "256",
    "--total-iters", "7750", "--warmup-iters", "500", "--schedule-by-wall",
    "--lr-schedule", "wsd", "--wsd-decay-frac", "0.8", "--lr-min-ratio", "0.067",
    "--muon-lr", "8e-3", "--adam-lr", "1.2e-2", "--embed-lr", "2.4e-2",
    "--weight-decay", "0.1", "--ema-decay", "0.999", "--muon-momentum-warmup", "300",
    "--ce-mode", "quack-softcap", "--logit-softcap", "20", "--z-loss", "0",
    "--attn-backend", "fa4op", "--rmsnorm", "quack", "--fused-rope", "--fused-qkv",
    "--grad-clip", "0", "--cautious-wd", "--bf16-mt", "--value-embeds-k", "5",
    "--x0-lambdas", "--unet-skips", "--smear", "--second-embed", "--xsa",
    "--data-sampling", "shuffled", "--eval-ctx", "512",
]


@app.local_entrypoint()
def wave_b200():
    """B200-native wave on the record config: seed replicates + k=7 tables.
    B200 is now the home frame (local box occupied; single-hardware comparisons)."""
    legs = {
        "b200_ve5_s1": ["--seed", "1"],
        "b200_ve5_s2": ["--seed", "2"],
        "b200_ve7":    ["--value-embeds-k", "7"],
    }
    hs = [train.spawn(n, "transformer_lm.train_sol", VE5CAP20 + extra) for n, extra in legs.items()]
    for h in hs:
        name, rc = h.get()
        print(f"DONE {name}: exit={rc}")


@app.function(image=image, gpu="B200", timeout=1800, volumes={"/data": vol})
def canon_b200(ckpt_dir: str, softcap: float, k: int = 5, layers: int = 20, form: str = "tanh"):
    """Canonical full-sweep eval of a volume checkpoint, on B200 (the home frame)."""
    import re
    import subprocess
    import sys
    arch = ["--num-layers", str(layers), "--value-embeds-k", str(k),
            "--x0-lambdas", "--unet-skips", "--smear", "--second-embed", "--xsa"]
    out = []
    for tag, extra in [("raw", []), ("ema", ["--ema", f"/data/ckpt/{ckpt_dir}/ema_final.pt"])]:
        import glob
        cks = sorted(glob.glob(f"/data/ckpt/{ckpt_dir}/iter_*_final.pt"))
        if not cks:
            return {"error": f"no ckpt in {ckpt_dir}"}
        r = subprocess.run([sys.executable, "scripts/eval_canon.py", cks[-1],
                            "--val-tokens", "/data/owt_valid.npy",
                            "--softcap", str(softcap), "--softcap-form", form, *extra, *arch],
                           cwd="/root", capture_output=True, text=True)
        m = re.search(r"val_loss=([0-9.]+)", r.stdout)
        out.append((tag, float(m.group(1)) if m else None))
    return dict(out)


@app.local_entrypoint()
def wave_b200_2():
    """Second B200 wave: k-bracket, depth/ffn/rope geometry at the ve5 config, cheap-stack LOO."""
    legs = {
        "b200_ve3":     ["--value-embeds-k", "3"],
        "b200_L16ve5":  ["--num-layers", "16", "--total-iters", "9300"],
        "b200_dff3584": ["--d-ff", "3584", "--total-iters", "8100"],
        "b200_prope75": ["--partial-rope", "0.75"],
        "b200_noxsa":   ["--no-xsa"],
        "b200_nosmear": ["--no-smear"],
    }
    hs = []
    for n, extra in legs.items():
        # LOO legs: our flags are store_true; emulate removal by rebuilding the arg list
        base = list(VE5CAP20)
        if extra == ["--no-xsa"]:
            base.remove("--xsa"); extra = []
        if extra == ["--no-smear"]:
            base.remove("--smear"); extra = []
        hs.append(train.spawn(n, "transformer_lm.train_sol", base + extra))
    for h in hs:
        name, rc = h.get()
        print(f"DONE {name}: exit={rc}")


L16BASE = [a for a in VE5CAP20]
L16BASE[L16BASE.index("20", L16BASE.index("--num-layers")) ] = "16"
L16BASE[L16BASE.index("7750", L16BASE.index("--total-iters"))] = "9300"


@app.local_entrypoint()
def wave_b200_3():
    """Wave 3: rebase on L16+ve5+cap20 (B200 record 2.9784). New levers + depth bracket + seeds."""
    legs = {
        "b200_L16_s1":    ["--seed", "1"],
        "b200_L16_s2":    ["--seed", "2"],
        "b200_L16_sig":   ["--logit-softcap", "23", "--softcap-form", "sigmoid"],
        "b200_L16_gates": ["--ve-gates", "per-head"],
        "b200_L16_both":  ["--logit-softcap", "23", "--softcap-form", "sigmoid", "--ve-gates", "per-head"],
        "b200_L14":       ["--num-layers", "14", "--total-iters", "10400"],
        "b200_L18":       ["--num-layers", "18", "--total-iters", "8400"],
    }
    hs = [train.spawn(n, "transformer_lm.train_sol", L16BASE + extra) for n, extra in legs.items()]
    cn = canon_b200.spawn("b200_L16ve5", 20.0, 5, 16)  # canon raw+EMA of the record ckpt
    for h in hs:
        name, rc = h.get()
        print(f"DONE {name}: exit={rc}")
    print("CANON b200_L16ve5:", cn.get())


@app.local_entrypoint()
def wave_b200_4():
    """Wave 4: compose L14 x per-head gates; bracket L12; replicate the winners."""
    legs = {
        "b200_L14g":    ["--num-layers", "14", "--total-iters", "10400", "--ve-gates", "per-head"],
        "b200_L14g_s1": ["--num-layers", "14", "--total-iters", "10400", "--ve-gates", "per-head", "--seed", "1"],
        "b200_L12":     ["--num-layers", "12", "--total-iters", "11900"],
        "b200_L12g":    ["--num-layers", "12", "--total-iters", "11900", "--ve-gates", "per-head"],
        "b200_L16g_s1": ["--ve-gates", "per-head", "--seed", "1"],
        "b200_L14_s1":  ["--num-layers", "14", "--total-iters", "10400", "--seed", "1"],
    }
    hs = [train.spawn(n, "transformer_lm.train_sol", L16BASE + extra) for n, extra in legs.items()]
    cn = canon_b200.spawn("b200_L16_gates", 20.0, 5, 16)
    for h in hs:
        name, rc = h.get()
        print(f"DONE {name}: exit={rc}")
    print("CANON b200_L16_gates:", cn.get())
