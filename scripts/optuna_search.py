"""Optuna TPE search over the champion (L20) config using SCALED-REPLICA trials.

Design decisions, earned the hard way:
  * Trials are 900s scaled replicas (total-iters ~3050, warmup/momentum/decay all
    scaled proportionally) -- NOT truncated runs. Truncation amputates the decay
    tail and we measured that ranking under truncation lies (cautious-wd inverted
    between a 2200-step probe and the full run). A schedule-complete miniature
    preserves ranking far better; the top-3 get full 45-min validation after.
  * Trial scoring uses the FIXED-protocol subsample eval (post-c06a0ef): same val
    windows for every trial, so the objective is comparable across trials with
    sigma ~0.003 (TPE tolerates that noise at these trial counts).
  * SQLite storage + N worker processes (one per GPU) = parallel ask/tell.

Run one worker per free GPU, e.g.:
  CUDA_VISIBLE_DEVICES=4 python scripts/optuna_search.py --gpu-tag g4 &
  CUDA_VISIBLE_DEVICES=5 python scripts/optuna_search.py --gpu-tag g5 &
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna

E = "/tmp/claude-1012/-home-brayden/49c069ec-6b97-4caf-8b5f-c5a81a8dd40f/scratchpad/sol_exp"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = f"sqlite:///{E}/optuna_sol.db"
TRIAL_WALL = 900.0
TRIAL_ITERS = 3050          # ~295ms/step on B300 at the L20 config
FULL_ITERS = 9350           # the 45-min budget these HPs must transfer to


def run_trial(trial: optuna.Trial) -> float:
    muon_lr = trial.suggest_float("muon_lr", 4e-3, 2e-2, log=True)
    adam_ratio = trial.suggest_float("adam_ratio", 1.0, 2.5)
    embed_ratio = trial.suggest_float("embed_ratio", 1.5, 4.5)
    decay_frac = trial.suggest_float("wsd_decay_frac", 0.55, 0.92)
    lr_min = trial.suggest_float("lr_min_ratio", 0.02, 0.15, log=True)
    warmup_frac = trial.suggest_float("warmup_frac", 0.02, 0.09)
    wd = trial.suggest_float("weight_decay", 0.03, 0.3, log=True)
    ema_decay = trial.suggest_categorical("ema_decay", [0.998, 0.999, 0.9995])
    mom_warm_frac = trial.suggest_float("mom_warm_frac", 0.01, 0.06)
    mom_decay_last = trial.suggest_categorical("mom_decay_last_frac", [0.0, 0.005, 0.015])

    warmup = max(20, int(warmup_frac * TRIAL_ITERS))
    name = f"opt_t{trial.number}"
    cmd = [
        f"{E}/.venv/bin/python", "-m", "transformer_lm.train_sol",
        "--train-tokens", f"{E}/data/owt_train_full.npy",
        "--val-tokens", f"{E}/data/owt_valid.npy", "--vocab-size", "32000",
        "--d-model", "1024", "--num-layers", "20", "--num-heads", "8", "--d-ff", "4096",
        "--max-seq-len", "512", "--context-length", "512", "--batch-size", "256",
        "--total-iters", str(TRIAL_ITERS), "--max-wall-sec", str(TRIAL_WALL),
        "--warmup-iters", str(warmup),
        "--lr-schedule", "wsd", "--wsd-decay-frac", f"{decay_frac}",
        "--lr-min-ratio", f"{lr_min}", "--weight-decay", f"{wd}",
        "--muon-lr", f"{muon_lr}", "--adam-lr", f"{muon_lr * adam_ratio}",
        "--embed-lr", f"{muon_lr * embed_ratio}",
        "--ce-mode", "quack", "--logit-softcap", "0", "--z-loss", "0",
        "--attn-backend", "fa4op", "--rmsnorm", "quack", "--fused-rope", "--fused-qkv",
        "--muon-momentum-warmup", str(max(10, int(mom_warm_frac * TRIAL_ITERS))),
        "--muon-momentum-decay-last", str(int(mom_decay_last * TRIAL_ITERS)),
        "--grad-clip", "0", "--cautious-wd", "--bf16-mt", "--value-embeds",
        "--x0-lambdas", "--unet-skips", "--smear", "--second-embed", "--xsa",
        "--ema-decay", f"{ema_decay}", "--eval-ctx", "512",
        "--val-interval", "100000", "--val-batches", "40",
        "--data-sampling", "shuffled",
        "--wandb-mode", "disabled", "--compile",
        "--checkpoint-dir", f"{E}/ckpt/{name}", "--checkpoint-interval", "1000000",
        "--compile-cache", f"{E}/ckpt/optuna_mega.cache",  # trials share ONE graph: compile once, kill the 3x compile storms
    ]
    env = {**os.environ, "TORCHINDUCTOR_CACHE_DIR": "/tmp/ti_optuna",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=TRIAL_WALL + 600)
    out = r.stdout + r.stderr
    vals = re.findall(r"val/loss=([0-9.]+)", out)
    if r.returncode != 0 or not vals:
        raise optuna.TrialPruned()  # crashed configs just get pruned
    return float(vals[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-tag", default="g?")
    ap.add_argument("--n-trials", type=int, default=20)
    args = ap.parse_args()
    study = optuna.create_study(
        study_name="l20_hp_v1", storage=DB, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=None),
    )
    # anchor trial: the current champion HPs (so TPE starts from the known optimum)
    if len(study.trials) == 0 and args.gpu_tag.endswith("4"):
        study.enqueue_trial({"muon_lr": 8e-3, "adam_ratio": 1.5, "embed_ratio": 3.0,
                             "wsd_decay_frac": 0.8, "lr_min_ratio": 0.067,
                             "warmup_frac": 500 / FULL_ITERS, "weight_decay": 0.1,
                             "ema_decay": 0.999, "mom_warm_frac": 300 / FULL_ITERS,
                             "mom_decay_last_frac": 0.0})
    study.optimize(run_trial, n_trials=args.n_trials, gc_after_trial=True)
    print(f"[{args.gpu_tag}] done. best={study.best_value:.4f} params={study.best_params}")


if __name__ == "__main__":
    main()
