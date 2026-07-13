import argparse
import logging
import math
import os
import sys
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import wandb

from transformer_lm.modules import (
    AdamW,
    TransformerLm,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)
from transformer_lm.generate import sample as generate_sample
from transformer_lm.tokenizer import Tokenizer

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer LM on tokenized data.")

    # --- data ---
    data = p.add_argument_group("data")
    data.add_argument("--train-tokens", type=str, required=True,
                      help="Path to .npy/.bin of uint16 (or int) tokenized train data.")
    data.add_argument("--val-tokens", type=str, default=None,
                      help="Path to tokenized validation data. Optional.")
    data.add_argument("--tokens-dtype", type=str, default="uint16",
                      choices=["uint16", "int32", "int64"])

    # --- model ---
    model = p.add_argument_group("model")
    model.add_argument("--vocab-size", type=int, required=True)
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--num-heads", type=int, default=16)
    model.add_argument("--d-ff", type=int, default=1344)
    model.add_argument("--max-seq-len", type=int, default=256)
    model.add_argument("--rope-theta", type=float, default=10000.0)
    model.add_argument("--use-rmsnorm", action=argparse.BooleanOptionalAction, default=True,
                       help="Use RMSNorm; --no-use-rmsnorm replaces it with Identity.")
    model.add_argument("--pos-enc", type=str, default="rope", choices=["rope", "nope"])
    model.add_argument("--ffn", type=str, default="swiglu", choices=["swiglu", "silu"])

    # --- optimization ---
    opt = p.add_argument_group("optimization")
    opt.add_argument("--batch-size", type=int, default=32)
    opt.add_argument("--context-length", type=int, default=256)
    opt.add_argument("--total-iters", type=int, default=10_000)
    opt.add_argument("--max-wall-sec", type=float, default=None,
                     help="If set, break the loop once wall-clock elapsed >= this many seconds.")
    opt.add_argument("--lr-max", type=float, default=3e-4)
    opt.add_argument("--lr-min", type=float, default=3e-5)
    opt.add_argument("--warmup-iters", type=int, default=200)
    opt.add_argument("--cosine-cycle-iters", type=int, default=None,
                     help="Defaults to --total-iters.")
    opt.add_argument("--weight-decay", type=float, default=0.1)
    opt.add_argument("--beta1", type=float, default=0.9)
    opt.add_argument("--beta2", type=float, default=0.95)
    opt.add_argument("--eps", type=float, default=1e-8)
    opt.add_argument("--grad-clip", type=float, default=1.0,
                     help="Max L2 norm. Set <=0 to disable.")

    # --- checkpointing ---
    ckpt = p.add_argument_group("checkpointing")
    ckpt.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ckpt.add_argument("--checkpoint-interval", type=int, default=1000)
    ckpt.add_argument("--resume-from", type=str, default=None)

    # --- logging ---
    log = p.add_argument_group("logging")
    log.add_argument("--log-interval", type=int, default=10)
    log.add_argument("--val-interval", type=int, default=500)
    log.add_argument("--val-batches", type=int, default=20)
    log.add_argument("--log-level", type=str, default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    log.add_argument("--log-file", type=str, default=None,
                     help="Also write logs to this file.")
    log.add_argument("--wandb-project", type=str, default=None,
                     help="If set, log to this W&B project.")
    log.add_argument("--wandb-entity", type=str, default=None,
                     help="W&B entity (user or team). Omit to use your default.")
    log.add_argument("--wandb-run-name", type=str, default=None)
    log.add_argument("--wandb-mode", type=str, default="online",
                     choices=["online", "offline", "disabled"])
    log.add_argument("--sample-interval", type=int, default=0,
                     help="Generate a text sample every N steps. 0 = disabled.")
    log.add_argument("--sample-vocab", type=str, default=None,
                     help="Tokenizer vocab.pkl (required for --sample-interval).")
    log.add_argument("--sample-merges", type=str, default=None,
                     help="Tokenizer merges.pkl (required for --sample-interval).")
    log.add_argument("--sample-max-new-tokens", type=int, default=100)
    log.add_argument("--sample-temperature", type=float, default=0.8)
    log.add_argument("--sample-top-p", type=float, default=0.9)

    # --- runtime ---
    rt = p.add_argument_group("runtime")
    rt.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    rt.add_argument("--dtype", type=str, default="float32",
                    choices=["float32", "bfloat16", "float16"])
    rt.add_argument("--seed", type=int, default=0)
    rt.add_argument("--compile", action="store_true")

    args = p.parse_args()
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.total_iters
    return args


def setup_logging(level_name: str, log_file: str | None) -> None:
    level = getattr(logging, level_name.upper())
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)


def setup_wandb(args: argparse.Namespace) -> bool:
    """Returns True if wandb is active."""
    if args.wandb_project is None:
        logger.info("wandb disabled (no --wandb-project set)")
        return False

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )
    logger.info("wandb run started: %s", wandb.run.url if wandb.run else "(no url)")
    return True


def log_metrics(metrics: dict[str, Any], step: int, wandb_active: bool) -> None:
    """Log to console + wandb."""
    pieces = []
    for k, v in metrics.items():
        if isinstance(v, float):
            pieces.append(f"{k}={v:.4f}")
        else:
            pieces.append(f"{k}={v}")
    logger.info("step=%d | %s", step, " | ".join(pieces))

    if wandb_active and wandb.run is not None:
        wandb.log(metrics, step=step)


def load_tokens(path: str, dtype_str: str) -> np.ndarray:
    if path.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
    else:
        dtype = {"uint16": np.uint16, "int32": np.int32, "int64": np.int64}[dtype_str]
        arr = np.memmap(path, dtype=dtype, mode="r")
    logger.info("loaded %s (%d tokens, dtype=%s)", path, arr.shape[0], arr.dtype)
    return arr


def build_model(args: argparse.Namespace) -> TransformerLm:
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model = TransformerLm(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        theta=args.rope_theta,
        use_rmsnorm=args.use_rmsnorm,
        use_rope=(args.pos_enc == "rope"),
        activation=args.ffn,
        device=args.device,
        dtype=dtype,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model built: %.2fM params on %s (%s)", n_params / 1e6, args.device, args.dtype)
    return model


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_tokens: np.ndarray, args: argparse.Namespace) -> float:
    model.eval()
    losses = []
    for _ in range(args.val_batches):
        inputs, targets = get_batch(val_tokens, args.batch_size, args.context_length, args.device)
        logits = model(inputs)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    wandb_active = setup_wandb(args)

    train_tokens = load_tokens(args.train_tokens, args.tokens_dtype)
    val_tokens = load_tokens(args.val_tokens, args.tokens_dtype) if args.val_tokens else None

    model = build_model(args)
    if args.compile:
        if args.device == "mps":
            logger.info("torch.compile(backend='aot_eager') — Inductor not supported on MPS")
            model = torch.compile(model, backend="aot_eager")
        else:
            logger.info("torch.compile(default/Inductor)")
            model = torch.compile(model)
    optimizer = build_optimizer(model, args)

    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        logger.info("resumed from %s at iter %d", args.resume_from, start_iter)

    logger.info("naive loss floor log(vocab)=%.4f — step 0 should be near this", math.log(args.vocab_size))

    # Load tokenizer for periodic sample generation, if requested.
    sample_tokenizer: Tokenizer | None = None
    if args.sample_interval > 0:
        if not (args.sample_vocab and args.sample_merges):
            logger.warning("--sample-interval set but vocab/merges missing — disabling sampling")
        else:
            sample_tokenizer = Tokenizer.from_files(
                args.sample_vocab, args.sample_merges, special_tokens=["<|endoftext|>"]
            )
            logger.info("periodic sampling every %d steps", args.sample_interval)

    model.train()
    t_start = time.time()
    # Moving window: keep last N log samples as (iter, timestamp) for instantaneous throughput.
    recent: deque[tuple[int, float]] = deque(maxlen=20)
    recent.append((start_iter, t_start))

    clip_count = 0
    steps_done = 0
    stop_reason = "total_iters"
    last_it = start_iter

    for it in range(start_iter, args.total_iters):
        last_it = it
        inputs, targets = get_batch(train_tokens, args.batch_size, args.context_length, args.device)

        lr = get_lr_cosine_schedule(it, args.lr_max, args.lr_min, args.warmup_iters, args.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        logits = model(inputs)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()

        # Always compute grad norm (for logging). Clip only when max_l2_norm > 0.
        max_norm = args.grad_clip if args.grad_clip > 0 else float("inf")
        grad_norm = gradient_clipping(list(model.parameters()), max_norm).item()
        if args.grad_clip > 0 and grad_norm >= args.grad_clip:
            clip_count += 1
        steps_done += 1

        optimizer.step()

        now_step = time.time()
        elapsed_step = now_step - t_start
        if args.max_wall_sec is not None and elapsed_step >= args.max_wall_sec:
            stop_reason = "wall_clock"
            logger.info("wall-clock stop: elapsed=%.1fs >= max_wall_sec=%.1fs at iter %d",
                        elapsed_step, args.max_wall_sec, it)
            break

        if it % args.log_interval == 0:
            now = time.time()
            elapsed = now - t_start
            ips_avg = (it - start_iter + 1) / max(elapsed, 1e-9)

            recent.append((it, now))
            first_it, first_t = recent[0]
            window_iters = it - first_it
            window_time = now - first_t
            ips_window = window_iters / max(window_time, 1e-9) if window_iters > 0 else ips_avg

            remaining = max(args.total_iters - it - 1, 0)
            eta_sec = remaining / max(ips_window, 1e-9)
            eta_h, rem = divmod(int(eta_sec), 3600)
            eta_m, eta_s = divmod(rem, 60)

            # Total weight L2 norm — divergence often shows as weights exploding.
            with torch.no_grad():
                weight_norm = torch.sqrt(sum((p ** 2).sum() for p in model.parameters())).item()

            log_metrics(
                {
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm,
                    "train/weight_norm": weight_norm,
                    "train/clip_count": clip_count,
                    "train/clip_fraction": clip_count / max(steps_done, 1),
                    "lr": lr,
                    "ips_window": ips_window,
                    "ips_avg": ips_avg,
                    "tokens_per_sec": ips_window * args.batch_size * args.context_length,
                    "elapsed_sec": elapsed,
                    "eta": f"{eta_h:d}:{eta_m:02d}:{eta_s:02d}",
                },
                step=it,
                wandb_active=wandb_active,
            )

        if val_tokens is not None and it > 0 and it % args.val_interval == 0:
            val_loss = evaluate(model, val_tokens, args)
            log_metrics({"val/loss": val_loss}, step=it, wandb_active=wandb_active)

        if sample_tokenizer is not None and it > 0 and it % args.sample_interval == 0:
            # Unwrap torch.compile so sampling doesn't trigger dynamic-shape recompiles.
            inner = getattr(model, "_orig_mod", model)
            inner.eval()
            try:
                text = generate_sample(
                    inner,
                    sample_tokenizer,
                    prompt="",
                    max_new_tokens=args.sample_max_new_tokens,
                    temperature=args.sample_temperature,
                    top_p=args.sample_top_p,
                    device=args.device,
                    max_seq_len=args.max_seq_len,
                )
            except Exception as e:
                text = f"[sampling failed: {type(e).__name__}: {e}]"
            inner.train()
            one_line = text.replace("\n", " ⏎ ")
            logger.info("[sample@%d] %s", it, one_line)
            if wandb_active and wandb.run is not None:
                wandb.log({"samples/generation": wandb.Html(f"<pre>{text}</pre>")}, step=it)

        if it > 0 and it % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"iter_{it:07d}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
            logger.info("checkpoint saved: %s", ckpt_path)

    final_iter = last_it + 1 if stop_reason == "wall_clock" else args.total_iters
    final_path = os.path.join(args.checkpoint_dir, f"iter_{final_iter:07d}_final.pt")
    save_checkpoint(model, optimizer, final_iter, final_path)
    total_elapsed = time.time() - t_start
    logger.info("final checkpoint saved: %s (stop_reason=%s, elapsed=%.1fs, iters=%d)",
                final_path, stop_reason, total_elapsed, final_iter)

    if val_tokens is not None:
        val_loss = evaluate(model, val_tokens, args)
        log_metrics({"val/loss": val_loss, "elapsed_sec": total_elapsed},
                    step=final_iter, wandb_active=wandb_active)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    logger.info("args: %s", vars(args))
    try:
        train(args)
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
