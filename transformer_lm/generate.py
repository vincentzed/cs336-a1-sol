import argparse
import os
import sys

import torch
import torch.nn.functional as F

from transformer_lm.modules import TransformerLm, load_checkpoint, AdamW
from transformer_lm.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample text from a trained Transformer LM checkpoint.")

    # --- what to load ---
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a training checkpoint (.pt).")
    p.add_argument("--vocab", type=str, required=True,
                   help="Path to vocab.pkl (used at tokenizer training time).")
    p.add_argument("--merges", type=str, required=True,
                   help="Path to merges.pkl.")
    p.add_argument("--eot-token", type=str, default="<|endoftext|>",
                   help="End-of-text special token string.")

    # --- model hparams (must match training) ---
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--rope-theta", type=float, default=10000.0)

    # --- generation config ---
    p.add_argument("--prompt", type=str, default="",
                   help="Prompt text. Empty → seeded with <|endoftext|> token.")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max tokens to generate after the prompt.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Softmax temperature. 0 = greedy, higher = more random.")
    p.add_argument("--top-k", type=int, default=0,
                   help="Keep only top-k logits before sampling. 0 = disabled.")
    p.add_argument("--top-p", type=float, default=1.0,
                   help="Nucleus sampling threshold. 1.0 = disabled.")
    p.add_argument("--no-stop-at-eot", dest="stop_at_eot", action="store_false",
                   help="Keep generating past the EOT token. Default: stop at EOT.")
    p.set_defaults(stop_at_eot=True)
    p.add_argument("--num-samples", type=int, default=1,
                   help="How many independent samples to generate.")

    # --- runtime ---
    p.add_argument("--device", type=str,
                   default="mps" if torch.backends.mps.is_available()
                   else "cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility. Omit for non-deterministic sampling.")

    return p.parse_args()


def build_model(args: argparse.Namespace) -> TransformerLm:
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    return TransformerLm(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        theta=args.rope_theta,
        device=args.device,
        dtype=dtype,
    )


# ---------- notebook-friendly helpers ----------

TS_DEFAULT_CONFIG = dict(
    vocab_size=10000,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    max_seq_len=256,
    theta=10000.0,
)


def load_model(checkpoint_path: str, device: str = "mps", dtype: torch.dtype = torch.float32, **model_kwargs) -> TransformerLm:
    """Construct a TransformerLm and load weights from a checkpoint.

    Tolerates checkpoints saved from torch.compile-wrapped models (strips the
    '_orig_mod.' prefix that torch.compile inserts into state_dict keys).
    """
    cfg = {**TS_DEFAULT_CONFIG, **model_kwargs}
    model = TransformerLm(**cfg, device=device, dtype=dtype)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def load_tokenizer(vocab_path: str, merges_path: str, eot: str = "<|endoftext|>") -> Tokenizer:
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens=[eot])


def sample(
    model: TransformerLm,
    tokenizer: Tokenizer,
    prompt: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    stop_at_eot: bool = True,
    eot: str = "<|endoftext|>",
    device: str = "mps",
    max_seq_len: int = 256,
    seed: int | None = None,
) -> str:
    """One-shot: prompt string in, generated string out."""
    if seed is not None:
        torch.manual_seed(seed)
    eot_id = tokenizer._special_ids[eot]
    prompt_ids_list = tokenizer.encode(prompt) if prompt else [eot_id]
    prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long, device=device)
    out_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_seq_len=max_seq_len,
        eot_id=eot_id,
        stop_at_eot=stop_at_eot,
    )
    if not prompt and len(out_ids) > 0 and out_ids[0].item() == eot_id:
        out_ids = out_ids[1:]
    return tokenizer.decode(out_ids.tolist())


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = topk_vals[..., -1:].expand_as(logits)
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    # Mask tokens whose cumulative prob exceeds p (keep the first that crosses the threshold).
    keep = cumprobs <= p
    keep[..., 0] = True
    sorted_logits = torch.where(keep, sorted_logits, torch.full_like(sorted_logits, float("-inf")))
    # Unsort back to original order.
    return torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)


@torch.no_grad()
def generate(
    model: TransformerLm,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    max_seq_len: int,
    eot_id: int | None,
    stop_at_eot: bool,
) -> torch.Tensor:
    """
    Args:
        prompt_ids: (T,) long tensor on model's device
    Returns:
        (T + <=max_new_tokens,) long tensor
    """
    ids = prompt_ids.unsqueeze(0)  # (1, T) — model expects batched input

    for _ in range(max_new_tokens):
        # Truncate context window if it exceeds max_seq_len (drop oldest tokens).
        context = ids[:, -max_seq_len:]
        logits = model(context)            # (1, T', vocab)
        next_logits = logits[0, -1]        # (vocab,) — logits at last position

        if temperature == 0:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            next_logits = apply_top_k(next_logits, top_k)
            next_logits = apply_top_p(next_logits, top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)

        if stop_at_eot and eot_id is not None and next_id.item() == eot_id:
            break

    return ids[0]


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"[loading tokenizer] vocab={args.vocab} merges={args.merges}", file=sys.stderr)
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[args.eot_token])
    eot_id = tokenizer._special_ids[args.eot_token]

    print(f"[loading model] {args.checkpoint} on {args.device} ({args.dtype})", file=sys.stderr)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model = load_model(
        args.checkpoint,
        device=args.device,
        dtype=dtype,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        theta=args.rope_theta,
    )

    # Encode prompt (or seed with EOT for a "blank" start).
    if args.prompt:
        prompt_ids_list = tokenizer.encode(args.prompt)
    else:
        prompt_ids_list = [eot_id]
    print(f"[prompt] {len(prompt_ids_list)} tokens", file=sys.stderr)

    prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long, device=args.device)

    for sample_idx in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n===== sample {sample_idx + 1}/{args.num_samples} =====")
        out_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_seq_len=args.max_seq_len,
            eot_id=eot_id,
            stop_at_eot=args.stop_at_eot,
        )
        # Drop the seed EOT token if no user prompt was given.
        if not args.prompt and len(out_ids) > 0 and out_ids[0].item() == eot_id:
            out_ids = out_ids[1:]
        text = tokenizer.decode(out_ids.tolist())
        print(text)


if __name__ == "__main__":
    main()
