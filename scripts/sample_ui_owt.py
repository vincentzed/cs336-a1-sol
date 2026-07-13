"""Gradio UI for prompting an OWT-trained checkpoint.

Run:  uv run python scripts/sample_ui_owt.py

The checkpoint directory defaults to `checkpoints/owt_final/` relative to the
repo root. Override with the TRANSFORMER_LM_CKPT_DIR env var.
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch

from transformer_lm.generate import load_model, load_tokenizer, sample

REPO = Path(__file__).resolve().parent.parent
CKPT_DIR = Path(os.environ.get("TRANSFORMER_LM_CKPT_DIR", REPO / "checkpoints" / "owt_final"))
VOCAB = REPO / "tokenizers" / "owt" / "vocab.pkl"
MERGES = REPO / "tokenizers" / "owt" / "merges.pkl"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

OWT_CONFIG = dict(
    vocab_size=32000,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=2048,
    max_seq_len=1024,
    theta=10000.0,
)

MODELS: dict[str, dict] = {
    "owt_final (val=3.25)": dict(
        dir=".", val=3.25,
        note="45-min B200 run. 6169 iters at B=320, ctx=1024 (~2.02 B tokens). LR=2.5e-3, warmup=1000."),
}

_tokenizer = load_tokenizer(str(VOCAB), str(MERGES))
_model_cache: dict[str, torch.nn.Module] = {}


def _load(name: str) -> torch.nn.Module:
    if name in _model_cache:
        return _model_cache[name]
    meta = MODELS[name]
    ckpt_dir = CKPT_DIR / meta["dir"]
    # The final checkpoint filename depends on exact iteration count; just grab the _final.pt
    finals = list(ckpt_dir.glob("*_final.pt"))
    if not finals:
        raise FileNotFoundError(f"no *_final.pt in {ckpt_dir}")
    ckpt = finals[0]
    model = load_model(
        str(ckpt),
        device=DEVICE,
        dtype=torch.float32,
        **OWT_CONFIG,
    )
    _model_cache[name] = model
    return model


def describe(name: str) -> str:
    m = MODELS[name]
    return (
        f"**{name}**\n\n"
        f"- val loss: `{m['val']:.3f}`\n"
        f"- arch: d_model=768, L=12, H=12, d_ff=2048 (~124M params)\n"
        f"- {m['note']}"
    )


def run(name: str, prompt: str, max_new: int, temperature: float, top_p: float, seed_s: str):
    model = _load(name)
    seed = int(seed_s) if seed_s.strip() else None
    text = sample(
        model, _tokenizer,
        prompt=prompt,
        max_new_tokens=int(max_new),
        temperature=float(temperature),
        top_p=float(top_p),
        device=DEVICE,
        max_seq_len=1024,
        seed=seed,
    )
    return text


with gr.Blocks(title="OWT Prompt Playground") as demo:
    gr.Markdown("# OpenWebText Prompt Playground")
    gr.Markdown(
        f"Device: `{DEVICE}` · checkpoint dir: `{CKPT_DIR}`. "
        "Model loads lazily on first generate."
    )
    with gr.Row():
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=next(iter(MODELS.keys())),
                label="Model",
            )
            info = gr.Markdown()
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="In a shocking discovery, scientists",
                lines=3,
                value="The city of San Francisco",
            )
            max_new = gr.Slider(1, 512, value=200, step=1, label="max_new_tokens")
            temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="temperature (0 = greedy)")
            top_p = gr.Slider(0.05, 1.0, value=0.9, step=0.01, label="top_p (1 = disabled)")
            seed = gr.Textbox(label="seed (blank = random)", value="")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            out = gr.Textbox(label="Generated", lines=22)

    model_dd.change(describe, inputs=model_dd, outputs=info)
    demo.load(describe, inputs=model_dd, outputs=info)
    btn.click(
        run,
        inputs=[model_dd, prompt, max_new, temperature, top_p, seed],
        outputs=out,
    )


if __name__ == "__main__":
    demo.launch(server_port=7861, inbrowser=True, theme=gr.themes.Soft())
