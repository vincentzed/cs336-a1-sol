"""Gradio UI for prompting a TinyStories-trained checkpoint.

Run:  uv run python scripts/sample_ui.py

The checkpoint directory defaults to `checkpoints/tinystories/` relative to the
repo root. Override with the TRANSFORMER_LM_CKPT_DIR env var.
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch

from transformer_lm.generate import load_model, load_tokenizer, sample

REPO = Path(__file__).resolve().parent.parent
CKPT_DIR = Path(os.environ.get("TRANSFORMER_LM_CKPT_DIR", REPO / "checkpoints" / "tinystories"))
VOCAB = REPO / "tokenizers" / "tinystories" / "vocab.pkl"
MERGES = REPO / "tokenizers" / "tinystories" / "merges.pkl"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

TS_CONFIG = dict(
    vocab_size=10000,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    max_seq_len=256,
    theta=10000.0,
)

_tokenizer = load_tokenizer(str(VOCAB), str(MERGES))
_model: torch.nn.Module | None = None


def _load() -> torch.nn.Module:
    global _model
    if _model is not None:
        return _model
    finals = sorted(CKPT_DIR.glob("*_final.pt"))
    if not finals:
        raise FileNotFoundError(f"no *_final.pt in {CKPT_DIR}")
    _model = load_model(str(finals[-1]), device=DEVICE, dtype=torch.float32, **TS_CONFIG)
    return _model


def run(prompt: str, max_new: int, temperature: float, top_p: float, seed_s: str):
    model = _load()
    seed = int(seed_s) if seed_s.strip() else None
    return sample(
        model, _tokenizer,
        prompt=prompt,
        max_new_tokens=int(max_new),
        temperature=float(temperature),
        top_p=float(top_p),
        device=DEVICE,
        max_seq_len=256,
        seed=seed,
    )


with gr.Blocks(title="TinyStories Prompt Playground") as demo:
    gr.Markdown("# TinyStories Prompt Playground")
    gr.Markdown(f"Device: `{DEVICE}` · checkpoint dir: `{CKPT_DIR}`.")
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", lines=3, value="Once upon a time, there was a little")
            max_new = gr.Slider(1, 512, value=200, step=1, label="max_new_tokens")
            temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="temperature (0 = greedy)")
            top_p = gr.Slider(0.05, 1.0, value=0.9, step=0.01, label="top_p (1 = disabled)")
            seed = gr.Textbox(label="seed (blank = random)", value="")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            out = gr.Textbox(label="Generated", lines=22)

    btn.click(run, inputs=[prompt, max_new, temperature, top_p, seed], outputs=out)


if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=True, theme=gr.themes.Soft())
