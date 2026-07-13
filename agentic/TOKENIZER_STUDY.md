# Tokenizer Lever — Quantified

The leaderboard scores cross-entropy **per token** under a self-trained 32k BPE.
Since the tokenizer defines the token, it defines the metric's units — so before
spending GPU-hours, we measured how much the number moves from units alone.
CPU-only study; no model trained. Estimator: fit a token unigram (add-0.5) on a
fixed 500 MB train slice, report CE on tokenized `owt_valid`; a real model's
CE shifts like the unigram CE to first order. `bits/byte` is the unit-free control.

## Results (`scripts/tokenizer_study.py`)

| Tokenizer | trained on | bytes/token | vocab used | unigram CE/token | **bits/byte** (unit-free) | est. leaderboard shift |
|---|---|---:|---:|---:|---:|---:|
| **current** (baseline's) | unknown | 4.367 | 99.6% | 7.576 | 2.503 | — (reference) |
| weak | 50 MB slice | 4.342 | 98.9% | 7.568 | 2.515 | −0.008 |
| vweak (pathological) | 2 MB slice | 4.190 | 94.1% | 7.506 | **2.584** | −0.070 |
| full | 12 GB (whole corpus) | *(training; slow pretokenize)* | — | — | — | *(pending)* |

## Reading

**The gaming direction is real but small and self-defeating.** Deliberately
crippling the tokenizer (2 MB → worse merges → shorter, more-predictable tokens)
lowers per-token CE by only **~0.07**, and it does so by making tokens carry
fewer bytes — the honest metric, **bits/byte, gets *worse* (2.503 → 2.584)**.
That is the signature of pure unit inflation: the model predicts the corpus
*less* well per byte, but the leaderboard's per-token yardstick shrank faster
than its skill did. None of that 0.07 is modeling; all of it is measurement.

**The legitimate direction (full-corpus BPE) most likely moves the number the
wrong way.** Better merges compress harder (more bytes/token), which *raises*
per-token CE even as bits/byte improves — you would train a genuinely better
model and post a genuinely worse leaderboard score. So the honest tokenizer is
not a lever *toward the target*; it is a lever toward truth, at the target's
expense.

**Magnitudes, for planning:** the whole tokenizer axis is worth **≤0.07** and
only in its indefensible extreme; a mild "weak" tokenizer buys **~0.01**, indistinguishable
from seed noise. This is a small lever wearing a large disguise.

## Is it allowed?

The leaderboard rules (verbatim): *"you may only use the OpenWebText training
dataset that we provide"*, *"The code must clearly be your own work"*, and the
verification check that "your vocab is correct with **32k tokens**". A
self-trained 32k BPE on the provided OWT is squarely inside those lines — every
entrant trains their own; the assignment is a from-scratch tokenizer + model.
So **`full` is unambiguously legitimate** (and everyone should use it — it is
just correct engineering; that our inherited tokenizer's provenance is unknown
is the actual finding here).

**`weak`/`vweak` are legal-but-dishonest.** No rule forbids a badly-trained 32k
BPE, and the loss would "seem too good" exactly as the README's own caveat warns
— which is the point: the check exists because this hole exists. Deliberately
degrading your tokenizer to shrink the yardstick keeps the letter of "32k vocab,
your own work" while breaking the spirit of what the number is supposed to
certify (how well you model English). We flag it, quantify it, and recommend
against it: 0.07 of borrowed nothing is not worth the asterisk, and it would not
survive a bits/byte audit.

## Artifacts

- `scripts/tokenizer_study.py` — `train` / `analyze` / `npy` subcommands.
- `tokenizers/owt_weak/`, `tokenizers/owt_vweak/` — the gaming tokenizers (kept for
  the record, not for use). `tokenizers/owt_full/` pending (12 GB BPE still running).
- No `.npy` produced: the study shows no variant is worth a training leg toward the
  target. If `full` finishes, its `owt_*_full.npy` are worth one *honesty* run — to
  confirm the real model improves in bits/byte while its leaderboard number worsens —
  but not a target run.

**Bottom line:** the tokenizer is a near-dead lever for beating the leaderboard,
and a live lever only for lying to it. Spend the GPU-hours on model and optimizer.
