# Canonical record book

Protocol: deterministic full sweep of every non-overlapping ctx-512 window of the OWT
validation set (129,711 windows, 66.4M tokens), token-weighted mean CE, softcap applied as
part of the model forward, EMA weights unless marked raw. `scripts/eval_canon.py`.

## B200 (leaderboard hardware) — the reference book (campaign closed 2026-07-14)
| Config | canon-EMA | canon-raw |
|---|---|---|
| L14 + d_ff 5632 + ve5 + per-head gates + cap20 (`b200_L14g_dff5632`) | **2.95712** | 2.95964 |
| L14 + ve5 + per-head gates + cap20 (`b200_L14g`) | 2.96551 | 2.97803 |
| L16 + ve5 + cap20 (`b200_L16ve5`) | 2.97083 | 2.97329 |

Run-protocol anchors (fixed-window subsample, EMA): L14g mean 2.9731 ± 0.0003 (n=4 seeds —
B200 replicates are that tight); dff5632 leg 2.96454 (−0.0086, ~28σ); muon-lr at L14:
6.5e-3 = 2.97491, 1.0e-2 = 3.00793 (slow node, 8353/10400 iters — confounded but
directionally consistent), 8e-3 confirmed; L20+ve5+cap20 mean 2.98713 ± 0.0002 (n=3);
L16 family 2.9790 ± 0.0009; L14 no-gates 2.97566. Final-wave dead levers: EMA decay
0.9995 (2.99926) and 0.9985 (2.97241), LR-tail-to-zero (2.97297), momentum decay-back
last-2k (2.97422), d_ff 4864 (2.97429 — yet 5632 wins big; the FFN optimum is not
monotone in width, and 5632 costs only ~12% throughput for +37.5% FFN FLOPs on B200).

## B300 (development box) — closed book
| Config | canon-EMA | canon-raw |
|---|---|---|
| L20 + ve5 + cap23 (`r19_ve5`) | **2.94966** | 2.95537 |
| L20 + cap20 (`r18_cap20`) | 2.97424 | 2.97964 |
| L20 + cap23 (`r17_cap23`) | (EMA lost, pre-fix) | 2.97822 |
| L20, no softcap (`r13_s7`) | (EMA lost) | 2.99864 |

Reference points: baseline reproduced at 3.2500 (B200, ctx-1024 self-eval, their protocol);
best-ever leaderboard entry 3.03543 (B200); measured B300→B200 hardware delta ≈ +0.037.
