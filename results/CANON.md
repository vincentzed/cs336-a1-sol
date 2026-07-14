# Canonical record book

Protocol: deterministic full sweep of every non-overlapping ctx-512 window of the OWT
validation set (129,711 windows, 66.4M tokens), token-weighted mean CE, softcap applied as
part of the model forward, EMA weights unless marked raw. `scripts/eval_canon.py`.

## B200 (leaderboard hardware) — the reference book
| Config | canon-EMA | canon-raw |
|---|---|---|
| L14 + ve5 + per-head gates + cap20 (`b200_L14g`) | **2.96551** | 2.97803 |
| L16 + ve5 + cap20 (`b200_L16ve5`) | 2.97083 | 2.97329 |

Run-protocol anchors (fixed-window subsample, EMA): L14g mean 2.9731 ± 0.0003 (n=4 seeds —
B200 replicates are that tight); muon-lr at L14: 6.5e-3 = 2.97491, 1.0e-2 = 3.00793
(slow node, 8353/10400 iters — confounded but directionally consistent), 8e-3 confirmed;
L20+ve5+cap20 mean 2.98713 ± 0.0002 (n=3); L16 family 2.9790 ± 0.0009; L14 no-gates 2.97566.
Frontier still moving — see the W&B project for runs newer than this snapshot.

## B300 (development box) — closed book
| Config | canon-EMA | canon-raw |
|---|---|---|
| L20 + ve5 + cap23 (`r19_ve5`) | **2.94966** | 2.95537 |
| L20 + cap20 (`r18_cap20`) | 2.97424 | 2.97964 |
| L20 + cap23 (`r17_cap23`) | (EMA lost, pre-fix) | 2.97822 |
| L20, no softcap (`r13_s7`) | (EMA lost) | 2.99864 |

Reference points: baseline reproduced at 3.2500 (B200, ctx-1024 self-eval, their protocol);
best-ever leaderboard entry 3.03543 (B200); measured B300→B200 hardware delta ≈ +0.037.
