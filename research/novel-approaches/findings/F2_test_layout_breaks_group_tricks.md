# F2: Test-time prompt-group tricks are fitted to train's layout and need not transfer
Status: **SUPPORTED, triangulated** (official text + independent LB reports + simulation + local replication + literature).
Hypothesis link: H2 (supported).

## Thesis
Train has a complete 18-answers-per-prompt layout. The official pages state that per-model test counts are unequal and differ between public and private. Centring, Hungarian/injective assignment and Gaussian group heads gain +0.08 to +0.12 on train-layout CV but about 0 on the public LB. With 1–2 answers per prompt they are neutral or harmful.

## Evidence
| Source | Type | Label |
|---|---|---|
| sources/01 official Evaluation: unequal counts, public≠private mix | primary | FACT |
| sources/21 (751802, 755733): LB 0.350 vs 0.353 with matching | discussion | SOURCE CLAIM |
| sources/23 (757855, 751871): +0.10 CV vs +0.002 LB; 0.425 vs 0.427 | discussion (2 teams) | SOURCE CLAIM |
| sources/22, 25 (750213, 752708, 753053): simulation and thinning OOF; heavy centring −0.03 to −0.05 at m=2–3 | discussion with code | SOURCE CLAIM |
| sources/09 local replication: full centring at m=1 → 0.056; Hungarian gain +0.089 at m=18, +0.003 at m=2 | primary (local) | FACT (weak model) |
| sources/73 Veilleux: transductive methods drop below inductive when label marginals are arbitrary | academic | SOURCE CLAIM |

## Implication
- dev1 contains capped agglomerative grouping, Gaussian group heads, prompt-centred heads and Sinkhorn at mix 0.50 (sources/06). Their private-split contribution is unknown, and part of the public 0.682 may rest on them. INFERENCE.
- UNKNOWN: true answers-per-prompt in test, and whether public/private is split by prompt.
