# Kaggle runs (quota tracking; week resets Sat 00:00 UTC)
| run | acc | start UTC | end UTC | h | note |
|---|---|---|---|---|---|
| dev1 | T4x2 | 2026-11-10 12:08 | 12:09 | 0.02 | submission 58200298 COMPLETE; public score 0.682 (verified 2026-11-12); submitted 2026-11-10 12:12:49 UTC |
| dev2 v1 | TPU v5e-8 | 2026-11-10 12:31 | | | smoke: Qwen3-8B-base k24 LoRA SPMD DP |

| dev3 v1 | T4x2 | | | | flashpm/ntl-dev3; submission 58281588 COMPLETE; public score 0.676; Sinkhorn off |
| dev4 v1 | T4x2 | | | | flashpm/ntl-dev4; submission 58281594 COMPLETE; public score 0.657; test-time grouping off |
| dev10 v1 | T4x2 | 2026-11-14 | | | flashpm/ntl-dev10; submission 58348789 COMPLETE; public score 0.691 (long-mount paths fix); verbatim fork of lightningv08 0.694 (GLM-5.3 likelihood stage); new base/anchor |
| dev11 v2 | T4x2 | 2026-11-14 | | | flashpm/ntl-dev11; submission 58348797 COMPLETE; public score 0.691 (same as dev10; dev9 either skipped or no effect, hidden rerun has no log); dev10 + dev9 at fixed 10% linear share of parent before native/Sinkhorn; fail-safe (late>5.5h, error, 20-min timeout, train-parity <0.6 -> skipped = dev10) |
