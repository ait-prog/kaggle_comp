# 33_runtime_hardware_hidden_rerun
- title: TPU v5e-8 trains Qwen3-8B LoRA ~14x faster (754657); Late post-processing cell never affects hidden rerun (758021); Scoring-run hardware (758259)
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/754657
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/758021
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/758259
- type: discussion
- author(s) / date: GoPain 2026-11-01; vishu phalswal 2026-11-11, 2026-11-12
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 3/5 | Recency 5/5 | Bias: low
## Verbatim quotes
> "A stage that takes me about 7 h on 2xT4 comes out at about 0.5 h." (GoPain, 754657)
> "But every submission that should have used it scored exactly the same as the base (0.682). That suggests the cell always falls back in the hidden rerun." (vishu phalswal, 758021)
> "Does anyone know if the scoring rerun has the same machine as a normal commit: same RAM, same /dev/shm size, and the same local disk space in /tmp?" (vishu phalswal, 758259)
## What it supports
- SOURCE CLAIM: TPU v5e-8 (20 h/week) is the practical training machine. Inference is on T4×2.
- UNKNOWN: hidden-rerun RAM, disk and timing parity with commit runs. A silent fallback is a real risk for any late, heavy stage, so new components need explicit success markers that can be inspected (e.g. a fail-loud mode in a test submission).
