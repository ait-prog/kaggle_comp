# 27_gemini37_routing_label_noise
- title: Gemini-3.7 Flash requests routed to Gemini-3.8 Flash
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/751370
- type: discussion
- author(s) / date: Darragh1Walsh 2026-10-21; comment GoPain 2026-10-22
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 3/5 | Recency 5/5 | Bias: low
## Verbatim quotes
> "any requests through it’s endpoint are automatically routed to Gemini 3.8 Flash." (Darragh1Walsh, 751370)
> "Short version: the host's labels show no sign of it." (GoPain, 751370)
> "3.7 against 3.8 is simply the hardest pair: the LoRA gets about half of each right." (GoPain, 751370)
## What it supports
- SOURCE CLAIM: provider routing can silently mislabel self-generated or pool data. The served-model field must be checked (also 751871: "check `modelVersion` on every response").
- SOURCE CLAIM: host labels appear clean. The 3.7/3.8 confusion is intrinsic difficulty plus pool imbalance (7 of 12 sources lack 3.7).
