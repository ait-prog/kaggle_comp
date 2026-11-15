# 25_centring_thin_prompts
- title: Centring members inside content clusters (752708); Why grouped CV over-picks test-time centring (753053)
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/752708
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/753053
- type: discussion
- author(s) / date: GoPain 2026-10-26, 2026-10-27
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 4/5 (OOF plus LB checks) | Recency 5/5 | Bias: low
## Verbatim quotes
> "Heavy centring of a strong member hurts once prompts are thin (LoRA at w = 1: −0.03 to −0.05 at m = 2 and 3)." (752708)
> "the centring weight my prompt-grouped CV liked best (all members at
0.5, OOF 0.646) was the worst of four on the public LB" (753053)
> "What worked in every case was to put the validation passages into the unlabelled test batch,
cluster everything together, and score only the validation passages." (753053)
## What it supports
- SOURCE CLAIM: any cluster-dependent test-time rule should be tuned with "validation-in-test-batch" clustering. Full-group grouped CV over-selects.
- INFERENCE: this is a concrete, in-notebook validation protocol the team can adopt (Experiment 1 in REPORT.md).
