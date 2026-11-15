# 34_grouped_cv_reverse_leakage
- title: Random-split CV underestimates you here (749593); Random K-fold is pessimistic here (756477)
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/749593
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/756477
- type: discussion
- author(s) / date: Aleksey Osipenko 2026-10-15; Renan Pinheiro 2026-11-06
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 4/5 (replicated locally in sources/09) | Recency 5/5 | Bias: low
## Verbatim quotes
> "The group split tracks the LB within 0.014, the random split misses it by 0.14." (749593)
> "The lab is much easier than the exact model: lab-level balanced accuracy 0.587 vs 0.369." (749593)
> "In my submissions, grouped CV and public LB moved together every time (r ≈ 0.99)." (756477)
## What it supports
- SOURCE CLAIM, triangulated with sources/09 (0.145 random vs 0.351 grouped): prompt-grouped CV is mandatory and tracks the LB for per-passage models. That correlation does not extend to test-time group tricks (sources/21–25).
