# 23_group_tricks_lb_transfer_and_validation
- title: Notes from a solo pipeline (757855); Notes from our runs: what transferred to the LB (751871)
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/757855
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/751871
- code: https://github.com/pedrohbenetton/kaggle-name-that-llm (linked, not fetched)
- type: discussion
- author(s) / date: pedrohbenetton 2026-11-11; Daniel Andres Miranda 2026-10-22
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 3/5 | Recency 5/5 | Bias: low-med
## Verbatim quotes
> "Hungarian assignment within prompt groups gave +0.10 CV but +0.002 LB." (pedrohbenetton, 757855)
> "On the public LB: 0.425 with it vs 0.427 without." (Daniel Andres Miranda, 751871)
> "The 8-layer and the single-layer readouts score the same but disagree on 29 % of passages, so blending them adds ~2 points." (Daniel Andres Miranda, 751871)
> "Adding seyit + lucas (13 of 18 labels) to frozen-feature classifiers: +4 points overall, but the 5 uncovered labels (GLM ×2, kimi-k3, qwen3.8-max, gemini-3.7-flash) **lost** 6 points of recall" (Daniel Andres Miranda, 751871)
## What it supports
- SOURCE CLAIM (two independent teams): grouping tricks don't transfer to the LB.
- SOURCE CLAIM: unbalanced external-data coverage creates a "new prompt ⇒ not label X" shortcut. Coverage matters more than volume.
