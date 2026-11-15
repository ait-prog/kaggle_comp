# 26_openpool_v14_generation_config
- title: Open Pool v14: real kimi-k3 was written with reasoning on, glm-5.3 is fine, and 284 mislabelled rows are gone
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/752766
- type: discussion (dataset author)
- author(s) / date: Carrot Production (crtproduction) 2026-10-26
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 3/5 (weak 33%-BA yardstick; the author flags it as untested by training) | Recency 5/5 | Bias: med (dataset author)
## Verbatim quotes
> "So only about 2,700 of the 9,700 pool `kimi-k3` passages look like the real model." (752766)
> "**Real** `glm-5.3` already reads like Claude: the train-only classifier puts 40% of its probability on the Claude labels." (752766)
> "In one source, the author's own model column shows that 284 passages were written by models outside the 18 labels" (752766)
> "**None of this is tested by training yet.** It shows which passages look real, not that dropping the others raises your score." (752766)
## What it supports
- SOURCE CLAIM: pool label noise (model substitution) and generation-config mismatch exist and are measurable per source and config.
- CONFLICT: 751871 (sources/32) reports that Kimi-K3 *without* reasoning matches train's em-dash rate better. The kimi config direction is contested. Treat it as an open empirical question.
