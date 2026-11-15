# 24_sinkhorn_cost_uneven_mix
- title: Sinkhorn balancing: what it costs when the test mix is uneven
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/752365
- code: https://www.kaggle.com/code/locdunhat/sinkhorn-rebalancing-under-balanced-accuracy (linked by commenter; not fetched)
- type: discussion
- author(s) / date: Boris Konstantinovskii 2026-10-24; comments HALX2158, GoPain
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 4/5 (code included in post) | Recency 5/5 | Bias: low
## Verbatim quotes
> "**Full balancing is a bet that the test is close to even.** If it is, you gain a few thousandths. If it is not, you can lose several points." (Boris Konstantinovskii, 752365)
> "On a sparse CV scheme (about 75 passages per pool, 1 to 4 per prompt, uneven mix), full balancing took the same blend from 0.645 to 0.475." (752365)
> "Estimating the test prior by EM and balancing towards it was worse than both. That is expected, because for balanced accuracy the optimal decision does not depend on the test prior." (GoPain, 752365)
> "Offsets learned that way do not care what the test mix is." (GoPain, 752365)
## What it supports
- SOURCE CLAIM: Sinkhorn toward equal or target marginals is risky under an uneven mix. Cross-fitted per-class offsets learned on OOF capture most of the benign gain.
- SOURCE CLAIM: EM test-prior estimation is worse for balanced accuracy. This refutes the lead's H4 as originally stated (see REPORT.md).
- RELEVANCE: dev1 raised the Sinkhorn target-mix strength from 0.35 to 0.50 (sources/06), which increases exposure to this risk on the private split.
