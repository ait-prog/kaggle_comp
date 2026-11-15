# 22_grouping_simulation_known_truth
- title: Test-time prompt grouping: simulation vs. LB, and a question about the test layout (with GoPain's known-truth simulation)
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/750213
- code: https://www.kaggle.com/code/locdunhat/when-does-test-time-prompt-grouping-help (linked; not fetched)
- type: discussion
- author(s) / date: Karthikeya Thatipamula 2026-10-17; comment GoPain 2026-10-21
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 4/5 (reproducible simulation with code, plus real OOF check) | Recency 5/5 | Bias: low
## Verbatim quotes
> "So a CV on train's layout promises +0.08 to +0.12, and a test with 1-2 answers per prompt gives about nothing, or a loss for centring." (GoPain, 750213)
> "Estimating the prompt effect from the *other* members only (a cavity estimate) avoids this and never lost to plain decoding when the groups were true." (GoPain, 750213)
> "for balanced accuracy the best decision ignores the author frequencies, because every author's recall counts the same." (GoPain, 750213)
> "subsample each validation prompt to 1-3 answers before measuring any grouping trick, and recover the groups on that subsample the same way as on the test." (GoPain, 750213)
## What it supports
- SOURCE CLAIM: group-trick gains depend on answers per prompt, and the LB behaviour fits a test with few answers per prompt.
- Triangulated by the lead's own oracle-group replication (sources/09) and the LB reports in sources/21 and 23.
