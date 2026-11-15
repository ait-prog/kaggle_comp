# 05_public_leaderboard_snapshot
- title: Name That LLM public leaderboard (computed snapshot)
- url: https://www.kaggle.com/competitions/name-that-llm/leaderboard ; API `competitions.LeaderboardService/GetLeaderboard`
- code: none
- type: primary (platform data)
- author(s) / date: Kaggle; snapshot 2026-11-12
- accessed: 2026-11-12. Raw summary: sources/raw/kaggle_public_lb_summary_2026-11-12.json
- scores: Credibility 5/5 (data) | Recency 5/5 | Bias: low (public split only, 40% of test)
## Verbatim quotes (fields copied from the raw JSON)
> {"teamId":17167595,"submissionId":58239760,"rank":1,"displayScore":"0.768"
> "submissionId":58200298,"rank":534,"displayScore":"0.682","medal":"BRONZE"
## Computed facts (from the full JSON; lead-agent computation)
- 5,607 teams. Score at rank 1/10/50/100/200/500 = 0.768/0.747/0.708/0.690/0.684/0.682.
- 72 teams ≥ 0.700. 116 ≥ 0.685. 319 teams sit at exactly 0.684 and 141 at exactly 0.682.
- The best public-notebook score in the top-60-by-votes kernel list is 0.684 (sources/10).
## What it supports
- FACT: the team's 0.682 is rank 534. About 70+ teams score above any public notebook, by up to +0.084.
- INFERENCE: the 0.682–0.684 spikes are forks of public notebooks. The 0.70–0.77 tail uses non-public methods or data. The plateau is a property of the public family, not evidence of a Bayes ceiling.
- CAVEAT: public scores can overfit the 40% split. Private ranks are unknown.
