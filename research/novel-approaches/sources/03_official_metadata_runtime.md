# 03_official_metadata_runtime
- title: Name That LLM, competition metadata (GetCompetition JSON) and Code Requirements page
- url: https://www.kaggle.com/competitions/name-that-llm/overview/code-requirements ; API `competitions.CompetitionService/GetCompetition` {"competitionName":"name-that-llm"}
- code: none
- type: primary (host metadata)
- author(s) / date: Kaggle
- accessed: 2026-11-12. Raw: sources/raw/kaggle_getcompetition_2026-11-12.json ; pages JSON key "Code Requirements"
- scores: Credibility 5/5 | Recency 5/5 | Bias: low
## Verbatim quotes (JSON fields are quoted exactly)
> "leaderboardPercentage":40
> "maxCpuRuntimeMinutes":540,"maxGpuRuntimeMinutes":540
> "deadline":"2027-01-14T23:59:00.817Z"
> "evaluationAlgorithm":{"id":501,"name":"Balanced Accuracy Score"
> "Freely & publicly available external data is allowed, including pre-trained models"
## What it supports
- FACT: public LB = 40% of the hidden test (≈1,600 rows, about 89 per class on average but unequal). Private = 60%.
- FACT: CPU or GPU notebook ≤ 9 h. Internet disabled. Output is `submission.parquet`. Notebook-only submissions (`onlyAllowKernelSubmissions: true`). 5 submissions/day; 2 final selections. Deadline 2027-01-13/14 23:59 UTC. Team merger and entry deadline 2027-01-06.
- UNKNOWN: the exact accelerator of the hidden rerun. The page names only CPU/GPU limits and does not mention TPU for submission. The team's deployed run uses T4×2 (k/dev1/kernel-metadata.json). Forum questions on rerun hardware are unanswered (sources/33).
- INFERENCE: the public-LB binomial SE for BA≈0.7 with ≈1,600 rows is ≈0.011. Single-submission steps of 0.001–0.004 are noise.
