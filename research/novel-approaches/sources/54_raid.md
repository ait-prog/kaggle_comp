# 54_raid
- title: RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors
- url: https://arxiv.org/abs/2405.07940
- code: https://github.com/liamdugan/raid (stated in search result only; fetched abs page listed no GitHub link)
- type: academic
- author(s) / venue / date: Liam Dugan, Alyssa Hwang, Filip Trhlik, Josh Magnus Ludan, Andrew Zhu, Hainiu Xu, Daphne Ippolito, Chris Callison-Burch; ACL 2024; arXiv 13 May 2024
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 3/5 | Bias: low
## Verbatim quotes
> "Many commercial and open-source models claim to detect machine-generated text with extremely high accuracy (99% or more)."
> "In this work we present RAID: the largest and most challenging benchmark dataset for machine-generated text detection."
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: benchmark of generations varying model, domain, decoding strategy, repetition penalty and adversarial attacks; detectors fail under those shifts (details of the 6M, 11 models, 8 domains, 4 decoding strategies figures come from the search summary, not the fetched abstract).
- regime (text length, #classes, domains): binary detection.
- generalization evidence: SOURCE CLAIM: detectors fooled by adversarial attacks, sampling changes, unseen models.
- cost / code: dataset and leaderboard.
- relevance to Name That LLM: INFERENCE: disconfirming for sampling-sensitivity; if test data was produced with differing temperature or reasoning modes, features keyed to decoding statistics are fragile.
