# 89_gltr_statistical_detection
- title: GLTR: Statistical Detection and Visualization of Generated Text
- url: https://arxiv.org/abs/1906.04043 (abstract page; PDF also fetched)
- code: https://github.com/HendrikStrobelt/detecting-fake-text (repo exists per GitHub API: "Giant Language Model Test Room")
- type: academic
- author(s) / venue / date: Sebastian Gehrmann, Hendrik Strobelt, Alexander Rush; ACL 2019 demo (arXiv 1906.04043)
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 2/5 | Bias: low
## Verbatim quotes
> "GLTR applies a suite of baseline statistical methods that can detect generation artifacts across common sampling schemes."
> "improves the human detection-rate of fake text from 54% to 72% without any prior training"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: per-token rank/probability under a reference LM (top-10/100/1000 buckets, entropy) separates sampled from human text; sampling scheme changes the rank histogram. INFERENCE: binary human-vs-machine on GPT-2 era models; multi-class use requires training a classifier on rank-histogram features, which is cheap (a 10-20 dim feature vector) but low capacity for 18 near-identical RLHF models.
- regime: Any open reference LM; token-level rank histogram.
- generalization / failure evidence: GPT-2/GPT-3-era generators; not tested on frontier reasoning models. Humans improved 54%->72% only.
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: one forward pass of a 1-7B proxy LM over 4,000 short passages: minutes on a T4.
- relevance to Name That LLM: Source of the rank-bucket feature idea; complementary, not primary.
