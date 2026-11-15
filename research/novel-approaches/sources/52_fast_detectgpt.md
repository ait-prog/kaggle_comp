# 52_fast_detectgpt
- title: Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature
- url: https://arxiv.org/abs/2310.05130
- code: https://github.com/baoguangsheng/fast-detect-gpt
- type: academic
- author(s) / venue / date: Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, Yue Zhang; ICLR 2024; arXiv 8 Oct 2023 (v3 Dec 2024)
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 3/5 | Bias: low
## Verbatim quotes
> "we introduce the concept of conditional probability curvature to elucidate discrepancies in word choices"
> "substitutes DetectGPT's perturbation step with a more efficient sampling step"
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: curvature of conditional token probability under sampled alternatives from a scoring LM, a single-pass statistic.
- regime (text length, #classes, domains): binary; speed and 340x and 28% figures come from search-result summary, not the fetched abstract.
- generalization evidence: not retrieved from fetch.
- cost / code: repo exists; one forward pass per scoring model.
- relevance to Name That LLM: INFERENCE: per-token log-prob and entropy profile statistics (mean, curvature) from Qwen scorers are cheap scalar features; hidden-state pooling may already encode them, so expect small additive gain.
