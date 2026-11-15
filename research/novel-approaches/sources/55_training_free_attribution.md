# 55_training_free_attribution
- title: A Training-free Method for LLM Text Attribution
- url: https://arxiv.org/abs/2501.02406
- code: none found
- type: academic
- author(s) / venue / date: Tara Radvand, Izak Duenyas, Ambuj Tewari; arXiv 4 Jan 2025 (v6 Sep 2026), stat.ML
- accessed: 2026-11-12
- scores: Credibility 3/5 | Recency 5/5 | Bias: low (theory-leaning)
## Verbatim quotes
> "We model LLM-generated text as a sequential stochastic process and develop training-free statistical tests"
> "We prove that both Type I and Type II errors decay exponentially with text length"
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: hypothesis tests on next-token probabilities between known LLM sets; errors decay exponentially in text length. Search result (not fetch) added: 300-token texts and within-family similarity as the main difficulty.
- regime (text length, #classes, domains): requires log-probs of candidate models.
- generalization evidence: not retrieved.
- cost / code: none found.
- relevance to Name That LLM: INFERENCE: exponential dependence on length means 40-word texts are in the hard regime; per-length-bucket calibration or length-aware ensembling is justified.
