# 85_ippolito_reverse_decoding
- title: Reverse-Engineering Decoding Strategies Given Blackbox Access to a Language Generation System
- url: https://arxiv.org/abs/2309.04858 (abstract page; PDF also fetched)
- code: none found
- type: academic
- author(s) / venue / date: Daphne Ippolito, Nicholas Carlini, Katherine Lee, Milad Nasr, et al.; arXiv 2309.04858
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 3/5 | Bias: low
## Verbatim quotes
> "we present methods to reverse-engineer the decoding method used to generate text (i.e., top-$k$ or nucleus sampling)"
> "Our ability to discover which decoding strategy was used has implications for detecting generated text."
> "We perform our attack on several families of open-source language models, as well as on production systems (e.g., ChatGPT)."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: with black-box prompt access one can infer whether top-k or nucleus sampling (and its parameter) was used, via crafted prompts that induce near-uniform token distributions. INFERENCE: requires repeated queries to the generator; here we only have 4,000 fixed passages, so the attack itself cannot be run. Value is conceptual: decoding settings (temperature, top-p) leave a footprint in token-rank statistics.
- regime: Query access to the generator, many samples per prompt.
- generalization / failure evidence: Production-system (ChatGPT) results are in the paper; the fetched page only gives the abstract. No multi-class lab attribution.
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: n/a (attack not applicable offline). A passive analogue (GLTR-like rank histograms) is cheap on a small proxy LM.
- relevance to Name That LLM: Supports the idea that per-model decoding config (e.g. high temp 1.0 / top-p 0.95 for GLM and Kimi) is a latent feature; but cannot be extracted directly from single passages without the generator.
