# 49_ghostbuster
- title: Ghostbuster: Detecting Text Ghostwritten by Large Language Models
- url: https://arxiv.org/abs/2305.15047
- code: none found
- type: academic
- author(s) / venue / date: Vivek Verma, Eve Fleisig, Nicholas Tomlin, Dan Klein; NAACL 2024; arXiv 24 May 2023 (v3 Apr 2024)
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 3/5 | Bias: low
## Verbatim quotes
> "Our method works by passing documents through a series of weaker language models"
> "Crucially, Ghostbuster does not require access to token probabilities from the target model"
> "Ghostbuster achieves 99.0 F1 when evaluated across domains, which is 5.9 F1 higher than the best preexisting model."
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: token probabilities from weaker LMs, then structured search over combinations of vector/scalar functions yields features for a linear classifier.
- regime (text length, #classes, domains): binary human vs AI; cross-domain evaluation (essays, news, stories).
- generalization evidence: SOURCE CLAIM: 99.0 F1 across domains.
- cost / code: forward passes of small LMs; no repo link on fetched page.
- relevance to Name That LLM: INFERENCE: binary task; the weaker-LM token-probability feature search could be run per class with open scoring LMs to produce a small, domain-robust feature block for a GBDT stacker.
