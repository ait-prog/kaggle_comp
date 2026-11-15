# 61_luar_universal_authorship_representations
- title: Learning Universal Authorship Representations
- url: https://aclanthology.org/2021.emnlp-main.70
- code: https://github.com/LLNL/LUAR (stated in search result, not on fetched page)
- type: academic
- author(s) / venue / date: Rivera-Soto, Miano, Ordonez, Chen, Khan, Bishop, Andrews / EMNLP 2021 / 2021-11
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 2/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "we conduct the first large-scale study of cross-domain transfer for authorship verification"
> "We find that although a surprising degree of transfer is possible between certain domains"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Transformer authorship embedding trained contrastively on same-author pairs; zero-shot transfer tested across Amazon, fanfiction, Reddit.
- regime: SOURCE CLAIM: Cross-domain (topic/genre) authorship verification.
- when it fails: SOURCE CLAIM/INFERENCE: Transfer works for some domain pairs, not others (abstract). Later work (StyleDistance) finds LUAR also captures content.
- cost / code: Code at LLNL/LUAR per search result only.
- relevance to Name That LLM: INFERENCE: Positive precedent that author signal transfers across domains, but not guaranteed; supports pretraining/adding a LUAR-like embedding as a feature, with caution about content leakage.
