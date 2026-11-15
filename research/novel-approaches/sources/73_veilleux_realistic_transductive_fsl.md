# 73_veilleux_realistic_transductive_fsl
- title: Realistic Evaluation of Transductive Few-Shot Learning
- url: https://arxiv.org/abs/2204.11181
- code: https://github.com/oveilleux/Realistic_Transductive_Few_Shot
- type: academic
- author(s) / venue / date: Veilleux, Boudiaf, Piantanida, Ben Ayed / NeurIPS 2021 / 2022-04-24 (arXiv)
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 3/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "The current few-shot benchmarks use perfectly class-balanced tasks at inference."
> "unlabeled query sets come with arbitrary and unknown label marginals."
> "We introduce and study the effect of arbitrary class distributions within the query sets"
> "substantial performance drops, even below inductive methods in some cases"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Evaluates transductive methods (incl. OT/Sinkhorn class-balanced ones) when query sets have arbitrary label marginals.
- regime: SOURCE CLAIM: Transductive classification with imbalanced unlabeled batches.
- when it fails: SOURCE CLAIM/INFERENCE: Hard balance assumptions collapse; search snippet (not fetched in-page) also says PT-MAP drops ~18-20 pts under Dirichlet imbalance.
- cost / code: Code available.
- relevance to Name That LLM: INFERENCE: Strongest counter-evidence for rigid Hungarian/equal-marginal Sinkhorn when per-group class presence is unequal. Use soft/relaxed marginals, or at-most-once (inequality) constraints, and test by simulating dropped classes per prompt.
