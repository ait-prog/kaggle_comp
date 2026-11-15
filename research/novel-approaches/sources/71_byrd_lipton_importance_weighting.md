# 71_byrd_lipton_importance_weighting
- title: What is the Effect of Importance Weighting in Deep Learning?
- url: https://arxiv.org/abs/1812.03372
- code: none found
- type: academic
- author(s) / venue / date: Byrd, Lipton / ICML 2019 (arXiv) / 2018-12-08
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 2/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "little is known about how it impacts over-parameterized, deep neural networks"
> "its effect diminishes over successive epochs"
> "restore some of the impact of importance weighting"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Empirical/theory: reweighted loss in overfit nets converges to the same solution as unweighted; effect vanishes once data are separable.
- regime: SOURCE CLAIM: Overparameterised nets trained to zero loss.
- when it fails: SOURCE CLAIM/INFERENCE: Fix: regularisation / early stopping / weight on logits.
- cost / code: None on page.
- relevance to Name That LLM: INFERENCE: OPPOSITION to class-weighted loss for balanced accuracy in fine-tuned LMs that fit train perfectly. Prefer post-hoc logit adjustment of priors.
