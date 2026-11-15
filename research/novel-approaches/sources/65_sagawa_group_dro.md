# 65_sagawa_group_dro
- title: Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization
- url: https://arxiv.org/pdf/1911.08731 (abstract via arxiv.org/abs/1911.08731; quotes via ar5iv)
- code: https://github.com/kohpangwei/group_DRO
- type: academic
- author(s) / venue / date: Sagawa, Koh, Hashimoto, Liang / ICLR 2020 (arXiv) / 2019-11-20
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 3/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "naively applying group DRO to overparameterized neural networks fails"
> "achieve substantially higher worst-group accuracies, with 10–40 percentage point improvements."
> "Our results suggest that regularization is important for worst-group generalization in the overparameterized regime"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Minimise worst-group loss with online group reweighting; needs strong L2 or early stopping to avoid memorising training groups.
- regime: SOURCE CLAIM: Known groups, CelebA/Waterbirds/MultiNLI, overparameterised nets.
- when it fails: SOURCE CLAIM/INFERENCE: Without strong regularisation DRO is no better than ERM; groups must be known; test groups must resemble training groups.
- cost / code: Official code available.
- relevance to Name That LLM: INFERENCE: Groups = prompts (or prompt clusters, or classes for macro-recall). Class-level DRO is effectively a balanced-accuracy objective; prompt-level DRO targets unseen-prompt robustness. Expect modest gains; must use heavy regularization.
