# 70_azizzadenesheli_rlls
- title: Regularized Learning for Domain Adaptation under Label Shifts
- url: https://arxiv.org/abs/1903.09734
- code: none found
- type: academic
- author(s) / venue / date: Azizzadenesheli, Liu, Yang, Anandkumar / ICLR 2019 / 2019-03-22
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 2/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "We propose Regularized Learning under Label shifts (RLLS), a principled and a practical domain-adaptation algorithm"
> "this is the first generalization bound for the label-shift problem"
> "Experiments on the CIFAR-10 and MNIST datasets show that RLLS improves classification accuracy"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: BBSE-style weight estimation with regularisation toward 1, then weighted retraining.
- regime: SOURCE CLAIM: Label shift, small target samples.
- when it fails: SOURCE CLAIM/INFERENCE: Alexandari find it not to beat calibrated EM; retraining with weights has the Byrd-Lipton weakness.
- cost / code: None on page.
- relevance to Name That LLM: INFERENCE: Regularised weights are useful when target class counts are tiny (private split mix); otherwise EM/BCTS suffices.
