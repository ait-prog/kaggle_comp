# 68_lipton_bbse_label_shift
- title: Detecting and Correcting for Label Shift with Black Box Predictors
- url: https://arxiv.org/abs/1802.03916
- code: none found
- type: academic
- author(s) / venue / date: Lipton, Wang, Smola / ICML 2018 / 2018-02-12
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 2/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "BBSE works even when predictors are biased, inaccurate, or uncalibrated, so long as their confusion matrices are invertible."
> "We prove BBSE's consistency, bound its error, and introduce a statistical test that uses BBSE to detect shift."
> "Experiments demonstrate accurate estimates and improved prediction, even on high-dimensional datasets of natural images."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Estimate importance weights w = C^{-1} mu_pred using held-out confusion matrix C and predicted-label histogram.
- regime: SOURCE CLAIM: Label shift, black-box predictor.
- when it fails: SOURCE CLAIM/INFERENCE: Confusion matrix must be invertible and estimated on held-out data matching train conditions; 18x18 confusion from 150 prompts is noisy; fails if p(x|y) shifts (e.g. new-prompt degradation).
- cost / code: No official code found on page.
- relevance to Name That LLM: INFERENCE: Use OOF (leave-prompt-out) confusion matrix, which already includes unseen-prompt degradation - a good fit. Compare with EM; cheap.
