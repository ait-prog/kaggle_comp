# 69_alexandari_bcts_label_shift
- title: Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation
- url: https://arxiv.org/abs/1901.06852
- code: https://github.com/kundajelab/labelshiftexperiments
- type: academic
- author(s) / venue / date: Alexandari, Kundaje, Shrikumar / ICML 2020 / 2019-01-21
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 3/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "and the lack of calibration can decrease the effectiveness of EM."
> "the best results were achieved when the calibration method involves class-specific bias parameters"
> "they did not tend to outperform EM when the probabilities were well-calibrated."
> "outperforms both BBSL and RLLS across diverse datasets and distribution shifts"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Fit temperature scaling plus class-specific bias (BCTS) on held-out data, then run Saerens EM; use training priors carefully.
- regime: SOURCE CLAIM: Label shift, deep nets, vision and genomics.
- when it fails: SOURCE CLAIM/INFERENCE: Miscalibrated posteriors degrade EM (quoted); calibration must be fit on data resembling target-time conditions.
- cost / code: Code available (kundajelab).
- relevance to Name That LLM: INFERENCE: Directly applicable: fit BCTS on leave-prompt-out logits, then EM on test. Class-specific biases also compensate for per-model over/under-prediction.
