# 72_menon_logit_adjustment
- title: Long-tail learning via logit adjustment
- url: https://arxiv.org/abs/2007.07314
- code: none found
- type: academic
- author(s) / venue / date: Menon, Jayasumana, Rawat, Jain, Veit, Kumar / ICLR 2021 / 2020-07-14
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 3/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "Real-world classification problems typically exhibit an imbalanced or long-tailed label distribution"
> "Our techniques revisit the classic idea of logit adjustment based on the label frequencies"
> "Such adjustment encourages a large relative margin between logits of rare versus dominant labels."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Post-hoc subtract tau*log(pi_y) from logits, or equivalent loss, which is Bayes-optimal for balanced error.
- regime: SOURCE CLAIM: Long-tailed training priors; balanced-error objective (fetched page did not mention term; INFERENCE from paper title/content).
- when it fails: SOURCE CLAIM/INFERENCE: Needs known training priors; tau tuned.
- cost / code: No code on page.
- relevance to Name That LLM: INFERENCE: Balanced accuracy optimum = argmax p(y|x)/pi_train. If community 200k rows are imbalanced, apply logit adjustment with the training prior, then optional test-prior estimate.
