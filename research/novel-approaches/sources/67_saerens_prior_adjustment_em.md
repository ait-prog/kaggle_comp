# 67_saerens_prior_adjustment_em
- title: Adjusting the Outputs of a Classifier to New A Priori Probabilities: A Simple Procedure
- url: https://mlanthology.org/neco/2002/saerens2002neco-adjusting
- code: none found
- type: academic
- author(s) / venue / date: Saerens, Latinne, Decaestecker / Neural Computation 14(1) / 2002
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 1/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "Experimental results also indicate that the classifier with adjusted outputs always performs better than the original one"
> "As a by-product, estimates of the new a priori probabilities are also obtained."
> "The gain in classification accuracy can be significant."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: EM alternates: reweight posteriors by estimated prior ratio, re-estimate priors as mean posterior, until convergence. No retraining.
- regime: SOURCE CLAIM: Prior shift, p(x|y) fixed, calibrated posteriors.
- when it fails: SOURCE CLAIM/INFERENCE: Needs calibrated posteriors (see Alexandari). Our test shift also shifts p(x|y) (new prompts) so assumption only holds approximately (INFERENCE).
- cost / code: Trivial to implement; no code needed.
- relevance to Name That LLM: INFERENCE: Core tool for estimating test class mix from model probabilities. For balanced accuracy, estimate priors then DIVIDE them out (predict argmax p(y|x)/pi_test*uniform) rather than multiply in. Pool across public/private split separately.
