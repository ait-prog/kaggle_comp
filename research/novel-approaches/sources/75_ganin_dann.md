# 75_ganin_dann
- title: Domain-Adversarial Training of Neural Networks
- url: https://arxiv.org/abs/1505.07818
- code: none found
- type: academic
- author(s) / venue / date: Ganin et al. / JMLR 17 / 2016
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 1/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "We introduce a new representation learning approach for domain adaptation"
> "a new gradient reversal layer"
> "no labeled target-domain data is necessary"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Domain classifier head with gradient reversal makes features domain-invariant (here: prompt/topic-invariant).
- regime: SOURCE CLAIM: Unsupervised DA with unlabeled target.
- when it fails: SOURCE CLAIM/INFERENCE: DANN can remove useful information and is unstable; Gulrajani found no consistent gain over ERM (PAGE not same fetch).
- cost / code: No repo on page.
- relevance to Name That LLM: INFERENCE: Test passages are unlabeled target data - DANN with prompt as the adversarial label is the adversarial topic-removal option, but with 150 prompts it is a low-prior bet; try last.
