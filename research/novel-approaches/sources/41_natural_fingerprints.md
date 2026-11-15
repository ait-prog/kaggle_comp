# 41_natural_fingerprints
- title: Natural Fingerprints of Large Language Models
- url: https://arxiv.org/abs/2504.14871 (full text read at https://arxiv.org/html/2504.14871v2)
- code: none found
- type: academic
- author(s) / venue / date: Teppei Suzuki, Ryokan Ri, Sho Takase; arXiv 21 Apr 2025 (v2 19 Sep 2025)
- accessed: 2026-11-12
- scores: Credibility 3/5 | Recency 4/5 | Bias: low (pretrained-from-scratch small models, not frontier APIs)
## Verbatim quotes
> "We refer to these unintended, distinctive characteristics as natural fingerprints."
> "even when LLMs are trained on exactly the same dataset, their outputs remain distinguishable"
> "The Unigram-based classifier achieved 39.0% accuracy and the Transformer-based classifier reached 55.6%."
> "The accuracy of even simple unigram-based methods indicates bias in their unigram distributions."
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: classifiers distinguish outputs of LLMs trained on identical data; differences arise from training dynamics, data order, random seed.
- regime (text length, #classes, domains): SOURCE CLAIM: max 512 output tokens; six families at 7B plus 0.5B/1B/3B models; fetch found no prompt/domain generalization test.
- generalization evidence: none in fetched text ("does not test across prompts or domains" per fetch summary).
- cost / code: no repo found.
- relevance to Name That LLM: INFERENCE: supports that sibling models from the same lab are separable by distributional bias (unigram classifier already well above chance), so sibling confusion is not purely irreducible; accuracy grows logarithmically with training data, matching the use of the 200k noisy rows.
