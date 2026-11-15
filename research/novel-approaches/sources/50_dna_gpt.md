# 50_dna_gpt
- title: DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text
- url: https://arxiv.org/abs/2305.17359
- code: https://github.com/Xianjun-Yang/DNA-GPT
- type: academic
- author(s) / venue / date: Xianjun Yang, Wei Cheng, Yue Wu, Linda Petzold, William Yang Wang, Haifeng Chen; arXiv 27 May 2023 (v2 Oct 2023)
- accessed: 2026-11-12
- scores: Credibility 3/5 | Recency 2/5 | Bias: med (needs sampling from target)
## Verbatim quotes
> "propose a novel training-free detection strategy called Divergent N-Gram Analysis (DNA-GPT)."
> "robust under the revised text attack and can additionally solve model sourcing."
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM (abstract only; mechanism detail from my own knowledge, not fetched): truncate the text, regenerate continuations from the candidate model, compare n-gram overlap; model sourcing via overlap with each model's regeneration.
- regime (text length, #classes, domains): not retrieved.
- generalization evidence: not retrieved.
- cost / code: repo exists; requires generation from each candidate model.
- relevance to Name That LLM: INFERENCE: regeneration-overlap is impossible for closed labels offline, but a variant using open models' regenerations as features is possible; low priority.
