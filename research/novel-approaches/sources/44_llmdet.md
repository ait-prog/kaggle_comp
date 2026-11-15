# 44_llmdet
- title: LLMDet: A Third Party Large Language Models Generated Text Detection Tool
- url: https://arxiv.org/abs/2305.15004 (full text read at https://arxiv.org/html/2305.15004)
- code: https://github.com/TrustedLLM/LLMDet
- type: academic
- author(s) / venue / date: Kangxi Wu, Liang Pang, Huawei Shen, Xueqi Cheng, Tat-Seng Chua; arXiv 24 May 2023 (v3 Nov 2023); Findings of EMNLP 2023
- accessed: 2026-11-12
- scores: Credibility 3/5 | Recency 2/5 | Bias: low (old open models only)
## Verbatim quotes
> "we record the next-token probabilities of salient n-grams as features"
> "LLMDet exhibits outstanding overall detection performance, with an F1-Macro score of 88.14%"
> "the F1 score for detecting texts from other sources is at least 76.39%"
> "resulting in suboptimal detection performance for some short texts."
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: precompute dictionary of salient n-grams to next-token probabilities per candidate LLM; proxy perplexity per model becomes a feature vector fed to a classifier; no model access at detection.
- regime (text length, #classes, domains): SOURCE CLAIM: Human plus 8 open LLMs (GPT-2, LLaMA, OPT, UniLM, BART, T5, Bloom, GPT-Neo).
- generalization evidence: no domain test of its own in fetched text.
- cost / code: cheap lookup; repo exists.
- relevance to Name That LLM: INFERENCE: per-model log-likelihood features need open candidates; here the open labels (Qwen, GLM, Kimi if weights fit) could supply scoring-LM likelihood features, but closed labels cannot. Cross-model likelihood vectors are a different view from hidden-state pooling.
