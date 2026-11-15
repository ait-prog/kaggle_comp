# 42_fdllm
- title: FDLLM: A Dedicated Detector for Black-Box LLMs Fingerprinting
- url: https://arxiv.org/abs/2501.16029 (full text read at https://arxiv.org/html/2501.16029v4)
- code: none found
- type: academic
- author(s) / venue / date: Zhiyuan Fu, Junfan Chen, Lan Zhang et al.; arXiv 27 Jan 2025 (v4 19 Jan 2026), cs.CR, no venue listed
- accessed: 2026-11-12
- scores: Credibility 3/5 | Recency 5/5 | Bias: med (own dataset; in-domain headline)
## Verbatim quotes
> "FDLLM (Q) achieves a Macro F1 of 96.56 %, representing a relative gain of 29 points over the strongest baseline (OpenAI-D)." [LEAD VERIFICATION 2026-11-12: not re-located in abs/html full text; treat as unverified]
> "FDLLM also exhibits strong generalization to newly released models, achieving an average accuracy of 95% on unseen models."
> "The Macro F1 score dropped to 19.43%, suggesting a notable gap between domains."
> "Our study relies on data collected between September 2024 and May 2025, which imposes temporal limitations."
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: LoRA-tuned Qwen2.5-7B-Instruct classifier, one forward pass; bilingual 90k-sample dataset over 20 LLMs including GPT-4o, Gemini-1.5, Claude3.5-haiku (per Table I).
- regime (text length, #classes, domains): SOURCE CLAIM: 20 classes; English and Chinese; multiple domains.
- generalization evidence: SOURCE CLAIM: in-distribution 96.56 Macro F1 but a cross-domain setting collapses to 19.43 Macro F1 (which exact setting is not clear from fetch; INFERENCE: treat as strong disconfirming evidence for unseen-prompt transfer).
- cost / code: 7B LoRA; no repo found.
- relevance to Name That LLM: INFERENCE: nearest architectural twin of the current approach, and its own cross-domain collapse is the main warning that in-prompt CV overstates unseen-prompt performance; validate with prompt-grouped folds.
