# 53_smaller_lms_detectors
- title: Smaller Language Models are Better Black-box Machine-Generated Text Detectors
- url: https://arxiv.org/abs/2305.09859
- code: none found
- type: academic
- author(s) / venue / date: Niloofar (Fatemehsadat) Mireshghallah, Justus Mattern, Sicun Gao, Reza Shokri, Taylor Berg-Kirkpatrick; arXiv May 2023 (v4 Feb 2024), no venue listed on page
- accessed: 2026-11-12
- scores: Credibility 3/5 | Recency 2/5 | Bias: low
## Verbatim quotes
> "We find that overall, smaller and partially-trained models are better universal text detectors:"
> "Most of these methods need access to the logits of the target model or need the ability to sample from the target."
> "For instance the OPT-125M model has an AUC of 0.81 in detecting ChatGPT generations,"
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: small/partially trained scoring LMs give better likelihood-based detection of other models' text.
- regime (text length, #classes, domains): binary detection.
- generalization evidence: universal detector across generators.
- cost / code: tiny LMs; none found.
- relevance to Name That LLM: INFERENCE: using a small, cheap (e.g., 0.5B Qwen / GPT-2) scoring model's token-level surprise statistics is complementary and nearly free on T4s.
