# 93_reader_llm_provenance
- title: READER: Dynamic LLM Provenance from Query-Varying Interactions
- url: https://arxiv.org/abs/2606.10794 (abstract page; PDF also fetched)
- code: none found (paper cites github.com/LeoJeshua/READER, but GitHub API returned Not Found at access time)
- type: academic
- author(s) / venue / date: Jiaxu Liu, Sunnan Mu, Dong Huang, Liuyin Wang; arXiv 2606.10794
- accessed: 2026-11-12
- scores: Credibility 3/5 | Recency 5/5 | Bias: low-med (new, un-replicated; includes self-built benchmark Agent500)
## Verbatim quotes
> "READER recovers comparability through a frozen proxy LLM."
> "On 100-way attribution, READER reaches $50.4\%$ accuracy from one response and $96.2\%$ from 100"
> "Controlled response-length and Math100 domain shifts expose the limits of zero-retraining transfer."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: projects a frozen proxy LLM's hidden states aligned with response tokens onto length-normalised DC and first-AC modes; enrollment-trained linear probe; Bayesian accumulation across responses. INFERENCE: at the single-response level, 100-way accuracy is only 50.4% (on their benchmark), so hidden-state probes give moderate per-document accuracy but are a legitimate high-capacity feature set for 18 classes.
- regime: Frozen open proxy LM hidden states plus a linear probe trained on labelled outputs of each candidate; no target internals.
- generalization / failure evidence: Single-response accuracy is the relevant number for us (one passage each): 50.4% at 100 classes; degrades under length and domain shift.
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: one forward pass through a 1-8B proxy per passage plus mean-pooled states; minutes to ~1 h on 2xT4 for 4,000 passages.
- relevance to Name That LLM: Supports training a probe on frozen proxy-LM hidden states as the main complementary signal; length shift caveat applies to 40-250 word range.
