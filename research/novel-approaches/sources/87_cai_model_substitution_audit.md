# 87_cai_model_substitution_audit
- title: Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs
- url: https://arxiv.org/abs/2504.04715 (abstract page; PDF also fetched)
- code: https://github.com/sunblaze-ucb/llm-api-audit (repo exists per GitHub API; also linked from the abstract)
- type: academic
- author(s) / venue / date: Will Cai et al.; arXiv 2504.04715
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 4/5 | Bias: low
## Verbatim quotes
> "statistical tests on text outputs are query-intensive and fail against subtle substitutions, while methods using log probabilities are defeated by inherent inference nondeterminism in production environments."
> "Providers may covertly substitute cheaper alternatives (e.g., quantized versions, smaller models) to reduce costs while maintaining advertised pricing."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: text-only statistical detectors need many queries and miss subtle substitutions; log-prob methods are unreliable in production due to nondeterminism. INFERENCE: the Kaggle data may include API-side substitution (e.g. gemini-3.7 routed to 3.8, see 95), which is label noise from the perspective of a pure model-identity classifier.
- regime: Black-box API audits with many queries.
- generalization / failure evidence: Opposition: shows software-only text/logprob signals are fragile for fine-grained identity (subtle substitutions are not detectable).
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: n/a.
- relevance to Name That LLM: Warns that sibling-version discrimination (same lab, subtle variants) is intrinsically hard from text alone and that label noise from routing is plausible.
