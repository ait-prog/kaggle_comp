# 94_knnproxy_proxy_mismatch
- title: kNNProxy: Efficient Training-Free Proxy Alignment for Black-Box Zero-Shot LLM-Generated Text Detection
- url: https://arxiv.org/abs/2604.02008 (abstract page; PDF also fetched)
- code: none found
- type: academic
- author(s) / venue / date: Kahim Wong, Kemou Li, Haiwei Wu; arXiv 2604.02008
- accessed: 2026-11-12
- scores: Credibility 3/5 | Recency 5/5 | Bias: low-med
## Verbatim quotes
> "the reliability of zero-shot methods fundamentally relies on the assumption that an off-the-shelf proxy LLM is well aligned with the often unknown source LLM, a premise that rarely holds in real-world black-box scenarios."
> "a lightweight datastore is constructed once from a target-reflective LGT corpus"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: zero-shot detectors depend on proxy-source alignment, which rarely holds; kNN-LM style datastore interpolated with the proxy distribution aligns the proxy without fine-tuning. INFERENCE: this is the nearest fetched statement of the own-model vs proxy gap (the paper states proxy mismatch hurts, the fetched pages do not give an own-model/proxy accuracy table that I extracted).
- regime: Proxy LM token distribution interpolated with kNN retrieval from labelled generated text.
- generalization / failure evidence: Supports that proxy-only likelihood features are weak when the generator is unknown; supervised alignment (using the training set) is the fix.
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: kNN-LM datastore for 4,000-passage scale is tiny; cheap.
- relevance to Name That LLM: Argues for using the labelled training data to adapt a proxy rather than raw perplexity.
