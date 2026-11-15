# 86_gao_model_equality_testing
- title: Model Equality Testing: Which Model Is This API Serving?
- url: https://arxiv.org/abs/2410.20247 (abstract page; PDF also fetched)
- code: https://github.com/irenasaracay/model-equality-testing (repo exists per GitHub API, description "Test equality between a black-box LLM API and a reference distribution"; link to the paper inferred, not stated in the fetched abstract)
- type: academic
- author(s) / venue / date: Irena Gao et al.; arXiv 2410.20247
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 4/5 | Bias: low
## Verbatim quotes
> "a test built on a simple string kernel achieves a median of 77.4% power against a range of distortions, using an average of just 10 samples per prompt."
> "We formalize detecting such distortions as Model Equality Testing, a two-sample testing problem"
> "finding that 11 out of 31 endpoints serve different distributions than reference weights released by Meta."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Maximum Mean Discrepancy two-sample test with a Hamming-distance string kernel over completions detects quantisation, watermarking and fine-tuning distortions of an API versus reference samples. INFERENCE: MMD with string kernels is a distribution-level test (many samples per prompt); with one passage per prompt and 18 classes it is not directly usable, but the idea of kernel two-sample matching of each test passage against per-class reference pools is a cheap kNN-style feature.
- regime: Needs ~10 samples per prompt from both sides; distribution-level, not per-document.
- generalization / failure evidence: Evidence that serving infrastructure (quantisation, endpoint) changes output distribution: 11/31 Llama endpoints differed from reference weights. Supports the community report that API outputs differ from released weights (e.g. 3.7->3.8 routing).
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: string-kernel computation is CPU-cheap.
- relevance to Name That LLM: Explains why a local open-weight copy would be an imperfect proxy for API-served generators, and why quant-level differences may blur the exact-generator likelihood.
