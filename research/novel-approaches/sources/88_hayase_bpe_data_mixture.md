# 88_hayase_bpe_data_mixture
- title: Data Mixture Inference: What do BPE Tokenizers Reveal about their Training Data?
- url: https://arxiv.org/abs/2407.16607 (abstract page; PDF also fetched)
- code: https://github.com/alisawuffles/tokenizer-attack (repo exists per GitHub API: "Official implementation of Data Mixture Inference...")
- type: academic
- author(s) / venue / date: Jonathan Hayase, Alisa Liu, Yejin Choi, Sewoong Oh, Noah A. Smith; arXiv 2407.16607
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 3/5 | Bias: low
## Verbatim quotes
> "Our key insight is that the ordered list of merge rules learned by a BPE tokenizer naturally reveals information about the token frequencies in its training data."
> "GPT-4o and Mistral NeMo's tokenizers are much more multilingual than their predecessors, training on 39% and 47% non-English language data, respectively"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: the ordered merge list of a BPE tokenizer lets one solve a linear program for the training-data mixture. INFERENCE: this infers data mix from a tokenizer; it does not attribute generated text. Transfer to this task is indirect: tokenizer vocabularies differ across labs (fetched configs: GLM vocab 154,880; Kimi 163,840; Qwen3.8 248,320), so a text tokenised by lab-specific tokenizers yields lab-specific compressibility features (tokens per word, rare-token counts). Closed labs (OpenAI, Anthropic, Google) have no public tokenizers for the newest models, limiting this.
- regime: Needs the tokenizer merge list and category sample data.
- generalization / failure evidence: Not a text attribution method; no generalisation claims for our use.
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: tokenise 4,000 passages with 3-4 open tokenizers: seconds on CPU.
- relevance to Name That LLM: Low direct relevance; the cheap derived feature is multi-tokenizer compression ratios as an extra stylometric channel for open-lab families.
