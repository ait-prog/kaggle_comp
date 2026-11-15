# 40_idiosyncrasies_llms
- title: Idiosyncrasies in Large Language Models
- url: https://arxiv.org/abs/2502.12150 (full text read at https://arxiv.org/html/2502.12150)
- code: https://github.com/locuslab/llm-idiosyncrasies
- type: academic
- author(s) / venue / date: Mingjie Sun, Yida Yin, Zhiqiu Xu, J. Zico Kolter, Zhuang Liu; arXiv 17 Feb 2025 (rev. 16 Jun 2025); associated with ICML 2025
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 4/5 | Bias: low (closest prior art to this task; only 5 chat APIs, long responses)
## Verbatim quotes
> "we achieve 97.1% accuracy on held-out validation data in the five-way classification problem"
> "These patterns persist even when the texts are rewritten, translated, or summarized by an external LLM"
> "Our further investigation reveals that these idiosyncrasies are rooted in word-level distributions."
> "Within the same model family, we obtain a non-trivial 59.8% accuracy across 4 model sizes in Qwen-2.5 series"
> "Intriguingly, for chat APIs and instruct LLMs, we observe around 50% accuracy using only a single text token."
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: fine-tune an LLM2vec (decoder-only) sequence embedding classifier on responses to the same prompts; signal in word-level distributions, markdown/formatting, and semantics. Word-shuffled text still classifies well ("word-shuffled responses yields high accuracies comparable to the original ones").
- regime (text length, #classes, domains): SOURCE CLAIM: first 512 tokens used; "Performance begins to saturate beyond an input sequence length of 256." 5 chat APIs (ChatGPT, Claude, Grok, Gemini, DeepSeek) plus open model families/sizes; instruct vs base.
- generalization evidence: SOURCE CLAIM: "We find that our classifiers generalize robustly to responses beyond their training distribution." Paraphrase-trained classifiers stay near original accuracy.
- cost / code: SOURCE CLAIM: public repo; LLM2vec finetune (8B-class), single-GPU scale. INFERENCE: same family as the pooled-hidden-state LoRA already in use.
- relevance to Name That LLM: INFERENCE: confirms word-level/unigram distribution is a major source of signal, so a bag-of-words / TF-IDF head is a legitimately complementary view to contextual embeddings (word shuffling barely hurts). First-token signal (~50% from one token) suggests explicit opening-token/first-sentence features and the 59.8% sibling-size figure sets a realistic ceiling for sibling labels.
