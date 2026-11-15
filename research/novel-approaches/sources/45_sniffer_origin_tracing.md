# 45_sniffer_origin_tracing
- title: Origin Tracing and Detecting of LLMs
- url: https://arxiv.org/abs/2304.14072 (full text read at https://arxiv.org/html/2304.14072)
- code: https://github.com/OpenLMLab/ (org page only; paper-specific repo not verified)
- type: academic
- author(s) / venue / date: Linyang Li, Pengyu Wang, Ke Ren, Tianxiang Sun, Xipeng Qiu; arXiv 27 Apr 2023; "working in progress"
- accessed: 2026-11-12
- scores: Credibility 2/5 | Recency 2/5 | Bias: low (old models, Chinese docs)
## Verbatim quotes
> "the core idea of Sniffer is to find different features between models."
> "it grows harder to detect and trace origins when the LLMs are stronger;"
> "texts from both Alpaca and Dolly tend to be categorized as ChatGPT-generated texts"
> "Sniffer can generalize to trace GPT-3 model origins"
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: perplexity of the sample under several known models, contrasted between models, as features for a tracer.
- regime (text length, #classes, domains): SOURCE CLAIM: known models GPT2-xl, GPT-Neo, GPT-J, LLaMA; "relatively long documents (more than 200 Chinese characters per document)".
- generalization evidence: SOURCE CLAIM: traces GPT-3 origins from known models.
- cost / code: needs forward passes of several reference LMs.
- relevance to Name That LLM: INFERENCE: reference-LM perplexity-difference features; also the finding that distilled/derived models collapse into their teacher is analogous to sibling/distilled label confusion.
