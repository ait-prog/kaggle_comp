# 63_styledistance_content_independent
- title: StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples
- url: https://arxiv.org/html/2410.12757v2
- code: https://huggingface.co/StyleDistance/styledistance (model only)
- type: academic
- author(s) / venue / date: Patel, Zhu, Qiu, Horvitz, Apidianaki, McKeown, Callison-Burch / NAACL 2025 / 2024-10-16
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 5/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "However, these methods are limited by the imperfect nature of data acquired under such assumptions."
> "does not explicitly train content-independent representations but captures both content and style features."
> "only models specifically trained for content-independence are able to capture style features better than content features."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Contrastive training on synthetic paraphrase-like parallel examples that hold content fixed and vary one style feature; evaluates with STEL-or-Content.
- regime: SOURCE CLAIM: Style embeddings, content-independence probe.
- when it fails: SOURCE CLAIM/INFERENCE: Disconfirming: LUAR-type same-author/different-topic training still encodes content (STEL-or-Content). Fetch agent reported LUAR score near 0 vs high on STEL; exact figure not verified (search snippet gave a different number).
- cost / code: HF model only.
- relevance to Name That LLM: INFERENCE: OPPOSITION. Do not assume author-style embeddings are topic-free; validate with leave-prompt-out. Also supports content-matched (same-prompt) positives.
