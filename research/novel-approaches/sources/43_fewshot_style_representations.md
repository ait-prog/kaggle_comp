# 43_fewshot_style_representations
- title: Few-Shot Detection of Machine-Generated Text using Style Representations
- url: https://arxiv.org/abs/2401.06712 (full text read at https://arxiv.org/html/2401.06712v1)
- code: https://github.com/LLNL/LUAR/tree/main/fewshot_iclr2024
- type: academic
- author(s) / venue / date: Rafael Rivera Soto, Kailin Koch, Aleem Khan, Barry Chen, Marcus Bishop, Nicholas Andrews; arXiv 12 Jan 2024 (v3 May 2024); ICLR 2024 (venue per search result and repo path; abs page itself did not state it)
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 3/5 | Bias: low
## Verbatim quotes
> "Instead, we propose to leverage representations of writing style estimated from human-authored text."
> "given a handful of examples composed by each of several specific language models of interest"
> "most documents considered in this work have length around 128 tokens according to a fixed tokenizer" [LEAD VERIFICATION 2026-11-12: not re-located in abs/html full text; treat as unverified]
> "trained to ignore features that evolve over time, such as topic and domain"
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: LUAR/UAR authorship style embeddings trained on human authors with contrastive learning to be topic-invariant; few-shot classifier on support examples per LLM.
- regime (text length, #classes, domains): SOURCE CLAIM: ~128 token documents; evaluation across M4's five domains with only Reddit seen in training; Table 1 pAUC 0.905 at N=5 and 0.9806 at N=10 documents per episode (that is detection with multi-document episodes, not single-paragraph).
- generalization evidence: SOURCE CLAIM: trained on Reddit only, evaluated across five domains.
- cost / code: small encoder (RoBERTa-sized); code available.
- relevance to Name That LLM: INFERENCE: a topic-invariant style embedding (LUAR) as extra frozen features is cheap, orthogonal to LLM hidden states, and targets exactly the unseen-prompt shift. Its strong numbers rely on multi-document support sets.
