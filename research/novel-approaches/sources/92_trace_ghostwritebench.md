# 92_trace_ghostwritebench
- title: Who Wrote the Book? Detecting and Attributing LLM Ghostwriters (GhostWriteBench, TRACE)
- url: https://arxiv.org/abs/2603.28054 (abstract page; PDF also fetched)
- code: none found
- type: academic
- author(s) / venue / date: Anudeex Shetty, Qiongkai Xu, Olga Ohrimenko, Jey Han Lau; arXiv 2603.28054
- accessed: 2026-11-12
- scores: Credibility 3/5 | Recency 5/5 | Bias: low-med (new, un-replicated; long-form books, not short paragraphs)
## Verbatim quotes
> "We also propose TRACE -- a novel fingerprinting method that is interpretable and lightweight -- that works for both open- and closed-source models."
> "TRACE creates the fingerprint by capturing token-level transition patterns (e.g., word rank) estimated by another lightweight language model."
> "dropping by as much as 70%"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: TRACE builds a fingerprint matrix of consecutive-rank transitions (binned into equal-probability clusters) under a lightweight evaluator LM; ten frontier LLMs, ~49K words per book. SOURCE CLAIM (PDF text, fragment): existing attribution methods degrade substantially in unseen-author scenarios, "dropping by as much as 70%".
- regime: Proxy LM rank-transition features + classifier; works with closed generators.
- generalization / failure evidence: Opposition/transfer limit: documents are 50K-word books; our passages are 40-250 words, so transition statistics are far noisier. OOD-author degradation is not our setting (all 18 classes are seen), which favours supervised methods.
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: one forward of a small LM per passage; minutes on a T4.
- relevance to Name That LLM: Best direct match for a proxy-LM complementary signal (rank transitions), including the closed-model case.
