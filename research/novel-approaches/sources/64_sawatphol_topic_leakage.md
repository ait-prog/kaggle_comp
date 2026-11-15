# 64_sawatphol_topic_leakage
- title: Addressing Topic Leakage in Cross-Topic Evaluation for Authorship Verification
- url: https://arxiv.org/abs/2407.19164
- code: none found
- type: academic
- author(s) / venue / date: Sawatphol, Udomcharoenchaikit, Nutanong / TACL 2024 / 2024-07-27
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 4/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "we argue that there can still be topic leakage in test data, causing misleading model performance"
> "Heterogeneity-Informed Topic Sampling (HITS), which creates a smaller dataset with a heterogeneously distributed topic set"
> "HITS-sampled datasets yield a more stable ranking of models across random seeds and evaluation splits."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Even 'cross-topic' splits leak topic similarity; heterogeneous topic sampling (HITS) and a topic-shortcut test (RAVEN) give stable model rankings.
- regime: SOURCE CLAIM: Evaluation design for AV.
- when it fails: SOURCE CLAIM/INFERENCE: Their remedy shrinks the dataset; AV rather than closed-set attribution.
- cost / code: No code link on page.
- relevance to Name That LLM: INFERENCE: Our CV must be GroupKFold by prompt and ideally by prompt-topic cluster; random-row CV will overstate and mis-rank. Check near-duplicate prompt topics across folds.
