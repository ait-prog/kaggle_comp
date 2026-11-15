# 11_atlas_paper
- title: Who's Behind the Text? Robust Model Authorship Attribution (ATLAS; MAABench)
- url: https://arxiv.org/abs/2610.19583 (metadata via https://export.arxiv.org/api/query?id_list=2610.19583)
- code: none verified
- type: academic (preprint)
- author(s) / venue / date: Minseok Ko, Hyunseok Lee, Woomin Song, Beomjun Kim, Jinwoo Shin, Kunwoong Kim; arXiv 2026-10-14
- accessed: 2026-11-12 (abstract only)
- scores: Credibility 3/5 (preprint) | Recency 5/5 | Bias: low-med (authors' method)
## Verbatim quotes
> "existing MAA methods often exhibit substantial performance degradation under such distribution shifts."
> "ATLAS learns how text from each source family is generated and attributes a given text by comparing its likelihood under each learned source condition."
> "when both the generation domain and the model version change on MAABench, ATLAS achieves 74.27% accuracy, 9.00pp higher than the strongest baseline."
> "ATLAS can also help identify distillation source models by measuring which source model a target LLM most closely resembles."
## Notes
- SOURCE CLAIM: generative (source-conditioned likelihood) attribution is more robust to domain and version shift than discriminative classifiers.
- FACT (sources/06): an ATLAS adaptation is already in dev1 at 10% on low-margin rows, and the public lineage measured +0.002 to +0.007 (sources/29). The proxy-LM generative family is not new to this competition.
- INFERENCE: the distillation-tracing claim bears on GLM↔Claude confusability. If one label resembles another's teacher, likelihood under a conditioned proxy will also confuse them, unless the true generator's own likelihood is available (sources/80+).
