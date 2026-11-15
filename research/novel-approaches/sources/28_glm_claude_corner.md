# 28_glm_claude_corner
- title: My classifier thinks GLM-5.3 is more Claude than GLM (755314); Running GLM-5.3 / GLM-5.3-Flash inside the 9h limit (757199); The glm/claude-fable corner (752197)
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/755314
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/757199
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/752197
- type: discussion
- author(s) / date: AyoGG 2026-11-03; vishu phalswal 2026-11-09; Meni David Gottesman 2026-10-23; commenters Vijay Konda, John K
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 3/5 | Recency 5/5 | Bias: low
## Verbatim quotes
> "GLM-5.3? Classified as Claude more often than as GLM-5.3 itself (68 vs 34 out of 150)." (AyoGG, 755314)
> "GLM-4-9B-0414 + Qwen3-8B-Base log-likelihood features (-0.002 in a prompt-grouped CV stack)" (vishu phalswal, 757199, listed under things that did NOT help)
> "I noticed public datasets with IQ3_XXS GGUF layer segments of GLM-5.3 and GLM-5.3-Flash, so the real model's likelihood looks like the missing signal." (vishu phalswal, 757199)
> "The workable route is layer streaming over the whole test set: load one layer, run all passages' hidden states through it, free it, next layer. That reads the weights once." (Vijay Konda, 757199)
## What it supports
- SOURCE CLAIM: the GLM↔Claude corner is the dominant cross-lab error for strong systems too (glm-5.3 OOF recall 0.32 in a ~0.68 system).
- SOURCE CLAIM: proxy-LM likelihood (an older GLM or Qwen) did not help, but quantised segments of the actual GLM-5.3 weights are reportedly public on Kaggle. Exact-generator likelihood is untried publicly. See REPORT.md, unconventional bet.
- UNVERIFIED: existence, licence and size of those GGUF datasets (not fetched; see sources/80+ for the open-weight check).
