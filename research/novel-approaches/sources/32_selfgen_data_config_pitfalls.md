# 32_selfgen_data_config_pitfalls
- title: Notes from our runs §4 (751871); Will self-generated extra data help or hurt here? (755371); 9,719 OpenAI responses with exact prompts and provenance (753974); Reconstructed train prompts (749455)
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/751871
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/755371
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/753974
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/749455
- type: discussion
- author(s) / date: Daniel Andres Miranda 2026-10-22; Ruslan Akhmetzianov 2026-11-03; Enai 2026-10-29; pjmathematician 2026-10-15
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 3/5 | Recency 5/5 | Bias: med
## Verbatim quotes
> "On OpenRouter only `kimi-k3` accepts `reasoning.enabled=false`." (751871)
> "Kimi-K3 without reasoning matches train's em-dash rate much better than with reasoning." (751871)
> "Data generated under a different setup might teach the model our prompt template instead of the model's style." (Ruslan Akhmetzianov, 755371)
> "I tried with a little bit of extra data but my val score decreased, might be due my generation setup not being exact(system prompt and other factors)" (somerandomguy, 755371)
> "These are LLM guesses, not ground truth." (pjmathematician, 749455, on reconstructed train prompts)
## What it supports
- SOURCE CLAIM: generation configuration (reasoning on/off/effort, route, system prompt) is only partly controllable and is contested (kimi: sources/26 vs this). Naive self-generation can hurt.
- SOURCE CLAIM: prompt reconstructions of all 150 train prompts exist publicly (LLM-inferred). Fresh prompt-panel datasets with provenance exist for 4 OpenAI labels.
- INFERENCE: own data is valuable only if config is matched *and* validated against official train rows by prompt-grouped fidelity tests. It also needs a served-model check.
