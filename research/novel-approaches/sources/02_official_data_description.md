# 02_official_data_description
- title: Name That LLM, Data description and Description (official)
- url: https://www.kaggle.com/competitions/name-that-llm/data ; https://www.kaggle.com/competitions/name-that-llm/overview
- code: none
- type: primary (host page)
- author(s) / date: Kaggle (host)
- accessed: 2026-11-12 via `competitions.PageService/ListPages`. Raw: sources/raw/kaggle_pages_2026-11-12.json keys "data-description", "Description"
- scores: Credibility 5/5 | Recency 5/5 | Bias: low
## Verbatim quotes
> "Every passage is an authentic LLM completion. Only light formatting cleanup was applied (for example, removing a `**bold**` wrapper around a whole passage)."
> "~4000 passages in the hidden rerun. Test passages are written for prompts that do not appear in **train.csv**."
> "Pinpointing the exact source model (a task known as authorship attribution) is much harder when prompts are unseen and passage lengths vary randomly between 40 and 250 words."
> "You can't rely on simple character-count shortcuts or topic keywords."
## What it supports
- FACT: train = 2,699 passages over 150 prompts. Test ≈ 4,000 passages on prompts disjoint from train. Test has no prompt_id.
- FACT: lengths are randomised to 40–250 words (a length target is part of the generation protocol).
- FACT (local check, sources/09): train is 149 prompts × 18 + 1 prompt × 17; each model answers each prompt at most once.
- UNKNOWN: system prompt, temperature, reasoning settings, API route and generation dates for the official data. The host has not documented them (forum questions 750408, 752280 are unanswered in the captured threads).
