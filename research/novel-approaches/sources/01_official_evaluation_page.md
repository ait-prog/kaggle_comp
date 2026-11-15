# 01_official_evaluation_page
- title: Name That LLM, Overview > Evaluation (official)
- url: https://www.kaggle.com/competitions/name-that-llm/overview/evaluation
- code: none
- type: primary (host page)
- author(s) / date: Kaggle (host); page content as served 2026-11-12
- accessed: 2026-11-12 via anonymous Kaggle internal JSON API `competitions.PageService/ListPages` (WebFetch of the SPA page returned only the title). Raw copy: sources/raw/kaggle_pages_2026-11-12.json key "Evaluation"
- scores: Credibility 5/5 | Recency 5/5 | Bias: low (authoritative rules text)
## Verbatim quotes
> "Submissions are evaluated on **balanced accuracy**: the average, over all candidate models, of the fraction of that model's test passages you labelled correctly (per-class recall)."
> "The number of test passages per model is **not** equal, and the mix of models differs between the public and private leaderboard. Balanced accuracy gives every model the same weight."
> "When your notebook is submitted for scoring, it is re-run against the full hidden test set (~4000 passages)."
> "A submission is rejected if it is missing the `id` or `model` column, contains null values, repeats an `id`, or predicts a label outside the 18 listed above."
## What it supports
- FACT: metric = balanced accuracy (macro recall) over K = 18 labels.
- FACT: per-class test counts are unequal, and the public and private mixes differ.
- INFERENCE: for balanced accuracy under pure label shift, the Bayes decision is argmax p(x|y), i.e. train-prior-corrected posteriors. It does not depend on the unknown test prior, so forcing predicted marginals toward any fixed target is a bet, not a correction.
- INFERENCE: any post-processing that assumes "one passage per model per prompt" or equal class counts has no official support.
