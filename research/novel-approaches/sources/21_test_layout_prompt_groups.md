# 21_test_layout_prompt_groups
- title: Prompt Group Sizes (755733); Does the test set keep the "one passage per model per prompt" structure? (751802); Clarification on test set structure (750768)
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/755733
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/751802
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/750768
- type: discussion
- author(s) / date: ナハト・ ファウスト 2026-11-04; Noussair Mighri 2026-10-22; Sujal Bist 2026-10-19; commenters John K, Ace
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 3/5 (participant LB reports; no host answer on layout) | Recency 5/5 | Bias: med
## Verbatim quotes
> "On train out-of-fold predictions this gave a large boost: **0.374 → 0.481 BA**." (Noussair Mighri, 751802)
> "On the real test it didn’t help at all: **LB 0.350 vs 0.353** without matching." (Noussair Mighri, 751802)
> "Public and private do not promise a full 18-model group, and model counts are unequal, so anything that forces one of each label per prompt on test is fitting a structure the test set does not have." (John K, 755733)
> "So “at most once” is likely true; “exactly once” probably isn’t." (Ace, 750768)
## What it supports
- SOURCE CLAIM: clustering plus Hungarian gains about +0.1 on train-layout OOF but about 0 on the public LB.
- UNKNOWN (no host answer captured): the number of answers per test prompt, whether "at most once per model per prompt" holds, and whether the public/private split is by prompt or by passage.
