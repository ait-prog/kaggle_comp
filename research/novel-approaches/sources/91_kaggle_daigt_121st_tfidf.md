# 91_kaggle_daigt_121st_tfidf
- title: Lizhecheng02/Kaggle-LLM-Detect_AI_Generated_Text: 121st place, TF-IDF solution (README)
- url: https://raw.githubusercontent.com/Lizhecheng02/Kaggle-LLM-Detect_AI_Generated_Text/main/README.md (repo: https://github.com/Lizhecheng02/Kaggle-LLM-Detect_AI_Generated_Text)
- code: https://github.com/Lizhecheng02/Kaggle-LLM-Detect_AI_Generated_Text
- type: code
- author(s) / venue / date: Lizhecheng02; GitHub README; competition 2023-24
- accessed: 2026-11-12
- scores: Credibility 2/5 | Recency 3/5 | Bias: med (single author write-up, mid-table result)
## Verbatim quotes
> "[121st Solution] You Can Achieve PB 0.929 Only Use TF-IDF"
> "Although it's another drop on the private leaderboard, fortunately, I managed to hold on to the silver medal."
> "Therefore, we can discover that giving ``CatBoost`` a higher weight can lead to better performance on the PB."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: char/word TF-IDF n-grams (3,5) with MultinomialNB, SGD, LGBM, CatBoost, RandomForest, MLP; tables show public LB 0.970 vs private 0.907-0.929, i.e. large public-private shake-up; more robust CatBoost weighted higher. SOURCE CLAIM: GPT-3.5 rewriting of human text for extra data "could not get a good result". README also mentions training a new tokenizer with tree models (search snippet; not confirmed on fetched page).
- regime: Sparse n-gram features plus classical classifiers; cheap CPU.
- generalization / failure evidence: Shake-up evidence: public 0.970 -> private 0.907 for the chosen submission; models tuned on public LB overfit the data distribution.
- cost on 2×T4 (INFERENCE if estimated): CPU only, minutes.
- relevance to Name That LLM: TF-IDF char n-grams are a cheap strong baseline and give diversity in an ensemble; warns against over-trusting public LB when generator mix differs.
