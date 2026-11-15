# 20_host_rulings_data_and_test_time
- title: Open rules questions in one place: generated data (§2.6), Benchmarks quota, winner licence, test-time inference (host replies)
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/752084
- type: primary (host/staff replies by MartynaPlomecka; thread author martin osmani)
- author(s) / date: MartynaPlomecka (Kaggle staff), 2026-10-28, 2026-10-29, 2026-11-03
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 5/5 (host) | Recency 5/5 | Bias: low
## Verbatim quotes
> "Effectively "buying" more training data via token spend beyond the value of a possible prize ($5k) would be considered unreasonable." (MartynaPlomecka, 752084)
> "Self-generated training data (within reasonable cost, as noted earlier) does not need to be made public, so there is no publication deadline for it." (MartynaPlomecka, 752084)
> "The Benchmarks proxy quota is reserved for benchmark evaluation runs, not for generating competition training data" (MartynaPlomecka, 752084)
> "Feel free to explore any relationships in the data that you find useful." (MartynaPlomecka, reply to the test-time clustering/pseudo-labelling question, 752084)
## What it supports
- HOST-FACT: team-generated data via provider APIs is allowed up to roughly $5k of spend and need not be published. The Benchmarks proxy is not to be used for training data. Existing public community datasets and public adapters are fine.
- HOST-FACT: test-time use of relationships between hidden test passages (clustering, pseudo-labelling) is permitted.
- INFERENCE: the top 0.70–0.77 teams may be using private self-generated data, which the rules allow. Prompt- and config-matched own data is a legitimate lever.
