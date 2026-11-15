# Name That LLM: verified competition facts, public solution families and their failure assumptions (as of 2026-11-12)

Labels used below: **OFFICIAL** = Kaggle competition pages, competition metadata or a host/admin (Kaggle staff) forum reply. **PUBLIC-CLAIM** = a participant's notebook or discussion post (unverified by me; numbers are the author's). **INFERRED** = my own reasoning from the cited evidence. **LOCAL** = direct read-only inspection of the competition files already downloaded in this repo (`data/name-that-llm.zip`, dated 2026-10-14).

How the evidence was obtained, and its limits:
- The Kaggle HTML pages are rendered by JavaScript, so WebFetch returned only the page title. The Chrome extension was not connected. The Kaggle CLI (v2.2.4) returned "Authentication required" for every call, so `kaggle competitions files`, `kaggle kernels list` and `kaggle kernels pull` could not be used. I did not open or repair credentials.
- Instead, the official page text, competition metadata, public leaderboard, notebook list and discussion threads were read from Kaggle's own unauthenticated public web endpoints (`https://www.kaggle.com/api/i/...`, the same calls the website makes) on 2026-11-12 around 10:00–11:00 UTC. Twenty public notebook sources were downloaded through the public "download code" link into `/private/tmp/claude-501/-Users-i2gunshaker-orca-workspaces-kaggle-comp-name-that-llm/715c8dc3-c85b-4607-80fd-cd053d781a3e/scratchpad/kaggle_public/notebooks/`. They were treated as untrusted and only read, never run. Raw JSON for 47 discussion threads is in `.../scratchpad/kaggle_public/topics/` (flattened copies are in `topics_md/`).
- I did not join, accept rules, submit, push or change anything on Kaggle.

## 1. What are the exact metric, label set, submission format, and hidden-rerun compute constraints?

### Takeaway
This is a notebook-only code competition scored on **balanced accuracy over 18 fixed model labels**. The submission notebook writes `submission.parquet` (`id`, `model`) and is rerun on about 4,000 hidden passages, with internet off and a 9-hour limit on CPU or GPU. The public leaderboard uses 40% of the test set, and per-class counts are deliberately unequal and differ between the public and private parts. No official model-size cap exists beyond the runtime. The officially listed accelerator options are only "CPU" and "GPU", and the pages do not state whether TPU submissions are allowed.

### Cited Findings
- OFFICIAL: The metric is balanced accuracy, "the average, over all candidate models" of per-class recall, with K = 18. The sklearn implementation is named in the metric metadata (`evaluationAlgorithm` = "Balanced Accuracy Score", id 501). Scores are truncated to 3 decimals. — [Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation); competition metadata from the public competition endpoint (`scoreTruncationNumDecimals: 3`)
- OFFICIAL: "The number of test passages per model is **not** equal, and the mix of models differs between the public and private leaderboard." — [Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation)
- OFFICIAL: The 18 valid labels by provider are listed below. A submission is rejected if it lacks an `id` or `model` column, contains nulls, repeats an `id`, or predicts a label outside the 18. — [Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation)
  - OpenAI: `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`
  - Google: `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`
  - Anthropic: `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`
  - Qwen: `qwen3.8-max`
  - ZAI: `glm-5.3`, `glm-5.3-flash`
  - Moonshot AI: `kimi-k3`
- OFFICIAL: It is a Code Competition. The notebook reads `/kaggle/input/competitions/name-that-llm/test.csv`, writes `submission.parquet` to the working directory, and on submission "is re-run against the full hidden test set (~4000 passages)." — [Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation)
- OFFICIAL: The metadata shows `onlyAllowKernelSubmissions: true`, `usesSynchronousReruns: true`, `maxCpuRuntimeMinutes: 540`, `maxGpuRuntimeMinutes: 540`, `requiredSubmissionFilename: submission.parquet`, `leaderboardPercentage: 40`, `maxDailySubmissions: 5` and `maxTeamSize: 5`. Up to two final submissions may be selected. — competition metadata (public competition endpoint); [Rules §2.2](https://www.kaggle.com/competitions/name-that-llm/rules)
- OFFICIAL: The Code Requirements are: CPU notebook ≤ 9 h, GPU notebook ≤ 9 h, internet access disabled, "Freely & publicly available external data is allowed, including pre-trained models", and the output must be named `submission.parquet`. — [Code Requirements](https://www.kaggle.com/competitions/name-that-llm/overview/code-requirements)
- OFFICIAL: Timeline: the start date was 2026-10-14, and the entry and team-merger deadlines are 2027-01-06 23:59 UTC (from metadata; the Timeline page shows these only as unrendered template variables). The Timeline page gives the final submission deadline as **January 13, 2027** at 23:59 UTC, but the metadata `deadline` field reads `2027-01-14T23:59:00Z`. These conflict, so treat Jan 13 as binding until clarified. — [Timeline](https://www.kaggle.com/competitions/name-that-llm/overview/timeline); competition metadata
- OFFICIAL: The prize pool is $50,000, paid as $5,000 to each of places 1–10. The sponsor is Google LLC and the host is Kaggle. Identity verification is required. — [Prizes](https://www.kaggle.com/competitions/name-that-llm/overview/prizes); [Rules §1](https://www.kaggle.com/competitions/name-that-llm/rules)
- OFFICIAL (minor inconsistency): The Data page says test.csv has "three example rows during interactive sessions". The Evaluation page says "a single example row". LOCAL: the downloaded `test.csv` has 3 rows (`id`, `text`), and `sample_submission.parquet` has 3 rows (`id`, `model`, string dtype). — [Data](https://www.kaggle.com/competitions/name-that-llm/data); [Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation)
- OFFICIAL (admin reply, 2026-11-03): asked whether a notebook may use relationships between hidden-test passages (clustering, pseudo-labelling), the admin said "Feel free to explore any relationships in the data that you find useful." Transductive inference at test time is therefore allowed. — [Discussion 752084](https://www.kaggle.com/competitions/name-that-llm/discussion/752084)
- PUBLIC-CLAIM: Public top notebooks run inference on "T4 ×2 · internet off". One author says the code "budgets itself to 8 hours", and another reports a full scoring rerun of about 8 h. — [prvsiyan notebook](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author); [zbigniewkirchner 0.664](https://www.kaggle.com/code/zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public)
- PUBLIC-CLAIM: Weight loading from `/kaggle/input` can be very slow on some GPU sessions. Qwen3.5-9B took 1,206 s to load against about 22 s in the author's log, and forks timed out. Suggested mitigations are copying weights to local disk, uploading as a Kaggle "Model" rather than a "Dataset", and time-guarding optional members. — [Discussion 751268](https://www.kaggle.com/competitions/name-that-llm/discussion/751268); thread 750924 (title: "Extremely slow /kaggle/input reads on some GPU sessions")
- PUBLIC-CLAIM: The free TPU v5e-8 (about 20 h/week) trains a Qwen3-8B LoRA classifier about 14× faster than 2×T4, and bf16 removes fp16 overflow. Queue waits can reach about 23 h, and "Inference stays on the GPU notebook." — [Discussion 754657](https://www.kaggle.com/competitions/name-that-llm/discussion/754657)
- PUBLIC-CLAIM: 2×T4 sessions have about 30 GB RAM. A Qwen3-14B in fp16 fits across both T4s, and about 4,000 rows take 17–20 min at about 0.25 s/passage. — [Discussion 757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199); [Discussion 750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390)

### Inferences
- INFERRED: Because the classes are reweighted, a passage of a rare test class counts for more than a passage of a common one. Mistakes on classes with few hidden passages are therefore costly, and the weighting differs between public and private. This makes the metric sensitive to both the class mix and any prior correction (see Q6).
- INFERRED: The binding constraint is runtime and memory: about 4,000 passages, 9 h, 2×T4 fp16 with no bf16, and finite RAM and disk. There is no official parameter cap. Models up to about 14B fp16, or about 27B quantized, are being run in public notebooks.
- INFERRED: With about 4,000 hidden rows and 40% public, the public LB has about 1,600 passages, roughly 90 per class on average with unequal counts. The private part has about 2,400 passages.

### Gaps
- No official statement on whether a TPU notebook can be a scoring submission. Code Requirements list only CPU and GPU, and public evidence shows TPU used for training only.
- No official RAM, disk, or GPU-type guarantee for the rerun beyond "GPU Notebook ≤ 9 hours". The 2×T4 setting is inferred from public notebooks.
- The Jan 13 vs Jan 14 final-deadline conflict is unresolved.

## 2. How was the data built, and does the hidden test differ from train?

### Takeaway
The hosts disclose little about how the data was generated. Officially: 18 models from six labs, prompts covering philosophy, physics, mathematics, biology, computer science and creative prose, passages of 40–250 words, only light formatting cleanup, and hidden-test prompts disjoint from train prompts. Train is a near-complete 150-prompt × 18-model grid. The hidden test has unequal per-model counts, and participants infer from leaderboard behaviour (not from any official statement) that it has few answers per prompt. Hosts have not answered questions on system prompt, temperature, reasoning setting, API route, collection dates, or how the public/private split was drawn.

### Cited Findings
- OFFICIAL: Passages are "self-contained paragraphs generated by 18 language models"; "Every passage is an authentic LLM completion. Only light formatting cleanup was applied" (for example, removing a whole-passage `**bold**` wrapper). — [Data](https://www.kaggle.com/competitions/name-that-llm/data)
- OFFICIAL: There are about 4,000 test passages across 18 models from OpenAI, Google, Anthropic, Qwen, ZAI and Moonshot AI. Prompts cover "philosophy, physics, mathematics, biology, computer science, and creative prose." Passage lengths "vary randomly between 40 and 250 words". — [Overview/Description](https://www.kaggle.com/competitions/name-that-llm/overview)
- OFFICIAL: train.csv has "2,699 passages written for 150 prompts" with columns `id, prompt_id, text, model`. test.csv has only `id, text`. "Test passages are written for prompts that do not appear in train.csv." — [Data](https://www.kaggle.com/competitions/name-that-llm/data)
- LOCAL: 150 prompts. 149 have 18 passages, one per model, and one has 17. Every model has 150 rows except `claude-fable-5` with 149. There are no duplicate texts. Text length is 230–1,900 characters (median 670). Per-model median length ranges from 583 characters (`gemini-3.8-flash`) to 747.5 (`gpt-5.6-sol`). — local `data/train.csv`
- PUBLIC-CLAIM: In train, every prompt is answered by every model exactly once (one missing `claude-fable-5`). — [Discussion 750706](https://www.kaggle.com/competitions/name-that-llm/discussion/750706); [Discussion 751802](https://www.kaggle.com/competitions/name-that-llm/discussion/751802)
- PUBLIC-CLAIM: Train prompts were reconstructed by an LLM from each prompt's 17–18 answers. These are guesses, not ground truth, with average self-rated confidence 0.81. They look mostly like pop-science "explain X vividly" prompts (85 explain, 45 describe/narrate, 10 argue, 8 story, 1 compare) on specific named topics. — [Discussion 749455](https://www.kaggle.com/competitions/name-that-llm/discussion/749455)
- PUBLIC-CLAIM: 2,677 of 2,699 train passages end with a period, so short passages are complete short answers, not truncations. Augmenting training with random or sentence truncations lowered grouped CV (0.343 → 0.335 / 0.331). — [Discussion 750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390)
- PUBLIC-CLAIM: Participants asked whether hidden prompts keep the 18-answer structure, whether the public/private split is by prompt or by passage, and whether train and test used the same request format, length targets, model snapshots, reasoning level, temperature and system prompt. No host answer appears in these threads. — [Discussion 750213](https://www.kaggle.com/competitions/name-that-llm/discussion/750213); [750768](https://www.kaggle.com/competitions/name-that-llm/discussion/750768); [752280](https://www.kaggle.com/competitions/name-that-llm/discussion/752280); [750408](https://www.kaggle.com/competitions/name-that-llm/discussion/750408)
- PUBLIC-CLAIM: Google's API reportedly began routing `gemini-3.7-flash` requests to 3.8 after 2026-10-08. One participant saw `modelVersion: gemini-3.8-flash` returned for 3.7 calls. Whether the host's 3.7 labels predate this was asked but not answered. A grouped-OOF confusion analysis found no sign of 3.7→3.8 label contamination in host train, and the errors run in both directions. — [Discussion 751370](https://www.kaggle.com/competitions/name-that-llm/discussion/751370); [751871](https://www.kaggle.com/competitions/name-that-llm/discussion/751871); [751954](https://www.kaggle.com/competitions/name-that-llm/discussion/751954)
- PUBLIC-CLAIM: Under a fixed TF-IDF yardstick trained on host train, real `kimi-k3` most resembles community kimi passages generated with reasoning "high": 25% recognised against 8–11% for default/off settings, where real kimi passages are recognised 31% of the time. This suggests the host used reasoning-on for kimi. A different participant found that kimi without reasoning matches train's em-dash rate better. These two claims conflict. — [Discussion 752766](https://www.kaggle.com/competitions/name-that-llm/discussion/752766); [Discussion 751871](https://www.kaggle.com/competitions/name-that-llm/discussion/751871)
- PUBLIC-CLAIM: On OpenRouter, several labels (`glm-5.3`, `glm-5.3-flash`, `gemini-3.7-flash`, `qwen3.8-max`) reportedly refuse `reasoning.enabled=false`. Generation protocols therefore cannot always be matched to whatever the host used. — [Discussion 751871](https://www.kaggle.com/competitions/name-that-llm/discussion/751871)

### Inferences
- INFERRED: The known train/test differences are: (a) prompts are disjoint, which is official; (b) per-model counts are unequal and the mix shifts between public and private, which is official; (c) the number of answers per prompt is probably much lower in test. Item (c) is a participant inference from leaderboard behaviour (Q5/Q6), not an official fact. Generation settings are not officially stated to be the same for train and test, but nothing suggests they differ.
- INFERRED: The 40–250 word statement plus complete-sentence endings points to an explicit length instruction in the host's prompt template. If so, a model's length behaviour under that instruction is itself a label signal. Per-model median lengths differ by about 28% in train, which fits this, but the Overview warns against relying on "simple character-count shortcuts".

### Gaps
- Not public: system prompt, temperature, top-p, reasoning or effort settings, API route (direct vs proxy), collection dates, model snapshot checks, and the exact cleanup procedure.
- Not public: the hidden-test passages-per-prompt distribution, whether the public/private split is by prompt or by passage, and the per-class counts.

## 3. Which external data and pretrained models are allowed (including community "open pool" data)?

### Takeaway
External data and pretrained models are allowed if publicly available and equally accessible at no cost, or if they meet a host-applied "Reasonableness" standard. Hosts clarified four points: public community datasets on Kaggle, including the merged open pools, are fine to train on; public weights such as LoRA adapters on Kaggle are fine; self-generated data within reasonable cost need not be published; and spending more than about $5k (one prize) on generation is unreasonable. Using the Kaggle Benchmarks proxy quota to generate training data is not intended, but past use is not penalised.

### Cited Findings
- OFFICIAL: Rules §2.6.a allows External Data that is "publicly available and equally accessible to use by all Participants … at no cost", or that satisfies the Reasonableness criteria of §2.6.b. Under §2.6.b, use must be "reasonably accessible to all" and of "minimal cost". A proprietary dataset license costing more than a prize is given as unreasonable, and a small subscription is given as acceptable. — [Rules §2.6](https://www.kaggle.com/competitions/name-that-llm/rules)
- OFFICIAL: The Code Requirements state "Freely & publicly available external data is allowed, including pre-trained models". — [Code Requirements](https://www.kaggle.com/competitions/name-that-llm/overview/code-requirements)
- OFFICIAL (admin reply, 2026-10-28): "This is intended to be an ML prediction competition, not a data generation competition." Generation spend "beyond the value of a possible prize ($5k) would be considered unreasonable." "Existing public community datasets on Kaggle are fine to train on." — [Discussion 752084](https://www.kaggle.com/competitions/name-that-llm/discussion/752084)
- OFFICIAL (admin reply, 2026-10-29): Public community datasets on Kaggle, including those generated via the Benchmarks proxy, are fine to train on. Self-generated data (within reasonable cost) "does not need to be made public". "model weights (such as LoRA adapters) that are publicly available to all participants on Kaggle are fine to use." The Benchmarks proxy quota is not intended for non-benchmark work, but earlier use is not penalised. A 2026-11-03 reply confirmed that publishing proxy data generated before the clarification is OK. — [Discussion 752084](https://www.kaggle.com/competitions/name-that-llm/discussion/752084)
- OFFICIAL (admin reply, 2026-11-03): AI coding assistants may be used ("use whatever workflow you prefer"). — [Discussion 752084](https://www.kaggle.com/competitions/name-that-llm/discussion/752084)
- OFFICIAL: The winner license is CC-BY 4.0 in the header, with an OSI-approved open-source license for code. Winners must deliver training and inference code plus environment documentation. Competition data is CC BY-SA 4.0. — [Rules §1, §2.5, §2.8](https://www.kaggle.com/competitions/name-that-llm/rules)
- PUBLIC-CLAIM: The community "Open Pool" by crtproduction (v14) has 223,840 passages from 61 source datasets, with per-row generation metadata (`gen_reasoning`, `gen_route`, `gen_served_model`, `gen_date`) and a `flag_kb_proxy_after_ruling` flag on 17,876 rows. Its curator removed 284 rows written by models outside the 18 labels, such as `gemini-3.5-flash-lite` labelled as `gemini-3.6-flash`. — [Discussion 752766](https://www.kaggle.com/competitions/name-that-llm/discussion/752766); [dataset](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool)
- PUBLIC-CLAIM: Other free community sets include prvsiyan's Extended dataset (v8: 13,500 passages, 810 prompts, all 18 labels, no system prompt, requested and served model id logged, 5,070 in complete clean blocks), an "18-model wide" corpus (135k responses, 1,902 prompts answered by all 18 models, mixed provenance and licenses), seyitkaangunes' about 5k Claude-only passages, and a 9,719-row OpenAI set. — [prvsiyan notebook](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author); [Discussion 749629](https://www.kaggle.com/competitions/name-that-llm/discussion/749629); [750224](https://www.kaggle.com/competitions/name-that-llm/discussion/750224); [753974](https://www.kaggle.com/competitions/name-that-llm/discussion/753974)
- PUBLIC-CLAIM: The most-voted thread (146 votes, 70 comments) is a fairness debate about whether the winner will be whoever buys the most generated data. — [Discussion 749447](https://www.kaggle.com/competitions/name-that-llm/discussion/749447)

### Inferences
- INFERRED: The "open pool" style data is explicitly permitted (OFFICIAL admin reply). The real risk is quality and protocol fidelity (Q5, family J), not legality. Self-generated API data under $5k is also permitted and need not be published, so top private teams may hold unpublished, better-matched data that the public stack does not have.
- INFERRED: Public open-weight relatives of label models (for example Qwen-family or GLM open checkpoints) are allowed as pretrained models if they are freely available on Kaggle or elsewhere. The constraint is compute, not rules.

### Gaps
- Hosts did not answer whether provider terms of service restricting the use of outputs matter (thread 750512), or questions on geo-restricted models (752270).
- There is no official per-team dollar figure below the $5k remark, and no disclosure requirement beyond winner documentation.

## 4. What do the public leaderboard and its score distribution look like?

### Takeaway
On 2026-11-12 about 10:58 UTC, 5,607 teams were on the public LB. The top score is 0.768, rank 10 is 0.747, gold starts at about 0.733 (about 21 teams), and only 72 teams are at or above 0.700. A very large spike at 0.680–0.684 matches the scores of forkable public notebooks: 319 teams are at exactly 0.684 and 141 at 0.682. The visible plateau is therefore a public-notebook artifact, not evidence of a Bayes ceiling. Teams using private data or methods are 0.05–0.08 higher.

### Cited Findings
- OFFICIAL (public LB snapshot via the leaderboard endpoint): n = 5,607. Rank 1 = 0.768, 2 = 0.765, 3 = 0.756, 5 = 0.752, 10 = 0.747, 20 = 0.733, 50 = 0.708, 100 = 0.690, 200 = 0.684, 500 = 0.682, 1000 = 0.673, 2000 = 0.652, 3000 = 0.574. The median is 0.596. 116 teams are above 0.684, 72 at or above 0.70, 32 at or above 0.72 and 7 at or above 0.75. Medal thresholds as displayed are gold ≥ 0.733 (21 teams), silver ≥ 0.684 (259) and bronze ≥ 0.682 (280). — [Leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard)
- OFFICIAL: The most common exact scores are 0.684 (319 teams), 0.682 (141), 0.680 (128), 0.322 (122, the official starter score), 0.663 (116), 0.681 (108) and 0.662 (105). Teams scoring 0.682 occupy ranks 489–629. — [Leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard)
- OFFICIAL: Metadata reports 6,091 competitors, 14,489 joined users and 67,017 submissions. — competition metadata
- PUBLIC-CLAIM: The highest-scoring public notebooks are at 0.684 ("SOTA" ensembles that fork one another). The same 0.684 code is published by at least five accounts. — [alexchilton 0.684](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed); [prvsiyan](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author)
- PUBLIC-CLAIM: Self-reported best single models are about 0.628 public for a Qwen3-8B LoRA on train plus Open Pool, "~0.670 Qwen3-14B" with some private data, 0.649 for Qwen3.5-9B, and 0.575 LB (0.616 CV) for ModernBERT. The author of the 0.628 result says nearly all the gain came from data, not the model. — [Discussion 754860](https://www.kaggle.com/competitions/name-that-llm/discussion/754860)
- PUBLIC-CLAIM: Baselines are about 0.35 for TF-IDF + LR on train only (grouped CV 0.34–0.37), 0.42 for an n-gram SVM with pool data, 0.466 for Qwen3-14B LoRA on train only, and 0.592 for two-stage LoRA (pool, then train) plus n-gram. — [Discussion 750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390); [nishantdahal](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool)

### Inferences
- INFERRED: The repo's own dev1 score (0.682, per README) sits inside the public-notebook cluster, at about rank 489–629. dev1's attached inputs (LOCAL: `k/dev1/kernel-metadata.json`) overlap heavily with the public 0.68x stack's inputs (for example `nlm-l24-adapters-v5`, `ntl-q14-lora-v4`, `ultrabert-380m-ntl`, Qwen3 models). That score therefore says little about any ceiling.
- INFERRED: A gap of about 0.08 between the public-notebook plateau (0.684) and rank 1 (0.768) on the same hidden test suggests that ingredients missing from public code give substantial headroom: data, protocol match, or different modelling. It does not say which ingredient matters. Public LB noise (Q6) is about 0.01, far smaller than 0.08.

### Gaps
- No top-team writeups exist yet (the competition is still running), so the methods behind 0.70–0.77 are unknown.

## 5. Which public approach families exist, and what assumption makes each one hit a ceiling?

### Takeaway
The public lineage converged on one forkable mega-ensemble with these parts:
- multi-depth LoRA readouts on truncated Qwen3-8B/14B and Llama-3.1-8B, trained in two stages (community pool, then host train)
- a 5-fold Qwen3-Embedding-8B LoRA expert
- n-gram SVC/LR on pool data
- encoders
- ATLAS source-conditioned likelihood on low-margin rows
- test-time prompt clustering with Gaussian "shared-prompt" heads and centring
- typography priors and family gates
- soft Sinkhorn class balancing

Public evidence suggests the biggest early lever was data volume and cleanliness, with diminishing returns after that. Prompt-grouping tricks that assume train's 18-per-prompt layout repeatedly lost on the LB. Ensemble growth is now within noise. Grouped-CV on train has stopped predicting LB differences for pool-trained or stacked members.

### Cited Findings (family by family; D = data limit, V = validation mismatch, C = compute bottleneck, R = model-representation limit)

**A. Char/word TF-IDF and stylometry with linear models (LR, LinearSVC, LightGBM on typography features)**
- PUBLIC-CLAIM: On train only, grouped CV is 0.34–0.37 and LB 0.353–0.361. 55–58% of errors fall within the correct lab, and lab-level accuracy is about 0.58–0.61. Adding typography and rhythm features raised overall accuracy but raised the within-lab error share. — [Discussion 750664](https://www.kaggle.com/competitions/name-that-llm/discussion/750664); [749593](https://www.kaggle.com/competitions/name-that-llm/discussion/749593); [756477](https://www.kaggle.com/competitions/name-that-llm/discussion/756477)
- PUBLIC-CLAIM: Short passages are hardest: ≤60 words gives 0.27 and >160 words gives 0.46. — [Discussion 750664](https://www.kaggle.com/competitions/name-that-llm/discussion/750664); [750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390)
- PUBLIC-CLAIM: Typography rules that are absolute in train (no curly quotes in Claude/GLM, no em-dash in Gemini) do not hold on freshly generated proxy text. A hard punctuation veto gave +0.001 and lost 0.003–0.005 in forks, so it was removed. A soft version (−0.70 log-score) is in a 0.684 fork, unscored. — [lucasboesen notebook](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering); [kirilltym](https://www.kaggle.com/code/kirilltym/lora-trained27-evidence-for-close-calls)
- PUBLIC-CLAIM: In a strong blend, the n-gram SVC (OOF 0.473) was worth −0.011 when removed. Weak but differently trained members carry more LB weight than their OOF suggests. — [Discussion 753177](https://www.kaggle.com/competitions/name-that-llm/discussion/753177)
- Key assumption and failures:
  - Surface statistics transfer across prompts and generation settings.
  - Fails on siblings within a lab (R).
  - Fails on short passages, where little evidence is available (D/R).
  - Rules mined from 150 prompts overfit, because the train prompts are not representative of test prompts (V).

**B. Fine-tuned encoders (DeBERTa-v3, ModernBERT, UltraBERT-380M, ELECTRA, desklib)**
- PUBLIC-CLAIM: DeBERTa-v3-base fine-tuned on train only did not beat char/word TF-IDF. — [Discussion 751871](https://www.kaggle.com/competitions/name-that-llm/discussion/751871)
- PUBLIC-CLAIM: ModernBERT pretrained on the pool, then given LoRA on train, reached CV 0.616 and LB 0.575. Adding ModernBERT at 10% or 20% to the 0.683 stack gave 0.672 and 0.669. The author's offline stand-in had predicted +0.003. Explanations given are prompt-overlap memorisation in its pool, a deployed member different from the checked one, and double counting. — [Discussion 754860](https://www.kaggle.com/competitions/name-that-llm/discussion/754860); [prvsiyan notebook](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author)
- PUBLIC-CLAIM: Encoders added to a 0.636 base were flat or negative because the same encoders were already in its complementary stack. — [Discussion 752357](https://www.kaggle.com/competitions/name-that-llm/discussion/752357)
- Key assumption and failures:
  - Small encoders capture the generator fingerprint.
  - Capacity is below that of LLM readouts (R).
  - Members are correlated with the ensemble's existing encoders (V).
  - Members trained on pool prompts near train prompts give inflated train-side checks (V).

**C. Frozen decoder-LLM features with shallow heads (multi-layer mean pooling, LR, Gaussian heads)**
- PUBLIC-CLAIM: On frozen Qwen3-4B, early layers beat late ones: layer 3 scores 0.391, the last layer 0.311, and six layers concatenated 0.414. Blended with TF-IDF it reached CV 0.439 and LB 0.420. — [Discussion 750462](https://www.kaggle.com/competitions/name-that-llm/discussion/750462)
- PUBLIC-CLAIM: On Qwen3.5-9B, frozen mid-layers plus TF-IDF gave CV 0.447 and LB 0.427. Layer 18 frozen scores about 0.42 against about 0.34 for layer 32. — [Discussion 751871](https://www.kaggle.com/competitions/name-that-llm/discussion/751871)
- Key assumption and failures:
  - A pretrained LM's intermediate states linearly encode generator style.
  - The plateau is about 0.41–0.45 without adaptation (R).
  - Topic dominates deep layers (R/V).

**D. LoRA classifiers on truncated decoder LLMs, trained in two stages (community pool, then host train)**
- PUBLIC-CLAIM: The recipe is Qwen3-8B (first 24 layers) or Qwen3-14B (first 27 layers), LoRA r16 on all projections, mean pooling at 8 depths, a linear head, final norm disabled, and max_len 384. Stage 1 on about 35k pool passages scores 0.483 grouped CV. After stage 2 the 8B and 14B each score 0.569. They disagree on 22% of passages, and the blend with n-gram plus centred twin plus balancing gave 0.593 CV and 0.592 LB. — [nishantdahal notebook](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool)
- PUBLIC-CLAIM: Retraining a Qwen2.5-14B member on the 120k pool raised OOF from 0.569 to 0.605, but the LB fell from 0.651 to 0.647. Keeping the old member and adding the new one at 0.4 gave 0.650. Member OOF is "a poor guide to LB value", and diversity of data mix matters. — [Discussion 753177](https://www.kaggle.com/competitions/name-that-llm/discussion/753177)
- PUBLIC-CLAIM: fp16 on T4 overflowed as the adapter grew. More than half of batches were skipped and OOF fell by 0.10, which bf16 on TPU fixed. — [Discussion 754657](https://www.kaggle.com/competitions/name-that-llm/discussion/754657)
- PUBLIC-CLAIM: The current 0.684 core stack has v5 multi-depth LoRA readouts on Qwen3-14B, Qwen3-8B and Llama-3.1-8B, a two-adapter Qwen3-14B blend, Qwen2.5-14B LoRA and a DeBERTa-v3 encoder. — [alexchilton 0.684](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed)
- Key assumptions and failures:
  - Assumption: community pool text is close enough to the host's generation protocol that stage 1 learns transferable fingerprints, and 2,699 rows suffice to recalibrate. Failures:
    - Pool protocol mismatch, for example in reasoning settings, chat vs essay genre, and length instruction (D).
    - Mislabelled or rerouted pool rows (D).
    - Grouped CV overstates pool-trained gains (V).
    - T4 fp16 numerics and training throughput (C).
  - Assumption: mean-pooled hidden states of a single other-family LM carry the fingerprint (R). Family-level separation is easy. Siblings within a lab (Gemini 3.7 vs 3.8, GLM vs Claude) stay hard.

**E. Native Qwen3-Embedding-8B LoRA expert (5 prompt folds, all 36 layers, normalisation at 8 depths, plain and prompt-centred heads)**
- PUBLIC-CLAIM: Adding this expert at a 40% geometric share took the public stack from about 0.668–0.673 to 0.680. A 50% share scored 0.681 (0.679 in a repeat), and 30% scored 0.674. — [lightningv08 0.680](https://www.kaggle.com/code/lightningv08/0-680-modernbert-lora-sparse-ensemble); [alexchilton 0.684](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed)
- Key assumption and failures:
  - A different pretraining objective (an embedding model) gives errors partly decorrelated from causal-LM readouts.
  - The share is tuned on the public LB (V).
  - Centring uses lexical prompt groups whose purity in test is unknown (V).

**F. Likelihood and source-conditioned methods (ATLAS; proxy-LM log-likelihood features; real open-weight label models)**
- PUBLIC-CLAIM (paper): ATLAS (Ko et al., arXiv 2610.19583, 2026-10-14) treats attribution as "source-conditioned text modeling" with clipped token-level evidence. It reports 74.27% on MAABench under combined domain and version shift. — [arXiv 2610.19583](https://arxiv.org/abs/2610.19583)
- PUBLIC-CLAIM: A Qwen3-8B-Base ATLAS adapter trained 5 epochs on the 2,699 official rows, mixed at 10% on low-margin rows, gave +0.002 to +0.007 across six public parents. A 20% mix gave +0.003 (0.660 → 0.663). — [josephadamski91 notebook](https://www.kaggle.com/code/josephadamski91/teach-the-model-10-atlas); [omerfarukmerey 0.668](https://www.kaggle.com/code/omerfarukmerey/0-668-verifier-matched-rows-pool-prompts)
- PUBLIC-CLAIM: GLM-4-9B-0414 plus Qwen3-8B-Base log-likelihood features gave −0.002 in a prompt-grouped stack. A participant asks about running the real GLM-5.3 / GLM-5.3-Flash from public IQ3_XXS GGUF segments. Replies say per-passage mmap will not fit in 9 h, and layer-streaming over the whole test set with 128–256 tokens is suggested. — [Discussion 757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199)
- Key assumptions and failures:
  - Assumption: a proxy LM's (conditional) likelihood carries generator identity. Failures:
    - The proxies are not the generators. Most labels are closed models (R).
    - ATLAS here is trained only on 2,699 rows (D).
    - It is applied only to a time-budgeted subset of rows (C).
  - Assumption: open-weight label-family models can be run. Failure: model size against 9 h on 2×T4 (C).

**G. Large quantized LLM classifiers (Qwen3.8-27B "GSQ3", Qwen3.5-27B NF4)**
- PUBLIC-CLAIM: A Qwen3.8-27B GSQ3 LoRA readout, trained on the 119,899-row public corpus and then 2 epochs on train, contributes only 3% of the parent mixture in the 0.684 stack. A 12% share variant scored 0.682. Swapping a q8v5 member for a Qwen3.5-27B NF4 member (0.642 alone) lowered nlm-v27 from 0.651 to 0.638. — [lightningv08 0.661](https://www.kaggle.com/code/lightningv08/0-661-qwen3-8-27b-lora-spectral-ensemble); [alexchilton 0.684](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed)
- Key assumption and failures:
  - Bigger backbones extract more fingerprint.
  - Quantization and fp16 on T4 add noise (C).
  - Load time and runtime limit coverage (C).
  - Being trained on the same pool makes them redundant with existing members (D/V).

**H. Test-time structure exploitation (prompt clustering, Hungarian one-per-model assignment, centring, "centred twin" heads, Gaussian shared-prompt heads, Sinkhorn balancing)**
- PUBLIC-CLAIM: Clustering plus Hungarian gave about +0.10 in train-layout CV but lost on the LB: 0.353 → 0.344, and 0.329 with centring added. Other reports: 0.427 → 0.425, +0.002 LB, and −0.009 to −0.017 across several parents. — [Discussion 750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390); [751871](https://www.kaggle.com/competitions/name-that-llm/discussion/751871); [757855](https://www.kaggle.com/competitions/name-that-llm/discussion/757855); [alexchilton 0.684](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed)
- PUBLIC-CLAIM: A known-truth simulation found that centring small groups removes the passage's own author signal. With one or two answers per prompt, centring loses up to −0.10 and Hungarian gives about 0. A "cavity" estimate (prompt effect from the other group members only) never lost to plain decoding when groups were correct. — [Discussion 750213](https://www.kaggle.com/competitions/name-that-llm/discussion/750213)
- PUBLIC-CLAIM: Grouped CV over-selects centring because a fold of about 30 prompts is less crowded than the full test batch. Mixing validation passages into the unlabelled test batch before clustering gave the lowest worst-case regret in simulation (0.014 vs 0.037). The proposer had not yet run this inside a submission. — [Discussion 753053](https://www.kaggle.com/competitions/name-that-llm/discussion/753053)
- PUBLIC-CLAIM: Gaussian shared-prompt heads repeat about +0.005 each across the public lineage. — [Discussion 752357](https://www.kaggle.com/competitions/name-that-llm/discussion/752357)
- PUBLIC-CLAIM: Under Dirichlet-uneven test mixes, full Sinkhorn balancing lost up to 0.07 (0.636 → 0.570 at alpha = 1), while 1–3 iterations cost under 0.01. On the LB, full balancing was worth +0.002 on one blend, and the current public stack uses a softened balance (strength 0.30, target mix 0.35). — [Discussion 752365](https://www.kaggle.com/competitions/name-that-llm/discussion/752365); [alexchilton 0.684](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed)
- Key assumptions and failures:
  - The test keeps train's dense one-answer-per-model-per-prompt layout and a near-even class mix.
  - Both assumptions are contradicted officially for counts and mix, and by participants for density (V).
  - Gains depend on cluster purity in a crowded batch of about 4,000 rows (V).

**I. Hierarchical and family gates (lab first, then sibling; Claude/GPT/Gemini/GLM gates)**
- PUBLIC-CLAIM: A Claude gate, a 5-way within-Claude model trained on train Claude rows plus about 5k Claude passages where every prompt has all 5 Claude labels, gave clear gains. Adding Claude-only prompts into the 18-way head instead reduced Claude recall. A GPT gate needed matched extra data, and without it gave −0.0004 OOF. A lab → line → version hierarchy gave +2.4 points for a CPU stylegram. A Gemini gate and four intra-family gates were flat on the LB (0.552 → 0.552; 0.559 → 0.559). — [Discussion 750224](https://www.kaggle.com/competitions/name-that-llm/discussion/750224); [lucasboesen](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering); [alexchilton 0.684](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed)
- PUBLIC-CLAIM: GLM-5.3 is classified as Claude more often than as GLM (68 vs 34 of 150 in one 9B classifier). A train-only TF-IDF puts 40% of real glm-5.3 probability on Claude labels, and community GLM data shows the same (42%), so the resemblance is the model's own style, not a pool label error. — [Discussion 755314](https://www.kaggle.com/competitions/name-that-llm/discussion/755314); [752766](https://www.kaggle.com/competitions/name-that-llm/discussion/752766)
- PUBLIC-CLAIM: One competitor's OOF recall is 0.32 for glm-5.3, 0.44 for glm-5.3-flash and 0.48 for claude-fable-5, with confusion mostly among these three. — [Discussion 757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199)
- Key assumption and failures:
  - The lab is reliably identified, and sibling differences can be learned from prompt-matched within-family data.
  - Fails where the "lab" boundary itself is blurred, as with GLM vs Claude (R/D).
  - Fails where no prompt-matched sibling data exists (D).

**J. External-data generation, curation and pooling**
- PUBLIC-CLAIM: Across 134 scored public versions, a bigger or cleaner pool (35k → 84k) gave +0.028, then +0.003 and +0.002. — [Discussion 752357](https://www.kaggle.com/competitions/name-that-llm/discussion/752357)
- PUBLIC-CLAIM: Same-size data quality ablation, using Qwen3-1.7B LoRA with 45k rows and training only on pool data, scored on all of train:
  - Removing junk and off-genre passages (about 1 in 4 rows) gave +0.026 to +0.027, consistent across 2 seeds.
  - Recall rose most for the dirtiest labels: kimi-k3 by 16 points and claude-opus-4-8 by 13.
  - Keeping only the 30% most "train-like" passages gave no gain (−0.004).
  - Dropping rows by cross-source agreement hurt.
  — [Discussion 753779](https://www.kaggle.com/competitions/name-that-llm/discussion/753779); [752766](https://www.kaggle.com/competitions/name-that-llm/discussion/752766)
- PUBLIC-CLAIM: Coverage matters more than volume. Adding data for only 13 of 18 labels cost the 5 uncovered labels 6 recall points, because the model learns "new prompt ⇒ not GLM". — [Discussion 751871](https://www.kaggle.com/competitions/name-that-llm/discussion/751871)
- PUBLIC-CLAIM: Pool fidelity relative to real passages is lowest for kimi-k3 (46%), qwen3.8-max (51%) and claude-opus-4-8 (54%), and 79–102% for GPT labels. Seven of 12 pool sources have no gemini-3.7 rows, which biases models toward 3.8 unless classes are reweighted. — [Discussion 752766](https://www.kaggle.com/competitions/name-that-llm/discussion/752766); [751370](https://www.kaggle.com/competitions/name-that-llm/discussion/751370)
- PUBLIC-CLAIM: Some participants found their own extra data lowered validation, attributing this to a generation-setup mismatch. — [Discussion 755371](https://www.kaggle.com/competitions/name-that-llm/discussion/755371); [757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199)
- Key assumption and failures:
  - More labelled text from "the same model name" equals more of the host's distribution.
  - Fails through protocol mismatch in reasoning, system prompt, genre, length instruction and serving route (D).
  - Fails through label noise and rerouting (D).
  - Fails through partial label coverage creating shortcuts (D).
  - Returns diminish.
  - The rules cap spend at about $5k (D).

**K. Mega-ensembles and forks of the public stack (the 0.68x cluster)**
- PUBLIC-CLAIM: Adding more LoRA backbones or members to a strong ensemble moved the LB by at most 0.004 either way. One team's blend scored 0.6865 in nested CV but 0.56 on the LB, because the folds used true 18-passage prompt groups. — [alexchilton 0.684](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed)
- PUBLIC-CLAIM: For stack changes, "a gain measured on train passages is not evidence". The authors report they trust only public A/B results. — [prvsiyan notebook](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author)
- PUBLIC-CLAIM: Scoring OOF with only 2 answers per prompt tracked the board far better than full groups (correlation 0.91 vs 0.13 over 15 submissions). — [alexchilton 0.684](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed), citing a comment on [Discussion 753177](https://www.kaggle.com/competitions/name-that-llm/discussion/753177)
- PUBLIC-CLAIM: Pseudo-labelling the test set was negative in simulation. — [Discussion 757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199)
- Key assumption and failures:
  - Adding members adds independent evidence.
  - Members trained on the same pool are correlated (D/R).
  - Weights are tuned on a noisy 40% public LB (V).
  - The 9-hour budget and slow input mounts can cause timeouts (C).

### Inferences
- INFERRED: Failure assumptions by type:
  - Data limits (D) are the most cited ceiling. There are only 150 host prompts, and community data is generated under unknown and different protocols, partly mislabelled, unevenly covered, and full of off-genre text.
  - Validation mismatch (V) is the second. Train-layout grouped CV inflates test-time grouping tricks and pool-trained members, and the noisy 40% public LB is now the de facto selector.
  - Compute (C) mainly limits running larger or label-family open-weight models and full-coverage likelihood scoring within 9 h on T4.
  - Representation (R) limits show up as persistent within-lab confusions (Gemini 3.7/3.8, GPT-5.6 siblings, GLM↔Claude) that many backbones share. These may be partly irreducible, because GLM writes in a Claude-like style per the community analysis. No public evidence shows these confusions are irreducible.
- INFERRED: Under-explored in public code, judging from the notebooks and threads read:
  - contrastive or content-controlled training that uses the prompt-matched structure of train and of matched community panels, beyond centring
  - generation-protocol-aware data selection or conditioning (for example by reasoning setting)
  - validation that reproduces the test's sparse, crowded layout (the embedded-validation proposal in 753053 was untested)
  - likelihood under real open-weight relatives of label models, which only reached the question stage

### Gaps
- Public notebooks show methods only up to 0.684. Nothing public explains 0.70–0.77.
- Many claims are single public submissions inside the about 0.01 noise band (Q6). I could not verify any author's numbers.
- I read the markdown of 20 notebooks and keyword-scanned their code. I did not audit full code paths, for example how groups are formed at test time in each fork.

## 6. What do discussions say about data quirks, leakage, shake-up risk, and the public/private split?

### Takeaway
The public/private split is officially 40/60, with different class mixes. The split method (by prompt or by passage) is not disclosed. Participants measure about 0.01 of run-to-run or public-sampling noise even on identical code. Random-row CV is pessimistically biased ("reverse leakage"), grouped CV by `prompt_id` is required but overstates pool-trained and test-grouping gains, and balancing and centring choices carry real private-LB risk. A notable shake-up should be expected for teams whose public-LB gains come from tuning or test-layout exploitation.

### Cited Findings
- OFFICIAL: `leaderboardPercentage: 40`. The class mix differs between public and private. — competition metadata; [Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation)
- PUBLIC-CLAIM: Random K-fold is pessimistic, at 0.109–0.22 vs 0.347–0.369 grouped by `prompt_id`. The cause is that 17 same-prompt siblings with other labels sit in training ("reverse leakage"). Moving 2, 4 or 8 siblings back into training drops accuracy from 0.347 to 0.264, 0.215 and 0.156. Grouped CV tracked the LB for train-only models (0.343 → 0.353; 0.370 → 0.361). — [Discussion 756477](https://www.kaggle.com/competitions/name-that-llm/discussion/756477); [749593](https://www.kaggle.com/competitions/name-that-llm/discussion/749593); [750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390)
- PUBLIC-CLAIM: For pool-trained models, grouped CV reads 0.03–0.04 above the LB: SVM +0.097 CV vs +0.062 LB from pool data, and LoRA 0.576 CV vs 0.538 LB. Prompt leakage from the pool was checked and ruled out (no shared prompt ids or texts, and 0.3% of pool passages with content similarity ≥ 0.4 to any train prompt). The cause is unresolved. — [Discussion 750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390) (comment, 2026-10-23)
- PUBLIC-CLAIM: The prompt-cluster-robust standard error is about 0.009 for CV and about 0.014 for the public LB. Simulated CV − LB has a standard deviation of 0.016–0.023, and all submissions share the same public rows. — [Discussion 750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390)
- PUBLIC-CLAIM: Identical code resubmitted scored up to 0.009 apart (0.645 vs 0.654; 0.643 vs 0.651). Rescoring the same saved predictions gives the same score. The variance is attributed to time-budgeted components such as ATLAS covering a variable set of rows, so it is rerun variance, not metric noise. — [alexchilton 0.684](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed); [Discussion 753177](https://www.kaggle.com/competitions/name-that-llm/discussion/753177)
- PUBLIC-CLAIM: The hidden test "appears to have few per prompt". This is an inference from repeated LB losses of train-layout tricks. Hosts have not confirmed it, and one commenter notes the worst case could be all separate prompts. — [Discussion 755733](https://www.kaggle.com/competitions/name-that-llm/discussion/755733); [prvsiyan notebook](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author)
- PUBLIC-CLAIM: Data quirks raised in discussion:
  - Typography differs by lab: Claude and GLM never use curly apostrophes in train, and Gemini rarely uses em-dashes. — [Discussion 752780](https://www.kaggle.com/competitions/name-that-llm/discussion/752780); [750664](https://www.kaggle.com/competitions/name-that-llm/discussion/750664); [757855](https://www.kaggle.com/competitions/name-that-llm/discussion/757855)
  - GLM resembles Claude. — [Discussion 755314](https://www.kaggle.com/competitions/name-that-llm/discussion/755314)
  - Gemini 3.7 may have been routed to 3.8 by the API. — [Discussion 751370](https://www.kaggle.com/competitions/name-that-llm/discussion/751370)
  - Kimi's reasoning setting matters. — [Discussion 752766](https://www.kaggle.com/competitions/name-that-llm/discussion/752766)
  - Train has one 17-answer prompt. — [Discussion 755733](https://www.kaggle.com/competitions/name-that-llm/discussion/755733)
- PUBLIC-CLAIM: An unverified forum remark says the #1 poster said part of an earlier pre-external-data score "may have been probing". I could not locate the original statement. — [Discussion 752357](https://www.kaggle.com/competitions/name-that-llm/discussion/752357)

### Inferences
- INFERRED: The public LB has about 1,600 rows and about 90 per class on average. A balanced-accuracy SE of about 0.012–0.014 means differences under about 0.02 between teams are not reliable for the private ranking. That covers the whole 0.680–0.684 cluster and much of the gap between neighbouring ranks in the top 20.
- INFERRED: The highest shake-up risk falls on:
  - strong Sinkhorn or uniform-prior balancing, because the private class mix differs officially
  - aggressive prompt-group centring or Hungarian assignment, because the test layout is unknown and probably sparse
  - weights and shares tuned on many public submissions
  Protocol-matched data and representation gains should transfer more robustly than these, though that is untested.
- INFERRED: The "reverse leakage" structure also means any extra data whose prompts overlap or nearly overlap the train prompts inflates train-side validation, as prvsiyan's ModernBERT post-mortem suggests. External rows should be grouped by source prompt and excluded from the fold of matching train prompts.

### Gaps
- There is no official information on the split method (prompt vs passage), per-class hidden counts, or answers-per-prompt in test.
- No private-LB data exists yet. Any shake-up estimate is a model, not an observation.
- The Kaggle Discord (official per thread 749382, but unmonitored by staff) was not read.
