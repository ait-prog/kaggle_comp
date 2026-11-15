# Data fidelity, noise, leakage, shift, confusability and validation design (Name That LLM)

Labels used throughout: **MEASURED** (computed by me from local files with the script named), **INFERRED** (my reasoning from measured numbers), **UNKNOWN** (cannot be determined locally), **RELAYED** (a claim from a sibling researcher's notes file that cites Kaggle pages; I did not re-verify it).

Scripts (all read-only, CPU, at most about 7 min each) live in the session scratchpad, which is temporary: `/private/tmp/claude-501/-Users-i2gunshaker-orca-workspaces-kaggle-comp-name-that-llm/715c8dc3-c85b-4607-80fd-cd053d781a3e/scratchpad/` (`s1_overview.py`, `s2_cv.py`, `s3_style.py`, `s4_fid.py`, `s5_ablate.py`, `s6_adv.py`, `s7_kper.py`). Each finding below restates the script's logic so the result can be reproduced without them.

Abbreviations: BA = balanced accuracy (the official metric), CV = cross-validation, OOF = out-of-fold, LR = logistic regression, SVC = linear support-vector classifier, sd = standard deviation, CI = confidence interval, LB = leaderboard, pt = percentage point.

Local roots used as sources:
- Competition data: [data/](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/data)
- Open-pool tables: [ext/openpool/](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool)
- Code: [src/](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/src), [k/](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/k)
- Sibling notes: [competition_rules_public_families.md](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/research/novel-approaches/deep-research/research_notes/Name%20That%20LLM%20novel%20approaches/competition_rules_public_families.md)

---

## Q1. What lives under data/ and ext/openpool: schemas, counts, labels, class balance, lengths, metadata columns?

### Takeaway
Train is a near-complete **150-prompt × 18-model grid**: one answer per (prompt, model), balanced classes, single-paragraph English, 36–274 words. The only metadata column is `prompt_id`; there are no source, temperature, reasoning or model-version columns. The open pool's passage-level parquet is **not present locally**. Only its summary tables, five figures and a 10,838-row Gemini-3.7 side file are local, so every pool-level statement beyond those tables is UNKNOWN or relayed.

### Cited Findings
- MEASURED: `data/` holds `train.csv` (2,699 × 4: `id, prompt_id, text, model`), `test.csv` (3 × 2: `id, text`), `sample_submission.parquet` (3 × 2: `id, model`), and `name-that-llm.zip`, which contains the same three files. — [data/](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/data) (s1_overview.py)
- MEASURED: train has 18 labels with 150 rows each, except `claude-fable-5` with 149 (max/min = 1.007). There are 150 `prompt_id`s: 149 have exactly 18 rows and 1 has 17. Every prompt has 17–18 distinct labels, and every (prompt, label) cell has exactly 1 row. There are no nulls. — [train.csv](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/data/train.csv) (s1)
- MEASURED: train characters: min 230, median 670, mean 755, max 1,900. Whitespace words: min 36, median 101, mean 115, max 274. Per-label median characters run from 583 (`gemini-3.8-flash`) to 748 (`gpt-5.6-sol`). Per-label mean words run from 100.7 (`gemini-3.8-flash`) to 123.6 (`claude-opus-4-8`), a 23-word spread against a within-class sd of about 55 words. — train.csv (s1, s3)
- MEASURED: train format is one paragraph in 100% of rows: newline rate 0, markdown or list rate 0, CJK rate 0, and a mean non-ASCII character fraction of 0.002. — train.csv (s3_style.py)
- MEASURED: the 3 `test.csv` rows have 321, 760 and 1,077 characters (48/111/149 words) and are single paragraphs. Per the constraints, nothing about the hidden test is inferred from them. — [test.csv](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/data/test.csv)
- MEASURED: `ext/openpool/` holds `dataq_results.csv` (18 rows), `dataq4_runs.csv` (10), `dataq4_paired.csv` (20), `fidelity_by_label.csv` (18), `fidelity_by_source.csv` (509 rows, 57 sources), `fidelity_by_reasoning.csv` (122), `gemini37_unverified.parquet` (10,838 × 19) and five PNG figures. — [ext/openpool/](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool)
- MEASURED: `gemini37_unverified.parquet` columns are `id, text, model, lab, prompt_group, source_dataset, source_file, source_license, origin_hint, n_words, also_in, n_sources, max_cos_train, max_sim_train_prompt, style_score, flag_kb_proxy, flag_gemini37_routing_risk, flag_near_train_prompt, flag_train_prompt_regen`. All rows are `gemini-3.7-flash`, from 9 sources (`harshitkmr/ntl-google-gen-v1` 7,547 rows). Words run 36–253 (median 92). — gemini37_unverified.parquet (s1)
- MEASURED: the code reads `ext/openpool/ntl_open_pool.parquet` with columns `text, model, gen_reasoning, style_score, match_weight, flag_kb_proxy, prompt_canon`. That file is absent locally. — [src/pe.py](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/src/pe.py), [k/dev2/ntl-dev2.py](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/k/dev2/ntl-dev2.py)
- MEASURED: the pool tables sum to 223,840 passages (`fidelity_by_label.csv`). Per-label counts range from 3,541 (`gemini-3.7-flash`) to 23,190 (`gemini-3.8-flash`). — fidelity_by_label.csv (s4_fid.py)
- MEASURED (from figure text): the v11 figure gives v10 = 198,507 → v11 main = 190,588, after −7,620 rows from sources no longer public, +10,294 new rows, and −10,593 Gemini-3.7 rows without a Vertex AI route moved to the side file. The pool carries `prompt_canon`: 13,021 prompts with full text, 592 answered by all 18 labels, 4,472 with missing labels. — [fig5_v11_overview.png](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/fig5_v11_overview.png)
- MEASURED: how the local code uses the data: `src/pe.py` and `src/tpu_train.py`/dev2 train **on the pool only** and evaluate BA **on all 2,699 train rows** (`ev = np.arange(len(T_ids))`). dev2 keeps only `kimi-k3` rows with `gen_reasoning=='high'` and caps each label at 12,000. — [src/tpu_train.py](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/src/tpu_train.py)
- RELAYED: officially, the hidden test is about 4,000 passages from prompts disjoint from train, with unequal per-model counts that differ between public (40%) and private. Lengths "vary randomly between 40 and 250 words". The hosts allow exploiting relationships between test rows. — [sibling notes](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/research/novel-approaches/deep-research/research_notes/Name%20That%20LLM%20novel%20approaches/competition_rules_public_families.md) citing [Kaggle Data](https://www.kaggle.com/competitions/name-that-llm/data), [Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation), [Discussion 752084](https://www.kaggle.com/competitions/name-that-llm/discussion/752084)

### Inferences
- INFERRED: the pool tables (223,840 passages) do not match the v11 main-file count (190,588). They probably describe main + side file or a later version (the sibling notes relay v14 = 223,840). Any reproduction must pin the pool version.
- INFERRED: train's word maximum of 274 exceeds the official 250 because whitespace splitting counts dash-separated or punctuation tokens differently. This is a counting artefact, not evidence of different length targets.
- INFERRED: because train is a balanced grid, prompt content carries **zero marginal label information** inside train. Every prompt appears with every label once. That shapes all the leakage results below.

### Gaps
- UNKNOWN: the pool passages themselves, so I could not measure pool-level duplicates, pool-vs-train overlap outside Gemini-3.7, pool length or typography per label, or `prompt_canon` overlap with train.
- UNKNOWN: train generation settings (system prompt, temperature, reasoning effort, API route, date). No column exists, and per the relayed sibling notes the hosts have not disclosed them.

---

## Q2. What do the open-pool quality and fidelity tables say? How faithful is the pool, and what are the noise indicators?

### Takeaway
The pool is uneven. 7 of 18 labels have a fidelity ratio below 0.75. `kimi-k3` (0.46), `qwen3.8-max` (0.51) and `claude-opus-4-8` (0.54) are the worst, and 35.7% of passages (passage-weighted) sit in (source, label) cells with ratio below 0.75. The largest controllable noise factors are **generation configuration** (kimi-k3: ratio 0.81 for `high` reasoning vs 0.27 for `off`) and **API routing** (76.6% of the Gemini-3.7 side-file rows carry the routing-risk flag). Cleaning and train-matching gave a repeatable gain of about 2–3 BA points on a fixed small model.

### Cited Findings
- MEASURED (my reading of the columns): in `fidelity_by_label.csv`, `fidelity_ratio = hit_pool / hit_real` holds arithmetically (kimi-k3: 0.143/0.313 = 0.457). I read `hit_real` as a reference classifier's recall on real train passages of that label and `hit_pool` as the same classifier's recall on pool passages labelled that model. A ratio below 1 means pool passages look less like the real label than real passages do. The exact reference classifier is not documented in the tables (fig5 says "Tests use a small 1.7B model"). — [fidelity_by_label.csv](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/fidelity_by_label.csv), [fig5](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/fig5_v11_overview.png)
- MEASURED: fidelity ratio by label, ascending: kimi-k3 0.457; qwen3.8-max 0.509; claude-opus-4-8 0.539; gemini-3.6-flash 0.666; claude-opus-4-6 0.697; glm-5.3-flash 0.733; gemini-3.1-pro-preview 0.737; glm-5.3 0.764; gpt-6-astra 0.791; gpt-5.6-sol 0.855; gemini-3.7-flash 0.856; claude-fable-5 0.882; claude-sonnet-5 0.887; gpt-5.6-luna 0.902; gemini-3.8-flash 0.954; claude-opus-5 0.965; gpt-5.5 1.004; gpt-5.6-terra 1.024. Mean 0.79. — fidelity_by_label.csv (s4)
- MEASURED: `fidelity_by_source.csv` has 509 (source, label) cells from 57 sources. The ratio has median 0.815 and IQR 0.68–0.96, ranging 0.04–1.74. Passage-weighted shares: ratio < 0.5 is 9.3%, < 0.75 is 35.7%, > 1.0 is 18.1%. — [fidelity_by_source.csv](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/fidelity_by_source.csv) (s4)
- MEASURED: within-label spread across sources is large. Among cells with ≥ 300 passages, `claude-opus-4-8` runs 0.17–0.82 (worst: `adurey/ntl-proxy-gen-20261031` 0.17, `kilupy/ntl-claude-family-passages` 0.24 on 3,209 rows). `gemini-3.8-flash` runs 0.42–1.34, kimi-k3 0.29–0.90 and qwen3.8-max 0.39–0.69. — fidelity_by_source.csv (s4)
- MEASURED: 47 (source, label) cells with ≥ 100 passages have a family-level hit rate more than 0.2 below real. They concentrate in Gemini labels (for example `adurey` gemini-3.6-flash fam_hit 0.54 vs real 0.96) and kimi/qwen/glm in `kilupy/ntl-18-model-corpus` (kimi fam_hit 0.014 vs 0.313). These are candidate **mislabelled or off-protocol** blocks. — fidelity_by_source.csv (s4)
- MEASURED: source↔label confounding. No source covers all 18 labels; 2 sources are single-label, and 12 is the most common label count (13 sources). The top 10 sources hold 59.9% of passages (`drjcfmd/ntl-generated-v2` 20,400; `harshitkmr/ntl-google-gen-v1` 17,798; `trnhip/ntl-api-gen` 16,818). — fidelity_by_source.csv (s4)
- MEASURED: reasoning config is a strong fidelity factor. kimi-k3 ratio by `gen_reasoning`: high 0.81 (2,673 rows), default 0.37 (2,871), off 0.27 (1,702), other 0.23, low 0.11, unknown 0.34. Inside one source (willrice harvest), kimi high 0.67 (988) vs off 0.24 (1,077). qwen3.8-max with `low` scores 0.00 (70 rows). gemini-3.8-flash within willrice: minimal 0.97, high 0.94, default 0.77, low 0.47. `gen_reasoning == unknown` covers 45.6% of passages. — [fidelity_by_reasoning.csv](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/fidelity_by_reasoning.csv) (s4)
- MEASURED (figure text): a Gemini-3.7 vs 3.8 classifier inside the harshitkmr source has AUC 0.498, i.e. the two are indistinguishable there, versus 0.582 on real train (n = 300). The figure attributes this to non-Vertex APIs serving 3.8 for "gemini-3.7" requests since 2026-10-08. Moving to Vertex-only 3.7 lifted 3.7 recall from 0.21 to 0.42 and 3.8 from 0.59 to 0.64 but dropped 3.6 from 0.63 to 0.47. — [fig5](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/fig5_v11_overview.png)
- MEASURED: in the side file, `flag_gemini37_routing_risk` is set on 8,297 / 10,838 rows (76.6%): 100% of harshitkmr and willrice rows, 0% elsewhere. `flag_kb_proxy` is set on 268 rows. `flag_near_train_prompt` and `flag_train_prompt_regen` are set on 0 rows. — [gemini37_unverified.parquet](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/gemini37_unverified.parquet) (s3)
- MEASURED: data-quality experiments (45,000 passages, 2 seeds, scored as BA on 2,699 train rows):
  - Cleaning: R5 → C5 (minus junk and off-genre sources) adds +0.0273 (seed 0) and +0.0275 (seed 1). R7 → C7 adds +0.0319 and +0.0233.
  - Prompt-centred BA gains are similar: +0.024 and +0.036.
  - Round 4 paired deltas vs v10, with 95% CI from prompt resampling:
    - typography/length-matched sampling (T): +0.0263 [0.0119, 0.0393]
    - Gemini-3.7 Vertex-only (G): +0.0093 [0.000, 0.0186]
    - most-labelled prompts first (P): +0.0089 [−0.0048, 0.0222]
    - adding prvsiyan rows (X): −0.0092 [−0.0226, 0.0041]
  - On the "two answers per prompt" metric, every CI includes 0.
  - Sources: [dataq_results.csv](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/dataq_results.csv), [dataq4_paired.csv](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/dataq4_paired.csv), [dataq4_runs.csv](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/dataq4_runs.csv)
- MEASURED: removing Kaggle-Benchmarks proxy rows (U → UK) gave −0.0097 and +0.0081 BA across the two seeds, which is inconclusive. Removing the 30% least train-like passages per label (UK → US) gave +0.0004 and −0.0074, with no gain. — dataq_results.csv
- MEASURED (figure text): raw vs cleaned pool vs real train style rates. Contains digit: raw about 29%, clean about 21%, real about 10%. we/our/us: clean about 15.5%, real about 31%. Straight `"`: raw 19%, clean 9%, real 4%. API dump or refusal: raw about 8%, clean about 0%, real 0%. The figure notes the we/our/us gap "comes from the host's prompts, not from junk". — [fig3_style_gap.png](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/fig3_style_gap.png)
- MEASURED: train has 0 exact or whitespace/case-normalised duplicates. A crude refusal/boilerplate regex ("as an ai", "i'm sorry", "here is", "certainly!", ...) hits 0.7% of train rows and 0.2% of the Gemini-3.7 side file. These are upper bounds that likely include benign "here is" phrases; I did not inspect them. — s1, s3
- RELAYED: the pool curator removed 284 rows written by models outside the label set (for example `gemini-3.5-flash-lite` labelled `gemini-3.6-flash`). — sibling notes citing [Discussion 752766](https://www.kaggle.com/competitions/name-that-llm/discussion/752766)

### Inferences
- INFERRED: label noise in the pool is mostly **label-conditional configuration noise** (wrong reasoning effort, wrong route or served model, off-protocol prompting), not random flips. It concentrates on the labels that are already hardest on real data (kimi, qwen, opus-4-8, GLM, gemini-3.6). Filtering or reweighting by config should therefore lift exactly the weak classes, which BA rewards.
- INFERRED: since `hit_pool` is scored by a classifier fitted on real-like data, a low ratio could also mean the reference classifier is weak on that label rather than that the pool is wrong. For kimi and qwen, the within-source split by reasoning (0.67 vs 0.24 on the same source) rules that out. For opus-4-8 the cross-source spread (0.17–0.82) points at source-specific contamination.
- INFERRED: the data interventions' effects (+2.3 to +3.2 pt for cleaning, +2.6 pt for typography matching) are larger than 2× the validation sd I measured (about 0.009, Q5/Q7). The marginal ensemble tweaks in the deployed notebook (0.681 → 0.682 by mixing weights) are below it.

### Gaps
- UNKNOWN: how the reference classifier behind `hit_pool`/`hit_real` was trained (on train? OOF?) and on which pool version. Without that, ratios above 1 (gpt-5.6-terra 1.02) cannot be interpreted strictly.
- UNKNOWN: within-pool "same prompt, different label" contradictions. They require `prompt_canon` in the absent parquet.

---

## Q3. Prompt/source leakage: does the same prompt or source appear across classes or splits? Would a random split leak prompt identity?

### Takeaway
Yes. Every train prompt appears in all 18 classes, so a random split leaks prompt identity into every fold. But in this balanced grid the leak runs **backwards**: a row's prompt-mates in the training fold all carry other labels, so content similarity actively points to wrong labels. A random split therefore **underestimates** a content-sensitive model (BA 0.123 random vs 0.353 grouped). This has nothing to do with test-time behaviour, so random-split CV on train is invalid in both directions. For the pool, source identity is tied to label (no source has all 18), so random splits within the pool would leak source style.

### Cited Findings
- MEASURED: 150 / 150 prompts are shared across ≥ 17 classes. Under random StratifiedKFold(5), each validation row has on average about 13.6 of its 17 prompt-mates in the training fold, each with a different label (INFERRED arithmetic: 17 × 0.8). — train.csv (s1)
- MEASURED: word TF-IDF cosine (stop words removed): same-prompt pairs mean 0.226 (p10 0.111), different-prompt pairs mean 0.012 (p99 0.065). The 1-nearest neighbour shares the prompt for 99.78% of rows and the label for 0.11% of rows. — s2_cv.py
- MEASURED: char 1–4 + word 1–2 TF-IDF with LogisticRegression(C = 10):
  - random StratifiedKFold5: BA 0.1233, family accuracy 0.661
  - prompt GroupKFold5: BA 0.3530, family accuracy 0.711
  - Source: s2_cv.py, which fits the vectorisers inside each fold
- MEASURED: the gap persists with LinearSVC(C = 0.3), char 1–4: random 0.1552 vs grouped 0.3130. — s5_ablate.py
- MEASURED: prompt identity is recoverable without labels. Average-link agglomerative clustering (cosine, k = 150) on word TF-IDF recovers `prompt_id` with ARI 0.909; median cluster size is 18 and the maximum is 54 (some prompts merge). — s2
- MEASURED: the Gemini-3.7 side file was already filtered against train prompts (`flag_near_train_prompt` = 0, `flag_train_prompt_regen` = 0, `max_cos_train` max 0.299, p99 0.256). Only 6 / 10,838 rows share any 8-word shingle with train, and 0 match a normalised train text. — s3
- MEASURED: pool source↔label confounding (Q2). 0 / 57 sources cover all 18 labels, and the label count per source ranges from 1 to 17. — s4
- RELAYED: a ModernBERT member pretrained on the pool reached CV 0.616 but LB 0.575, and one explanation offered is prompt-overlap memorisation in its pool. — sibling notes citing [Discussion 754860](https://www.kaggle.com/competitions/name-that-llm/discussion/754860)

### Inferences
- INFERRED: the "random − grouped" BA gap is a **content-dependence meter** in this dataset. A negative gap means a model reads prompt content (penalised under random splits). A gap near 0 means its features are prompt-invariant (see Q5: the typography-only gap is +0.0004).
- INFERRED: the opposite, inflating leak happens when a model is trained on **pool rows answering train prompts** and evaluated on train. Prompt content is then label-informative wherever the pool's per-prompt label coverage is incomplete (4,472 pool prompts have missing labels, per fig5). That is why pool→train evaluation must exclude near-train-prompt pool rows (the curator's `flag_near_train_prompt`/`flag_train_prompt_regen`).
- INFERRED: within the pool, a random split shares both prompt (`prompt_canon`) and source between folds. Since sources differ in label coverage and style, pool-internal CV will overstate generalisation. Use leave-source-out and prompt groups.

### Gaps
- UNKNOWN: the number of pool `prompt_canon` groups that duplicate or paraphrase train prompts in the main pool (the parquet is absent). fig5 reports only pool-wide prompt counts.
- UNKNOWN: whether the hidden public/private split is by prompt or by passage (relayed as undisclosed).

---

## Q4. Near duplicates within and across classes, and between train and the pool

### Takeaway
No exact or near-exact duplicates exist in train. The most similar pairs are always **same prompt, different label**, so "near-duplicate" risk in train is purely prompt-mate similarity, not copy leakage. Gemini-3.7 pool rows have no duplicates against train or within themselves. Main-pool duplicates could not be measured locally.

### Cited Findings
- MEASURED: exact duplicate rows in train: 0. After lowercasing and collapsing whitespace: 0. — s1
- MEASURED: char 5-gram (`char_wb`) TF-IDF cosine, each row's maximum similarity to any other row: median 0.367, p99 0.645, max 0.782. 0 rows exceed 0.8, and 5 rows exceed 0.7, all of them same prompt and different label. — s2
- MEASURED: Gemini-3.7 side file vs train: 0 normalised exact matches and 6 rows sharing any 8-word shingle. Within the side file there are 0 normalised duplicates. The curator's `n_sources` (number of source datasets that list the row) has a mean of 2.28 and a maximum of 11. 2,617 / 10,838 rows (24.2%) appear in more than one source, so they are cross-source copies that the curator already merged into one row. — s3, gemini37_unverified.parquet

### Inferences
- INFERRED: near-duplicate risk lives across *sources* of the pool (the `also_in`/`n_sources` columns show the same passage republished in up to 11 datasets). Deduplication must happen before any leave-source-out split, or the same text lands on both sides.
- INFERRED: the char-5 maximum of 0.78 for same-prompt pairs shows that prompt-mates share long phrases (named entities, topic terms). This is the content channel that grouped CV must block.

### Gaps
- UNKNOWN: exact and MinHash near-duplicate rates in the 190k–224k main pool, and pool-vs-train near-duplicates for 17 of 18 labels. The parquet is not local, and I did not download it, per the read-only, no-new-data scope.

---

## Q5. Which classes are hard to tell apart? Stylometric proxies, grouped vs random confusion, shortcut gap

### Takeaway
The confusable sets are (1) Gemini 3.6/3.7/3.8 Flash, (2) the GPT-5.6 luna/sol/terra tiers (plus astra), (3) **GLM ↔ Claude**: GLM passages are predicted "Claude" more often than "GLM", and they share em-dash and straight-apostrophe typography, and (4) kimi-k3 and qwen3.8-max collapsing into GPT. Typography alone is strongly family-diagnostic and prompt-invariant, but it cannot separate siblings.

### Cited Findings
- MEASURED: per-label rates of passages containing each marker (s3_style.py):
  - em dash `—`: Claude 0.63–0.83, GLM 0.79–0.82, kimi 0.52, GPT-5.6 0.23–0.37, gpt-5.5 0.13, gpt-6-astra 0.09, Gemini 0.00–0.01, qwen 0.03
  - spaced em dash ` — `: only claude-opus-4-6 (0.27) and claude-opus-5 (0.25); everything else ≤ 0.03
  - curly apostrophe `’`: Claude 0.00, GLM 0.00, Gemini 0.06–0.31, GPT 0.42–0.53, kimi 0.32, qwen 0.29
  - curly double quote `“`: GPT 0.07–0.15, kimi 0.09, others ≤ 0.02
  - hyphenated word: Claude/GLM/GPT about 0.4–0.5 vs Gemini 0.09–0.18 vs gpt-6-astra 0.21
  - largest across-class ranges: em dash 0.83, straight apostrophe 0.73, colon 0.61, curly apostrophe 0.53, semicolon 0.47
- MEASURED: grouped OOF (char + word TF-IDF + LR, GroupKFold5 by prompt), confusions out of the 300 rows of each pair:
  - gemini-3.7/3.8: 85
  - gpt-5.6-luna/sol: 58
  - gemini-3.6/3.7: 58
  - glm-5.3/glm-5.3-flash: 55
  - luna/terra: 49
  - gemini-3.6/3.8: 46
  - gpt-5.5/kimi-k3: 45
  - opus-4-8/sonnet-5: 44
  - opus-5/glm-5.3: 42
  - sol/terra: 41
  - Source: s2
- MEASURED: family confusion (rows = true, grouped):
  - GLM: 134 → Claude, 114 → GLM
  - Claude: 95 → GLM
  - kimi: 55 → GPT, 45 → kimi
  - qwen: 59 → GPT, 57 → qwen
  - Gemini: 567/600 correct-family
  - GPT: 601/750
  - Source: s2
- MEASURED: grouped per-class recall, lowest first: fable-5 0.16, opus-4-8 0.18, glm-5.3-flash 0.19, glm-5.3 0.21, gpt-5.6-terra 0.23, gpt-5.6-sol 0.24, kimi 0.30. Highest: gemini-3.1-pro 0.65, opus-5 0.61, astra 0.50. — s2
- MEASURED: these weak labels match the curator's `hit_real` (per-label recall of a stronger model on real train): fable 0.107, glm 0.19–0.20, sol 0.193, terra 0.213, opus-4-8 0.227. `fam_hit_real` for GLM is 0.37–0.38. — fidelity_by_label.csv
- MEASURED: content-vs-style ablation ladder (LinearSVC C = 0.3, balanced, 5 folds, same folds for every row):

  | feature set | random BA | grouped BA | random − grouped |
  |---|---|---|---|
  | A raw char 1–4 | 0.1552 | 0.3130 | −0.1578 |
  | B content words only (stop words removed) | 0.0759 | 0.2252 | −0.1493 |
  | C function words + punctuation, content words masked to `X` | 0.2193 | 0.2911 | −0.0719 |
  | D char 1–4 after normalising `— – ’ ‘ “ ”` to ASCII | 0.1448 | 0.3045 | −0.1597 |
  | E 16 typography/length rates only (LR) | 0.2482 | 0.2478 | +0.0004 |
  | F masked content + typography normalised | 0.2163 | 0.2811 | −0.0648 |

  Source: s5_ablate.py
- MEASURED: grouped BA rises with length (char SVC). Word-length quartiles Q1 (≤ 64 words) 0.247, Q2 0.285, Q3 0.330, Q4 (> 157) 0.375. — s5

### Inferences
- INFERRED: typography (E) is content-invariant (gap about 0) and alone reaches 0.25 BA, about 80% of the raw char model's grouped BA. But normalising typography away (D) costs only about 0.9 pt. So the char model also carries the same family signal through other channels, and typography is mostly **family-level** signal, not sibling-level.
- INFERRED: whether typography is a generator fingerprint or a host post-processing artefact matters. If the host's "light formatting cleanup" (relayed) differs between train and test, or pool producers normalise quotes differently, a typography-heavy model breaks. The guards hard-coded in the deployed notebook (`GUARDS` in `core_math.py`, the em dash / `’` / `"` / `“` / `–` label sets) are exactly these rules, which makes the 0.682 system exposed to that risk. — [k/dev1/ntl-dev1.ipynb](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/k/dev1/ntl-dev1.ipynb)
- INFERRED: the hard pairs are sibling versions or tiers that share tokenizer, post-training and typography. Distinguishing them needs either per-config pool data that matches the host (kimi high, Vertex 3.7) or likelihood/residual features. More capacity on the same mismatched data will not help much. The GLM↔Claude confusion (GLM written in Claude-like typography) is a candidate for a pairwise specialist.
- INFERRED: the length dependence means any validation comparison must be length-stratified. A change that helps only long passages can look neutral overall or flip under a different test length mix.

### Gaps
- UNKNOWN: confusion structure of the deployed 0.682 stack on held-out data. No OOF outputs are local, and the 3-row preview cannot be used.
- The TF-IDF probe has only 120 training rows per class per fold, so it understates the absolute accuracy of the strong models. The pair ranking is a proxy only.

---

## Q6. Domain shift: train/test-preview vs open pool (length, formatting, markdown, language)

### Takeaway
The cleaned pool already matches train on gross format: single paragraph, no markdown, English, complete sentences. It still differs measurably in length, digits, hyphenation and first-person-plural use, which point to different *prompts*, not junk. An adversarial classifier separates real Gemini-3.7 train rows from Gemini-3.7 pool rows with AUC 0.85 (char TF-IDF), or 0.73 from 13 typography/length rates alone.

### Cited Findings
- MEASURED: Gemini-3.7 side file vs train (all labels / train gemini-3.7 rows):
  - words: mean 103 vs 115 / 117; median 92 vs 101
  - word hyphenation: 0.506 vs 0.351 / 0.18
  - digit: 0.168 vs 0.099 / 0.04
  - we/our/us: 0.10 vs 0.307 / 0.307
  - colon: 0.205 vs 0.521 / 0.227
  - curly apostrophe: 0.40 vs 0.21 / 0.31
  - em dash: 0.000 vs 0.39 / 0.007
  - newline: 0 vs 0
  - non-ASCII fraction: 0.001 vs 0.002
  - Source: s3
- MEASURED: shift varies by pool source for the same label. Hyphenated-word rate runs 0.21 (drjcfmd) to 0.68 (kilupy), digits 0.00 (earthmaker) to 0.22 (willrice / michaelsemenoff pool-v3), and we/our/us 0.07 (harshitkmr) to 0.19 (earthmaker). — s3
- MEASURED: adversarial validation of train gemini-3.7-flash (150) vs 1,500 sampled side-file rows, 5-fold OOF AUC: 0.852 with char 1–4 TF-IDF + LR (s5), and 0.726 with 13 typography/length rates + LR (s6_adv.py; zero-variance columns dropped).
- MEASURED (figure text): typography/length-matched sampling (`match_weight`) moves pool rates to real: word hyphens (Gemini) 44% → about 12%, digits 21% → about 10%. The we/our/us gap remains (15.5% vs 31%). That variant was the only round-4 change with a 95% CI excluding 0 on all-prompt BA (+0.026). — fig5, dataq4_paired.csv
- MEASURED: test preview (n = 3) is single-paragraph, 48–149 words. No further inference is drawn. — test.csv

### Inferences
- INFERRED: the remaining shift is mainly the **prompt/topic distribution and instruction template** (host prompts ask for vivid explanations; the relayed reconstruction says 85/150 are "explain X vividly"). That is why the we/our/us gap cannot be cleaned away. Prompt-matched regeneration (the host-style template, 40–250 word length instruction) is the data fix. Filtering cannot do it.
- INFERRED: AUC 0.73 from typography alone means pool producers' rendering and normalisation (quotes, dashes) is not identical to the host's. Typography-based rules learned on pool data can transfer wrongly per source.
- INFERRED: the hidden test also comes from new prompts (official, relayed), so the train→test shift in topic is real too. Train is the only in-protocol sample (2,699 rows), which makes it the validation target, not the main training set.

### Gaps
- UNKNOWN: shift for the other 17 labels in the main pool (parquet absent). UNKNOWN: hidden-test length and topic mix beyond the official 40–250 words and topic list.

---

## Q7. Frozen validation schemes and ablations that separate causal generator signal from content shortcuts

### Takeaway
Use three frozen schemes:
- **V1** prompt-grouped repeated CV on train, scored at several answers-per-prompt (k) and class-mix settings
- **V2** pool→train transfer, with train-prompt-near pool rows removed and leave-source-out
- **V3** a length × family stratified shift report

Never use random splits. A key measured caution: within-prompt tricks (centring, assignment) are worth +7 to +11 pt **only when all 18 answers per prompt are present**. At 2–3 answers per prompt they are worth −6.6 to +1.1 pt, so validation must simulate the test's (unknown, probably small) k.

### Cited Findings
- MEASURED: grouped OOF (LR) BA 0.3530. With Hungarian one-label-per-row assignment inside oracle prompt groups: 0.4630 (+11.0 pt). Inside recovered clusters (ARI 0.909): 0.4430. Per-prompt log-probability centring: 0.4215 (+6.9 pt). Random split + Hungarian: 0.1652. — s2
- MEASURED: subsample k rows per prompt from the grouped OOF probabilities (30 draws) and apply each trick:

  | k | plain | centred | Hungarian |
  |---|---|---|---|
  | 1 | 0.3545 | — | — |
  | 2 | 0.3534 | 0.2876 | 0.3602 |
  | 3 | 0.3519 | 0.3409 | 0.3629 |
  | 6 | 0.3530 | 0.3882 | 0.3801 |
  | 12 | 0.3523 | 0.4156 | 0.4113 |
  | 18 | 0.3530 | 0.4215 | 0.4630 |

  Source: s7_kper.py
- RELAYED: "Prompt-grouping tricks that assume train's 18-per-prompt layout repeatedly lost on the LB", and participants infer few answers per prompt in test. — sibling notes ([competition_rules_public_families.md](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/research/novel-approaches/deep-research/research_notes/Name%20That%20LLM%20novel%20approaches/competition_rules_public_families.md)) citing [Discussion 750213](https://www.kaggle.com/competitions/name-that-llm/discussion/750213) and others
- MEASURED: validation noise on train. Grouped char-SVC BA 0.3130 has prompt-bootstrap (500 resamples of the 150 prompts) 95% CI 0.2956–0.3308, sd 0.0091. Per-fold BA 0.276 / 0.330 / 0.322 / 0.337 / 0.300. Per-prompt accuracy has mean 0.313, sd 0.105, p10 0.167 and p90 0.450. — s5
- MEASURED: the deployed notebook already uses test-time content grouping (`capped_agglo(S, cap=18, ...)`), `linear_sum_assignment` and soft Sinkhorn class balancing toward "official rows' shares". All three assume a grid or class-share structure that V1's k- and mix-stress tests should probe. — [k/dev1/ntl-dev1.ipynb](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/k/dev1/ntl-dev1.ipynb)
- MEASURED: the curator's eval already scores pool-trained models on all 2,699 train rows. It reports `balanced_acc_prompt_centred`, a `half_B` variant ("half of prompts unused for fitting") and `two_answers_per_prompt`, and the last has CIs about ±0.03 wide. — dataq4_paired.csv, fig5

### Inferences (proposed designs; untested as a full protocol)

**V1: Prompt-grouped, k-aware, mix-aware CV on train (for anything fitted or calibrated on train: heads, stackers, weights, priors, post-processing)**
- Folds: GroupKFold(5) on `prompt_id`, repeated with 3 different group-to-fold seeds (shuffle the prompt list, then assign). Freeze them as a `fold` column CSV with a SHA-256 of `id,prompt_id,fold`. StratifiedGroupKFold is unnecessary: every prompt contains every label, so each fold is automatically balanced (MEASURED grid structure).
- Score each candidate at k ∈ {1, 2, 3, 18} answers per prompt (subsampled as in s7, 30 draws). Report k = 2 as primary, because the relayed public evidence suggests few answers per prompt, and k = 18 only as an upper bound.
- Class-mix stress: resample validation rows with Dirichlet(α = 2) class proportions (20 draws) and report mean and worst-decile BA. This tests prior or Sinkhorn corrections, since the test's unequal mix is official (relayed).
- Decision rule: adopt a change only if the paired prompt-bootstrap delta is greater than 0 with a 95% CI excluding 0 at k = 2 *and* it is non-negative at k = 1. With sd about 0.009 (MEASURED), deltas under about 0.02 are noise on 2,699 rows.

**V2: Pool→train transfer with leave-source-out (for anything trained on external data)**
- Train on the pool after (a) dropping rows with `flag_near_train_prompt` or `flag_train_prompt_regen`, or `max_sim_train_prompt` above a fixed threshold (for example 0.3), and (b) dropping exact or `also_in` duplicates. Evaluate on all 2,699 train rows (prompt-disjoint by construction), which is the curator's protocol.
- Leave-one-source-out (LOSO) for the top 10 sources (59.9% of rows): retrain without source s and measure Δ BA on train per label. This is a causal source-quality ranking to compare with `fidelity_ratio`.
- Pool-internal model selection, if any, uses GroupKFold on `prompt_canon` × source, never random rows.
- Report per-label recall next to `hit_real` so that gains on kimi, qwen, opus-4-8 and GLM are visible.

**V3: Length × family stratified shift report (attached to every V1/V2 result)**
- Report BA per word-length quartile (edges 64 / 101 / 157 words, MEASURED) and family-level vs within-family accuracy separately. Within-family accuracy is where headroom lives: the probe's family accuracy is 0.71 but BA is 0.35.
- Importance-weighted BA, with weights from an adversarial pool-vs-train classifier (AUC 0.85 for Gemini-3.7, MEASURED), to check that pool-trained gains don't come from pool-like train rows only.
- Truncation robustness: re-score on train passages cut to the first 40 / 80 / 120 words at a sentence boundary. This simulates the official random 40–250 word lengths.

**Ablations that separate generator signal from content shortcuts (run on V1, and on V2 where it applies)**
1. *Content-masking ladder* (A–F above, with the real model). Feed masked-content and typography-normalised text at **test time only** to the trained model. A model whose BA drops sharply under content masking but whose random-split BA is far below its grouped BA is reading content. Success criterion: grouped BA retained ≥ 85% under masking, with random − grouped gap ≥ −0.07 (the masked-TF-IDF level, MEASURED).
2. *Typography swap test.* Swap ASCII and curly quotes and dashes in validation text. Measure prediction flips. A large flip rate means the model relies on rendering, which may be pipeline-specific (pool AUC 0.73 on typography alone).
3. *k-sensitivity of transductive post-processing* (s7 protocol). Any centring, Hungarian or cluster step must beat plain at k = 2. At the measured TF-IDF level, centring loses 6.6 pt at k = 2.
4. *Config ablations on pool data* (V2): kimi `high` only vs all (ratio 0.81 vs 0.27–0.37); drop `qwen3.8-max low`; Gemini-3.7 Vertex-only vs all (fig5: +0.21 recall on 3.7 but −0.16 on 3.6). Report per-label recall.
5. *Train-prompt-overlap ablation* (V2): with vs without pool rows above a `max_sim_train_prompt` threshold. If BA on train falls when they are removed, the earlier "gain" was content leakage.
6. *Same-prompt pairwise test*: within each prompt, for every pair of sibling labels (for example gemini-3.7/3.8, luna/sol, glm/claude), measure two-alternative accuracy of the model's score difference. Content is held exactly constant, so this isolates generator signal for the hard pairs.

- INFERRED (headroom judgement): data and validation probably offer more headroom than bigger models or ensembles, for four reasons:
  - Cleaning and matching interventions show repeatable +2–3 pt BA gains on a fixed small model, with CIs excluding 0.
  - 7 of 18 labels have fidelity below 0.75 and include the weakest real-data classes.
  - Configuration filtering more than doubles fidelity for kimi-k3.
  - Train-side validation cannot even resolve the +0.001–0.003 ensemble-weight tweaks behind 0.681 → 0.682 (sd 0.009).
- Counterevidence: those data gains were measured with a 1.7B model on pool→train BA (about 0.49). They may shrink inside a 0.68 stack, and the relayed claim that grouped CV "has stopped predicting LB differences" for stacked members means V1 is also not guaranteed to track LB.

### Gaps
- UNKNOWN: the hidden test's answers per prompt and class mix, so the k = 2 primary setting is an assumption from relayed participant inference. It should be revisited if hosts disclose it.
- UNKNOWN: whether V1/V2 rankings correlate with LB. That needs a few deliberate paired submissions, which were out of scope (no submissions).
- NOT RUN: MinHash on the main pool, leave-source-out retraining, and the masking ladder on the real (LoRA/LLM) models. All need the absent pool parquet or GPU time beyond this task's CPU-only budget.
