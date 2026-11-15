# Own the validation, then attack siblings

No single idea in the evidence base reliably promises a drastic jump from 0.682. The best-supported route to a large gain has two parts. First, stop treating dev1 as a model we can tune: it is a public fork we cannot validate. Second, build an owned, validated member whose errors differ from the public stack's. Three moves come first, in this order:

1. Freeze a prompt-grouped validation harness that simulates the sparse answers-per-prompt test layout and an unequal class mix, and use it to re-audit dev1's decision layer.
2. Train an owned member on generation-config-filtered community data with a same-prompt contrastive objective, aimed at within-lab sibling confusions.
3. Measure dev1's own error profile on a fresh, never-seen "sentinel" set, so that blending decisions rest on data rather than on single public-leaderboard submissions inside the noise.

The stakes are concrete. **The public top score is 0.768 against a forkable-notebook plateau of 0.684**, a gap of about 0.08, while **the measured noise floor is about 0.02**. So headroom exists, but it is not visible in public code. dev1 is a near-verbatim public notebook whose single team change was a blend setting. That change pushes predictions harder toward a uniform class prior, and the official statement that the hidden class mix is unequal and differs between public and private contradicts that prior. The most specific large-error target is the GLM↔Claude confusion. For it, the deliberately unconventional bet is to score passages under GLM-5.3-Flash's real open weights, streamed layer by layer. It is feasible in principle, unproven in practice, and cheap to kill on 1,050 training rows before anyone builds it.

Evidence labels used throughout:

| Label | Meaning |
|---|---|
| **[Official]** | Kaggle competition pages, metadata or an admin reply |
| **[Local]** | Verified in this repo's code or data, or measured by a read-only diagnostic on local files |
| **[Claim]** | One paper's or participant's report, not reproduced by us |
| **[Inference]** | Our reasoning from cited facts |
| **[Proposal]** | An untested idea |

"Novel to the competition" is itself an inference from a read of 20 public notebooks and 47 discussion threads. It is not an exhaustive survey, and private teams may already do any of these things.

## dev1 is a resubmitted public fork that nobody can validate offline

The baseline audit separates what is known from what is assumed. The table below keeps those apart. Its main finding: **the team owns none of the modelling in the 0.682 submission and has no held-out predictions from it.**

| Item | Status | Evidence |
|---|---|---|
| dev1 provenance | **[Local]** It is lightningv08's public 0.681 notebook (v15), "run unchanged except for 2 mixing settings; nothing is retrained": Sinkhorn target mix 0.35→0.50 and GSQ share 6%→12%. It is titled "0.682 \| 0.681 + Qwen3.8-27B GSQ 12% + Mix 0.50". | `k/dev1/ntl-dev1.ipynb` cell 0; first committed README said "commit of public 0.682 fork" |
| Submission 58200298, public 0.682, 2026-11-10 | **[Local]** Corroborated only by hand-edited prose (an uncommitted `README.md:4` edit and `BRIEF.md`). There is no Kaggle API record, no hidden-rerun log and no private score. | `README.md`, `out/dev1/ntl-dev1.log` |
| Local dev1 output | **[Local]** 3 rows, all `claude-fable-5`. These are hard-coded by a preview gate that loads no weights (`model_weights_loaded: false`), so they carry no model information. | `out/dev1/submission.mode.json`, notebook cell 4 |
| dev1 architecture | **[Local]** Frozen inference-only ensemble with these members: LoRA read-outs of truncated Qwen3-14B, Qwen3-8B, Llama-3.1-8B and Qwen2.5-14B; DeBERTa; a 4-view LinearSVC (char, word, "shape", function-word) plus TF-IDF and "distort" heads; a complementary style stack (15%); ATLAS on rows with top-2 margin < 0.3 (10%); Qwen3.8-27B GSQ (12%); a 5-fold Qwen3-Embedding-8B expert mixed geometrically (40%). It ends with soft Sinkhorn balancing and a clipped typography score. | notebook cells 6, 14, 25, 39–51 |
| Train data inside dev1's fitted parts | **[Local]** GSQ saw official train for 2 epochs. The Gaussian heads assert `official_rows == 2699`. Typography rates and the Sinkhorn prior are fit on train at runtime. | cells 8, 14, 39 |
| Offline validity of dev1 | **[Inference]** None. Train is consumed, and per-member OOF predictions and fold maps are not local. The upstream blend weights were chosen by single public-LB submissions of 0.676–0.682 that the notebook itself warns to "read with care". | cell 0 |
| Hidden-rerun runtime | **[Unknown]** for 58200298. Public notebooks in this lineage report about 8 h of the 9 h limit. | [prvsiyan](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author), [zbigniewkirchner](https://www.kaggle.com/code/zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public) **[Claim]** |
| Hard punctuation guards | **[Local]** `GUARDS` is defined but never referenced, and the finalize docstring says "no final hard punctuation mask or Hungarian override". One sibling note asserted the guards were live; the code contradicts it. Typography enters only as a soft score clipped to ±0.30. | cells 8, 43 |
| Hungarian assignment in the scored path | **[Unknown]** One note found `linear_sum_assignment` in the notebook text, and the audit found no Hungarian override in finalize. Whether it runs on the scored path is unresolved. | cell 8 |
| dev2 | **[Unknown outcome]** A TPU v5e-8 *smoke* script: `smoke=True`, at most 36 optimizer steps, eval on 256 train rows. The model is a Qwen3-8B-base 24-layer LoRA with mean pooling at 7 depths and a linear head, trained on the pool only, with no class weights in the pushed version. There are no logs, checkpoints or metrics. Even a completed smoke run would only check plumbing. Its T4 inference path (`src/infer_t4.py`) is not connected to any submission notebook. | `k/dev2/ntl-dev2.py`, `README.md:5` |

Two audit findings change strategy rather than just describing it.

First, **dev1's only team-side change is a noise-level change that increases private-board risk**. Moving the Sinkhorn target mix from 0.35 to 0.50 pulls the predicted class marginals harder toward the uniform train prior. The notebook does this with 100 iterations, blended in log space at strength 0.30, whenever there are 18 or more rows. Yet the hosts state that "the number of test passages per model is **not** equal, and the mix of models differs between the public and private leaderboard" ([Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation)) **[Official]**. One participant simulated Dirichlet-uneven test mixes and found that full Sinkhorn balancing lost up to 0.07, while 1–3 iterations cost under 0.01 ([Discussion 752365](https://www.kaggle.com/competitions/name-that-llm/discussion/752365)) **[Claim]**. A theoretical point sharpens this **[Inference]**. Balanced accuracy is maximised by predicting argmax_y p(x | y), which does not depend on the test mix at all. What BA asks for is removing the *training* prior from each member's posterior. Forcing predicted counts toward equality is a different operation. It helps when members are biased by imbalanced pool data, and it hurts when the true mix is uneven.

Second, **runtime is already spent**. If dev1 runs near the 8 h that its lineage reports **[Claim]**, any new member must either fit in roughly 45 minutes or replace something. The obvious candidate to replace is GSQ, a 27B model run at batch size 1 over full texts that carries only a 12% share. The audit flags it as a likely dominant cost **[Inference]**.

## The 0.08 gap lives in data fidelity, test layout and siblings, not in ensemble size

The official frame is narrow and clear. The metric is balanced accuracy over 18 labels, with scores truncated to 3 decimals. The notebook-only submission is rerun on about 4,000 hidden passages within a 540-minute CPU or GPU limit, with internet off. The public board is 40% of the test set. Teams get 5 submissions a day and 2 final selections ([Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation); [Code Requirements](https://www.kaggle.com/competitions/name-that-llm/overview/code-requirements); [Rules](https://www.kaggle.com/competitions/name-that-llm/rules)) **[Official]**.

Test prompts are disjoint from train prompts, and passages "vary randomly between 40 and 250 words" ([Data](https://www.kaggle.com/competitions/name-that-llm/data); [Overview](https://www.kaggle.com/competitions/name-that-llm/overview)) **[Official]**. An admin allowed test-set transduction: "Feel free to explore any relationships in the data that you find useful". The same admin allowed public community datasets and self-generated data within roughly the value of one prize ($5k), adding "This is intended to be an ML prediction competition, not a data generation competition" ([Discussion 752084](https://www.kaggle.com/competitions/name-that-llm/discussion/752084)) **[Official]**.

The final deadline is listed as January 13, 2027 on the Timeline page but January 14 in the metadata. Treat Jan 13 as binding ([Timeline](https://www.kaggle.com/competitions/name-that-llm/overview/timeline)) **[Official, conflicting]**. The code requirements list only CPU and GPU notebooks, so TPU is a training-only resource **[Official / Inference]**.

The leaderboard explains why the fork cluster is uninformative. Of 5,607 teams, **319 sit at exactly 0.684 and 141 at 0.682**, and a 0.682 places a team around rank 489–629. Rank 1 is 0.768, rank 10 is 0.747, and gold starts near 0.733 ([Leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard)) **[Official snapshot, 2026-11-12]**. No top-team write-ups exist, so what explains 0.70–0.77 is unknown. Plausible candidates are better-matched private data, protocol match or different modelling, but which one matters cannot be told from public evidence **[Inference]**.

The noise floor caps what any single comparison can show. A prompt bootstrap on 2,699 train rows gives **sd 0.0091 (95% CI width about ±0.018)** for a char-SVC **[Local]**. A participant estimates a cluster-robust public-LB standard error of about 0.014 ([Discussion 750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390)) **[Claim]**. Identical code resubmitted has scored up to 0.009 apart ([alexchilton 0.684](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed)) **[Claim]**. Differences under about 0.02 therefore decide nothing, and that covers the whole 0.676–0.684 tuning history behind dev1.

The public families share a small set of failure assumptions. The table compresses them.

| Family | Assumption that caps it | Evidence of the cap |
|---|---|---|
| Char/word TF-IDF and stylometry | Surface statistics transfer across prompts and settings | Grouped CV 0.34–0.37; 55–58% of errors fall within the correct lab ([750664](https://www.kaggle.com/competitions/name-that-llm/discussion/750664)) **[Claim]**. A hard typography veto lost 0.003–0.005 in forks ([lucasboesen](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)) **[Claim]** |
| Two-stage LoRA on truncated decoders | Pool text matches the host protocol | A pool-trained member's OOF rose 0.569→0.605 while the LB fell 0.651→0.647 ([753177](https://www.kaggle.com/competitions/name-that-llm/discussion/753177)) **[Claim]** |
| Encoders | Small encoders capture the fingerprint | ModernBERT: CV 0.616 vs LB 0.575; adding it to the stack lowered the LB ([754860](https://www.kaggle.com/competitions/name-that-llm/discussion/754860)) **[Claim]** |
| Bigger quantized LLMs | Capacity extracts more fingerprint | Swapping in a Qwen3.5-27B NF4 member lowered a blend 0.651→0.638 ([alexchilton](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed)) **[Claim]** |
| Test-layout tricks (clustering, Hungarian, centring) | Test keeps train's 18-answers-per-prompt grid | About +0.10 in train-layout CV, but −0.009 to −0.017 on the LB ([750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390); [757855](https://www.kaggle.com/competitions/name-that-llm/discussion/757855)) **[Claim]** |
| Family gates | The lab is identified reliably | A Claude gate helped; Gemini gates were flat; a GPT gate needed matched data ([750224](https://www.kaggle.com/competitions/name-that-llm/discussion/750224)) **[Claim]** |
| Mega-ensembles | Members add independent evidence | ±0.004 per added member; one nested-CV blend scored 0.6865 offline but 0.56 on the LB ([alexchilton](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed)) **[Claim]** |
| External data | Same model name means same distribution | Pool growth gave +0.028, then +0.003, then +0.002 ([752357](https://www.kaggle.com/competitions/name-that-llm/discussion/752357)) **[Claim]** |

Local measurements turn three of those assumptions into hard numbers.

**The test-layout assumption.** Train is a 150-prompt × 18-model grid, so prompt content carries zero label information inside train. A random split also leaks prompt-mates that all carry *other* labels. The result is that random-split BA (0.123) sits far *below* prompt-grouped BA (0.353) for a TF-IDF LR **[Local]**. Within-prompt tricks look powerful only on dense grids. For the same OOF probabilities, centring moves BA from 0.353 to 0.422 at 18 answers per prompt but to 0.288 at 2 answers. Hungarian assignment gives 0.463 at 18 but only 0.360 at 2 **[Local]**.

| Answers per prompt (k) | Plain | Centred | Hungarian |
|---|---|---|---|
| 2 | 0.353 | 0.288 | 0.360 |
| 3 | 0.352 | 0.341 | 0.363 |
| 18 | 0.353 | 0.422 | 0.463 |

A participant reports that OOF scored at two answers per prompt tracked the board at correlation 0.91, against 0.13 for full groups, over 15 submissions ([alexchilton](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed), citing [753177](https://www.kaggle.com/competitions/name-that-llm/discussion/753177)) **[Claim, single source]**. dev1 still carries heads that fire only for test groups of 2–18 members: twin, Gaussian, native-centred and GSQ group codes **[Local]**.

**Where the errors are.** The hardest pairs are siblings. In a grouped TF-IDF OOF, the most confused pairs (out of 300 rows each) are gemini-3.7/3.8 (85), gpt-5.6-luna/sol (58), gemini-3.6/3.7 (58) and glm-5.3/glm-5.3-flash (55). **GLM passages are called Claude more often than GLM: 134 vs 114** **[Local]**. That matches public reports that GLM writes in Claude-like style ([Discussion 755314](https://www.kaggle.com/competitions/name-that-llm/discussion/755314)) **[Claim]**. Typography alone is prompt-invariant (random−grouped gap +0.0004) and reaches 0.248 BA, but it is family-level signal. Normalising dashes and quotes away costs the char model only 0.9 points **[Local]**.

**The data assumption.** Pool fidelity, meaning a reference classifier's recall on pool rows relative to its recall on real rows, is lowest for **kimi-k3 (0.46), qwen3.8-max (0.51) and claude-opus-4-8 (0.54)** ([`ext/openpool/fidelity_by_label.csv`](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/fidelity_by_label.csv)) **[Local]**. Generation config drives much of it. Kimi's ratio is **0.81 with reasoning "high" vs 0.27 with "off"**, and the gap persists inside a single source (0.67 vs 0.24) ([`fidelity_by_reasoning.csv`](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/fidelity_by_reasoning.csv)) **[Local]**. 76.6% of the Gemini-3.7 side-file rows carry a routing-risk flag. Inside the source with the most such rows, a 3.7-vs-3.8 classifier scores AUC 0.498, against 0.582 on real train **[Local]**.

On a fixed 1.7B probe, the repo's own pool experiments found repeatable gains. Junk and off-genre cleaning gave **+0.027 BA on both seeds**, and typography/length-matched sampling gave **+0.026 [0.012, 0.039]** ([`dataq_results.csv`](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/dataq_results.csv); [`dataq4_paired.csv`](file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/dataq4_paired.csv)) **[Local]**. These are the only interventions in the whole evidence base whose measured effect clears the 0.02 floor with a CI excluding zero. Two caveats apply. They were measured at a pool→train BA near 0.49, not inside a 0.68 stack, and their creator is not documented.

## The literature transfers objectives and views, not headline accuracies

Published LLM-attribution accuracies of 96–97% do not transfer, because this competition is built from the documented failure regimes.

| Failure regime | Published evidence | Why it applies here |
|---|---|---|
| Within-family siblings | Four Qwen2.5 sizes: 59.8% 4-way ([Sun et al., arXiv 2502.12150v2](https://arxiv.org/html/2502.12150v2)) **[Claim, corroborated directionally]** | 15 of 18 labels have a same-lab sibling |
| Inheritance from synthetic data | Fine-tuning on shared synthetic data cut separability 96.5%→59.8%, and a held-out Grok was labelled ChatGPT 82.8% of the time (same paper) | The GLM↔Claude resemblance |
| Short texts | Accuracy keeps rising up to about 256 tokens (same paper) | The 10th-percentile train passage is 332 characters **[Local]** |
| Small data | Accuracy converges at about 10K samples (same paper) | About 150 host passages per class |
| Same-family confusions in a large model set | Across 55 base models, most confusions were within a family and between adjacent sizes ([Antoun et al., arXiv 2309.13322v2](https://arxiv.org/html/2309.13322v2)) **[Claim]** | Same structure as the sibling pairs above |

What does transfer is a set of mechanisms. Three bear directly on our problem.

**Content-controlled contrastive training.** In human authorship work, drawing hard negatives from the same conversation by a different author yields style representations that separate style from content better than domain or no content control ([Wegmann et al., arXiv 2204.04907](https://arxiv.org/abs/2204.04907)) **[Established for human authorship]**. DeTeCtive adds a multi-level contrastive loss over AI vs human, model family and individual model, with kNN inference. It reports 92.19 AvgRec on unseen models and 89.63 on unseen domains with training-free database expansion, and reports that dropping the multi-level term costs about 2 points ([arXiv 2410.20964](https://arxiv.org/html/2410.20964)) **[Claim; RoBERTa-base, 2023-era generators]**.

**Thin evidence flag.** No paper we found ablates "same-prompt hard negatives vs random negatives" for LLM attribution. The case rests on transposing Wegmann's "conversation control" to "prompt control".

**Topic suppression in authorship.** Domain-adversarial training and text distortion cut a 55.4-point topic-swap loss to about 4 points ([Bischoff et al., arXiv 2005.14714](https://arxiv.org/abs/2005.14714)), and POS stylometry is the least topic-susceptible feature family ([Altakrori et al., arXiv 2104.08530](https://arxiv.org/abs/2104.08530)) **[Established]**. Two counterweights apply. Word-shuffled text still carries 88.9% of 5-way chat-API attribution, so the fingerprint is largely *lexical* and wholesale content masking throws signal away ([Sun et al.](https://arxiv.org/html/2502.12150v2)) **[Claim]**. And DomainBed found tuned ERM competitive with nine domain-generalisation methods ([Gulrajani & Lopez-Paz, arXiv 2007.01434](https://arxiv.org/abs/2007.01434)) **[Established]**.

**Likelihood features.** Using a proxy LM's likelihood to attribute among many modern chat generators is weakly evidenced:

- The supporting work is mostly 2023 detection: Ghostbuster ([arXiv 2305.15047](https://arxiv.org/abs/2305.15047)), Sniffer's cross-model contrast features ([arXiv 2304.14072](https://arxiv.org/abs/2304.14072)) and Binoculars ([arXiv 2401.12070](https://arxiv.org/abs/2401.12070)).
- In few-shot attribution, LogRank trails a style embedding (pAUC 0.775 vs 0.958 at N=10) ([Soto et al., arXiv 2401.06712](https://arxiv.org/html/2401.06712)) **[Claim]**.
- A training-free white-box statistical test shows error decaying with length, but its errors concentrate in same-family pairs, and it requires each candidate's weights ([arXiv 2501.02406v6](https://arxiv.org/html/2501.02406v6)) **[Claim; numbers not verified]**.

Since 15 or 16 of the 18 labels are closed models, a true likelihood ratio is only possible for the open-weight GLM (and, in principle, Kimi) labels **[Inference]**.

Two ideas the literature kills or weakens:

- **Temperature is close to invisible.** Five temperatures are only 37.9% separable (chance is 20%) ([Sun et al.](https://arxiv.org/html/2502.12150v2)) **[Claim]**, so decoding config is not the nuisance worth modelling. The *reasoning* setting is different, as the kimi fidelity split shows **[Local]**.
- **Passive tokenizer fingerprints have no supporting paper** (thin evidence). The concrete "truncation at a token cap" trace fails its precondition here: 2,677 of 2,699 train passages end with a period ([Discussion 750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390)) **[Claim]**, and passages are length-instructed complete answers **[Official / Inference]**.

## Ten ideas ranked by evidence per runtime minute

The ranking weighs four things: strength of evidence, cost of the decisive test, private-board robustness, and the likelihood of adding errors decorrelated from dev1's. Raw upside is not the criterion, because no idea has evidence of drastic upside. Every "upside" entry below is a hypothesis. None is a score forecast.

### What each idea is and why it differs

| # | Idea | Mechanism | How it differs from dev1 and the public families | Evidence strength | Novel to competition / to ML research |
|---|---|---|---|---|---|
| 1 | Config-aware, protocol-matched data rebuild for an owned member | Pool v14 with these filters: near-train-prompt rows removed; duplicates removed; kimi reasoning=high only; Vertex-only Gemini-3.7; qwen "low" dropped; typography/length match weights; whole sources down-weighted by leave-one-source-out (LOSO) ΔBA. Optionally (label × reasoning-setting) heads marginalised with EM test priors. Class-prior logit adjustment for pool imbalance | Public family J filters junk. dev2 trains on the raw pool with no prompt filter or class weights. Conditioning on config and LOSO causal source ranking are not seen publicly **[Inference]** | **Moderate**: +0.027 cleaning and +0.026 typography matching on a 1.7B probe with CIs excluding 0; kimi fidelity 0.81 vs 0.27 **[Local]**; public diminishing returns **[Claim]** | Partly / No |
| 2 | Same-prompt, family-hierarchical contrastive objective | CE plus multi-level supervised contrastive loss (family, then model). Positives: same model, different prompt. Hard negatives: other models, same prompt. Batches drawn from train's grid and the pool's prompt groups (13,021 prompts, 592 answered by all 18 **[Local]**). kNN/prototype head alongside softmax | Public stacks fight content at *test time* (centring, twin and Gaussian heads), which needs dense test groups. This removes content in *training* and needs no test layout | **Thin–moderate**: established for human authorship (Wegmann); DeTeCtive hierarchy **[Claim]**; no LLM-attribution ablation exists | Likely yes / Partly (same-prompt ablation unpublished) |
| 3 | Mix-invariant decision layer plus sparse-layout harness | Replace uniform-target Sinkhorn with per-member training-prior removal and a bias vector fit on prompt-grouped OOF to maximise BA. Transduction only as a "cavity" prior (prompt effect estimated from other group members) validated at k=2 in an embedded 4,000-row batch | dev1 moved *toward* a uniform target, and public tuning is LB-driven. This targets BA's mix-invariant optimum | **Moderate**: official unequal mix; the k-table above **[Local]**; −0.07 Dirichlet simulation **[Claim]**; BA decision rule **[Inference]** | Partly / No |
| 4 | Fresh sentinel set to profile dev1's errors | 100–150 new host-style prompts × 18 labels, generated under best-guess host protocol, checked to share no prompts with train or pool. Run dev1 unchanged and save its probabilities | dev1 has no OOF. Public teams pick weights on the 40% LB | **No external evidence**; the rules permit it **[Official]** | Unknown / No |
| 5 | Sibling-pair specialists trained on same-prompt pairs | Binary or 3-way heads for GLM↔Claude, Gemini 3.6/3.7/3.8 and GPT-5.6 luna/sol/terra, trained on content-matched pairs. Rescore the top-2 only when it falls in a known pair | Public gates are lab-level and trained on 18-way data. These are pairwise and content-matched | **Mixed**: Claude gate helped; Gemini gates flat **[Claim]**; SemEval cascade ([arXiv 2404.14183](https://arxiv.org/html/2404.14183)) | Partly / No |
| 6 | Vendor proxy panel with cross-model contrast features | 4–6 small open models, one per vendor where available. Features: per-token log-prob, rank and entropy quantiles, plus pairwise "share of tokens where A beats B" (Sniffer style). Stacked | Public tried a *single* proxy's log-likelihood (−0.002 [757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199)). The contrast between vendor models is untried | **Weak**: detection-era evidence; public null result | Moderate / Low |
| 7 | Content-invariant views *not already in dev1* | POS-tag n-grams, frozen LUAR style embeddings, per-class zstd/PPM cross-entropy | dev1 already has function-word, "shape" and "distort" views **[Local]**, so only POS, LUAR and compression are new | **Weak–moderate**: POS lineage signal ([arXiv 2502.06659](https://arxiv.org/abs/2502.06659)); compressor baseline 0.533 vs SVM 0.545 at PAN 2019 ([overview](https://ceur-ws.org/Vol-2380/paper_264.pdf)) | Likely / No |
| 8 | **Unconventional**: GLM-5.3-Flash's own likelihood, streamed layer by layer | Score the first 128–256 tokens of every test passage under the real open weights, streaming each layer once over the whole batch. Features normalised against a small reference model | Reached only the question stage publicly. It is the one signal that does not come from text surface | **Thin** (no direct evidence; inheritance risk) | Yes / No (engineering novelty only) |
| 9 | Distilled student to buy runtime | Distil *owned* members, plus dev1 on the sentinel only, into one LoRA student. Use the freed time for #8 or full-coverage ATLAS | Distilling dev1 itself over the pool would cost about 400 T4-hours: 200k rows at roughly 8 h per 4,000 **[Inference]** | Enabler only | No / No |
| 10 | Tokenizer re-segmentation features | Chars per token and the multi-piece-word rate under each public vendor tokenizer | Untried. Probably redundant with char n-grams | **None found** (thin) | Yes / Yes, but low prior |

### What each costs and whether it fits the hidden rerun

All budget numbers are assumptions unless they are cited. The T4 inference anchor is a public claim that a Qwen3-14B processes about 4,000 rows in 17–20 minutes at about 0.25 s per passage ([Discussion 757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199)). The TPU anchor is a public claim that a v5e-8 trains a Qwen3-8B LoRA about 14× faster than 2×T4, with about 20 h of quota a week and queues of up to about 23 h ([Discussion 754657](https://www.kaggle.com/competitions/name-that-llm/discussion/754657)). TPU FLOP figures assume 8 × 197 TFLOPs bf16 peak at 30–40% utilisation, about 150 tokens per passage and 1.5× padding.

| # | Upside hypothesis (not a forecast) | Data and artifacts needed | Training budget | Inference budget (about 4,000 passages, 2×T4) | Hidden-runtime fit |
|---|---|---|---|---|---|
| 1 | Lifts recall on the weak, low-fidelity classes (kimi, qwen, opus-4-8, GLM), which BA weights equally; errors decorrelated from the public pool-trained stack | Pool v14 parquet (not local); flag columns; frozen splits | 1.7B ablations: minutes per run on v5e-8, about 1–2 h on one T4. 8B 24-layer member: about 1–3 TPU-h per epoch over ≤216k rows | About 10–15 min per adapter | Fits only if it replaces a member (e.g. GSQ) |
| 2 | Better sibling separation and robustness to new prompts; a kNN head errs differently from softmax | Same as #1; prompt-group batch sampler | About 1.3–2× the CE step cost | Same as #1, plus minutes for the kNN lookup | Same as #1 |
| 3 | Protects the private score against an unequal mix; may recover losses from the 0.50 mix | train.csv; dev1's `balance_logits` code; OOF from any owned member | CPU, under 1 day | Seconds | Yes |
| 4 | Turns blend weights from LB guesses into measured choices; reveals dev1's per-class weaknesses | API access; generation spend well under the $5k reasonableness line (assumption: tens to low hundreds of USD for 1,800–2,700 short passages, to be checked against actual pricing) | None | 2–4 GPU-h interactive for dev1 runs on the sentinel | No effect on submission |
| 5 | Reduces the GLM↔Claude and Gemini-Flash confusions specifically | Same-prompt pairs from train and pool | Minutes to 1 h per head on frozen features | Under 10 min | Yes |
| 6 | Adds a non-lexical channel on pairs that differ in post-training | Proxy weights as a Kaggle dataset | Stacker only (CPU) | About 1M tokens per proxy: minutes each; under 30 min for the panel | Yes, tight alongside dev1 |
| 7 | Low error correlation with LoRA read-outs | spaCy small model and LUAR weights as datasets | CPU minutes | CPU minutes | Yes |
| 8 | If GLM likelihood separates GLM from Claude, the 2 GLM classes (11% of BA weight) plus their Claude spill-over improve. Ceiling arithmetic only: raising both GLM recalls by 0.4 adds 2×0.4/18 ≈ 0.044 BA. dev1's actual GLM recall is unknown | A quantized GLM-5.3-Flash on Kaggle (unverified); a layer-major runner that works on sm_75 | Pilot: one 3–6 h interactive session | About 0.5–1 h compute per GPU plus 10–40 min I/O (inference) | Only by replacing GSQ and the native expert, or after #9 |
| 9 | Recovers 1–3 h of the budget | Owned teacher outputs | TPU hours | Below the teacher's | Enabler |
| 10 | Probably none | Tokenizer files | CPU | CPU | Yes |

### Where each idea breaks and how to kill it fast

| # | Main risk | Smallest decisive experiment | Go / no-go |
|---|---|---|---|
| 1 | Gains measured near 0.49 BA vanish inside a 0.68 stack; the host's config is unknown | 2×2 factorial in Experiment 2 (the data arm) | Go if ΔBA ≥ +0.02 with paired CI > 0 at k=2 on 2 seeds *and* weak-class recall rises; no-go if within ±0.01 |
| 2 | Content control strips lexical fingerprint (word-shuffle keeps 88.9%); contrastive loss trades some classes away ([Contra-X, arXiv 2209.11887](https://arxiv.org/abs/2209.11887)) | Experiment 2 (the objective arm) | Go if ΔBA ≥ +0.02 vs CE, or blend(CE+SupCon) beats a blend of two CE seeds by ≥ +0.02, with the same-prompt pair test improving on ≥ 3 of 5 hard pairs |
| 3 | Sinkhorn may be correcting real member bias that a train-OOF bias vector cannot see | Experiment 1 | Adopt for one final pick if worst-decile Dirichlet BA rises ≥ 0.01 while mean k=2 BA drops ≤ 0.005 |
| 4 | Our protocol ≠ the host's, so the sentinel mis-states dev1's errors | Experiment 3 (validity gate) | Use only if the sentinel-vs-train adversarial AUC on typography/length features is ≤ 0.73, i.e. no less train-like than the cleaned pool's 0.726 |
| 5 | GLM's style really is Claude-like (an inheritance ceiling) | V1 pair-restricted test on 300 rows per pair | Go if pair accuracy rises ≥ 0.03 and overall paired CI > 0 |
| 6 | Proxies share pretraining, not the closed vendors' post-training | Stack 30–60 features on OOF under V1 at k=2 | Kill if gain < 0.009 (1 SE) and error correlation is high |
| 7 | Redundant with dev1's distort and function-word views | Same as #6 | Same as #6 |
| 8 | Inheritance; API serving ≠ open weights; I/O or kernels blow the 9 h | 128-token pilot on 1,050 train rows | See the unconventional section |
| 9 | The student loses the teacher's diversity | Student vs teachers on V1 | Go only if within 0.005 at ≤ 30% of runtime *and* #8 has passed |
| 10 | Redundant | Add to char LR on V1 | Kill if < 0.005 |

Several ideas are rejected outright. Scaling Hungarian assignment or centring fails because its gain is conditional on a dense layout that the evidence says is absent: k=2 numbers above **[Local]**, plus public LB losses. Pseudo-labelling the test set was negative in a public simulation ([757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199)) **[Claim]**. DNA-GPT-style regeneration and generating prompt-matched data *with* the open label models fail on closed models, no internet, and models of 320B to 2.8T parameters. Bigger backbones trained on the same pool failed per the family table. Hard typography vetoes lost in forks.

## Three experiments, in order, with kill switches

All three share one frozen yardstick, defined before any model is trained.

| Component | Specification |
|---|---|
| **F1** train folds | GroupKFold(5) on `prompt_id`, with 3 seeds of prompt-to-fold shuffling. Write `id, prompt_id, fold, seed` to CSV and record its SHA-256. Every fold is automatically class-balanced because each prompt holds every label **[Local]** |
| Sparse layout | Score held-out rows after subsampling k answers per prompt, k ∈ {1, 2, 3, 18}, with 30 draws. **k=2 is primary, and k=1 must not regress.** k=18 is reported only as an upper bound |
| Embedded batch | Before any transductive step, mix the k-subsampled held-out rows into a simulated 4,000-row batch. The filler is pool rows from prompts held out of training. This follows the untested proposal in [Discussion 753053](https://www.kaggle.com/competitions/name-that-llm/discussion/753053) |
| Mix stress | Dirichlet(α=2) and α=1 class proportions, 20 draws each. Report the mean and the worst decile |
| **V2** pool→train | Drop pool rows with `flag_near_train_prompt`, `flag_train_prompt_regen`, or `max_sim_train_prompt` > 0.3, and drop rows that `also_in` marks as duplicates. Evaluate on all 2,699 train rows, which are prompt-disjoint by construction. Pool model selection groups by `prompt_canon` × source |
| Strata | BA by word-length quartile (edges 64/101/157 words); family-level vs within-family accuracy; truncation at 40/80/120 words |
| Decision rule | Paired prompt-bootstrap delta. "Go" means the point estimate is ≥ +0.02 *and* the 95% CI excludes 0. Results of +0.01 to +0.02 are "promising, replicate", and anything less is noise |

### 1. Freeze the yardstick and re-audit dev1's decision layer (CPU, about 1 day)

This experiment separates *test-layout and prior artefacts* from generator signal, and it is the only one that acts on dev1 immediately.

Inputs are OOF probabilities on F1 from three owned models: the existing TF-IDF LR probe, a typography-only LR, and later any LoRA member. The arms are:

- plain argmax (the control)
- dev1's `balance_logits` copied verbatim, at target mix 0.35 and 0.50 with strength 0.30
- strength 0 (no balancing)
- 1–3 Sinkhorn iterations
- a per-class bias vector fit on the other folds to maximise BA
- centring vs cavity-prior in the embedded batch

There are two sanity controls. The harness must reproduce the measured k=2 centring loss (about −0.066) and the k=18 Hungarian gain (about +0.11). The random-split BA serves as a content-dependence meter.

Success is measurable. If any arm beats dev1's 0.50 setting on worst-decile Dirichlet BA by ≥ 0.01 while losing ≤ 0.005 of mean k=2 BA, build a "robust dev1" with that setting and designate it as one of the two final selections. Spend at most two LB submissions on it, and treat only a move of ≥ 0.02 as informative. If no arm wins, keep dev1 as is and record that its decision layer passed the stress test.

The caveat is that post-processing effects depend on how confident the members are. A weak TF-IDF probe is a stand-in, so repeat the audit once a LoRA member's OOF exists.

### 2. Separate generator signal from content with a 2×2 on a small trunk (TPU, about 1 week of quota)

This is the causal test and the first diversification test. Use Qwen3-1.7B with LoRA on 45k pool passages. That matches the curator's protocol, so results line up with the `dataq` tables. Train with 3 seeds and evaluate on V2 at k=1 and k=2.

| Factor | Arm A | Arm B |
|---|---|---|
| Data | Cleaned pool: junk and off-genre removed, near-train-prompt rows removed, deduplicated | Arm A plus the config filters (kimi high only, Vertex-only Gemini-3.7, qwen "low" dropped), match weights, and LOSO-flagged sources down-weighted |
| Objective | Cross-entropy with prior logit adjustment | Arm A's objective plus multi-level SupCon with same-prompt hard negatives (batches of P prompts × M labels from `prompt_canon` groups with ≥ 4 labels) and a family-level term |

Train's grid supports a causal battery that holds content exactly fixed:

- **Same-prompt two-alternative test** on the five hardest sibling pairs: gemini-3.7/3.8, luna/sol, gemini-3.6/3.7, glm-5.3/glm-5.3-flash, and GLM vs Claude.
- **Masking ladder at test time only.** Content words become `X`, and typography is normalised. Retaining ≥ 85% of grouped BA counts as generator-driven **[Local criterion]**.
- **Typography-swap flip rate.** Swap curly and straight quotes and dashes, then count how many predictions change.

The control for diversity is **error diversity between two seeds of the same arm**. A new objective counts as diversifying only if its double-fault rate with CE is clearly below the CE-vs-CE-seed double-fault rate.

Go rules:

- Data arm: go if ΔBA ≥ +0.02 with paired CI > 0 on at least 2 of 3 seeds, *and* recall rises on kimi, qwen, opus-4-8 or GLM.
- Objective arm: go if ΔBA ≥ +0.02, *or* blend(CE, SupCon) beats blend(CE seed 1, CE seed 2) by ≥ +0.02. In both cases the two-alternative test must improve on ≥ 3 of 5 pairs.

A passing combination is promoted to the Qwen3-8B 24-layer trunk through the dev2 path, with class weights turned on. That promotion costs about 1–3 TPU-h per epoch (assumption). The pool parquet is not local and must be downloaded first.

### 3. Profile dev1's errors on a fresh sentinel and test whether owned members diversify them (API spend plus 2–4 GPU-h)

Diversifying dev1 is impossible to measure without dev1's predictions on rows it has never seen. This experiment supplies them.

**Build the sentinel.** Generate 100–150 new prompts in the host's apparent style. A public reconstruction suggests mostly "explain X vividly" ([Discussion 749455](https://www.kaggle.com/competitions/name-that-llm/discussion/749455)) **[Claim]**. Request 40–250 word single paragraphs, no system prompt, Vertex-routed Gemini-3.7 and kimi reasoning on. The kimi setting is contested ([752766](https://www.kaggle.com/competitions/name-that-llm/discussion/752766) vs [751871](https://www.kaggle.com/competitions/name-that-llm/discussion/751871)), so log it as a factor. Reject any prompt whose cosine similarity to a train or pool prompt exceeds 0.3.

**Validity gate.** The sentinel-vs-train adversarial AUC on typography and length features must be ≤ 0.73, i.e. no less train-like than the cleaned pool **[Local benchmark]**.

**Run dev1.** Run dev1 unchanged, except for saving its final probability matrix, on three k=2 sparse batches. Score the Experiment 2 winner and the proxy-panel features (#6) on the same rows.

**Fit and test the blend.** Fit the geometric blend weight on one half of the sentinel prompts and test on the other, frozen half. Controls are dev1 alone, and dev1 plus the *CE* arm at the same weight; the second checks that the objective, rather than any extra member, adds value.

Go if the blended paired ΔBA is ≥ +0.02 with CI > 0 on the held-out half, no class's recall drops by more than 0.10, and the added member's projected runtime on 4,000 rows is ≤ 45 minutes or a named dev1 member is removed. Only then spend one or two LB submissions, read with an SE of about ±0.014.

## The long shot: score passages under GLM's own weights

The worst family-level confusion has one property that text-surface methods cannot use: GLM's weights are open. Secondary reports put GLM-5.3 at about 753B total / 40B active parameters and GLM-5.3-Flash at about 320B total / 18B active, released around 2026-08-28 under a custom license ([HF zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3); [The New Stack](https://thenewstack.io/zai-glm-weights-license/)) **[Claim; model card not fetched, figures conflict across reports]**. A participant has already asked about IQ3_XXS GGUF segments. The replies say per-passage memory mapping cannot fit in 9 h and suggest layer-streaming over the whole test set with 128–256 tokens ([Discussion 757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199)) **[Claim]**. Kimi K3, reported at about 1.5 TB, is out of reach **[Claim / Inference]**.

**Feasibility check [Inference].**

| Constraint | Estimate | Problem it creates |
|---|---|---|
| Weight size | GLM-5.3-Flash at 2–3 bits per weight is about 85–120 GB | Cannot fit in 2×16 GB VRAM plus about 30 GB RAM, so each layer must be loaded once and run over all about 1M test tokens before moving on |
| Activations between layers | About 8–12 GB | Must sit in RAM or on local disk |
| Compute | About 3.6e16 FLOPs, i.e. 0.5–1 h per T4 at a dequantisation-adjusted 5–10 TFLOPs | Turing has no bf16 or FP8 tensor cores, so i-quant or FP8 weights must be dequantised to fp16 |
| I/O | Reading about 100 GB takes 10–40 min | One participant logged a model load of 1,206 s vs 22 s on slow `/kaggle/input` sessions ([Discussion 751268](https://www.kaggle.com/competitions/name-that-llm/discussion/751268)) **[Claim]** |
| Prerequisites | Unverified | Whether a quantized Flash exists as a Kaggle dataset, Kaggle's dataset size limit, and license terms for competition use |

The verdict: **possible, unproven, and affordable only by evicting expensive dev1 members**, namely GSQ and possibly the native expert.

The kill test is small. In one interactive session, score the 150 glm-5.3, 150 glm-5.3-flash and 750 Claude train rows at 128 tokens, with a small reference model for normalisation. **Go** only if all three conditions hold:

- the GLM-vs-Claude AUC from likelihood features beats the owned text model's pairwise OOF AUC by ≥ 0.05
- glm-5.3 vs glm-5.3-flash AUC exceeds 0.6
- the measured runtime on one tenth of a 4,000-row batch projects to ≤ 4 h

The strongest reason it may fail is inheritance. If GLM learned from Claude-like outputs, its own likelihood may rate Claude text as highly as its own. Fine-tuning on shared synthetic data drives separability toward 59.8% ([Sun et al.](https://arxiv.org/html/2502.12150v2)), and teacher identification across "heavily entangled" lineages is hard ([arXiv 2607.09692](https://arxiv.org/abs/2607.09692)) **[Claim]**. A second risk is that host passages may be post-reasoning answers from an API, and that GLM reportedly refuses `reasoning.enabled=false` on OpenRouter ([751871](https://www.kaggle.com/competitions/name-that-llm/discussion/751871)) **[Claim]**. Scoring the answer text without its reasoning context may then mismeasure GLM's real conditional distribution.

## How this plan fails, and what would change it

**Idea #1 (data rebuild).** It may fail because its evidence comes from a 1.7B probe at about 0.49 BA, and public pool gains shrank from +0.028 to +0.002 as pools grew ([752357](https://www.kaggle.com/competitions/name-that-llm/discussion/752357)). The stack may already have absorbed most of the clean-data benefit. Config filtering also shrinks some classes sharply: kimi "high" is only 2,673 rows **[Local]**. Moving gemini-3.7 to Vertex-only routing raised its recall but dropped gemini-3.6 from 0.63 to 0.47 in the curator's figure **[Local]**, so a per-class gain can be a zero-sum shuffle within the Gemini trio.

**Idea #2 (contrastive objective).** It may fail because the generator fingerprint is mostly lexical, so pushing away same-prompt neighbours can also push away the fingerprint words those neighbours share. DomainBed-style evidence says careful ERM is hard to beat **[Established]**.

**Idea #3 (decision layer).** It may fail because Sinkhorn's public LB gains, though small, are real evidence that it corrects member bias, and a train-OOF bias vector cannot be fit for dev1's members without their OOF.

**The harness.** It may not track the private board. The 0.91 correlation at k=2 comes from one participant over 15 submissions **[Claim, thin]**, and grouped CV reportedly "stopped predicting LB differences" for stacked members. The split method, by prompt or by passage, is also undisclosed, so the private part *could* contain denser prompt groups than the public part.

**The sentinel.** It inherits the same protocol uncertainty that limits the pool, because the hosts never disclosed system prompt, temperature, reasoning setting or route ([750408](https://www.kaggle.com/competitions/name-that-llm/discussion/750408)) **[Claim]**.

**Runtime.** It is a silent killer throughout. dev1's margin under 9 h is unknown, and slow mounts have timed out forks.

Specific evidence would change the recommendation:

- **Hosts disclose dense prompt groups or near-equal class counts.** Test-layout tricks and uniform balancing come back, and Experiment 1 drops in priority.
- **Experiment 2 returns nulls on both factors.** That would mean small-trunk text views are saturated, so the remaining levers are protocol-matched *new data* within the $5k cap and the GLM pilot.
- **The GLM pilot clears its AUC bar.** It becomes the top priority, because it is the only idea aimed at a named, large confusion with a signal the stack does not contain.
- **The sentinel shows dev1 already strong on GLM↔Claude and the Gemini trio.** Effort should shift to whichever classes the sentinel shows are weakest.
- **Top-team write-ups appear.** Re-rank everything against what explains 0.70–0.77.

## Source register and evidence labels

| Source | Type | Label | Supports | Thinness note |
|---|---|---|---|---|
| [Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation), [Code Requirements](https://www.kaggle.com/competitions/name-that-llm/overview/code-requirements), [Data](https://www.kaggle.com/competitions/name-that-llm/data), [Rules](https://www.kaggle.com/competitions/name-that-llm/rules), [Timeline](https://www.kaggle.com/competitions/name-that-llm/overview/timeline) | Kaggle pages | Official | Metric, unequal mix, 9 h, ~4,000 rows, disjoint prompts, external data | Deadline Jan 13 vs Jan 14 conflict |
| [Discussion 752084](https://www.kaggle.com/competitions/name-that-llm/discussion/752084) | Admin replies | Official | Transduction allowed; data spend about $5k; community data OK | — |
| [Leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard) | Snapshot 2026-11-12 | Official | 0.768 top, 0.684 plateau, 0.682 rank band | Moves daily |
| [alexchilton 0.684](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed), [prvsiyan](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author), [nishantdahal](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool) | Public notebooks | Claim | Public stack contents, k=2 correlation, failures | Authors' own numbers, single submissions |
| Discussions [750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390), [750213](https://www.kaggle.com/competitions/name-that-llm/discussion/750213), [752365](https://www.kaggle.com/competitions/name-that-llm/discussion/752365), [753053](https://www.kaggle.com/competitions/name-that-llm/discussion/753053), [753177](https://www.kaggle.com/competitions/name-that-llm/discussion/753177) | Participant posts | Claim | Noise SEs, sparse-layout losses, Sinkhorn simulation, embedded validation | 753053 untested in a submission |
| Discussions [752766](https://www.kaggle.com/competitions/name-that-llm/discussion/752766), [751871](https://www.kaggle.com/competitions/name-that-llm/discussion/751871), [751370](https://www.kaggle.com/competitions/name-that-llm/discussion/751370), [755314](https://www.kaggle.com/competitions/name-that-llm/discussion/755314), [757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199) | Participant posts | Claim | Kimi config, Gemini routing, GLM↔Claude, GLM GGUF question | Kimi claims conflict |
| `ext/openpool/*.csv`, `gemini37_unverified.parquet`, `data/train.csv`, `k/dev1`, `k/dev2`, `src/` | Repo files plus read-only diagnostics | Local | Fidelity, config split, data ablations, k-table, confusions, dev1/dev2 audit | The pool tables' creator and reference classifier are undocumented |
| [Sun et al., arXiv 2502.12150v2](https://arxiv.org/html/2502.12150v2) | ICML 2025 paper | Claim (core finding corroborated) | Lexical signal, sibling 59.8%, SFT inheritance, temperature 37.9% | Internal table/text inconsistencies |
| [Wegmann et al., arXiv 2204.04907](https://arxiv.org/abs/2204.04907) | Paper | Established (human AA) | Content-controlled negatives | Not tested on LLMs |
| [DeTeCtive, arXiv 2410.20964](https://arxiv.org/html/2410.20964) | NeurIPS 2024 | Claim | Multi-level contrastive plus kNN | RoBERTa-base, 2023-era generators |
| [Bischoff et al., arXiv 2005.14714](https://arxiv.org/abs/2005.14714); [Altakrori et al., arXiv 2104.08530](https://arxiv.org/abs/2104.08530) | Papers | Established (human AA) | Topic suppression, POS robustness | Human authors only |
| [Gulrajani & Lopez-Paz, arXiv 2007.01434](https://arxiv.org/abs/2007.01434) | Paper | Established | ERM is hard to beat | Not text attribution |
| [Soto et al., arXiv 2401.06712](https://arxiv.org/html/2401.06712) | ICLR 2024 | Claim | Style embeddings beat LogRank few-shot | About 128-token documents |
| [Ghostbuster 2305.15047](https://arxiv.org/abs/2305.15047); [Sniffer 2304.14072](https://arxiv.org/abs/2304.14072); [Binoculars 2401.12070](https://arxiv.org/abs/2401.12070) | Papers | Claim | Proxy-likelihood features | Detection-only, older; abstracts only |
| [arXiv 2501.02406v6](https://arxiv.org/html/2501.02406v6) | Preprint | Claim | White-box likelihood attribution | Experiments section not verified |
| [Wadhwa et al., arXiv 2502.06659](https://arxiv.org/abs/2502.06659); [arXiv 2607.09692](https://arxiv.org/abs/2607.09692) | Papers | Claim | POS lineage; entangled lineages | Small students; abstract only for 2607 |
| [Antoun et al., arXiv 2309.13322v2](https://arxiv.org/html/2309.13322v2) | Paper | Claim | Same-family confusions | Base models, 2023 |
| [PAN 2019 overview](https://ceur-ws.org/Vol-2380/paper_264.pdf) | Shared-task report | Established | Compressor baseline; the value of diversity | Fanfiction, human authors |
| [HF zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3); [The New Stack](https://thenewstack.io/zai-glm-weights-license/) | Model hub and news | Claim | GLM open weights and sizes | Not fetched; secondary reports conflict |
| [ATLAS, arXiv 2610.19583](https://arxiv.org/abs/2610.19583) | Preprint | Claim | Source-conditioned likelihood (already in dev1) | — |

Two kinds of novelty need to be kept apart. **Novel to ML research:** only the same-prompt hard-negative ablation for LLM attribution (unpublished as far as we found), passive tokenizer attribution (unpublished and low prior), and the layer-major streaming *engineering* for a frontier MoE on T4 qualify. Everything else is established technique. **Novel to this competition:**

- Ideas #2, #6, #7 (POS and LUAR only) and #8 are probably unexploited publicly.
- Ideas #1, #3 and #5 are refinements of public families.
- Whether private top teams already use any of them is unknowable.

## Conclusion

Our 0.682 is not a position. It is a timestamp of someone else's notebook, and the only change we made to it adds private-board risk without measurable public gain. The research shifts the question from "which member to add" to "which measurements we lack". There are three: dev1's errors on unseen rows, a yardstick that reproduces the sparse, unequally mixed test, and a causal test of whether a model is reading the generator or the prompt. All three can be built in under two weeks on free compute. Each one also guards against the failure mode behind most public plateaus, which is adopting changes because of noise-level LB moves.

The deeper implication is that the 0.08 gap to the top is probably not closed by any single representational trick. The evidence points to compounding: config-matched data that lifts the low-fidelity classes, an objective that learns generator identity without leaning on prompt content, and a decision layer that respects BA's indifference to the test mix. If one genuinely different signal exists in this problem, it is GLM's own likelihood. That is a long shot whose kill test costs one afternoon, so it is worth running early.
