# REPORT: Name That LLM, where the next gains are likely to come from (decision report, deep meta-research)

Date: 2026-11-12. Method: `meta-research` v2.14.7 at deep depth. The plan is in `plan.md`, 79 per-source files are in `sources/`, the index is `sources.csv`, atomic theses are in `findings/`, and the update protocol is in `refresh_targets.md`. This file is the skill's final "decision" report.

**Evidence labels used throughout:**
- **FACT:** an official page, platform data, a verified model card, or code read directly.
- **SOURCE CLAIM:** what a paper or participant reports, not reproduced here.
- **INFERENCE:** my reasoning from the evidence.
- **PROPOSAL:** an untested plan.

Every `[NN]` maps to `sources/NN_*.md`, which holds short verbatim quotes. A thesis backed by fewer than 3 independent, differently-typed sources is marked **insufficient evidence**.

## 0. Executive recommendation

1. **The 0.682 plateau belongs to the public family, not to the problem.**
   - FACT: 72 teams are at 0.700 or above and the leader is at 0.768 [05]. No listed public notebook exceeds 0.684 [10].
   - FACT: dev1 is a mixing-only fork of a public notebook [06].
   - SOURCE CLAIM, triangulated: the measured lever is training data (coverage, generation-config match, label cleanliness), not more members [29][31][08][26]. See F1.
2. **Part of dev1's score rests on test-structure assumptions the official page contradicts.**
   - FACT: the official page says per-model test counts are unequal and the public/private mix differs [01].
   - SOURCE CLAIM: prompt-group tricks gained about 0 on the LB for several teams [21][23]. Simulation and a local replication show they vanish with 1–2 answers per prompt [22][09].
   - SOURCE CLAIM: fixed Sinkhorn balancing can lose several points under uneven mixes [24].
   - Fixing the decision layer is cheap private-split insurance (F2, F3).
3. **Ranked bets:**
   - (1) fidelity-calibrated, prompt-panel training data;
   - (2) a train-time content-controlled member built for error diversity;
   - (3) a layout-agnostic decision layer;
   - (4) generation config as latent sub-classes;
   - (5) unconventional: exact-family likelihood from open-weight label models, staged Qwen3.8-27B first, then GLM-5.3-Flash.
4. **First three experiments:**
   - E1: frozen folds plus a thin-prompt and uneven-prior harness; pick the decision rule by worst-case BA.
   - E2: a 2×2 data-curation × capacity ablation of the dev2 recipe, scored on all 2,699 official rows.
   - E3: a prompt-panel contrastive member vs a seed-matched control, judged by blend gain and error disagreement.

## 1. Baseline audit

### 1.1 Competition facts (FACT)

| Item | Status | Evidence |
|---|---|---|
| Metric | Balanced accuracy over 18 labels (macro recall), scores truncated to 3 decimals | [01][03] |
| Test size and split | ~4,000 hidden rows; public LB 40%, private 60% | [01][03] |
| Test class mix | Not equal per model; public and private mixes differ | [01] |
| Test prompts | Disjoint from train; no `prompt_id` in test | [02] |
| Passage length | Varies randomly between 40 and 250 words | [02] |
| Train | 2,699 rows: 149 prompts × 18 + 1 prompt × 17; each model answers each prompt once | [02][09] |
| Formatting | "Only light formatting cleanup"; exact procedure undocumented | [02] |
| Runtime | CPU or GPU notebook ≤ 9 h, internet off, notebook-only, `submission.parquet`; 5 submissions/day, 2 final picks | [03][04] |
| Deadlines | Final 2027-01-13; team merger and entry 2027-01-06 | [03] |
| External data | Allowed if public or "reasonable". Host: self-generated data up to about $5k needn't be published; Benchmarks quota not for training; public adapters OK | [04][20] |
| Test-time relationships | Allowed by the host | [20] |
| Information at inference | Only `id` and `text` | [01][02] |

### 1.2 Unknowns

| Unknown | Why it matters | Best available evidence |
|---|---|---|
| Answers per test prompt; whether public/private is split by prompt | Decides whether group tricks help or hurt | No host answer. LB behaviour fits few answers per prompt [21][23] (INFERENCE) |
| Official generation config (reasoning, temperature, route, dates) | Decides how to generate or curate data | Undocumented [02]. Provider defaults: GLM forced thinking [99]; Kimi effort default max [98]; Gemini thinking levels [95] |
| Whether host gemini-3.7 labels are true 3.7 | The API routes 3.7 to 3.8 [95] | Host labels look distinct from 3.8 [27][09] |
| Hidden-rerun hardware, timing parity | Late stages may silently fall back [33] | Unanswered [33] |
| dev1's hidden-run wall time | Bounds what can be added | Unknown; the local run was the 3-row preview only [06] |
| Whether TPU is allowed at submission | Inference planning | The page names CPU and GPU only [03] |

### 1.3 Team code state
- **dev1 (deployed, scored 0.682):** lightningv08's public 0.681 notebook with GSQ share at 12% and Sinkhorn target mix at 0.50. Nothing in it was retrained by the team.
  - Members: TF-IDF 4-view SVC; LoRA24 readouts (Qwen3-14B/8B, Llama-3.1-8B); two-stage Qwen3-14B and Qwen2.5-14B; encoders; a stylegram stack; ATLAS; a Qwen3.8-27B GSQ3 classifier; a Qwen3-Embedding-8B 5-fold expert at 40%.
  - Test-time steps: Gaussian prompt-group heads, capped test clustering, centring, Sinkhorn, typography guards [06].
  - Locally only the 3-row preview ran, with weights not loaded [06].
- **dev2 / src (smoke-only, unvalidated):** Qwen3-8B-base truncated to 24 layers, LoRA r16, 7-layer pooling, trained on the pool only. The smoke config runs 12 steps per bucket and evaluates on 256 rows. The src copy adds class weights. `infer_t4.py` is untested and `pe.py` is not runnable here [07].
  - **dev2 state: UNKNOWN** (no end time, no outputs, CLI blocked).
- **Score:** 0.682, rank 534 of 5,607 [05]. The public-LB standard error is about 0.011 (INFERENCE [03]).

### 1.4 Local diagnostics (FACT on this data; weak model at 0.35 BA) [09]
- Prompt-grouped BA is 0.351 vs 0.145 with random folds [34].
- Length alone scores 0.107, punctuation alone 0.234, and topic-masked character n-grams 0.307.
- 52% of errors are within the same lab. glm-5.3 → claude-opus-5 accounts for 38 of 150 glm-5.3 rows.
- Thinning prompts with oracle groups: Hungarian matching gains +0.089 at 18 answers per prompt but only +0.003 at 2. Full centring falls to 0.056 at 1 answer per prompt.
- No cross-prompt near-duplicates (maximum cosine 0.225).

## 2. Public families and where each breaks

| Family | What it assumes | Failure / ceiling | Class of limit |
|---|---|---|---|
| F1 stylometry [10][06] | Surface n-grams transfer across prompts | Weak alone (0.32–0.45); TF-IDF shake-up in DAIGT [91] | representation |
| F2 truncated LoRA, multi-depth, two-stage [23][30] | Pool ≈ official data | Pool noise [26][08]; correlated members [31] | data / validation |
| F3 Qwen3-Embedding expert, prompt-centred [06] | Centring helps on test | Hurts with thin prompts [25][09] | validation mismatch |
| F4 large classifier (Qwen3.8-27B) [06] | Capacity helps | Steps of about 0.001, within noise [06][29] | compute |
| F5 ATLAS proxy likelihood [11] | A proxy LM can model each source | +0.002 to +0.007 [29]; proxy likelihood −0.002 [28] | representation |
| F6 transductive tricks [06] | 18 per prompt, even mix | Contradicted by [01]; about 0 on LB [21][23]; Sinkhorn risk [24] | validation mismatch |
| F7 encoders [10] | Diversity | Counted twice inside stacks [29] | — |

INFERENCE: the evidence points to data limits and validation mismatch. Compute isn't binding for training (TPU is about 14× 2×T4 [33]). Representation limits show up in the sibling corners (F4).

## 3. Hypothesis verdicts

| H | Verdict | Basis |
|---|---|---|
| H1: data over capacity | Supported, not proven (no equal-compute ablation) | F1 |
| H2: fragile structural assumptions | Supported | F2 |
| H3: complementary fingerprint signal | Under-determined | For: [40][41][93]. Against: [28][42][87] |
| H4: EM test-prior estimation | Refuted as stated | [24][22][72], F3 |

## 4. Ranked idea table

Assumptions: about 170 tokens per passage. The test is about 0.68M tokens, train about 0.46M. The capped pool is about 216k rows. A TPU v5e-8 does about 12.5k padded tokens/s for an 8B LoRA [33]. All figures are planning estimates.

| # | Idea | Mechanism | Difference from dev1 | Evidence | Upside (hypothesis) | Data / artifacts | Train budget | Inference | Risk | Smallest decisive experiment | Go / no-go | Runtime fit | Novelty (competition / ML) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Fidelity-calibrated prompt-panel data | Pool curation (remove mislabels [26], Vertex-only 3.7 [08], config gating with a train-only yardstick, coverage balance [23]); own panels where all 18 models answer each new prompt, config set per label by the same loop, served-model checks [27][95], 40–250 words, same cleanup | dev1 uses public pool states as-is | Strong for the lever (F1); weak for own generation [32] | Largest lever, since it lifts every member | Pool v14 config columns; reconstructed prompts [32]; API spend ≤ about $5k [20] (prices not verified) | 1–2 h per epoch per 8B member on TPU | No extra | Config can't be fully matched [98][99]; Kimi direction contested [26][32] | E2 curation arm; own-generation pilot of 30 prompts × 18 × 2 configs, scored for fidelity first | Curated ≥ +0.015 BA on official rows at equal row count (CI > 0) | Neutral | Not novel / not novel |
| 2 | Prompt-panel contrastive member | Cross-entropy + supervised contrastive (positives: same label, different prompt; hard negatives: same prompt, different label) + gradient-reversal prompt head + topic masking [62] | dev1 imposes content invariance at test time | Moderate in literature [60][63][62] plus [09]; competition transfer insufficient | Diverse errors; robust to test layout | Pool `prompt_canon` / `prompt_group`, or idea-1 panels | About 2–3 h TPU | About 10–30 min on T4×2 | Gains often under 1 point [66]; embeddings still carry content [63] | E3 | blend(A,B) − blend(A,A′) ≥ +0.008 and lower prompt-probe accuracy | Fits as a swap | Partial / not novel |
| 3 | Layout-agnostic decision layer | Cross-fitted offsets instead of Sinkhorn 0.50 [24]; light cavity centring [22]; no injective constraint; tune with validation rows clustered inside the test batch [25] | dev1 bets on an even mix and full groups | Strong for the risk (F2, F3) | Private-split protection | Team-owned OOF log-probs | CPU | Negligible | Could lose public points if the test is grouped | E1 | Worst case +0.01 with a loss ≤ 0.005 at 18 answers per prompt and an even mix | Trivial | Not novel / not novel |
| 4 | Config as latent sub-classes | Label × config sub-classes, unknown configs marginalised, summed to 18 at inference | dev1 treats each label as one class | Weak to moderate [08][26][95]–[99] | Better use of noisy rows for kimi, qwen and opus-4-8 | Pool `gen_reasoning` (about 54% known) | One run | Same | Thin sub-classes; noisy config labels | E2 add-on | Low-fidelity labels +0.03 mean recall, overall BA not lower | Fits | Novel as far as seen / minor |
| 5 | Exact-family likelihood (§6) | Clipped per-token log-ratio under a label model vs a reference LM | dev1 uses proxy-LM ATLAS | Thin; weights exist [81][83]; proxy version failed [28] | Moves the GLM↔Claude corner | Qwen3.8-27B; GLM-5.3-Flash IQ3_XXS (about 128 GB) [84] | TPU for train rows; streaming for GLM | Qwen: shared forward (unverified); GLM: 0.5–1.5 h (unverified) | Engineering, quantisation, served ≠ released weights [86] | Stage 1 and 2 (§6) | Qwen AUC ≥ 0.75 and stack +0.005; GLM AUC ≥ 0.70 | Stage 1 plausible, Stage 2 marginal | Novel / modest |
| 6 | Corner pairwise verifiers | Binary verifiers on prompt-matched sibling pairs | dev1 has family gates | Moderate | Gemini and GPT trios | Panels | Small | Small | A public pairwise-verifier notebook exists | Inside the E3 analysis | — | Fits | Not novel / not novel |

**Rejected:**
- Scalar proxy-LM detector statistics: −0.002 [28], weak on short texts [44], binary-only methods [49][51][52].
- DANN, IRM or GroupDRO as the main bet [66][65].
- EM test-prior estimation (F3).
- Kimi-K3 (2.78T) and full GLM-5.3 (753B) likelihood: infeasible [80][82].
- Character CNN [23].
- More same-pool backbones [31][29].
- Watermark detection: not investigated; assumed prerequisites unavailable (unverified).

## 5. First three experiments

**Shared protocol:**
- **Folds:** GroupKFold(5) on `prompt_id`, sorted, no shuffle. Save fold ids to a CSV with its sha256.
- **Pool-trained models:** score on all 2,699 official rows, after dropping rows flagged `flag_train_prompt_regen` or `flag_near_train_prompt`.
- **Statistics:** paired cluster bootstrap over the 150 prompts (2,000 resamples). Report per-class recall for the corner classes.
- **Leaderboard:** a sanity check only. A difference under 0.011 is noise.

**E1: harness and decision-layer audit (CPU)**
- Inputs: OOF from a TF-IDF member and from the E2 control.
- Grid: answers per prompt m ∈ {1, 2, 3, 6, 18} × 20 seeds; class mix Dirichlet(α) with α ∈ {1, 5, 100} × 40 pools; groups either oracle or recovered by clustering validation rows inside a synthetic 4,000-row test batch [25].
- Arms: plain argmax; Sinkhorn at 0.35 and 0.50; cross-fitted offsets; cavity centring at w ∈ {0.25, 0.5}; Hungarian.
- Go: worst case improves by at least 0.01 over dev1's rule, with a loss of at most 0.005 at m = 18 and an even mix.
- No-go: if all arms are within 0.005, keep the current rule.

**E2: data curation vs capacity, 2×2 (TPU, about 5–8 h)**
- Recipe: `src/tpu_train.py` with smoke off, 2 seeds.
- Data arms: D0 = random cap. D1 = curated at equal row count (mislabel removal, Vertex-only 3.7, config gating, typography-matched sampling, coverage balance).
- Capacity arms: C0 = Qwen3-8B truncated to 24 layers. C1 = Qwen3-14B.
- Metrics: BA on the official rows, per-class recall, lab-level vs within-lab BA, and a prompt-cluster probe.
- Go for idea 1: D1 − D0 ≥ +0.015 (CI > 0) and at least as large as C1 − C0.
- Refutes H1: C1 − C0 > D1 − D0 + 0.01.
- Add-on: one more D1 run with config sub-classes (idea 4).

**E3: contrastive member judged on diversity (TPU, about 3 h)**
- Arms: A = cross-entropy (the E2 model). A′ = A with another seed. B = A + supervised contrastive with same-prompt negatives + gradient-reversal prompt head (λ ∈ {0.1, 0.3}).
- Metrics: error disagreement and κ (B vs A, A′ vs A); blend(A,B) vs blend(A,A′); prompt-probe accuracy; corner recall.
- Go: blend gain ≥ +0.008 (CI > 0), lower prompt-probe accuracy, and no class losing more than 0.05 recall.
- No-go: if B only matches A′, the objective adds nothing beyond seed diversity.

## 6. Unconventional direction: exact-family likelihood

**What exists (FACT):**
- GLM-5.3-Flash: 321B total, MIT, "18B active", 45 layers, hidden size 4096, 288 experts with 8 per token [81].
- GLM-5.3: 753B [80]. Qwen3.8-27B: 27.8B, Apache-2.0 [83]. Kimi-K3: about 2.78T [82].
- A GLM-5.3-Flash IQ3_XXS GGUF of about 128 GB [84].
- Proxy-LM likelihood has failed here (−0.002) [28]. Scoring under the generators' own weights is untried publicly [28].

**Mechanism (PROPOSAL):** compute clipped per-token log-ratios under the label model vs a reference LM (e.g. Qwen3-8B-base), and feed the aggregates into a stacker gated to the GLM↔Claude and qwen corners.

**Feasibility:**
- **Stage 1 (Qwen3.8-27B → qwen3.8-max): plausible.** dev1 already runs a 3-bit 27B forward on a T4; whether the LM head is retained is unverified. Train-row features can be computed on TPU in bf16 (about 56 GB).
- **Stage 2 (GLM-5.3-Flash): marginal.**
  - Compute is about 2.5e16 FLOP, or about 0.5–1.5 h on two T4s.
  - Requirements: layer streaming [28]; an IQ3_XXS kernel on sm_75 (unverified); a 128 GB Kaggle dataset (size limits unverified); 9 h slack after dev1 (unknown).
  - Activations need about 5.6 GB of RAM.
- **Infeasible:** Kimi-K3 and full GLM-5.3.

**Counter-evidence:**
- Served models differ from released weights [86], and log-prob tests are fragile [87].
- Answers were generated with forced thinking and an unknown prompt [99], but we would score them unconditionally.
- Quantisation shifts the distribution [84].
- The two GLM labels share a recipe [99]. Possible GLM-from-Claude distillation cuts both ways [11][28].

**Falsification:**
- Stage 1 on the official rows: AUC ≥ 0.75 and stack gain ≥ +0.005. Otherwise stop the whole line.
- Stage 2 on a 300-row GLM/Claude subset: AUC ≥ 0.70, conditional on the existing stack's logits.

## 7. Adversarial review

1. **What would make "data first" wrong?**
   - The evidence is self-reported, plus measurements from a weak yardstick [26][29][08]. A stronger pool-trained member lowered the LB in one report [31].
   - Top teams may use bigger ensembles, or tricks that do suit a grouped test. E2 is designed to refute this.
2. **Where is the evidence thin?**
   - Own-generation benefit: participants report failures [32].
   - Kimi config direction is contested [26] vs [32] vs [98].
   - Contrastive transfer in this competition: insufficient evidence.
   - Exact-likelihood value: untested.
3. **What alternative explanation fits?**
   - The ≈0 LB gain from group tricks could come from weak base models or bad clustering.
   - The public split is only 40%, and private may have fuller prompts. E1 brackets this rather than assuming a layout.
4. **What wasn't searched?**
   - Threads beyond the 46 fetched; private top-team methods; hidden-rerun hardware and dev1's hidden time.
   - Provider prices; dataset size limits; train-prompt regeneration flags in the full pool; the dev2 outcome.

**Steel-manned counterpoints:**
- "Fork the next public notebook": forks have converged at 0.684 with correlated errors [05][10][31]. Fine as a fallback final pick only.
- "The host allows group tricks": legality isn't transfer [20][22][25][01].
- "Domain generalisation is a dead end": true as a main bet [66], so idea 2 is judged only by seed-controlled diversity.

**What would change the recommendation:**
- The host confirms full 18-model groups in test → keep group heads and demote idea 3.
- E2 shows capacity ≥ curation → prioritise capacity and readout diversity [23].
- E3 fails its seed control → drop idea 2.
- dev1's hidden time leaves under 1 h → drop Stage 2.
- Qwen Stage 1 AUC < 0.75 → abandon §6.

## 8. Triangulation summary

| Thesis | Sources | Status |
|---|---|---|
| F1 data lever | [29][31][23] discussion; [08][26] measurement; [90] prior competition; [42][46][48] academic | Triangulated; no ablation |
| F2 layout fragility | [01] primary; [21][23] discussion; [22][25] discussion with code; [09] local; [73] academic | Triangulated |
| F3 BA decision rule | [01] primary; [24][22] discussion; [72] academic | Triangulated |
| F4 error corners | [09] local; [28][34] discussion; [95][99] docs; [46][48] academic | Triangulated |
| F5 open weights | [80]–[84] plus a lead HF API check | Fact; usefulness untested |
| F6 train-time content control | [60][62][63][64][66] academic; [09] local; [31] discussion | Partial; competition transfer insufficient |
| Own-generation benefit | [20] host; [32] discussion; [90] prior competition | Insufficient evidence |

## 9. Verification log and limitations
- **Quote checks:** lead and forum quotes (01–34) were matched against `sources/raw/`. All pass; one JSON line is a parser artifact, checked by hand.
- **Literature (40–99):** every URL was re-fetched and resolves to the cited work. Unmatched quotes are tagged in files 42 and 43; file 62 has one partial match.
- **Access and tools:** the Kaggle CLI was unauthenticated, so Kaggle data came from the anonymous read-only JSON endpoints. An account rate limit interrupted two sub-agents and web search; the forum scope was redone by the lead and the likelihood sweep was rerun. No Skill tool was available.
- **REPORT.md:** the harness blocked the subagent's write, so the main session saved this text from the agent's returned output.
- **Excluded:** `research/novel-approaches/deep-research/` (separate run; not read, not modified).

Sources: [Kaggle competition](https://www.kaggle.com/competitions/name-that-llm), [Digg summary](https://digg.com/ai/chvbgdkg) (initial search only), plus all per-source URLs listed in `sources.csv`.
