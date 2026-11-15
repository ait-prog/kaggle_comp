# Unconventional high-upside directions and counterevidence against LLM attribution (Name That LLM)

Scope and conventions:
- Researcher notes, current to 2026-11-12.
- Labels:
  - ESTABLISHED: peer-reviewed or official, and fetched.
  - SOURCE-CLAIM: a claim in a paper, repo or news item, fetched or seen only as a search abstract (which one is stated).
  - LOCAL: my own read-only diagnostic on local data.
  - INFERRED: my reasoning.
  - UNTESTED-PROPOSAL.
- Nothing here promises a score gain.
- The Kaggle rules, data and discussion pages render client-side and returned only the page title to WebFetch. Competition facts below therefore come from the sibling notes file `competition_rules_public_families.md` (same folder), which read them through other means. I mark these "per sibling notes".
- Local context (LOCAL, `data/train.csv`):
  - 2,699 rows: 150 prompts × 18 models, one answer each, with one prompt missing `claude-fable-5`.
  - Text length: median 670 characters, 10th percentile 332, 90th percentile 1,327.
  - Labels: 5 Claude, 5 Gemini, 5 GPT, `glm-5.3`, `glm-5.3-flash`, `kimi-k3`, `qwen3.8-max`.
- Competition facts (per sibling notes):
  - Balanced accuracy over 18 labels, with about 4,000 hidden passages.
  - Test prompts are disjoint from train prompts, and per-class counts are unequal.
  - 9 h runtime, internet off. External data and pretrained models are allowed if free and public, and generation spend is capped at about $5k ("reasonableness").
  - A Kaggle admin said "Feel free to explore any relationships in the data" for transductive use of the test set.
  - Public notebooks plateau at 0.684. The top of the public LB is 0.768.

## Q1. Which unconventional directions are credible, and how feasible is each under a T4×2, 9 h, offline notebook?

### Takeaway
Most of the seed directions are either already public families or infeasible as stated. Two are infeasible as stated: re-generation probes, and running the actual candidate models, because 15 of the 18 labels are closed and the open-weight ones are 320B to 2.8T parameters. Two are already public: transductive grouping and pseudo-labelling.

Six directions survive with honest feasibility notes:
- **D1:** layer-streamed likelihood from the real open-weight GLM-5.3-Flash/GLM-5.3. Highest upside on the worst confusion (GLM↔Claude) and highest engineering risk.
- **D2:** vendor-matched small proxy panel with cross-model contrast features (Sniffer/Ghostbuster-style), rather than single-model log-likelihood.
- **D3:** content-invariant syntactic and authorship-style views (PoS templates, LUAR) chosen to diversify errors.
- **D4:** generation-protocol (reasoning-setting) latent-variable marginalization.
- **D5:** tokenizer re-segmentation features. Low prior; worth a quick kill test only.
- **D6:** distilling the ensemble into a fast student, as an enabler for D1 and to remove rerun variance.

Each one is cheap to falsify on train with sparse-layout grouped CV before any submission.

### Cited Findings

**Prerequisite facts about which labels are open-weight**
- SOURCE-CLAIM (news via search; sources conflict on date and license):
  - Z.ai published GLM-5.3 weights on Hugging Face around 2026-08-28. Reports describe a 753B-total / 40B-active MoE in FP8 and BF16 under a custom license.
  - A companion GLM-5.3-Flash is reported at 320B total / 18B active.
  - [HF zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3); [The New Stack](https://thenewstack.io/zai-glm-weights-license/); [noze.it](https://www.noze.it/en/insights/glm-5-3-weights-licence-benchmark-method/).
  - I did not fetch the model card itself, so parameter counts are unverified.
- SOURCE-CLAIM (news via search): Moonshot released Kimi K3 weights in late July 2026. Reports describe a 2.8T-parameter MoE, about 104B active, with a repository of roughly 1.56 TB. — [The New Stack](https://thenewstack.io/kimi-k3-open-weights/); [Gigazine](https://www.gigazine.net/gsc_news/en/20260728-kimi-k3-open); [llmgateway](https://llmgateway.io/blog/kimi-k3-open-weights)
- SOURCE-CLAIM (per sibling notes, Discussion 757199):
  - A participant asked about running real GLM-5.3 / GLM-5.3-Flash from public IQ3_XXS GGUF segments.
  - Replies say per-passage mmap will not fit in 9 h, and suggest layer-streaming over the whole test set with 128–256 tokens.
  - GLM-4-9B plus Qwen3-8B log-likelihood features gave −0.002 in a prompt-grouped stack.
  - [Discussion 757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199) (not fetched by me)

**Evidence for "generator-as-scorer" / proxy-likelihood features**
- SOURCE-CLAIM (search abstract):
  - Ghostbuster passes documents through "a series of weaker language models" (unigram up to non-instruction-tuned davinci) and runs a structured search over feature combinations, feeding a linear classifier.
  - It reports 99.0 F1 across domains for human-vs-AI detection and does not need the target model's probabilities.
  - It is a binary detection result, not 18-way attribution among frontier siblings.
  - [arXiv 2305.15047](https://arxiv.org/abs/2305.15047); [BAIR blog](https://bair.berkeley.edu/blog/2023/11/14/ghostbuster)
- SOURCE-CLAIM (search abstract):
  - Sniffer ("Origin Tracing and Detecting of LLMs") uses "contrastive features between different accessible language models". For each model pair, it computes the share of tokens where one model's perplexity is lower than the other's, plus correlations: 22 features from 4 models.
  - It claims origin tracing with limited supervised data.
  - LLMDet similarly uses proxy perplexities per candidate LLM.
  - [arXiv 2304.14072](https://arxiv.org/abs/2304.14072)
- SOURCE-CLAIM (search abstract; reported accuracy not fetched):
  - DNA-GPT truncates a text, regenerates the remainder with the candidate LLM, and compares by n-gram overlap (black-box) or probability divergence (white-box).
  - It claims it "can additionally solve model sourcing".
  - It requires query access to each candidate model.
  - [arXiv 2305.17359](https://arxiv.org/abs/2305.17359)

**Evidence for content-invariant and style views**
- ESTABLISHED (Findings of ACL 2025, fetched):
  - When tracing which teacher a distilled student learned from, "n-gram similarity alone is unreliable".
  - Part-of-speech templates, which students "mimic" from teachers, carry the lineage signal.
  - [Wadhwa et al., arXiv 2502.06659](https://arxiv.org/abs/2502.06659)
- ESTABLISHED (ICML 2025, fetched):
  - On 5-way chat-API attribution (97.1%), shuffling words still gives 88.9%.
  - A TF-IDF logistic regression gets 85.5%.
  - Markdown-only (all words removed) gets 73.1%.
  - Shuffling letters drops it to about 39%.
  - The signal is therefore mostly word-distribution plus formatting, and substantially order-free.
  - [Sun et al., "Idiosyncrasies in LLMs", arXiv 2502.12150v2](https://arxiv.org/html/2502.12150v2), Tables 8–9
- SOURCE-CLAIM (ICLR 2024, search abstract):
  - Style representations trained contrastively on human authorship (UAR/LUAR, trained on Reddit, Twitter and StackExchange) support few-shot detection.
  - Given "a handful of examples composed by each of several" LLMs, they predict which model wrote a document.
  - Code: LLNL/LUAR `fewshot_iclr2024`.
  - [Soto et al., arXiv 2401.06712](https://arxiv.org/abs/2401.06712)

**Evidence on decoding and reasoning configuration as nuisance**
- ESTABLISHED (fetched): Outputs of one LLM under 5 temperatures are only 37.9% separable (20% chance), and the best decoding-strategy pair reaches 59.6%. "outputs from the same LLM are not easily separable based on decoding strategies". — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2), §2.2, Table 5
- LOCAL (`ext/openpool/fidelity_by_reasoning.csv`, the repo's own pool-fidelity table):
  - The fidelity ratio varies by recorded reasoning setting within some labels:
    - glm-5.3: default 0.90 vs low 0.68
    - claude-opus-4-8: low 0.76 vs unknown 0.45
    - gemini-3.7-flash: default 0.90 vs "other" 0.33
  - It is flat for others, such as gpt-5.5 at 0.95–1.04.
  - I did not re-derive how this ratio is computed. See the sibling `local_baseline_audit.md`.

**Evidence on tokenizer fingerprints**
- SOURCE-CLAIM (search): The tokenizer-fingerprint work I found is *active*. UTF embeds under-trained tokens via fine-tuning ([arXiv 2410.12318](https://arxiv.org/abs/2410.12318)). Other work fingerprints models by querying them ([arXiv 2607.10252](https://arxiv.org/abs/2607.10252), not fetched). I found no paper on *passive* attribution of existing text from segmentation cost.

**Evidence on the distillation-as-enabler idea and data-centric moves**
- SOURCE-CLAIM (fetched README): The 1st-place solution of Kaggle "LLM – Detect AI Generated Text" fine-tuned about 10 diverse LLMs with a CLM objective on the PERSUADE corpus to *generate* matched training data. The detectors were (Q)LoRA LLMs, DeBERTa-v3-large with a ranking loss, and a supervised-contrastive embedding used for kNN. The README does not state lessons or ensemble weights. — [rbiswasfc/llm-detect-ai](https://github.com/rbiswasfc/llm-detect-ai)
- SOURCE-CLAIM (per sibling notes): Time-budgeted members such as ATLAS cover a variable set of rows, and identical code rescored up to 0.009 apart. — [alexchilton 0.684 notebook](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed) (not fetched by me)

### Inferences (each direction: mechanism, novelty, prerequisites, compute, rules risk, cheapest falsifier)

**D1. Real open-weight label-model likelihood via layer-major streaming (GLM-5.3-Flash first, then GLM-5.3). UNTESTED-PROPOSAL.**
- Mechanism:
  - Score each test passage's first 128–256 tokens under the actual open weights of the `glm-5.3*` labels, with a generic chat template and no prompt.
  - Use the per-token log-prob summaries, and ratios against a small reference model (Binoculars-style normalization), as features.
  - This is the closest thing to a Bayes likelihood available for any label.
- Why it targets the worst error:
  - In public reports GLM-5.3 is classified as Claude more often than as GLM (per sibling notes).
  - My probe gives family-correct rates of 0.317 for GLM and 0.287 for Kimi (Q2).
  - A real-model likelihood is the one feature that should not inherit the Claude-like style.
- Novel to the competition? Yes. It reached only the question stage publicly. Novel to ML? No (likelihood-based detection is old).
- Compute (INFERRED, rough):
  - GLM-5.3-Flash at about 2–3 bits per weight is about 85–120 GB. That cannot fit in 2×16 GB VRAM plus about 30 GB RAM.
  - Weights must therefore be streamed **layer by layer over the whole test batch**, so each weight is read once. Per-passage mmap re-reads everything per passage and is infeasible.
  - Activations: about 4,000 × 256 tokens × hidden size (unknown, assume 4–6k) × 2 bytes ≈ 8–12 GB. These must live in RAM or on local disk between layers.
  - FLOPs: about 2 × 18e9 × 1e6 tokens ≈ 3.6e16. At an effective 5–10 TFLOPs per T4 with dequantization overhead, that is roughly 0.5–1 h per GPU, before I/O.
  - I/O: reading about 100 GB from the input mount takes 10–40 min, depending on unverified mount throughput.
  - Plausible inside 9 h only if a custom layer-major MoE runner with i-quant or FP8-to-fp16 dequantization works on Turing (sm_75, no bf16 or FP8 tensor cores).
  - GLM-5.3 (40B active, about 2.4× the Flash size) roughly doubles everything. Kimi K3 (about 1.5 TB) is out of reach.
  - `qwen3.8-max` and the Claude, Gemini and GPT labels are closed (INFERRED from naming and the public discussion; not separately verified).
- Prerequisites:
  - A public Kaggle dataset of the quantized weights, if one exists, within dataset-size limits. The size limit is unverified; see Gaps.
  - A runner that works without internet.
- Rules risk: low. Public pretrained weights are allowed per sibling notes. Check the custom GLM license for competition use; the reported license change concerns very large providers.
- Cheapest falsifier:
  - In an interactive Kaggle GPU session, score only the 150 `glm-5.3` + 150 `glm-5.3-flash` + 750 Claude train rows.
  - Use 128 tokens under GLM-5.3-Flash, compared against a small reference LM.
  - Kill if the AUC for GLM-vs-Claude from the likelihood features is not clearly above what the current stack's OOF already achieves on that pair.
  - Also kill if the AUC for glm-5.3 vs glm-5.3-flash is near 0.5.
  - Use sparse-layout grouped CV (Q2). Run this before building any full-test runner.
- Main counterevidence:
  - A GLM trained on Claude-like data may also assign high likelihood to Claude text. See Q3 (SFT inheritance, distillation entanglement).
  - The host's API serving may differ from open weights: reasoning mode, system prompt, quantization.

**D2. Vendor-matched small proxy panel with cross-model *contrast* features (Sniffer/Ghostbuster). UNTESTED-PROPOSAL.**
- Mechanism:
  - Compute token log-probs under 4–6 small open models, one per vendor where one exists. Examples (INFERRED, not verified by me): a Gemma model for Google, an open GLM-4.x for ZAI, Qwen3 small for Qwen, an OpenAI open-weight model if it runs on T4, and a Moonshot small MoE. Anthropic has no open model.
  - Build pairwise features: the fraction of tokens where A beats B, correlations of the token-loss series, and Ghostbuster-style vector/scalar combinations.
- Difference from public use: public attempts used single-model log-likelihood as a feature (−0.002). The hypothesis is that the *contrast between vendors' models* encodes lineage that no single model does.
- Compute: small models over about 1M tokens take minutes to tens of minutes on T4. Feasible.
- Rules risk: low.
- Falsifier:
  - Add about 30–60 contrast features to the existing OOF logits in a stacker under sparse-layout grouped CV.
  - Kill if the gain is within the CV standard error (about 0.009 per sibling notes) **and** the features' error correlation with the stack is high.
- Counterevidence: the evidence is detection-only and from the 2023 era. The public single-proxy result was null.

**D3. Content-invariant syntactic and style views for error diversification. UNTESTED-PROPOSAL.**
- Mechanism:
  - (a) PoS-tag n-gram TF-IDF (spaCy small model shipped as a dataset), motivated by the distillation-tracing result that PoS templates carry lineage where n-grams fail.
  - (b) Frozen LUAR authorship embeddings with a linear head.
  - (c) An optional "markdown/punctuation skeleton" view.
- Purpose: low error correlation with the LoRA readouts, rather than a high standalone score. The 73.1% markdown-only and 88.9% word-shuffle results show that much of the signal survives without content.
- Compute: CPU minutes. Feasible.
- Falsifier: standalone balanced accuracy, plus the correlation of its error indicators with the current stack's OOF errors, plus the stacker gain under sparse-layout CV. Kill if the gain is within noise.
- Counterevidence:
  - The Wadhwa result is about student→teacher tracing with small students (GPT-2, Olmo-1B), not frontier siblings.
  - Public typography hard rules did not transfer to fresh text (sibling notes).

**D4. Latent generation-protocol marginalization. UNTESTED-PROPOSAL.**
- Mechanism:
  - Train the head on (label × recorded reasoning setting) classes from the pool metadata (`gen_reasoning`).
  - At test, either marginalize over configs with priors fitted by EM on the unlabelled test batch (transductive use is allowed by the admin), or reweight the pool toward the config mix that best matches host train.
- Rationale: temperature is nearly invisible (37.9% separable), so decoding is not the nuisance. Reasoning setting does shift pool fidelity for some labels (LOCAL table), so mixing configs blurs class-conditional densities.
- Compute: negligible beyond retraining a head.
- Falsifier: on host train (grouped CV), compare a config-aware head with marginalization against a plain head, for glm-5.3, claude-opus-4-8, gemini-3.7 and kimi-k3. Kill if there is no per-class recall gain.
- Counterevidence: if the host used one fixed config, simple pool reweighting captures all of the benefit (INFERRED). `unknown` is the largest config bucket for most labels (LOCAL), so the latent variable is mostly unobserved.

**D5. Tokenizer re-segmentation cost. UNTESTED-PROPOSAL, low prior.**
- Features: chars per token and the fraction of multi-token words under each public vendor tokenizer: OpenAI's tiktoken encodings, Gemma, Qwen3, GLM-5.3 and Kimi K3. Claude's current tokenizer is not public (INFERRED).
- Why the prior is low (INFERRED):
  - Any text can be tokenized by any tokenizer, so the only signal is lexical preference for vocabulary-efficient words.
  - Char/word n-gram models already see that.
  - No passive-attribution literature was found.
- Falsifier: CPU minutes on train; tokenizer files are megabytes. Kill if adding the features to a char-TF-IDF LR moves grouped-CV balanced accuracy by less than about 0.005.

**D6. Distil the ensemble into a fast student to buy runtime. UNTESTED-PROPOSAL, enabler only.**
- Mechanism:
  - Train one LoRA student on teacher soft labels over pool + train.
  - Use OOF teacher labels for train rows to avoid self-leakage.
  - Optionally add the test batch transductively.
  - Use the freed hours for D1 or for full-coverage ATLAS, which removes the up-to-0.009 rerun variance.
- The ceiling is at most the teacher's (INFERRED).
- Falsifier: student vs teacher on sparse-layout CV. Go only if the student is within about 0.005 at under about 30% of runtime **and** D1 or another expensive member has passed its own falsifier. Otherwise D6 has no purpose.

**Seed ideas rejected as stated (INFERRED):**
- Re-generation probes (DNA-GPT): 15 of 18 labels are closed and there is no internet at inference, so there is nothing to regenerate with offline. For open GLM, D1 likelihood dominates regeneration at equal compute.
- Generating prompt-matched data from the open label models: generating with 320B–2.8T models needs paid multi-GPU compute (excluded by this assignment). The community already generates via APIs, under the about $5k cap per sibling notes. So it reduces to a data-spend decision, not a new method.
- Pseudo-labelling or self-training on test: public simulation was negative (per sibling notes).

### Gaps
- I could not fetch the GLM-5.3 / GLM-5.3-Flash / Kimi K3 model cards. Parameter counts, licenses and hidden sizes come from conflicting secondary reports.
- I could not verify whether a quantized GLM-5.3-Flash exists as a Kaggle dataset, or Kaggle's per-dataset size limit and input-mount throughput. All are critical for D1.
- I did not verify which vendor small models (Gemma, OpenAI open weights, Moonshot small models) exist on Kaggle, or whether they run on T4. D2's panel is therefore a sketch.
- Ghostbuster, Sniffer, DNA-GPT, Soto and RAID were verified only at abstract or search level, not full text.
- No evidence was found for passive tokenizer attribution (D5), or for latent-config marginalization (D4) in attribution.

## Q2. Local diagnostic: does test-batch structure (prompt grouping and one-per-model assignment) offer headroom, and why has it failed publicly?

### Takeaway
On train layout, prompt grouping is nearly free and very powerful:
- The nearest neighbour shares the prompt 99.8% of the time.
- One-per-model Hungarian assignment within true groups lifts a weak TF-IDF model from 0.322 to 0.419 accuracy.

This matches the public "+0.10 in train-layout CV" and makes the repeated public LB *losses* informative: the hidden test very likely does not have dense 18-answer prompt groups. The gain is conditional on a layout that the evidence says is absent, so this direction should not be pursued except as a cavity-style soft prior validated under a sparse layout.

### Cited Findings
- LOCAL: the probe ran TF-IDF (char_wb 1–4 + word 1–2) with logistic regression, 5-fold GroupKFold by `prompt_id`, on all 2,699 rows (script: `/private/tmp/claude-501/-Users-i2gunshaker-orca-workspaces-kaggle-comp-name-that-llm/715c8dc3-c85b-4607-80fd-cd053d781a3e/scratchpad/assign_probe.py`, which is session scratch and may not persist; CPU, about 3 min).
  - Argmax accuracy is 0.322 and family accuracy 0.668.
  - Hungarian assignment within true prompt groups gives 0.419.
  - The word-TF-IDF nearest neighbour shares the prompt for 0.998 of rows.
  - Accuracy by family: Gemini 0.447 (family-correct 0.927), GPT 0.295 (0.757), Claude 0.324 (0.666), Kimi 0.287, Qwen 0.280, GLM 0.173 (family-correct 0.317).
  - Plain accuracy is not balanced accuracy, but classes are balanced in train, so the two nearly coincide.
- SOURCE-CLAIM (per sibling notes):
  - Clustering plus Hungarian gave about +0.10 in train-layout CV but lost on the LB (0.353 → 0.344; −0.009 to −0.017 across parents).
  - With one or two answers per prompt, Hungarian gives about 0 and centring loses up to 0.10.
  - OOF scored with 2 answers per prompt tracked the LB at correlation 0.91, vs 0.13 for full groups.
  - [Discussion 750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390); [Discussion 750213](https://www.kaggle.com/competitions/name-that-llm/discussion/750213); [alexchilton notebook](https://www.kaggle.com/code/alexchilton/name-that-llm-0-684-all-public-what-failed)

### Inferences
- INFERRED: Given 0.998 neighbour purity in train, any real dense prompt groups in test would be trivially found. The public losses are therefore evidence of sparse groups, not of bad clustering.
- INFERRED: Every Q1 falsifier should use the **sparse-layout grouped CV** protocol: hold out prompts, then keep only 1–2 answers per held-out prompt. Optionally mix the held-out rows into a simulated 4,000-row batch.
- INFERRED: The probe's family-correct rate for GLM (0.317) reproduces the public GLM↔Claude confusion with a trivial model. That confusion is the most specific target for D1 and D3.

### Gaps
- The hidden layout is not official. A different test-batch structure would revive this direction.

## Q3. When does LLM attribution fail? (counterevidence by mechanism)

### Takeaway
Attribution is robust to temperature, to paraphrase and translation, and to moderate cross-dataset shift. It is fragile in four cases:
- within a model family: 59.8% across 4 Qwen sizes
- when one model is trained on another's outputs: 96.5% → 59.8% after SFT on shared synthetic data, and a held-out Grok is called ChatGPT 82.8% of the time
- under summarization: 97.1% → 58.1%
- with very little data: 10 samples gives 40.3%; convergence takes about 10k samples

Name That LLM sits in the fragile part: 15 of 18 labels are within-family siblings, there are about 150 host samples per class, and passages are short. The strong 97% headline numbers do not transfer.

### Cited Findings
- **Within-family / derived models.** ESTABLISHED: Qwen2.5 7B–72B 4-way gets 59.8%, with the best pair at 85.5%. Base vs instruct of the same model gets 96.8%. Post-training separates models far more than size does. — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2), §2.1, Table 2
- **Synthetic-data inheritance and distillation.**
  - ESTABLISHED: SFT of different models on the same UltraChat data drops separability from 96.5% to 59.8%, and "training with synthetic data can propagate the idiosyncrasies in the source model". With Grok held out, 82.8% of its outputs are classified as ChatGPT. — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2), §4
  - ESTABLISHED (fetched): "n-gram similarity alone is unreliable for identifying teachers"; PoS templates work better. — [Wadhwa et al. 2025](https://arxiv.org/abs/2502.06659)
  - SOURCE-CLAIM (fetched abstract, June 2026 preprint):
    - Identifying a teacher in isolation is "highly challenging".
    - Near-perfect teacher recovery needs a reference checkpoint from the same lineage, and is shown only for single-teacher distillation.
    - Modern lineages are "heavily entangled".
    - [Rawat et al., arXiv 2607.09692](https://arxiv.org/abs/2607.09692)
- **Paraphrasing and rewriting.**
  - ESTABLISHED (NeurIPS 2023, fetched): The DIPPER paraphraser drops DetectGPT detection "from 70.3% to 4.6%" at 1% FPR. Retrieval over stored generations recovers 80–97%. This is a detection result, not an attribution result. — [Krishna et al., arXiv 2303.13408](https://arxiv.org/abs/2303.13408)
  - ESTABLISHED: For *attribution*, rewriting by GPT-4o-mini keeps most of the signal: 91.4% after paraphrase and 91.8% after translation. Summarization collapses it to 58.1% (71.5% with a Qwen rewriter). — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2), §3.3, Table 10, App. B.5
  - SOURCE-CLAIM (secondary review only): a fine-tuned BERT attributor loses 57.1% after one ChatGPT paraphrase (Tian et al.). — [liner review](https://liner.com/review/ship-theseus-curious-cases-paraphrasing-in-llmgenerated-texts)
  - SOURCE-CLAIM (search snippet only): detection falls 25.4% on average after five paraphrase stages. — [RANLP 2025](https://aclanthology.org/2025.ranlp-1.129)
- **Temperature and decoding.**
  - ESTABLISHED: Decoding changes leave the model fingerprint largely intact (5 temperatures only 37.9% separable). — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2), Table 5
  - SOURCE-CLAIM (search abstract): *Detectors* are vulnerable to sampling-strategy changes, repetition penalties and unseen generators (6M+ generations, 11 models, 11 attacks, 4 decoding strategies). — [RAID, Dugan et al., ACL 2024](https://arxiv.org/abs/2405.07940)
  - Conflict note: RAID concerns human-vs-machine detectors, often zero-shot likelihood ones, while Sun et al. concerns trained attributors. Likelihood-feature members such as D1 and D2 are the ones exposed to RAID-type decoding shifts.
- **Short texts.**
  - ESTABLISHED: Attribution accuracy rises with length and saturates past about 256 tokens. One token already gives about 50% for chat models. — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2), §2.2, Fig. 2
  - LOCAL: the competition's 10th-percentile train passage is 332 characters (roughly 70–80 tokens), and the median is 670 characters. Many passages sit below saturation.
- **Topic and domain shift.**
  - ESTABLISHED: Cross-dataset transfer among chat datasets stays at 88.3–99.8%. — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2), Table 3
  - ESTABLISHED (ACL 2024, abstract fetched): good performance "usually requires an access to the training data from the same domain and generators". — [M4GT-Bench, arXiv 2402.11175](https://arxiv.org/abs/2402.11175)
  - SOURCE-CLAIM (per sibling notes): pool-trained models read 0.03–0.04 higher on grouped CV than on the LB, and test prompts are disjoint from train.
- **Length/format control and convergence.**
  - ESTABLISHED: Controlling length lowers 5-way accuracy from 96.3% to 93.0%; controlling format lowers it to 91.4%. Markdown-only carries 73.1%. — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2), Tables 4, 9
  - SOURCE-CLAIM (NeurIPS 2025, search abstract): "inter-model homogeneity", with different models producing strikingly similar open-ended outputs across 70+ models. — [Artificial Hivemind, arXiv 2510.22954](https://arxiv.org/abs/2510.22954)
  - SOURCE-CLAIM (per sibling notes): the host applied "light formatting cleanup", and GLM-5.3 writes in a Claude-like style ([Discussion 755314](https://www.kaggle.com/competitions/name-that-llm/discussion/755314)).
- **Data volume.** ESTABLISHED: Accuracy converges at about 10K samples; 10 samples gives 40.3%. — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2), Fig. 3

### Inferences
- INFERRED: The competition is deliberately built from the fragile cases: same-family siblings, possible cross-lab synthetic-data inheritance (GLM↔Claude), short passages, formatting cleanup, disjoint prompts and small host data. The near-97% literature numbers are irrelevant as targets. The 0.684 → 0.768 LB spread says headroom exists, but nothing here says how much.
- INFERRED: Robustness to temperature is *good news* for using community pool data generated at unknown temperatures. Reasoning-setting mismatch is not covered by that evidence (D4).
- INFERRED: If GLM was trained on Claude-like outputs, *all* text-surface views, D3 included, may hit an inheritance ceiling on GLM↔Claude. Only views tied to the model itself (D1) escape it in principle. Even those may not, because an imitating model also scores the imitated text highly.

### Gaps
- No 2024–2026 paper I found measures attribution among **sibling versions of frontier closed models**, such as GPT-5.x variants. The closest evidence is Qwen sizes (59.8%).
- I found no direct evidence on whether GLM-5.3 was distilled from Claude. The resemblance is a public empirical observation only.
- The Hivemind paper and RAID were checked only at abstract level, so effect sizes are unverified.

## Q4. Adversarial stress test: how could each direction fail, and what would change the recommendation?

### Takeaway
Every direction has a credible failure mode traceable to Q3 evidence. Rank them by cost of the falsifier, not by upside:
1. D3, D5 and D4 cost minutes.
2. D2 costs an hour.
3. D1 costs a day of engineering plus one GPU session.
4. D6 only matters if D1 passes.

### Cited Findings
- See Q1–Q3. The key ones: SFT inheritance (96.5 → 59.8) and Grok → ChatGPT 82.8% ([Sun et al.](https://arxiv.org/html/2502.12150v2)); entangled lineages ([Rawat et al.](https://arxiv.org/abs/2607.09692)); RAID decoding sensitivity for likelihood detectors ([Dugan et al.](https://arxiv.org/abs/2405.07940)); public null result for single-proxy log-likelihood (per sibling notes, [Discussion 757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199)).

### Inferences
| Dir | Most likely failure (evidence) | What would change the call |
|---|---|---|
| D1 real-GLM likelihood | Claude text scores as high as GLM text (inheritance); API-served GLM ≠ open weights (reasoning mode, template); runner engineering on sm_75 or input I/O blows the 9 h budget | Go if the train-subset AUC for GLM vs Claude clearly beats the stack's pairwise OOF; no-go if near the stack, or if runtime > ~4 h measured on 1/10 of the batch |
| D2 proxy contrast | Proxies of closed vendors do not share their post-training, so contrast reflects pretraining corpora only; public single-proxy result was null | Go only if stacker gain > ~1 SE (≈0.009) under sparse-layout CV *and* low error correlation |
| D3 syntactic/style views | Signal is mostly lexical, so content-free views are weak; may add noise to a stack already at its limit | Go if error correlation with the stack is low and the stacker gain is positive on two seeds |
| D4 config marginalization | The host used one config, so plain reweighting suffices; `unknown` config dominates the pool | Go if per-class recall rises for glm-5.3, opus-4-8 or gemini-3.7 under grouped CV |
| D5 tokenizer cost | Redundant with char n-grams | Kill on less than ~0.005 grouped-CV change |
| D6 distillation | Student loses ensemble diversity, so it scores below the teacher | Only after D1 passes; go if within ~0.005 of the teacher at under 30% runtime |

- INFERRED: The general failure for all of them is validation mismatch. Train-layout grouped CV overstated transductive tricks and pool-trained members. Every falsifier must use sparse-layout grouped CV, with external rows grouped by source prompt.
- INFERRED: Selecting by the 40% public LB carries about ±0.012–0.014 noise (per sibling notes), and the private class mix differs. Gains under about 0.02 should not drive the final selection.

### Gaps
- I ran none of D1–D6. All six are untested proposals, and their falsifiers have not been executed.
- How much hidden-test headroom any of them reaches is unknown and cannot be estimated from the sources found.
