# Source-model attribution and LLM fingerprinting: literature map for Name That LLM

Scope: mechanisms for attributing a text to the specific LLM that generated it, judged for transfer to an offline Kaggle notebook (T4x2 or similar, ~9-12h, no internet). Researched 2026-11-12.

Evidence labels used throughout:
- ESTABLISHED: replicated across independent papers.
- SOURCE-CLAIM: a single paper's claim, not independently reproduced.
- INFERRED: my own reasoning from the cited material, untested.

"Older" means a result from 2023 or earlier, usually on GPT-2/OPT/davinci-era generators. Treat these as weaker evidence for modern chat models.

Verification method: arXiv abstracts were fetched through the arXiv API (two batches of 14 and 13 IDs) and arXiv HTML full texts through WebFetch. Bash had no network access, so no code was run and no repo was cloned. Repo URLs below come from the papers themselves and were not opened. Several full texts were truncated or failed; each case is listed under Gaps.

Citation caveat: titles, dates and numbers were verified against fetched abstracts or full texts. Some first-author surnames were not shown in the fetch output and come from my memory, so treat them as unverified: Hans, Bao, Mitchell, Verma, Mireshghallah, Wu, Li, Venkatraman, Ippolito, Dugan, Pasquini, He, Soto, Bitton, McGovern, Wang, Sun. The arXiv IDs are the reliable identifiers. Author lists that were verified by fetch: 2501.02406, 2603.28054, 2609.02379, 2601.02320 and 2504.11369.

Novelty framing: the competition baseline, per BRIEF.md, is fine-tuned text classifiers (a deployed submission scoring 0.682, and a Qwen3-8B-base LoRA experiment). "Novel to competition" below means "not a fine-tuned text classifier". I did not verify what public Kaggle notebooks actually do, so every novelty-to-competition statement is INFERRED.

---

## 1. Likelihood and residual fingerprints (proxy-LM log-prob, rank, entropy, cross-perplexity, curvature, feature search): how well do they separate generators, not just human vs AI?

### Takeaway
Proxy-LM likelihood statistics are well established as human-vs-AI detectors. As multi-class generator attributors, the evidence is thin and mostly older (2023, GPT-2/OPT/LLaMA-1 era). Per-candidate likelihood tests need each candidate's logits, which a closed-API label set cannot supply. The transferable form is a per-token feature vector (log-prob, rank, entropy, cross-entropy between two or three small open proxies) fed to a learned head alongside the text classifier. The aim is error diversity, not a replacement.

### Cited Findings
- **Binoculars (2024).** Zero-shot, no training data. It "detects over 90% of generated samples from ChatGPT (and other LLMs) at a false positive rate of 0.01%". This is a binary detector, and the abstract makes no attribution claim. — [Hans et al. 2024, arXiv 2401.12070](https://arxiv.org/abs/2401.12070). Detection: ESTABLISHED (widely used as a baseline). Attribution: not studied.
- **Fast-DetectGPT (2023, older).** Conditional probability curvature. About a 75% relative gain over DetectGPT and a 340x speedup, in white-box and black-box settings. Detection only. — [Bao et al., arXiv 2310.05130](https://arxiv.org/abs/2310.05130)
- **DetectGPT (2023, older).** Raised GPT-NeoX fake-news detection "from 0.81 AUROC ... to 0.95 AUROC". It needs the scoring model's log-probs and many perturbation passes (expensive). It is not framed as attribution. — [Mitchell et al., arXiv 2301.11305](https://arxiv.org/abs/2301.11305)
- **Ghostbuster (2023, older).** Runs a structured search over combinations of token-probability features from weaker LMs, then a linear classifier. It does not need the target model's probabilities. Reports "99.0 F1 when evaluated across domains" and "+4.4 F1" for cross-LM generalization. Detection, not attribution. — [Verma et al., arXiv 2305.15047](https://arxiv.org/abs/2305.15047). The feature-search idea is directly reusable for multi-class (INFERRED).
- **Smaller proxies generalize better (2023, older).** OPT-125M reached "an AUC of 0.81 in detecting ChatGPT generations" versus 0.45 for GPT-J-6B. Shared training data between detector and generator mattered little. — [Mireshghallah et al., arXiv 2305.09859](https://arxiv.org/abs/2305.09859). SOURCE-CLAIM. It argues for small, cheap proxies.
- **LLMDet (2023, older).** Model-specific source identification from per-LLM proxy perplexity (pre-computed n-gram probability dictionaries), covering GPT-2, OPT and LLaMA. Reports "98.54% precision" on human text, is "x5.0 faster", and extends to new open models. — [Wu et al., arXiv 2305.15004](https://arxiv.org/abs/2305.15004). SOURCE-CLAIM. The per-model attribution accuracy was not visible in the abstract.
- **Sniffer / origin tracing (2023, older).** Traces text to a specific LLM using contrastive features across a set of proxy LMs' likelihoods. Works white-box and black-box, needs "only limited data", extends to new origins, and claims to detect GPT-3 outputs "without the GPT-3 models". — [Li et al., arXiv 2304.14072](https://arxiv.org/abs/2304.14072). SOURCE-CLAIM. I did not retrieve its accuracy table.
- **GPT-who (2023, older).** Builds per-author signatures from Uniform-Information-Density (surprisal) features computed by a proxy LM. Outperforms GLTR, GPTZero, DetectGPT, the OpenAI detector and ZeroGPT "by over 20% across domains". — [Venkatraman et al., arXiv 2310.06202](https://arxiv.org/abs/2310.06202). SOURCE-CLAIM. It is multi-class with a cheap logistic head on proxy-LM statistics.
- **Training-free statistical attribution (2025, v6 Sep 2026).**
  - White-box statistic: the average log-perplexity Z_n^M = −log p^M(Y_n) under each candidate model. The text is assigned to set A if some A_i scores lower than every B_j.
  - Type I and Type II errors decay exponentially with text length.
  - The black-box variant estimates next-token probabilities from m samples per position; "by m=1000, performance nearly matches the exact-probability benchmark".
  - — [Radvand, Duenyas, Tewari, arXiv 2501.02406](https://arxiv.org/html/2501.02406v6). A search snippet of the same paper says errors concentrate in the same-family setting (Llama2-7B vs Llama3-8B) — [search result, arXiv 2501.02406](https://arxiv.org/pdf/2501.02406). SOURCE-CLAIM. My fetch truncated before the experiments, so the numeric accuracy was not verified.
- **TRACE (EMNLP 2026).** A fingerprint built from "token-level transition patterns" such as word rank, estimated by a separate lightweight LM. Attribution goes to the nearest reference fingerprint. Claims SOTA, says it "works well in limited training data scenarios" and is "robust in OOD settings" on long-form books. — [Shetty, Xu, Ohrimenko, Lau, arXiv 2603.28054](https://arxiv.org/abs/2603.28054). SOURCE-CLAIM. No numbers were visible in the abstract.
- **MultiGhostBench (Sep 2026, 928 books, 5 generators, 6 languages).** "No single method consistently performs best across settings", and performance "generally degrades under distribution shifts". "Statistical and fingerprint-based detectors are more language-dependent" than transformer detectors. — [Greco et al., arXiv 2609.02379](https://arxiv.org/abs/2609.02379); code at github.com/GrecoMT/MultiGhostBench (not opened). SOURCE-CLAIM, and counterevidence to relying on likelihood fingerprints alone.
- **Likelihood statistics in few-shot attribution are weak against style embeddings.** In the few-shot benchmark, pAUC at FPR ≤ 1% (N=5/N=10 support docs) was:
  - LogRank: 0.764/0.775
  - Rank: 0.569/0.558
  - Entropy: 0.498/0.498
  - UAR style embedding: 0.868/0.958
  - — [Soto et al., ICLR 2024, arXiv 2401.06712](https://arxiv.org/html/2401.06712). SOURCE-CLAIM, on ~128-token documents.
- **Temperature confounds likelihood detectors.** DetectLLM's appendix notes Log-Rank/Log-Likelihood performance "can become extremely high when the temperature drops". — [search snippet, arXiv 2306.05540](https://arxiv.org/pdf/2306.05540). Not fetched in full.

### Inferences
- Likelihood features capture how predictable the text is to the proxy. That is partly a function of decoding temperature and of model "peakedness" (instruct vs base), so they discriminate model families and configurations better than near-identical sibling versions. This is consistent with the same-family errors in 2501.02406 and the FDLLM family clustering (see Q4). INFERRED.
- Likelihood-ratio attribution (score under each candidate, pick the most probable) is only possible for candidates whose weights are available offline. With closed-API classes (GPT/Claude/Gemini), a proxy-feature + learned-head design (Sniffer/Ghostbuster/GPT-who style) is the only viable form. INFERRED.
- **Most promising novel-to-competition variant.** Run 2-3 small open proxies from different families (for example a Qwen, a Llama and a Gemma ~0.5-2B base model). For each text, compute token-level log-prob, rank, entropy and observer/performer cross-entropy (Binoculars-style). Aggregate them into histograms and quantiles, then stack the result with the existing classifier's logits. The proxies' tokenizers also differ, which links to Q2. Novelty: to ML research, low (2023-era idea family). To the competition, likely moderate, unverified. INFERRED.
- **Expected value.** Mostly error diversification on confusable pairs that differ in decoding config or base vs instruct. Gains on sibling versions are unlikely. INFERRED, and this should be tested cheaply before investing (see Q6).

### Gaps
- I found no paper reporting multi-class accuracy of Binoculars or Fast-DetectGPT features across many modern chat generators. This is a genuine gap, not a fetch failure.
- Sniffer and LLMDet per-model accuracy tables, and the experiment section of 2501.02406, were not retrieved (abstracts only, or truncated fetches).
- TRACE and MultiGhostBench numeric results were not visible in the abstracts. Full PDFs were not fetched.

---

## 2. Tokenizer fingerprints: do generator tokenizer boundaries or vocabulary leave detectable traces in output text?

### Takeaway
I found no primary paper that empirically isolates the generator's tokenizer as an attribution signal from plain output text. FDLLM names "tokenizer boundary behaviors" among its features but never tests the claim. Re-tokenization cost under candidate tokenizers is therefore an untested hypothesis, plausible only as a weak, length-normalized auxiliary feature. Thin evidence.

### Cited Findings
- FDLLM lists "rare word choices, tokenizer boundary behaviors, implicit punctuation patterns, and sampling noise" as identity cues. My full-text read found the tokenizer claim untested: the family clustering in t-SNE is only speculatively attributed to shared tokenizers or pretraining. — [FDLLM, arXiv 2501.16029v4](https://arxiv.org/html/2501.16029v4). SOURCE-CLAIM, unsupported by an ablation.
- Token-count fingerprinting works only at the API level (prompt-token counts returned by endpoints). That paper "validates token-count consistency as a fingerprint of a shared tokenization stack, but rejects its use as a standalone necessary test for model-family lineage". It needs API metadata, not text. — [search snippet, arXiv 2608.29930](https://arxiv.org/pdf/2608.29930). Not fetched in full.
- LLMs do not always emit canonical token sequences, which is why "canonical sampling" has been proposed. The same string has many valid segmentations, and post-trained models tolerate non-canonical inputs while base models degenerate. — [Broken Tokens?, NeurIPS 2025, arXiv 2506.19004](https://arxiv.org/html/2506.19004v1); [NeurIPS poster](https://neurips.cc/virtual/2025/poster/117561). Relevant mechanism, but it says nothing about recovering tokenization from detokenized text.
- Letter-level statistics carry almost no model signal. Shuffling letters drops 5-way chat accuracy to 39.1% (chance 20%), and letter frequencies are near-identical across models. Shuffling words keeps 88.9%. — [Sun et al., Idiosyncrasies in LLMs, arXiv 2502.12150](https://arxiv.org/html/2502.12150). SOURCE-CLAIM. This implies the signal lives at the word or phrase level, not in sub-word character patterns.

### Inferences
- Once output is detokenized to a string, the generator's segmentation is mostly unrecoverable. What survives are side effects:
  1. **Vocabulary coverage.** Models may avoid strings their tokenizer splits awkwardly (rare multi-token words, specific Unicode such as curly quotes, em-dashes, non-breaking spaces, CJK and emoji handling).
  2. **Whitespace and normalization artifacts.** Leading spaces, "\n\n" versus " \n", and byte-fallback glitches.
  3. **Glitch tokens or truncation at token boundaries** when max_new_tokens is hit. A text cut mid-word at a token boundary is a direct tokenizer trace (INFERRED).

  Cheap features for these: a per-text count of characters outside common ASCII; whether the text ends without terminal punctuation; and, for each candidate open tokenizer, tokens-per-character and the fraction of words split into ≥3 pieces. Inference cost is CPU-only and negligible. INFERRED. Novel to the competition: plausibly. To ML research: as a tested method, yes, but low prior.
- **Truncation-point analysis is the most concrete tokenizer trace.** If dataset generation used a max_new_tokens cap, texts that hit the cap end at exactly N tokens of the generator's tokenizer. Re-tokenizing each truncated text with each candidate tokenizer and checking which yields exactly the cap (for example 256/512/1024) could be near-deterministic for those texts. This depends entirely on how the competition data was built, so it must be checked against the data. INFERRED, high upside if the precondition holds, zero otherwise.

### Gaps
- I found no empirical paper on re-tokenization cost or segmentation artifacts as a text-only attribution feature. Two searches ("tokenizer fingerprint LLM generated text attribution", "non-canonical tokenization ... model identification") returned only adjacent work.
- Whether the competition data has length caps, truncation or normalization (for example stripping Unicode) was not checked here. The main report needs a data audit.

---

## 3. Generation-configuration modeling: can temperature, top-p, repetition penalty or the system prompt be detected from text, and does modeling config as a nuisance help?

### Takeaway
Configuration is weakly to moderately recoverable from text by a classifier. Same-model temperature classification sits well above chance but far below model identity. Temperature can be estimated by maximum likelihood with a proxy LM. Config shifts likelihood features strongly and fine-tuned classifiers only modestly, and no paper tests "config as nuisance variable improves attribution". Treat it as a hypothesis whose cheapest test is the per-config confusion of the current model.

### Cited Findings
- **Same model, different sampling.** Pairwise discrimination between sampling strategies "peaked at 59.6%". A 5-way temperature task reached 37.9% (chance 20%). In the same paper, model identity reached 97.1% 5-way. — [Sun et al., arXiv 2502.12150](https://arxiv.org/html/2502.12150). SOURCE-CLAIM. Config leaves a much weaker trace than identity.
- **FDLLM (20 LLMs; data generated at T ∈ {0, 0.3, 0.5, 0.7, 1}).**
  - Lower generation temperature gives higher accuracy: Tgen=0 at 93.20% versus Tgen=1 at 92.10% with 50% of the training data.
  - The paper calls the effect of inference temperature "marginal".
  - — [FDLLM, arXiv 2501.16029v4](https://arxiv.org/html/2501.16029v4). SOURCE-CLAIM. A fine-tuned attributor is fairly config-robust when training spans the configs.
- **Temperature estimation by maximum likelihood.** "This parameter can be estimated using maximum likelihood approach" relative to a given LM. The best estimator among small-to-medium LLMs tested was Qwen3-14B, and most corpora come out close to T≈1. — [Mikhaylovskiy, arXiv 2601.02320](https://arxiv.org/abs/2601.02320). SOURCE-CLAIM. The abstract gives no error figures and says nothing about proxy-generator mismatch.
- **Reverse-engineering decoding strategies (2023, older).** Infers top-k vs nucleus parameters given black-box generation access, with "implications for detecting generated text". — [Ippolito et al., arXiv 2309.04858](https://arxiv.org/abs/2309.04858). Needs query access, so it does not apply to a fixed test set.
- **RAID.** Spans "11 models, 8 domains, 11 adversarial attacks and 4 decoding strategies". Detectors are "easily fooled". — [Dugan et al., arXiv 2405.07940](https://arxiv.org/abs/2405.07940). ESTABLISHED that decoding config shifts detector behaviour (also seen in DetectLLM and EMNLP 2025 snippets below).
- **Temperature hurts detection.** An EMNLP 2025 paper reports that raising inference temperature significantly reduces detection accuracy. — [search snippet, aclanthology 2025.emnlp-main.1607](https://aclanthology.org/anthology-files/pdf/emnlp/2025.emnlp-main.1607.pdf). Not fetched.
- **TempTest.** A zero-shot detector targeting the local normalization distortion from temperature/top-k sampling, agnostic of the generator. — [arXiv 2503.20421](https://arxiv.org/html/2503.20421v1). Not fetched in full; search snippet only.
- **System and prompt effects.** In Idiosyncrasies, length-controlled and format-controlled prompts kept 93.0% and 91.4% (versus 96.3%) for instruct models. Format instructions shave a few points but do not erase identity. — [Sun et al., arXiv 2502.12150](https://arxiv.org/html/2502.12150). SOURCE-CLAIM.
- **Active fingerprinting.** LLMmap claims robustness to system prompts, sampling settings, RAG and CoT, identifying "42 different LLM versions with over 95% accuracy" from ~8 interactions. It requires active querying. — [Pasquini et al., arXiv 2407.15847](https://arxiv.org/abs/2407.15847). Not applicable offline, but it shows identity survives config changes.

### Inferences
- If the competition generated each model under several configs, a two-head model (identity plus auxiliary config or prompt-template head) could help in two ways: (a) through multi-task regularization, and (b) by letting likelihood features be interpreted conditional on estimated temperature. The evidence above suggests text classifiers already absorb most config variance (FDLLM), so expected gain is modest. Larger gains are possible only if label confusions correlate with config. INFERRED.
- A "temperature-normalized" likelihood feature, using the per-text ML temperature estimate under a proxy (2601.02320) as an explicit covariate, is a cheap way to decouple config from identity in Q1 features. Novel to the competition, low novelty to research. INFERRED.
- A repetition penalty should show up as suppressed n-gram and token repetition. Simple counts of repeated tokens or bigrams per length are a free proxy. INFERRED, untested here.

### Gaps
- I found no paper testing whether explicit nuisance or config modeling improves multi-class attribution accuracy.
- Error figures for 2601.02320 temperature estimation, and its proxy-mismatch sensitivity, are not visible from the abstract.
- RAID's exact decoding grid was not verified here; only "4 decoding strategies" was. My recollection is greedy/sampling × repetition penalty on/off, but that is unverified.

---

## 4. Attribution benchmarks: what accuracies are reported for fine-grained model and version discrimination, and what drives them?

### Takeaway
Across independent papers, distinguishing model families or vendors is easy (≈90-99%) for a fine-tuned LM-embedding classifier on in-distribution data. Distinguishing sizes or versions within one family is much harder (≈60% 4-way in Qwen2.5 sizes). Accuracy collapses under domain shift, and to about 50% on an out-of-distribution QA set in FDLLM. The main drivers are word choice and characteristic phrases, then the first words, then markdown and formatting, with semantic or topical preferences as a residual.

### Cited Findings
- **Idiosyncrasies in LLMs (Sun et al., 2025).** LLM2vec embeddings, LoRA rank 16, linear head.
  - Headline accuracy: ChatGPT/Claude/Grok/Gemini/DeepSeek reach 97.1% 5-way. Instruct models (Llama/Gemma/Qwen/Mistral) reach 96.3%; base models reach 87.3%.
  - Within Qwen2.5 sizes (7B/14B/32B/72B): 59.8% 4-way, with the best pair at 85.5%. Qwen2.5-7B base vs instruct: 96.8%.
  - Ablations for chat / instruct / base:
    - Removing special characters: 95.1 / 93.8 / 75.4
    - Shuffling words: 88.9 / 88.9 / 68.3
    - Markdown elements only: 73.1 / 77.7 / 38.5
  - A TF-IDF + logistic-regression baseline reached 85.5% (chat).
  - Characteristic phrases: ChatGPT favours "such as/certainly/overall"; Claude favours "here/according to/based on". Claude uses "the/and/to/of" far less often.
  - The first token alone gives about 50%, and accuracy saturates beyond ~256 tokens.
  - Robustness: paraphrase 91.4%, translation 91.8%, summarization 58.1%.
  - Cross-dataset transfer for instruct models ≈88-99.8%.
  - Code: github.com/locuslab/llm-idiosyncrasies (not opened).
  - — [arXiv 2502.12150](https://arxiv.org/html/2502.12150). The core finding (high vendor-level separability, driven by lexical choice) is ESTABLISHED: it is corroborated by FDLLM, McGovern et al. and Bitton et al. below. The specific numbers are SOURCE-CLAIM.
- **FDLLM / FD-Dataset (2025).** 90K samples, 20 LLMs (12 proprietary including GPT-4o vs GPT-4o-mini vs GPT-3.5, GLM4-Flash vs GLM4-Plus; 8 open), English and Chinese.
  - Qwen2.5-7B-Instruct + LoRA reaches 96.56% accuracy and 96.56 macro-F1, versus 73.84% for the strongest baseline (OpenAI-D).
  - Few-shot LoRA with 50 samples per new model: 95.63% accuracy on 4 unseen models (GPT-4.1 97.5, GPT-4.1-mini 95.0), "under 20 minutes on a single RTX 3090".
  - OOD Quora QA: only 50.25% accuracy.
  - Translation attack: attack success rate 66.55% (en).
  - Internal inconsistency: the abstract says "22.1% higher" but the text says "29 points"; the table supports ~22 points.
  - — [arXiv 2501.16029v4](https://arxiv.org/html/2501.16029v4). SOURCE-CLAIM.
- **Your LLMs Are Leaving Fingerprints (2024).** Lexical and morphosyntactic (n-gram/POS) fingerprints are "often persistent across models in the same model family" (for example llama-13b vs llama-65b). Chat-tuned models are easier to detect. Robust across 5 datasets. — [McGovern et al., arXiv 2405.14057](https://arxiv.org/abs/2405.14057). SOURCE-CLAIM, consistent with the within-family difficulty above.
- **Detecting Stylistic Fingerprints (2025).** Four families (Claude, Gemini, Llama, OpenAI) with a three-classifier unanimous ensemble: "precision (0.9988)" and "false-positive rate (0.0004)". Also distinguishes seen from unseen models. Precision comes at the cost of abstention. — [Bitton et al., arXiv 2503.01659](https://arxiv.org/abs/2503.01659). SOURCE-CLAIM.
- **Who Taught You That? (2025).** "n-gram similarity alone is unreliable for identifying teachers", while POS templates favoured by students "mimic those of their teachers". — [arXiv 2502.06659](https://arxiv.org/abs/2502.06659). SOURCE-CLAIM. Relevant if the competition includes distilled models: syntactic templates may beat lexical n-grams for lineage.
- **SemEval-2024 Task 8, Subtask B (older generators).** Human + davinci-003, ChatGPT, Cohere, Dolly-v2, BLOOMz, with 70 participants. Top system AISPACE reached "the highest accuracy of 90.85" against a baseline of 74.61%. Winning systems used ensembles of fine-tuned RoBERTa/DeBERTa/XLNet/Longformer/T5 with augmentation. The 3rd-place system used a hierarchical human → {ChatGPT, Cohere} vs {davinci, BLOOMz, Dolly} → ChatGPT vs Cohere cascade on LLaMA-2-70B embeddings. — [Wang et al., arXiv 2404.14183](https://arxiv.org/html/2404.14183). ESTABLISHED as a shared task. The cascade shows the hierarchical-specialist design in practice.
- **MAGE (2023, older).** The top detector "can identify 86.54% out-of-domain texts generated by a new LLM". — [Li et al., arXiv 2305.13242](https://arxiv.org/abs/2305.13242)
- **M4 (2023, older).** Detectors struggle on "instances from unseen domains or LLMs", often labelling machine text as human. — [Wang et al., arXiv 2305.14902](https://arxiv.org/abs/2305.14902). Generalization failure is ESTABLISHED (M4, RAID, MultiGhostBench, FDLLM OOD).
- **OpenTuringBench (2025).** Open-LLM attribution benchmark with tasks for manipulated text, OOD and unseen models. Its OTBDetector uses contrastive learning. Dataset: huggingface.co/datasets/MLNTeam-Unical/OpenTuringBench. — [La Cava & Tagarelli, arXiv 2504.11369](https://arxiv.org/abs/2504.11369). The HTML full text returned 404, so numbers were not retrieved.
- **Instance-level and family fingerprinting beyond text.**
  - FLIPS reports 96% closed-set accuracy over 237 instances from 25 LLMs (including abliterated variants) — [arXiv 2606.03330 via search](https://arxiv.org/html/2606.03330).
  - "One Token Is Enough" reports family recovery of 59.5% leave-one-out versus 18.4% chance, across 165 OpenRouter models — [arXiv 2607.10252 via search](https://www.alphaxiv.org/abs/2607.10252).
  - Hide and Seek reaches 72% family accuracy — [arXiv 2408.02871](https://arxiv.org/abs/2408.02871).
  - All three need active prompting or query access. Snippet-level verification only for the 2026 items.

### Inferences
- For the Kaggle task, the hard confusions will almost certainly sit within a family or between adjacent versions (sizes, -mini variants, successive releases). That is where all the papers above report the biggest drops. A hierarchical family → member cascade with specialist heads, trained on within-family pairs only, is the most directly evidenced structural change (SemEval 3rd place; Idiosyncrasies within-family numbers). INFERRED.
- Markdown and formatting carry 73-78% accuracy alone for chat and instruct models. Any preprocessing that strips formatting (for example markdown-to-plain conversion) throws away signal. Separate formatting-only features (counts of headers, bullets, bold, emoji, em-dashes) are cheap and complementary. INFERRED from SOURCE-CLAIM.
- Accuracy saturating at ~256 tokens, and ~50% from the first token alone, suggest two things: features from the opening words carry outsized signal, and a long-context model may give diminishing returns beyond ~256-512 tokens. Useful for runtime planning (Q6). INFERRED.
- Large OOD drops (FDLLM 96% to 50%; MultiGhostBench) mean validation must be prompt or source grouped. Otherwise local CV will overstate performance. This matches the brief's validation concern. INFERRED.

### Gaps
- I found no paper with a public, multi-version confusion matrix for frontier closed models (for example GPT-4o vs GPT-4.1 vs GPT-5-series, or Claude 3.5 vs 3.7 vs 4). FDLLM covers GPT-4o vs 4o-mini vs 3.5 but gives no per-pair confusion in what I read.
- OpenTuringBench numbers are unretrieved (HTML 404, PDF not fetched).
- No 2026 frontier-model-scale attribution benchmark with per-version numbers was found, beyond MultiGhostBench (5 generators, numbers not visible).

---

## 5. Contrastive and embedding-based generator fingerprints; open-set and few-shot attribution

### Takeaway
Contrastive "LLM as author" embeddings with kNN or prototype inference are the strongest-evidenced non-standard approach:
- DeTeCtive reaches ~92-93 AvgRec on unseen models and adds training-free incremental adaptation through its kNN database.
- UAR style embeddings beat likelihood features in few-shot settings.
- FDLLM adapts to new models with 50 samples per model.

For Kaggle, the transferable piece is the training objective (multi-level supervised contrastive loss over family and model labels) and kNN or prototype heads ensembled with softmax heads. It is not a new backbone.

### Cited Findings
- **DeTeCtive (NeurIPS 2024).**
  - Method: multi-level contrastive loss over human / model-family / individual-model labels, plus cross-entropy, on a SimCSE-RoBERTa-base backbone, with kNN (Faiss) inference.
  - In-distribution results: M4-mono 98.44 AvgRec; TuringBench 99.74.
  - OOD unseen models: 92.19, or 93.03 with Training-Free Incremental Adaptation (TFIA, which adds encoded samples to the database). Unseen domains: 82.60, or 89.63 with TFIA.
  - Replacing the multi-level loss with a plain contrastive loss drops AvgRec from 96.15 to 94.22.
  - TuringBench multi-class authorship attribution: accuracy 82.75, F1 83.05.
  - Code: github.com/heyongxin233/DeTeCtive.
  - — [He et al., arXiv 2410.20964](https://arxiv.org/html/2410.20964). SOURCE-CLAIM for the specific numbers.
- **Few-shot style representations (ICLR 2024).**
  - Method: UAR, an ~82M RoBERTa trained contrastively on ~5M Reddit authors. Given N support docs per LLM, query docs are scored by similarity.
  - pAUC at FPR ≤ 1%, N=5/N=10: UAR 0.868/0.958 (multi-domain UAR 0.886/0.968); ProtoNet 0.871/0.948; SBERT 0.621/0.716; LogRank 0.764/0.775.
  - In-distribution, a fine-tuned RoBERTa beats UAR (0.967 vs 0.855). Under topic and domain shift both collapse to ~0.5, with UAR slightly ahead.
  - Code: github.com/LLNL/LUAR/tree/main/fewshot_iclr2024.
  - — [Soto et al., arXiv 2401.06712](https://arxiv.org/html/2401.06712). SOURCE-CLAIM.
- **FDLLM few-shot.** 50 samples per new LLM via incremental LoRA gives 95.63% average accuracy on 4 unseen models. — [arXiv 2501.16029v4](https://arxiv.org/html/2501.16029v4). SOURCE-CLAIM.
- **MAGE.** 86.54% on OOD texts from a new LLM (detection framing). — [arXiv 2305.13242](https://arxiv.org/abs/2305.13242)
- **OpenTuringBench OTBDetector.** A contrastive framework that outperforms "most existing detectors" (no numbers retrieved). — [arXiv 2504.11369](https://arxiv.org/abs/2504.11369)
- **Open-set at the family level.** The stylistic-fingerprint ensemble distinguishes seen vs unseen models with unanimous-vote abstention. — [Bitton et al., arXiv 2503.01659](https://arxiv.org/abs/2503.01659)

### Inferences
- **For a closed label set**, the value of contrastive training is likely (a) better-separated within-family clusters through the family-level term, and (b) a kNN head whose errors differ from a softmax head's, giving ensemble diversity. Distance-to-prototype features can also flag test texts that look off-distribution. INFERRED.
- **Content-controlled contrastive pairs** are novel to the competition and only partly novel to research. If the data offers several generators answering the same prompt, positives can be defined as the same model on different prompts, and hard negatives as different models on the same prompt. This directly pushes the embedding away from topic and content, which the OOD drops in Q4 identify as the main failure. No paper above reports exactly this prompt-matched hard-negative scheme for attribution, so its evidence is INFERRED, extrapolated from DeTeCtive's multi-level gain and the OOD literature.
- **A kNN over an embedding database (TFIA)** lets external or new training data be added without retraining. That is cheap in an offline notebook: build the Faiss index at inference from pre-computed embeddings shipped as a dataset. INFERRED.

### Gaps
- No paper found reports DeTeCtive-style contrastive training applied to modern 7B+ decoder backbones for attribution among many frontier chat models. Its results use RoBERTa-base on 2023-era datasets.
- Open-set attribution (an "unknown model" class) is relevant only if the competition test set contains unseen generators. That is not verified here.

---

## 6. Runtime: which techniques fit a T4x2 offline Kaggle notebook (~9-12h, no internet)?

### Takeaway
Everything except perturbation-based curvature (DetectGPT) and active querying (LLMmap, Hide and Seek, FLIPS, One Token) fits comfortably:
- Small-proxy likelihood features (0.1-2B models, one forward pass per text), tokenizer and formatting statistics (CPU), and kNN or prototype heads are cheap add-ons to an existing 7-8B LoRA classifier.
- The per-candidate likelihood-ratio test is limited to open-weight candidates.
- Fast-DetectGPT and Binoculars each cost 1-2 forward passes.

No paper reports T4 throughput, so every budget figure below is INFERRED.

### Cited Findings
- Fast-DetectGPT is "340x" faster than DetectGPT. DetectGPT needs many perturbation passes per text. — [arXiv 2310.05130](https://arxiv.org/abs/2310.05130); [arXiv 2301.11305](https://arxiv.org/abs/2301.11305)
- Binoculars needs two LMs (observer and performer) over the same tokenization and no training. — [arXiv 2401.12070](https://arxiv.org/abs/2401.12070)
- Small proxies (OPT-125M) can be better than large ones for cross-generator detection. — [arXiv 2305.09859](https://arxiv.org/abs/2305.09859)
- FDLLM attribution is "a single forward pass without querying candidate LLM APIs", with incremental adaptation in "under 20 minutes on a single RTX 3090". — [arXiv 2501.16029v4](https://arxiv.org/html/2501.16029v4)
- Idiosyncrasies accuracy saturates beyond ~256 tokens, which bounds useful sequence length. — [arXiv 2502.12150](https://arxiv.org/html/2502.12150)
- The black-box (sampling) variant of the statistical test needs about m=1000 samples per position. That is infeasible offline, so only the white-box variant with open-weight candidates applies. — [arXiv 2501.02406v6](https://arxiv.org/html/2501.02406v6)
- LLMmap, Hide and Seek, FLIPS and One Token all require querying the models. Not applicable. — [arXiv 2407.15847](https://arxiv.org/abs/2407.15847); [arXiv 2408.02871](https://arxiv.org/abs/2408.02871)

### Inferences

Transferability matrix (all INFERRED; "novel-comp" means not a standard fine-tuned text classifier):

| Technique | Prereqs | Offline T4x2 inference cost (assumes ≤~50k texts × ≤512 tokens, fp16) | Evidence | Novel-comp / novel-research | Verdict |
|---|---|---|---|---|---|
| Small-proxy likelihood feature vector (log-prob/rank/entropy quantiles from 2-3 open 0.5-2B models) + GBDT/stacking | Proxy weights as Kaggle dataset; per-token logits | 1 forward pass per proxy per text; minutes to ~1h total | Detection ESTABLISHED; attribution SOURCE-CLAIM, older (Sniffer, GPT-who, LLMDet) | Likely novel to competition / old in research | Try as a cheap diversity add-on |
| Binoculars cross-perplexity as a feature | Two LMs sharing a tokenizer (e.g. base + instruct of one family) | 2 passes | Detection only | Novel-comp / not novel | Fold into the row above |
| Per-candidate likelihood ratio (white-box) | Weights of each candidate generator (open models only) | 1 pass per candidate; 7-8B models are heavy at fp16 on T4 (16GB) and need 4-bit or smaller siblings | SOURCE-CLAIM (2501.02406); same-family weakness | Novel-comp / moderate | Only for open-weight classes; check label set |
| DetectGPT perturbation curvature | Mask-filling model + scorer | Tens of passes per text | Older; detection only | — | Reject (runtime) |
| Temperature ML estimate as a covariate | One proxy LM | Reuses proxy logits; near free | SOURCE-CLAIM, no error figures | Novel-comp / new-ish (2026) | Low-cost covariate; test |
| Tokenizer re-tokenization and truncation-cap check | Candidate tokenizers (files only) | CPU, seconds to minutes | No direct evidence; truncation idea untested | Novel to both if it works | Check data for length caps first; near-free |
| Formatting and lexical stats (markdown counts, first words, TF-IDF LR) | None | CPU | ESTABLISHED as signal (Idiosyncrasies, McGovern) | Likely already captured by LM classifier | Cheap stacking feature |
| Multi-level contrastive training + kNN/prototype head | Family taxonomy of labels | Same as the classifier, plus a Faiss lookup | SOURCE-CLAIM (DeTeCtive), small backbones | Novel-comp / moderate | Strong candidate; adds error diversity |
| Prompt-matched hard-negative contrastive learning | Multiple generators per prompt in training data | Training-time only | INFERRED | Novel-comp / partly novel research | High upside if prompt-matched data exists |
| Hierarchical family → member specialist cascade | Family taxonomy | 1 extra small head pass | SemEval 3rd place; within-family gap ESTABLISHED | Possibly used publicly / not novel | Targets the hardest confusions |
| Active fingerprinting (LLMmap etc.) | Query access to models | — | — | — | Reject (no internet, fixed texts) |

- **Budget arithmetic** (INFERRED, assumption-laden). A T4 does roughly tens of TFLOPs fp16 in practice. One forward pass of a 1B model over 512 tokens is about 1e12 FLOPs, so 50k texts is about 5e16 FLOPs, which is minutes per proxy. A 7-8B model is ~7-8x that, fitting in ~1h per pass in 4-bit or fp16 with both GPUs. The binding constraint is the existing 7-8B classifier, not the proxy features. The actual test-set size and length were not verified here.

### Gaps
- The hidden test-set size, text length distribution and exact Kaggle runtime limit were not verified in this research thread. Other researchers or the main report should confirm them.
- No primary measurement of T4 throughput for these specific pipelines was found. All cost figures are estimates.
- Whether the competition label set includes open-weight models (which enables per-candidate likelihood) is unknown to this thread.
