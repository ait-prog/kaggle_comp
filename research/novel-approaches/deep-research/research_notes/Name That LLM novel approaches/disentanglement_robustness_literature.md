# Disentangling generator fingerprint from prompt/content: transferable methods from authorship attribution, domain generalization and MGT detection

Scope: methods that push a text classifier to attend to WHO generated the text (which LLM) rather than WHAT it is about. Research date 2026-11-12. All arXiv abstracts below were retrieved directly from the arXiv API or arXiv/ACL pages in this session unless marked otherwise.

Labels: **ESTABLISHED** = replicated or a primary result reported in a peer-reviewed or widely used benchmark paper. **SOURCE-CLAIM** = reported by one paper (often a preprint) and not independently replicated here. **INFERRED** = my reasoning about transfer to Name That LLM. "Novel to competition" means I believe no public Kaggle family uses it. That rests on the other researchers' notebook survey and was not verified here. "Novel to research" means absent from the literature I found.

Runtime reference: offline Kaggle notebook, about 2x T4 (16 GB each, no bf16, fp16 only), 9-12 h wall clock.

## Q1. Content-controlled contrastive learning (style embeddings, prompt-matched pairs)

### Takeaway
Content-controlled contrastive training is well established in human authorship work. Positives share an author but differ in content; negatives share content but differ in author. It measurably reduces topic leakage. The LLM-attribution literature already shows that same-prompt multi-model data plus contrastive or kNN objectives generalize well out of domain (DeTeCtive, Idiosyncrasies). No paper I found isolates "prompt-matched hard negatives" as an ablation for LLM attribution, so that specific claim is INFERRED/thin. For Name That LLM, the approach is probably novel to the competition but not to research.

### Cited Findings
- **ESTABLISHED (problem statement):** Authorship-verification-trained style representations "might also encode content information instead of style alone." Controlling for content by sampling negatives from the same conversation (same-conversation, different-author) gives representations that are "better than representations trained with domain or no content control" at representing style independently of content. — [Wegmann, Schraagen, Nguyen 2022, "Same Author or Just Same Topic?"](https://arxiv.org/abs/2204.04907)
  - Direct analogue (INFERRED): "same conversation" maps to "same prompt". The hard negative for a Name That LLM text is another model's answer to the same prompt. The positive is the same model's answer to a different prompt.
- **ESTABLISHED (evaluation tool):** STEL is a "content-controlled similarity-based" style evaluation framework. It tests whether a model prefers style over content using paired sentences. — [Wegmann & Nguyen 2021, STEL](https://arxiv.org/abs/2109.04817). A STEL-like probe could be built for LLM attribution: pairs where content is held fixed and the generator varies. (INFERRED)
- **SOURCE-CLAIM:** StyleDistance uses an LLM to make "near-exact paraphrases with controlled style variations" over 40 style features. It notes that ordinary contrastive triplets "may vary in both style and content, leading to potential content leakage." The authors report better content-independence and downstream results. Model at huggingface.co/StyleDistance/styledistance. — [Patel et al. 2024, StyleDistance](https://arxiv.org/abs/2410.12757)
- **SOURCE-CLAIM:** LISA builds interpretable style embeddings from LLM-annotated stylometry. — [Patel et al. 2023](https://arxiv.org/abs/2305.12696)
- **ESTABLISHED (LUAR):** "Learning Universal Authorship Representations" (Rivera-Soto et al., EMNLP 2021) is a large cross-domain study of authorship representations with zero-shot transfer across Amazon reviews, fanfiction and Reddit. "Transfer works surprisingly well between some domains but poorly between others." — [ACL Anthology 2021.emnlp-main.70](https://aclanthology.org/2021.emnlp-main.70/). Precursor deep-metric account-linking model: [Khan et al. 2021](https://arxiv.org/abs/2105.07263).
- **SOURCE-CLAIM (ICLR 2024):** Style representations learned from human-authored text (LUAR-family) work for few-shot detection of machine text: "features effective at distinguishing among human authors are also effective at distinguishing human from machine authors," including GPT-4 and Llama-2. — [Soto et al. 2024, Few-Shot Detection of MGT using Style Representations](https://arxiv.org/abs/2401.06712)
- **SOURCE-CLAIM:** Contra-X (contrastive fine-tuning for authorship attribution) improves accuracy by up to 6.8% over cross-entropy fine-tuning on human and machine authorship benchmarks. The abstract warns that it "improves overall accuracy at the cost of sacrificing performance for some" authors. — [Ai et al. 2022, Whodunit?](https://arxiv.org/abs/2209.11887)
- **SOURCE-CLAIM (closest LLM-attribution analogue, NeurIPS 2024):** DeTeCtive uses a multi-level contrastive loss with labels for human vs AI, model series/company, and specific model. It pulls same-model texts closest, then same-family texts, then other AI. Inference is kNN over a Faiss database of training embeddings. Training-Free Incremental Adaptation (TFIA) adds new embeddings to that database. Backbone: SimCSE-RoBERTa-base (about 125M parameters), 50 epochs on 8xV100. Reported results: M4-monolingual AvgRec 98.44; Unseen Domains 82.60, or 89.63 with TFIA (+7.03); Unseen Models 92.19, or 93.03 with TFIA. — [Guo et al. 2024, DeTeCtive](https://arxiv.org/abs/2410.20964) ([HTML](https://arxiv.org/html/2410.20964v1))
- **SOURCE-CLAIM (same prompts across models; key evidence on what the signal is):** Idiosyncrasies collected responses from every model to the same UltraChat prompts (11K per model, same split across models) and trained an LLM2vec + LoRA classifier with 512-token truncation. Reported accuracy: 97.1% for 5 chat APIs, 96.3% for 4 instruct models, 87.3% for base models. Cross-prompt-source generalization: about 88–99%. — [Sun et al. 2025, Idiosyncrasies in LLMs](https://arxiv.org/abs/2502.12150) ([HTML v2](https://arxiv.org/html/2502.12150v2)). Controls from Table 8–9 (chat / instruct / base):
  - Words shuffled: 88.9 / 88.9 / 68.3.
  - Special characters removed: 95.1 / 93.8 / 75.4.
  - Markdown-only skeleton: 73.1 / 77.7 / 38.5.
  - TF-IDF + LR: 85.5 / 83.7 (chat / instruct).
  - First token only: about 50%.
  - Rewriting by GPT-4o-mini: paraphrase 91.4 / 92.2 / 71.7; summarize 58.1 / 57.5 / 44.7.
  - Within-family difficulty: Qwen2.5 instruct 7B/14B/32B/72B 4-way only 59.8%; Qwen2.5-7B base vs instruct 96.8%.
  - Sampling-temperature identification for Llama3.1-8B: at most 59.6% pairwise and 37.9% for 5 temperatures.
  - The paper has internal inconsistencies. The text says letter-shuffling is 49–56% while Table 8 shows about 39%. The text calls base-model markdown accuracy "near chance" while the table shows 38.5%.
- **SOURCE-CLAIM:** Shared-prompt multi-LLM data has been used to study paraphrase effects, where "multiple authors write on the same subject." — [Tripto et al., Ship of Theseus](https://arxiv.org/pdf/2311.08374) (from search snippet only; not fetched)
- **SOURCE-CLAIM (newest, Nov 2026):** Yadav et al. model authors as mixtures of "multiple consistent but distinct writing styles." Only the abstract was retrieved. A search-engine summary says they report StyleDistance results and residual cross-subreddit confounding, but this was not verified on the abstract page. — [Yadav et al. 2026](https://arxiv.org/abs/2611.07111)
- **SOURCE-CLAIM (survey):** "author-supervised learning can retain content", and "evaluation protocols can reverse model rankings". The survey recommends designs with "different authors writing within the same conversation, and parallel texts with controlled content." — [Authorship identification under domain shift survey, 2310.00436 v2](https://arxiv.org/abs/2310.00436)

### Inferences
- **INFERRED (mechanism for Name That LLM):** If the training pool has, or can be grouped into, prompt-matched sets (same prompt, several generators), a supervised-contrastive or InfoNCE loss can use same-label/different-prompt positives and same-prompt/different-label hard negatives. This is Wegmann's "conversation control" transposed to generators. A cheaper variant keeps cross-entropy and adds a pairwise margin term only between same-prompt pairs, or uses listwise softmax across all K answers to one prompt.
- **INFERRED (prerequisite):** This needs prompt IDs or a reliable prompt-grouping proxy, such as near-duplicate prompt text or embedding clustering of the first user turn. Without grouping, "hard negatives" fall back to semantic-nearest neighbours of other classes, which is weaker but still feasible.
- **INFERRED (what Idiosyncrasies implies):** Most of the signal is in word choice and distribution. It survives word shuffling and paraphrasing. Formatting/markdown alone carries a lot for chat and instruct models. So a content-controlled objective should not remove lexical signal wholesale; masking all content words would throw away real fingerprint (see Q2). Content control should target prompt-specific vocabulary, not every word.
- **INFERRED (where it helps most):** The hard regime is within-family, between size or version siblings: 59.8% for 4 Qwen sizes in Idiosyncrasies, and "majority of confusions … same family … adjacent sizes" in [Text to Source](https://arxiv.org/html/2309.13322v2). Prompt-matched hard negatives give most leverage there, because sibling models answer the same prompt with near-identical content.
- **INFERRED (runtime):** Training a contrastive head on a DeBERTa-v3-base/large or a 1–8B LoRA encoder is training-side work and could be done offline as a dataset. Inference is one forward pass plus a linear or kNN head, which is cheap on T4. kNN over ≤1M stored 768-d embeddings with Faiss on CPU/GPU takes minutes.
- **Novelty:** Novel to research: no (DeTeCtive, Wegmann). Probably novel to the competition, if public notebooks are plain cross-entropy classifiers (verify against the notebook survey). The prompt-matched hard-negative ablation for LLM attribution appears not to be published (thin evidence; see Gaps).

### Gaps
- I found no paper that ablates "same-prompt hard negatives" against random negatives for LLM model attribution and reports out-of-topic gains. The evidence is indirect: Wegmann for human authors, DeTeCtive for hierarchy, Idiosyncrasies for cross-prompt-source transfer.
- The LUAR GitHub repo and its exact training recipe (episodes of multiple posts) were not verified in this session. The ACL page lists no code link.
- DeTeCtive's within-family attribution numbers (version-level) were not extracted. The reported metrics are mostly human-vs-AI AvgRec.

## Q2. Adversarial and invariance methods (GRL, IRM/V-REx/GroupDRO, counterfactual views/masking)

### Takeaway
In human authorship attribution, domain-adversarial training and text distortion (topic masking) both recover most of the accuracy lost to topic/domain swap: losses fall from up to 55.4 pp to about 4 pp. That is the strongest evidence in this section (ESTABLISHED, one paper plus PAN practice). Generic invariance penalties (IRM/V-REx/GroupDRO) have weak empirical support in DomainBed, where tuned ERM is competitive. Their text-attribution evidence is thin. For LLM attribution, aggressive content masking is risky because a large share of the fingerprint is lexical.

### Cited Findings
- **ESTABLISHED (key evidence):** In a fixed-author, swapped-domain setup, character-trigram attribution "suffer[s] drops of up to 55.4 percentage points." A domain-adversarial remedy and heuristic (text-distortion style) remedies reduce the loss to 3.6% and 3.9% respectively. — [Bischoff et al. 2020, The Importance of Suppressing Domain Style in Authorship Analysis](https://arxiv.org/abs/2005.14714)
- **ESTABLISHED:** The topic-confusion task swaps author-topic pairings between train and test. "Stylometric features with part-of-speech tags are the least susceptible to topic variations." Combining them with other features lowers topic confusion. BERT/RoBERTa "perform poorly on this task and are surpassed by simple features such as word-level n-grams." — [Altakrori et al. 2021, The Topic Confusion Task](https://arxiv.org/abs/2104.08530)
- **ESTABLISHED (practice):** The PAN 2019 cross-domain attribution overview lists "n-grams extracted from distorted versions" (Stamatatos 2017, text distortion) among popular features. Per the 2nd-place notebook, distorted-text n-grams were used by Bacciu et al. (0.680 macro-F1). — [Kestemont et al. 2019, Overview of PAN-2019 Cross-domain AA](https://ceur-ws.org/Vol-2380/paper_264.pdf); [Bacciu et al. 2019](https://iris.uniroma1.it/bitstream/11573/1446340/1/Bacciu_Authorship-Attribution_2019.pdf) (latter via search snippet only)
- **SOURCE-CLAIM:** ConDA frames generators as domains. It combines domain adaptation with contrastive learning to learn "domain invariant representations" from labeled source and unlabeled target data, reporting "average performance gains of 31.7%" over the best baselines on AI-news detection. — [Bhattacharjee et al. 2023, ConDA](https://arxiv.org/abs/2309.03992)
- **SOURCE-CLAIM:** DEER uses a disentangled mixture-of-experts with domain-local and domain-invariant experts and an RL router. It reports modest OOD gains (+2.92 F1, +2.26 acc on OOD averages). — [DEER 2025](https://arxiv.org/abs/2511.01192)
- **ESTABLISHED (invariance objectives, definitions):**
  - IRM learns a representation whose optimal classifier "matches for all training distributions" — [Arjovsky et al. 2019](https://arxiv.org/abs/1907.02893).
  - V-REx penalises "the variance of training risks" — [Krueger et al. 2020](https://arxiv.org/abs/2003.00688).
  - GroupDRO minimises worst-group loss and needs "stronger-than-typical L2 penalty or early stopping" to help. It reports 10–40 pp worst-group gains on NLI and images — [Sagawa et al. 2019](https://arxiv.org/abs/1911.08731).
- **ESTABLISHED (counter-evidence):** DomainBed found that carefully implemented ERM "shows state-of-the-art performance across all datasets" vs nine DG algorithms. Model selection is decisive. — [Gulrajani & Lopez-Paz 2020](https://arxiv.org/abs/2007.01434)
- **SOURCE-CLAIM (MGT detection under shift, 2026):** A plain fully fine-tuned RoBERTa matches specialized detectors in-distribution but "degrades sharply when the topic domain or generating model changes," and "simply adding more source data does not close the gap." — [Rethinking AI-Generated Text Detection, 2607.03680](https://arxiv.org/abs/2607.03680)
- **ESTABLISHED (benchmarks showing shift fragility):**
  - MAGE: OOD difficulty; best detector 86.54% on OOD texts from a new LLM — [Li et al. 2023](https://arxiv.org/abs/2305.13242).
  - RAID: detectors fooled by "variations in sampling strategies, repetition penalties, and unseen generative models"; 11 models, 8 domains, 4 decoding strategies — [Dugan et al. 2024](https://arxiv.org/abs/2405.07940).
- **SOURCE-CLAIM (counterfactual views on LLM attribution):** Idiosyncrasies' views give concrete numbers for counterfactual augmentations. Word shuffle keeps 88.9% (chat). Markdown skeleton keeps 73–78% for chat/instruct. Letter shuffle drops to about 39%. — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2)

### Inferences
- **INFERRED (most transferable recipe):** Text distortion / topic masking (Stamatatos-style: replace low-frequency or content words with placeholders, keep the top-k most frequent words and punctuation) is a cheap, proven anti-topic view in human AA. For Name That LLM, use it as an additional view or ensemble member, not a replacement. Idiosyncrasies shows lexical choice is a main LLM fingerprint, so full masking loses signal.
- **INFERRED:** A gradient-reversal topic adversary needs topic labels. Prompt-cluster IDs (k-means on prompt or response embeddings) can serve as pseudo-domains, and the adversary predicts the cluster from the attribution encoder's features. Bischoff et al. used real domain labels; pseudo-domains are untested. GRL training is notoriously unstable (λ schedule). This is training-only cost with no inference cost.
- **INFERRED:** IRM/V-REx/GroupDRO need environments. Natural environments here are data source (competition train vs external pools), prompt cluster, or length bucket. Given DomainBed, expect small or no average gain. GroupDRO is most defensible if the leaderboard metric penalises weak classes (e.g., macro-F1), since it targets worst groups. Use it as a cheap ablation (V-REx is one extra loss term), not a bet.
- **INFERRED (shortcut diagnostic, cheap and high value):** Run the Topic-Confusion-style split on local validation. Group-split by prompt or topic cluster so that no prompt in validation appears in training. Compare random-split vs group-split accuracy. A big gap is direct evidence of a content shortcut and is the precondition for any method in this section to matter.
- **Novelty:** Text distortion views and GRL are novel to research for human AA only (they date from 2017–2020). For LLM attribution they are lightly explored (ConDA, DEER for detection only). For this competition they are likely novel (unverified).

### Gaps
- No paper found that applies topic masking or GRL specifically to multi-class LLM model attribution and reports OOD gains. Detection (human vs AI) papers dominate.
- Original DANN paper (Ganin & Lempitsky, arXiv 1409.7495) and Stamatatos 2017 "Authorship attribution using text distortion" (EACL) were not fetched in this session (API timeout). They are cited only via secondary references above.
- IRM/V-REx/GroupDRO evidence on text authorship is absent from what I found.

## Q3. Multi-view sequence statistics (function words, punctuation/markdown, length/structure, compression)

### Takeaway
Cheap stylometric views (POS/function-word n-grams, punctuation, markdown skeleton, length) carry substantial LLM-identity signal and are among the most topic-robust features in human AA. Compression/PPM methods are competitive baselines in cross-domain AA (PAN 2019 compressor 0.533 macro-F1 vs SVM 0.545). The famous "gzip beats BERT" claim was inflated by a tie-breaking bug. Their value is ensemble diversity, not standalone accuracy.

### Cited Findings
- **SOURCE-CLAIM:** Simple n-gram and POS classifiers detect machine text well. LLMs leave "unique fingerprints" as frequency differences in lexical and morphosyntactic features. These fingerprints persist within families (e.g., llama-13b vs llama-65b) and hold in- and out-of-domain. Chat-tuned models are easier to detect. — [McGovern et al. 2024, Your LLMs Are Leaving Fingerprints](https://arxiv.org/abs/2405.14057)
- **SOURCE-CLAIM:** Stylometric lexical/syntactic/structural features can "augment pre-trained language model-based classifiers" for neural authorship attribution across proprietary and open LLMs. — [Kumarage & Liu 2023](https://arxiv.org/abs/2308.07305)
- **SOURCE-CLAIM:** Markdown/formatting-only views reach 73.1% (chat 5-way) and 77.7% (instruct 4-way) but 38.5% for base models. Removing special characters costs about 2–3 pp for chat/instruct. Prompting for plain-text output drops instruct accuracy from 96.3 to 91.4, and a length limit drops it to 93.0. — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2)
- **ESTABLISHED:** POS-based stylometry is the least topic-susceptible feature family in the topic-confusion task — [Altakrori et al. 2021](https://arxiv.org/abs/2104.08530)
- **ESTABLISHED (compression in AA, PAN 2019 open-set cross-domain fanfiction):**
  - Macro-F1: winner Muttenthaler et al. 0.690; majority-vote meta-baseline 0.686; Bacciu et al. 0.680; van Halteren 0.598; BASELINE-SVM (char 3-gram) 0.545; BASELINE-COMPRESSOR (PPM cross-entropy per author) 0.533; BASELINE-IMPOSTERS 0.395.
  - The compressor "achieves the best baseline results for English, French, and Italian."
  - The top cohort "almost exclusively adopted a highly similar approach" (SVM + char n-grams + thresholding). The organisers attribute the strong majority-vote result to "lack of diversity."
  - Source: [Kestemont et al. 2019](https://ceur-ws.org/Vol-2380/paper_264.pdf)
- **SOURCE-CLAIM:** Compression-model authorship verification gives competitive results without ML, NLP or feature engineering. — [Halvani et al. 2017](https://arxiv.org/abs/1706.00516)
- **CONTESTED:** gzip+kNN text classification claimed parity with non-pretrained DNNs and to beat BERT OOD — [Jiang et al. (arXiv 2212.09410)](https://arxiv.org/abs/2212.09410). Re-analysis found the reported accuracy relied on a lenient top-2 tie-break. Realistic tie-breaking cut kNN performance "by up to about 9 accuracy points on average," and bag-of-words matched or beat gzip. — [Opitz 2023, Gzip versus bag-of-words](https://arxiv.org/abs/2307.15002); [Sinai blog replication](https://liorsinai.github.io/coding/2023/08/13/gzip-knn.html) (secondary). Original Schutte posts were not fetched.
- **SOURCE-CLAIM:** An ensemble of three diverse classifiers over four LLM families (Claude, Gemini, Llama, OpenAI), predicting only on unanimous agreement, reaches precision 0.9988 and FPR 0.0004. This trades coverage for precision. — [Bitton et al. 2025, Detecting Stylistic Fingerprints of LLMs](https://arxiv.org/abs/2503.01659)

### Inferences
- **INFERRED (priority view set for diversity):**
  1. Markdown/format skeleton: headers, bullets, bold, code fences, emoji, list depth, line-break patterns.
  2. Function-word and POS n-grams.
  3. Punctuation sequences.
  4. Length and structure: tokens, sentences, paragraph counts, ratio of answer length to prompt length if available.
  5. Per-class PPM/zstd cross-entropy (one compressor dictionary per class, then compressed-size deltas).

  All are CPU-only, minutes on Kaggle, and fit a GBDT or LR stacker. They are likely to make errors decorrelated from a transformer's, which is what PAN 2019's majority-vote result rewards.
- **INFERRED:** Per-class compression is O(N_test × K) compressions. With zstd dictionaries trained per class, this is fast (seconds to minutes for 10^4–10^5 texts). Avoid pairwise NCD kNN, which is O(N_test × N_train) and slow, and whose evidence is weaker than claimed.
- **INFERRED (caveat):** Formatting signal depends on whether the competition's data pipeline preserved or stripped markdown/whitespace. Verify against the actual data before investing.
- **Novelty:** Stylometric views are standard in research. TF-IDF char/word n-grams are likely common in public Kaggle LLM notebooks. Markdown skeleton + POS/function-word + per-class compression as a stacked diversity branch is plausibly underexplored in the competition (unverified).

### Gaps
- No direct comparison found of PPM/compression vs fine-tuned transformers on multi-class LLM attribution.
- I did not find evidence quantifying ensemble-diversity gains from adding stylometric views to a strong transformer for LLM attribution specifically. Kumarage & Liu claim augmentation, but numbers were not extracted.

## Q4. Hierarchical / specialist models, MoE, open-set and abstention

### Takeaway
LLM-attribution errors concentrate within families and between adjacent sizes or versions (ESTABLISHED by two independent papers). That justifies coarse-to-fine or pairwise-specialist designs. Direct evidence that a two-stage hierarchy beats a flat classifier for LLM version attribution is thin. DeTeCtive's hierarchical contrastive loss is the closest positive evidence. Open-set/abstention is mature in PAN, but naive reject thresholds dominate.

### Cited Findings
- **SOURCE-CLAIM:** Across 55 base models (DeBERTaV3-base, 800 train samples per model), "The majority of confusions occurred between models from the same family," and "within a model family, the confusions tend to happen between models with adjacent sizes." Reported F1: source-model 17.7%, family (12 labels) 37%, size bins 38%. Quantization was not detectable (54.5%). — [Antoun et al. 2023, From Text to Source](https://arxiv.org/html/2309.13322v2)
- **SOURCE-CLAIM:** Within-family sizes are hard (Qwen2.5 7B–72B, 4-way 59.8%), but base vs instruct of the same model is easy (96.8%). Fine-tuning two different models on the same SFT data reduced their separability from 96.5% to 59.8%. This suggests post-training data is a major fingerprint source. — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2)
- **SOURCE-CLAIM:** DeTeCtive encodes the hierarchy (AI → company/series → model) inside the contrastive loss rather than as separate stage-wise classifiers — [DeTeCtive](https://arxiv.org/html/2410.20964v1)
- **SOURCE-CLAIM:** PhantomHunter models "family-level features shared between base LLMs and their fine-tuned descendants" for detecting text from privately fine-tuned models — [arXiv 2506.15683](https://arxiv.org/html/2506.15683v1) (search snippet only; not fetched)
- **SOURCE-CLAIM:** Family-level 4-class ensemble with unanimous-agreement abstention reaches very high precision — [Bitton et al. 2025](https://arxiv.org/abs/2503.01659)
- **SOURCE-CLAIM:** MoE for MGT detection: DEER's disentangled experts with instance-adaptive routing give about 1–3 pp gains — [DEER](https://arxiv.org/abs/2511.01192)
- **ESTABLISHED (open-set practice):** In PAN 2019 open-set AA, "none of the participants attempted to build a pure open-set classifier." All used closed-set classifiers with a reject option, e.g., the baseline rejects if the top-2 Platt-probability gap is under 0.1. The organisers recommend moving "towards purely open-set classifiers." — [Kestemont et al. 2019](https://ceur-ws.org/Vol-2380/paper_264.pdf)
- **ESTABLISHED (task context):** SemEval-2024 Task 8 Subtask B is exact-source multi-class detection with 70 participating teams — [Wang et al. 2024](https://arxiv.org/abs/2404.14183)

### Inferences
- **INFERRED (design):**
  - Option A: flat classifier plus pairwise/one-vs-one specialists trained only on the top confusable pairs found in a group-split OOF confusion matrix. Each specialist re-scores the flat model's top-2 when they fall in a known confusable pair.
  - Option B: hierarchical softmax / family-aware loss, which needs only a family map.
  - Option B is cheaper and lower-risk. Option A adds K small models but targets exactly where the errors are.
- **INFERRED (specialist inputs):** Specialists for sibling versions should use features that differ between versions, such as tokenizer/likelihood features (covered by other researchers), formatting habits and refusal phrasing. They should not re-learn content.
- **INFERRED (abstention):** Abstention matters only if the metric rewards it or there are unseen classes. If the metric is plain accuracy or log-loss over closed classes, use calibration (temperature scaling per stage) rather than reject options.
- **INFERRED (MoE by length):** Idiosyncrasies' first-token experiment (~50%) and its length-limit control both imply that short texts carry much less signal. A length-bucketed router (short texts to a stylometric/likelihood specialist, long texts to the transformer) is a cheap hypothesis to test.
- **Novelty:** Hierarchical heads are known in research (DeTeCtive). Pairwise confusable-class specialists are standard ML practice but plausibly unused in public notebooks for this competition (unverified).

### Gaps
- I found no paper directly comparing two-stage family→version classifiers against flat classifiers for LLM attribution. A search returned only family-level work.
- FAID (fine-grained human/LLM/collaborative + family) appeared only on an aggregator and is unverified.

## Q5. Label/noise robustness (configuration noise in external pools)

### Takeaway
Confident learning is an established, model-agnostic way to estimate the noisy-vs-true label joint distribution and prune likely mislabels. It needs out-of-fold probabilities only, so it is cheap. For external LLM pools, "noise" is often configuration mismatch rather than wrong labels: different system prompt, temperature or quantization. Idiosyncrasies and From Text to Source suggest temperature and quantization are weak signals, so they are less likely to cause harmful noise than a different post-trained checkpoint mislabelled as the same model.

### Cited Findings
- **ESTABLISHED:** Confident learning estimates the joint of given and true labels under a class-conditional noise assumption. It is "provably consistent," modality-agnostic, and found errors in MNIST, ImageNet and Amazon Reviews text. — [Northcutt et al. 2021, Confident Learning](https://arxiv.org/abs/1911.00068)
- **SOURCE-CLAIM:** Sampling temperature is only weakly identifiable for one model: at most 59.6% pairwise and 37.9% for 5 temperatures. — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2)
- **SOURCE-CLAIM:** Quantization was not detectable from text (54.5% binary). — [Antoun et al. 2023](https://arxiv.org/html/2309.13322v2)
- **SOURCE-CLAIM:** Post-training data dominates: same-SFT-data fine-tunes of different bases become hard to separate (96.5 → 59.8%). — [Sun et al. 2025](https://arxiv.org/html/2502.12150v2)
- **SOURCE-CLAIM:** Decoding strategy and repetition penalty variations fool detectors — [RAID](https://arxiv.org/abs/2405.07940). This conflicts in emphasis with Idiosyncrasies' weak temperature identifiability. Detectability of a config and robustness to it are different questions; both can hold.

### Inferences
- **INFERRED (recipe):**
  1. Train on competition-only data under a prompt/group split and get OOF probabilities on external pools.
  2. Run cleanlab `find_label_issues` with the external pool's claimed labels.
  3. Inspect flagged samples by source/config. Drop or down-weight whole sources whose flagged rate is anomalous, rather than individual samples.
  4. A/B against no-cleaning on the group-split validation.

  Cost: one extra OOF pass on CPU/GPU, under 1 h.
- **INFERRED:** Treat "external pool fidelity" as a domain-shift problem as well as noise. Sources can serve as environments for V-REx/GroupDRO (Q2), and the per-source validation gap is the diagnostic.
- **INFERRED (most damaging noise):** Mislabelled model version or checkpoint, e.g., an API alias that moved to a newer snapshot, is likely more harmful than temperature mismatch. Cleanlab's per-class-pair off-diagonal counts would surface it as a concentrated confusion between two sibling labels.

### Gaps
- Robust losses (GCE, symmetric CE, co-teaching) were not verified in this session.
- I found no study of label noise specifically in LLM-attribution training pools.

## Q6. Test-time adaptation / transductive learning on the hidden test set

### Takeaway
Transductive use of the unlabeled test set has precedent in PAN cross-domain AA (van Halteren 2019: confident pseudo-labels, then an in-domain classifier). It has recent support in AI-text detection (2026 TTA preprint exploiting "inference-time homogeneity"). In PAN it did not beat simpler methods. It is useful mainly when test distribution differs from training. Legality in a Kaggle code competition depends on that competition's rules. I did not verify the Name That LLM rules here; another researcher covers rules.

### Cited Findings
- **ESTABLISHED (precedent):** In PAN 2019, van Halteren exploited "multiple texts of unknown authorship in the target domain." He kept only the most confident cross-domain predictions, built "a new in-domain classifier" on them, and combined both. The method scored 0.598 macro-F1, 10th of 12 submissions plus baselines, below the 0.690 winner. — [Kestemont et al. 2019](https://ceur-ws.org/Vol-2380/paper_264.pdf)
- **SOURCE-CLAIM:** DeTeCtive's TFIA (adding new embeddings to the kNN database without retraining) gave +7.03 AvgRec on unseen domains. Note this uses labeled new samples, not unlabeled test. — [DeTeCtive](https://arxiv.org/html/2410.20964v1)
- **SOURCE-CLAIM (2026):** TTA for AI-text detection exploits "homogeneity among unlabeled samples observed at inference time" with semi-supervised (PU/PNU) learning. Example: Pangram detects "just 27.0%" of their adversarial AI text vs "93.3%" for TTA. Summaries of other versions give 24.1% vs 90.5%. — [Hitting a Moving Target, 2606.25152](https://arxiv.org/abs/2606.25152). The abstract says adaptation uses unlabeled test-distribution data, and the search snippet says the evaluation set itself is excluded.
- **ESTABLISHED (generic TTA):** Tent minimises prediction entropy at test time by updating normalisation affine parameters, giving SOTA on ImageNet-C in one epoch — [Wang et al. 2021](https://arxiv.org/abs/2006.10726). Image-domain evidence; transfer to transformers for text is not established here.
- **SOURCE-CLAIM:** K-shot MAML-LoRA adaptation needs labeled target examples, so it is not applicable to a hidden test — [2607.03680](https://arxiv.org/abs/2607.03680)

### Inferences
- **INFERRED (useful variants, cheapest first):**
  1. Class-prior re-estimation on the test set via EM over predicted probabilities (Saerens-style label-shift correction). Seconds of runtime, no retraining.
  2. Fit unsupervised feature transforms on train+test (TF-IDF vocabulary/IDF, PCA, compressor dictionaries). This is transductive but label-free.
  3. One or two rounds of pseudo-labeling with high-confidence test predictions, then fine-tune a light head or LoRA for 1 epoch. On 2xT4 this is probably 1–3 h for a DeBERTa-base on ≤100K texts (INFERRED, not measured).
  4. kNN/prototype refinement: add confident test embeddings to a class-prototype bank, DeTeCtive-TFIA-style but with pseudo-labels.
- **INFERRED (risk):** Pseudo-labeling amplifies systematic confusions, so sibling-version errors become self-reinforcing. Restrict pseudo-labels to classes with high OOF precision, and gate on agreement between two diverse models (transformer + stylometric), echoing the unanimous-agreement filter of Bitton et al.
- **INFERRED (legality):** Kaggle code competitions generally allow using the test inputs at inference unless rules forbid it or hand-labeling is involved. Verify the Name That LLM rules page and any "no test-set training" clause before relying on this. The PAN precedent shows the idea is legitimate in shared tasks.
- **Novelty:** Novel to research: no. Novel to competition: unknown; pseudo-labeling is a common Kaggle trick.

### Gaps
- I could not verify Kaggle "LLM - Detect AI Generated Text" claims that top solutions trained tokenizers or vectorizers on the test set. Search did not surface the write-ups, and the 1st-place repo README snippet did not mention it.
- No evidence found for entropy-minimisation TTA (Tent-style) on transformer text classifiers for LLM attribution.
- Name That LLM rules on test-set use were not checked by me.
