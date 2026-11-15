# Local baseline audit: Name That LLM (dev1 deployed vs dev2 planned)

Scope: read-only audit of `/Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm` (branch `i2gunshaker/name-that-llm`, HEAD `31eeb53`). All paths are relative to that repo root.

Citation conventions:
- `k/dev1/ntl-dev1.ipynb` is stored as one physical line of JSON (`git show --stat b89a905` shows `k/dev1/ntl-dev1.ipynb | 1 +`). Citations into it use **cell index (0-based) and line within the cell source**, written `nb c43:L151`.
- No repo instruction files exist: there is no CLAUDE.md or AGENTS.md anywhere in the repo. The full file list is README.md, data/, ext/openpool/, k/dev1/, k/dev2/, out/dev1/, src/ and research/novel-approaches/BRIEF.md.
- Labels: **[VERIFIED]** means seen in code, output or data. **[INFERRED]** means reasoned from verified facts. **[UNKNOWN]** means it cannot be established from local files.
- Local diagnostics were run with `python3 -I` from the scratchpad, on train.csv, test.csv, the parquets and the notebook JSON. Nothing in the repo was modified.

## Q1. What does dev1 do end to end? (models, features, classifier, labels, preprocessing, truncation, I/O)

### Takeaway
dev1 is not a team-built model. It is a near-verbatim copy of a public Kaggle notebook, "0.682 | 0.681 + Qwen3.8-27B GSQ 12% + Mix 0.50". That notebook runs lightningv08's public 0.681 ensemble with two changed blend settings. It is an inference-only, frozen, multi-family ensemble with roughly 30 third-party datasets and kernels attached:
- LoRA read-outs of truncated Qwen3-14B, Qwen3-8B, Llama-3.1-8B and Qwen2.5-14B, plus a DeBERTa encoder
- a sparse 4-view SVC with TF-IDF and "distort" heads
- a complementary style stack
- ATLAS, a likelihood expert run on low-margin rows
- a Qwen3.8-27B "GSQ" classifier
- a 5-fold Qwen3-Embedding-8B expert, mixed in geometrically

The blend is followed by transductive Sinkhorn class balancing and a typography correction, and the output is the argmax over 18 labels. The repo's own `src/` code (dev2's LoRA and T4 inference) is **not** wired into dev1.

### Cited Findings
**Provenance (dev1 is a public fork)**
- [VERIFIED] The notebook title is "0.682 | 0.681 + Qwen3.8-27B GSQ 12% + Mix 0.50". It describes itself as lightningv08's 0.681 notebook (version 15) "run unchanged except for 2 mixing settings; nothing is retrained", plus heliosli's mount shim — `nb c0:L1-L5`.
- [VERIFIED] The two changes are:
  - the Sinkhorn target mix goes from 0.35 to 0.50
  - the GSQ share goes from 6% to 12%

  The notebook says "The code cells below are byte for byte the version that scored 0.682" — `nb c0:L9-L12`.
- [VERIFIED] The first committed README described dev1 as a "commit of public 0.682 fork; submitted 58200298" (`git show 5ee2ff1:README.md`). The working-tree README (uncommitted change, `git diff README.md`) replaced that wording with the receipt text — `README.md:4`.
- [VERIFIED] The notebook credits lightningv08, CHEN Han, heliosli, aboodps and others. It says "Our own earlier, all-public line is 0.671 …" and links a third-party profile, so "our fork" in its A/B table refers to the public author, not this team — `nb c0:L16-L28, L34-L43`. [INFERRED] The team's own modelling contribution to dev1 is zero. dev1 is a resubmission of a public notebook.
- [VERIFIED] The kernel metadata says: notebook kernel, private, GPU on, TPU off, internet off, machine `NvidiaTeslaT4`. It attaches 20 dataset sources, 8 kernel sources and 6 model sources (Qwen3 8B/4B/14B base, Qwen3-Embedding-8B, Qwen2.5 7B/14B) — `k/dev1/kernel-metadata.json:6-10, 12-54, 56`.

**Label set and I/O schema**
- [VERIFIED] There are 18 labels: claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5, gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash, glm-5.3, glm-5.3-flash, gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra, kimi-k3 and qwen3.8-max — `nb c8:L7-L9`.
- [VERIFIED] The input `test.csv` must have exactly the columns `['id','text']`. Null text is filled with `''` — `nb c43:L69-L71`.
- [VERIFIED] The output is `/kaggle/working/submission.parquet` with columns `['id','model']`, written as the argmax of the final logits. Ids keep test order — `nb c4:L8`, `nb c43:L151-L156`, `nb c51:L14-L17`.
- [VERIFIED] Local data:
  - `data/train.csv` has 2699 rows with columns `id, prompt_id, text, model`, 150 prompt_ids and 150 rows per label (claude-fable-5 has 149)
  - `data/test.csv` has 3 rows with `id, text`
  - `data/sample_submission.parquet` has 3 rows with `id, model`, all `gpt-5.5`

  Source: local `python3 -I` diagnostic.
- [VERIFIED] Each train prompt has 18 rows, one per model: 149 prompts have all 18 and 1 prompt has 17. No (prompt_id, model) pair is duplicated (local diagnostic).

**Pipeline stages, in execution order (`ensemble.py`, `nb c43`)**
1. [VERIFIED] **Preview gate.** If test.csv has at most 3 rows and their ids, byte lengths and blake2s fingerprints match the public sample, the notebook writes a hard-coded 3-row parquet. All three predictions are `claude-fable-5`, and it loads no weights. Otherwise it runs the full ensemble — `nb c4:L9-L35`, `nb c51:L9-L13`.
2. [VERIFIED] **Transductive content grouping of the test set.** It fits TF-IDF on the test texts (`stop_words='english', min_df=2, max_df=0.05`). It then runs capped average-linkage agglomeration on cosine similarity, with clusters of at most 18 members and a threshold of 0.15 — `nb c8:L16-L35`, `nb c43:L75-L77`.
3. [VERIFIED] **Core LoRA24 members** (`CONFIG['models']`, `nb c6:L1`, parsed locally). Each member is a truncated decoder (`keep` layers) with LoRA merged in, mean pooling over several layers with LayerNorm, and linear "plain" and "centered" heads. The six members are:

   | name | base model | keep | layers | weight | bs | max_len | gauss | adapters |
   |---|---|---|---|---|---|---|---|---|
   | q14v5 | Qwen3-14B-Base | 27 | [2,4,6,8,12,16,20,24] | 3.0 | 8 | 384 | 0.7 | 5 folds + final (6) |
   | q8v5 | Qwen3-8B-Base | 24 | same | 2.0 | 16 | 384 | 0.44 | 6 |
   | ll8v5 | Llama-3.1-8B, 24-layer prefix | 24 | same | 1.0 | 16 | 384 | – | 6 |
   | n14b | Qwen3-14B | 27 | [2,4,6,9,13,17,22,27] | 2.5 | 8 | 447 | – | 1 |
   | q25_14 | Qwen2.5-14B | 32 | [3,5,8,11,16,21,27,32] | 1.0 | 8 | 447 | – | 1 |
   | deb | DeBERTa-v3-large head | – | – | 1.0 | – | – | – | – |

   All LoRA members have `twin` 0.35.
4. [VERIFIED] **Feature extraction details.** Texts are tokenized with `truncation=True, max_length=max_len-1` and EOS is appended. The trunk runs in fp16; a batch is redone in bf16 if the hidden states overflow. Masked mean pooling excludes the first token (`mask[:,0]=0`) — `nb c17:L58, L62, L104-L111`.
5. [VERIFIED] **CPU read-out** (`core_readout.py`). Per member it averages the plain log-softmax over adapters. It also computes a "twin" (prompt-group-centred features) for rows whose test group has ≥2 members. Gaussian group heads (fixed fitted states, `official_rows == 2699`) apply to groups of size 2–18. These are added to the sparse votes `svc4`, `tfidf` and `distort`, and the sum is divided by the number of active vote matrices — `nb c14:L15-L32, L35-L80, L92-L100`.
6. [VERIFIED] **Sparse heads.**
   - SVC4 is a LinearSVC over 4 TF-IDF views: char 3–5 (250k features), word 1–3 (180k), "shape" 2–6 (60k) and function-word 1–3 (50k) — `nb c11:L18-L23`, `nb c12:L20-L21`.
   - The cache manifest asserts `svc4.rows == 123828` and `matched_clean_rows_after_dedup == 24443` — `nb c46:L45-L46`.
   - The ntl666 prefit asserts `pool_rows == 190588` on Open Pool `dataset_version == 13` — `nb c47:L31-L32`.
   - `primary_svc_weight` is 12.0 — `nb c51:L7`.
7. [VERIFIED] **Complementary stack** (`legacy/comp_cached.py`): style heads, family gates and LoRA members with `LORA_MAX_LEN, LORA_BS = (447, 8)`; pooled states use max_len 512; there are 5 UltraBERT folds at max_length 512. It is blended as `prior = .85*softmax(core) + .15*softmax(comp)` — `nb c25:L22-L32, L336-L351, L600-L616`, `nb c43:L40-L48, L99-L100`.
8. [VERIFIED] **ATLAS** (a source-conditioned likelihood expert built on Qwen3-8B) runs only on rows whose top-2 margin is below 0.3, or on all rows if there are fewer than 50. It is mixed into those rows at 10% — `nb c43:L101-L109`, `nb c43:L44`.
9. [VERIFIED] **GSQ classifier.** Qwen3.8-27B in "GSQ3" packed weights, 64 decoder layers, LoRA rank 16 / alpha 32 on layers 56–63, read-outs at depths [8,16,…,64], `text_truncation False`. It was trained on 119,899 external rows plus 5,398 official rows (2 epochs × 2699). It is mixed linearly at `gsq_share = 0.12` — `nb c39:L19-L27`, `nb c40:L41-L44`, `nb c43:L46, L60, L114-L132`, `nb c51:L1`.
10. [VERIFIED] **Native Qwen3-Embedding-8B expert** (CHEN Han), 5 prompt-fold states:
    - LoRA rank 16 on all 7 projections of 36 layers (252 modules)
    - last-token pooling at depths (3,6,9,12,18,24,30,36), each LayerNormed, giving 32,768-dimensional features
    - plain and centered heads
    - instruction prefix "Identify the language model that generated the text based on its writing style"

    Source: `nb c49:L26-L42, L96-L99, L160-L169`.
11. [VERIFIED] **Native fusion.** Each fold's plain and centred heads are scaled by a fitted `plain_scale`/`centered_scale`. The centred share (≤0.5) applies only to test groups of size 2–18. The result is mixed geometrically with the parent: `softmax(0.6*log(parent) + 0.4*native_logp)` — `nb c50:L9-L39`, `nb c51:L3`.
12. [VERIFIED] **Per-fold finalize, then fold average.** `balance_logits` runs soft Sinkhorn toward a target of `(1-0.5)*predicted_marginal + 0.5*official_train_prior` for 100 iterations. It is blended in log space at strength 0.30 and only when there are ≥18 rows. Typography evidence then adds a Bayesian score over 37 punctuation and discourse markers fit on official train. That score is gated by margin, length and marker coverage and clipped to ±0.30. The 5 fold matrices are averaged and the argmax is taken — `nb c8:L42-L106`, `nb c50:L42-L57`, `nb c43:L157-L166`.
- [VERIFIED] Asserts pin the blend: `complementary_share == .15`, `gsq_share == 0.12`, `scored_reference_script_version == 363008892`, Q3.5 token-attention disabled and no modernbert member — `nb c43:L58-L63`. An intermediate cell sets `gsq_share = 0.05`, and the final cell overrides it to 0.12 — `nb c47:L37`, `nb c51:L1`.
- [VERIFIED] `GUARDS` (hard punctuation-to-label sets) is defined but referenced nowhere else in the notebook (`nb c8:L10-L14`, grep). The finalize docstring says "no final hard punctuation mask or Hungarian override" — `nb c8:L105`.

**Repo `src/` code: role relative to dev1**
- [VERIFIED] `src/infer_t4.py` loads a truncated Qwen3 (`keep` layers) in fp16 and merges LoRA `A/B` from a safetensors file. It applies per-layer LayerNorm, masked mean pooling over `cfg['layers']` and a linear head, using `max_length = buckets[-1]-1` plus EOS and batch 16, with bf16 fallback on non-finite logits. It saves `.npy` logits — `src/infer_t4.py:7-49`. Its expected inputs (`meta['cfg']`, `meta['labels']`, keys `head.norms.*`, `head.fc.*`, `*.A`, `*.B`) match what `tpu_train.py` saves — `src/tpu_train.py:219-226`.
- [VERIFIED] Nothing in the repo outside research/ references `infer_t4`, `pe.py` or `tpu_train` (grep). The dev1 notebook contains no such reference.
- [VERIFIED] `src/pe.py` is a Windows-path (`D:/hack/Name That LLM/`) TF-IDF experiment:
  - views: char 1–4 (400k features) and word 1–2 (300k)
  - classifier: `LinearSVC(C=0.3, class_weight='balanced')`
  - trained on `ext/openpool/ntl_open_pool.parquet`, with optional kimi-high filter and per-class cap
  - scored by balanced accuracy and per-class recall on `train.csv`

  Source: `src/pe.py:6-27`. [VERIFIED] `ext/openpool/ntl_open_pool.parquet` is **not** present locally (`ls ext/openpool`).

### Inferences
- [INFERRED] dev1 is a strong public ensemble with very broad view diversity: decoder LoRA read-outs from 4 backbone families, an encoder, sparse char/word/shape/function n-grams, typography, a likelihood expert, a 27B classifier and an embedding expert. The team does not own its training code, data recipes or fitted states. Every fitted state is frozen in third-party datasets: `trained_artifact_contract.hub_ref = lightningv08/ntl-olmo2-native24-final-inference-assets/16` (`nb c6:L1`).
- [INFERRED] Several heads (twin, Gaussian, native centred, GSQ group codes) only fire for test rows in content groups of size 2–18. They rely on the hidden test reproducing train's construction of 18 model responses per prompt.

### Gaps
- [UNKNOWN] How each upstream adapter, head and prefit was trained (data, splits, epochs). Only asserted metadata is visible, such as `rows_seen` and `outer_labels_used=False`.
- [UNKNOWN] Upstream contents of `ntl-olmo2-native24-final-inference-assets` beyond what the notebook asserts. Attached datasets are not mirrored locally.

## Q2. Receipt: submission 58200298, dev1, public 0.682, submitted 2026-11-10 12:12:49 UTC — what corroborates it, what is missing?

### Takeaway
The repo corroborates that dev1 was committed on Kaggle T4×2 for about 0.02 h and produced only the 3-row preview output. The score and submission id appear only in the hand-edited README (and BRIEF.md). No Kaggle API response, submission CSV/JSON or hidden-rerun log is stored locally.

### Cited Findings
- [VERIFIED] The README row reads: `dev1 | T4x2 | 2026-11-10 12:08 | 12:09 | 0.02 | submission 58200298 COMPLETE; public score 0.682 (verified 2026-11-12); submitted 2026-11-10 12:12:49 UTC` — `README.md:4`. This is an uncommitted working-tree edit. The committed version says only "commit of public 0.682 fork; submitted 58200298" (`git diff README.md`).
- [VERIFIED] BRIEF.md repeats the receipt and warns that the 3-row preview "has not loaded model weights" — `research/novel-approaches/BRIEF.md:9`.
- [VERIFIED] The commit log `out/dev1/ntl-dev1.log` contains:
  - the mount-shim line at t=13.4 s (`sources 30, moved 22, absent ["ntl-spectral-style-heads-paired-4404"], input_paths 132, missing []`)
  - the `%%writefile` runtime modules
  - nbconvert finishing at t≈22.8 s

  It has no ensemble stage, model load or inference output — `out/dev1/ntl-dev1.log:9, 10-42, 47-50`. [INFERRED] This is consistent with the preview path (cell 4) and with the README's 12:08–12:09 duration.
- [VERIFIED] The absent spectral dataset is expected: the notebook says version 15 "attaches but never reads" it — `nb c0:L47`. It is also not in `k/dev1/kernel-metadata.json:12-33`.
- [VERIFIED] `out/dev1/submission.mode.json:1` reads `{"mode": "exact_public_format_preview", "rows": 3, "model_weights_loaded": false, "hidden_runtime_verified": false, ...}`.
- [VERIFIED] `out/dev1/submission.parquet` has 3 rows, ids 00353fb1c541, 005f5a9a6139 and 0062a53d5487, all predicted `claude-fable-5`. These equal the hard-coded `PUBLIC_FIXTURE['preview_predictions']` — `nb c4:L9` — so they carry **no** model information.
- [VERIFIED] The notebook header claims "Public LB 0.682" for this exact notebook as published by its upstream author — `nb c0:L3, L26`. [INFERRED] The team's 0.682 reproduces the upstream public score, which is expected for an unchanged deterministic rerun.
- [VERIFIED] The saved notebook has 52 cells and **0 output cells** (all `outputs: []`). Kaggle's executed notebook (359,568 bytes) and `__results__.html` were written remotely per the log, but they are not stored locally — `out/dev1/ntl-dev1.log:48-50`.

### Inferences
- [INFERRED] The submission id, COMPLETE status, timestamp and 0.682 score are corroborated only by README/BRIEF prose ("verified 2026-11-12"). The code and output are consistent with them but cannot prove them.

### Gaps
- [UNKNOWN / missing] The Kaggle submissions API output (e.g. `kaggle competitions submissions` JSON), the kernel version number that was submitted, the hidden-rerun `inference-logs/*.log` and `execution-contract.json` (which would be written to `/kaggle/working/inference-logs`, `nb c43:L67-L68, L157`), the hidden-rerun runtime and the private score.
- [UNKNOWN] The fraction of the test set the public LB covers. The notebook says only "the public board scores a part of the test set" — `nb c0:L28`.

## Q3. What is dev2's actual state?

### Takeaway
dev2 is a TPU v5e-8 training script that was configured as a **smoke test** (`smoke=True` by default). Locally there are no logs, outputs, checkpoints, metrics or end time. Whether the v1 smoke run finished, failed or produced any metric is **unknown**. The local `src/tpu_train.py` is a later, divergent copy that adds class-weighted loss and was not the pushed `code_file`. Nothing connects dev2 to a T4 submission path except the standalone `src/infer_t4.py`.

### Cited Findings
- [VERIFIED] The README row reads `dev2 v1 | TPU v5e-8 | 2026-11-10 12:31 | (no end) | (no hours) | smoke: Qwen3-8B-base k24 LoRA SPMD DP` — `README.md:5`.
- [VERIFIED] Kernel metadata: script kernel `ntl-dev2.py`, TPU on, GPU off, internet off, dataset `crtproduction/name-that-llm-open-pool`, model `qwen-lm/qwen-3/Transformers/8b-base/1`, machine `TpuV5E8` — `k/dev2/kernel-metadata.json:4-15`.
- [VERIFIED] No `out/dev2/` exists. There is no dev2 log, `meta.json`, `train_logits.npy`, `train_feats.npy` or `lora_head.safetensors` anywhere in the repo; the full file listing has only `out/dev1/*`. dev2 is a `.py` script, so there are no notebook output cells.
- [VERIFIED] Defaults: `smoke=True, model='8b-base', keep=24, layers=[3,6,9,12,16,20,24], r=16, alpha=32, ldrop=0.05, cap=12000, kimi_high=True, epochs=1, gbs=64, buckets=[128,256,384], lr_a=1e-4, lr_h=1e-3, wd=0.01, warm=0.05, fsdp=False, smoke_steps=12, seed=0`. These can be overridden by a JSON argv — `k/dev2/ntl-dev2.py:4-10`.
- [VERIFIED] **Smoke plan.** It keeps at most `smoke_steps`=12 batches per length bucket, so at most 36 optimizer steps — `k/dev2/ntl-dev2.py:159-163`. Evaluation uses only the first 256 rows of `train.csv` — `k/dev2/ntl-dev2.py:208`. Those 256 rows have 9–21 rows per class (local diagnostic).
- [VERIFIED] **Model.** Qwen3-8B-base truncated to 24 of 36 layers, bf16. Custom LoRA (A random init / √in, B zeros) on q, k, v, o, gate, up and down in every kept layer. Per-layer gradient checkpointing via `torch_xla` — `k/dev2/ntl-dev2.py:93-114, 126-131`.
- [VERIFIED] **Head.** Per-layer LayerNorm over masked-mean-pooled hidden states at 7 depths, concatenated (7×4096 = 28,672 dims), dropout 0.1, one linear layer to 18 classes — `k/dev2/ntl-dev2.py:115-124`.
- [VERIFIED] **Training data.** Pool only. It globs `ntl_open_pool.parquet`, keeps `text, model, gen_reasoning`, and drops kimi-k3 rows unless `gen_reasoning=='high'`. Each epoch samples at most 12,000 rows per class — `k/dev2/ntl-dev2.py:49-55, 151-158`.
- [VERIFIED] **Truncation and batching.** `max_length=383` plus EOS. Length buckets are 128/256/384, with the batch size scaled to `min(gbs*384//b, 2*gbs)`, giving 128, 64 and 64 — `k/dev2/ntl-dev2.py:60-62, 68-83`.
- [VERIFIED] **SPMD.** The mesh is `(N,1)` over ('fsdp','model'). Inputs are sharded on batch. Parameters are sharded only if `fsdp=True` (default False), so this is pure data parallelism with replicated weights — `k/dev2/ntl-dev2.py:30-33, 134-137, 145-146`.
- [VERIFIED] **Outputs, if completed.** It writes `/kaggle/working/out/{train_logits.npy, train_feats.npy (fp16), lora_head.safetensors, meta.json}`, and `meta.json` includes `ba`, `losses` and `secs` — `k/dev2/ntl-dev2.py:216-224`.
- [VERIFIED] **Divergence from src.** `src/tpu_train.py` differs from `k/dev2/ntl-dev2.py` only by inverse-frequency class weights (`cw_pow`) in the cross-entropy — `src/tpu_train.py:164-166, 175` vs `k/dev2/ntl-dev2.py:172` (diff).
- [VERIFIED] `src/infer_t4.py` is the matching T4 inference: one GPU per process, `np.array_split` across `n` workers, batch 16, fp16 — `src/infer_t4.py:42-55`. No dev1-style notebook or kernel metadata consumes it.

### Inferences
- [INFERRED] The pushed v1 used `k/dev2/ntl-dev2.py` (the `code_file` in `k/dev2/kernel-metadata.json:4`), so it trained without class weights. Pool classes are capped at 12,000, so imbalance is limited to classes with fewer than 12,000 pool rows.
- [INFERRED] Even a successful smoke run of at most 36 steps would yield a near-untrained head. Its 256-row BA would be a plumbing check, not a model-quality signal.
- [INFERRED] A full run (`smoke=False`, 1 epoch) would see at most 18×12,000 = 216,000 pool passages and no official rows. Its eval on `train.csv` would therefore measure pool-to-official transfer.
- [INFERRED] With replicated weights (`fsdp=False`), the 24-layer 8B trunk (about 5.5 B params, about 11 GB in bf16) plus activations must fit in each v5e chip's HBM. The smoke run's memory logging (`k/dev2/ntl-dev2.py:189-194`) was presumably meant to test this. Whether it fit is UNKNOWN.

### Gaps
- [UNKNOWN] dev2 v1 outcome: completed, errored, OOM or timed out. Also unknown: the end time, TPU hours used, logged pool length p50/p99, losses and BA.
- [UNKNOWN] Version and row count of `crtproduction/name-that-llm-open-pool` as mounted for dev2. The parquet is not local; the dev1 prefits used Open Pool v13 with 190,588 rows (`nb c47:L31-L32`), and dev2 may have mounted a different version.
- [UNKNOWN] Whether dev2 used the `gemini37_unverified` rows or other flagged subsets. dev2 applies no filter beyond kimi reasoning (`k/dev2/ntl-dev2.py:53-54`).

## Q4. What validation schemes do dev1 and dev2 use, which local numbers exist, and are they comparable to the public LB?

### Takeaway
dev1 has no local validation at all: it is inference-only. Its blend settings were chosen by single public-LB submissions upstream, and multiple components were fit on the full official train set, so no clean out-of-fold estimate of the deployed ensemble exists. dev2 and `pe.py` use "train on external pool, score balanced accuracy on official train.csv" (a domain-transfer holdout with no prompt grouping). No dev2 or pe numbers are stored. The only local numeric validation tables are in `ext/openpool/` (BA about 0.46–0.50), and their provenance and comparability to the LB are unknown.

### Cited Findings
- [VERIFIED] **dev1 tuning evidence is LB-only.** The A/B table lists single submissions scoring 0.676–0.682, with the warning "read single steps of 0.001 with care: the public board scores a part of the test set" — `nb c0:L16-L28`.
- [VERIFIED] **Official train used in fitted dev1 components:**
  - GSQ `rows_seen = {"external": 119899, "official": 5398}`, i.e. official train for 2 epochs — `nb c39:L19-L20`, `nb c6:L1`
  - Gaussian states assert `official_rows == 2699` with source `nlm-v27-final-adapter-train-only` — `nb c14:L18-L19`
  - typography rates are fit on official train at runtime — `nb c8:L49-L70`, `nb c43:L142-L149`
  - the Sinkhorn target uses the official train class prior — `nb c8:L79-L81`
- [VERIFIED] **Upstream group CV signals.** The native expert has "five prompt-fold" classifiers — `nb c0:L32`. The fold calibration asserts `not state['outer_labels_used']` — `nb c50:L20`. Adapters come as `_f0`–`_f4` plus a final (`nb c6:L1`, `adapter_relative_paths`). [INFERRED] Upstream authors used prompt-grouped folds for some members, but no OOF scores are present locally.
- [VERIFIED] **dev2 scheme.** It trains on the pool only and evaluates `balanced_accuracy_score(T_y[ev], argmax)` and accuracy on `train.csv` (256 rows in smoke, all 2699 otherwise). There is no prompt grouping and no pool/train dedup — `k/dev2/ntl-dev2.py:64-65, 208-215`.
- [VERIFIED] **pe.py scheme.** Same pool-to-train transfer, reporting BA and per-class recall — `src/pe.py:20-27`. No output is saved.
- [VERIFIED] **`ext/openpool/dataq_results.csv`** has columns `round, variant, description, seed, passages, balanced_acc, balanced_acc_prompt_centred, balanced_acc_half_B, weak_label_recall`. BA ranges from 0.4597 (R5 raw pool v1-5, seed 1) to 0.5009 (U, licence/provenance filters plus 8.6k new passages, seed 0), all at 45,000 passages — `ext/openpool/dataq_results.csv:1-19`. These tables were added in the upload commit `b89a905` with no generating script in the repo.
- [VERIFIED] `ext/openpool/fidelity_by_label.csv` compares pool and real-data label hit rates per model; for example the claude-opus-4-8 fidelity ratio is 0.5394 and claude-opus-5 is 0.9654 — `ext/openpool/fidelity_by_label.csv:1-5`.
- [VERIFIED] The local `gemini37_unverified.parquet` has 10,838 rows, all `gemini-3.7-flash`. It carries leakage and quality flags: `flag_near_train_prompt` 0, `flag_train_prompt_regen` 0, `flag_kb_proxy` 268, `flag_gemini37_routing_risk` 8,297, plus `max_cos_train` and `max_sim_train_prompt` columns (local diagnostic).

### Inferences
- [INFERRED] Local numbers are **not comparable** to the public 0.682:
  - the pool-to-train BA numbers come from different, weaker single models (SVC or a smoke LoRA)
  - they are scored on official train, whose distribution relative to the hidden test is unknown
  - the competition metric is not established locally: dev2 and pe use balanced accuracy and dev1 code never computes a metric
  - the public LB is a partial-test sample
- [INFERRED] dev1 cannot be honestly re-validated on `train.csv`. GSQ, Gaussian, typography and the prior all consumed it, and per-member OOF predictions are not available. The only evaluation signal for dev1 is the LB.
- [INFERRED] For dev2, prompt-level leakage between pool and train cannot be ruled out without the flags on the full pool. The dev2 code does not filter on them.

### Gaps
- [UNKNOWN] The official competition metric (accuracy vs balanced accuracy vs other). It is not stated in any local file.
- [UNKNOWN] Who produced the `ext/openpool/*` tables, with which model and on which eval split. The column names suggest official train with a "half_B" split, but nothing confirms it.

## Q5. Which inference-time constraints are encoded in the code?

### Takeaway
dev1 hard-codes a 2×T4, internet-off, offline-wheel environment with 13 GiB per GPU. It runs isolated sequential GPU subprocesses, fp16 with bf16 overflow redo, and per-member token caps of 384/447/512. GSQ and the native expert read full text without truncation. There is **no explicit wall-clock budget or deadline** anywhere in the code. dev2's T4 path assumes one GPU per process, fp16, batch 16 and a 384-token cap.

### Cited Findings
**Hardware**
- [VERIFIED] `machine_shape: NvidiaTeslaT4`, `enable_internet: false` — `k/dev1/kernel-metadata.json:8-10, 56`. The notebook says "choose GPU T4 ×2 and Internet off" — `nb c0:L47`.
- [VERIFIED] `device_max_memory {"0":"13GiB","1":"13GiB"}` — `nb c6:L1`.
- [VERIFIED] The native expert asserts exactly 2 GPUs and splits Qwen3-Embedding-8B layers 0–15 to GPU0 and 16–35 to GPU1 — `nb c49:L54, L81-L82`.
- [VERIFIED] GSQ runs two isolated single-GPU workers on alternating rows, asserts SM ≥ 7.5, and uses batch size 1 — `nb c40:L22-L32`, `nb c43:L119-L124`.

**Offline operation**
- [VERIFIED] Each stage runs with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and its own `CUDA_VISIBLE_DEVICES`, and the child exits before the next stage — `nb c43:L18-L26, L80-L83`.
- [VERIFIED] Offline pip installs pinned wheels (transformers 5.14.1, bitsandbytes 0.50.2, etc.) from an attached dataset — `nb c6:L1` (`qwen_wheels`), `nb c43:L111-L113`.
- [VERIFIED] Large shards are staged to `/tmp`, with a free-disk check of `total + 2 GiB` — `nb c42:L13-L44`.

**Precision**
- [VERIFIED] Trunks load in fp16. Batches that overflow are redone in bf16 and counted, and dtype metadata records `float16-trunk-with-original-bf16-redo` — `nb c17:L58, L107-L109, L210`.

**Token caps**

| member | cap | source |
|---|---|---|
| L24 members q14v5 / q8v5 / ll8v5 | 384 tokens, incl. EOS | `nb c6:L1`, `nb c17:L62` |
| n14b, q25_14 | 447 | `nb c6:L1` |
| complementary LoRA | 447 | `nb c25:L32, L487` |
| complementary pooled states, UltraBERT | 512 | `nb c25:L336, L605` |
| ATLAS | `cfg['max_length']` from its bundle (value not visible locally) | `nb c34:L25` |
| native expert | 512-token covering windows, every token used (`discarded_tokens=0`), window scores averaged with token-count weights | `nb c49:L124-L144, L172-L175, L182` |
| GSQ | none (`truncation=False`); asserts length ≤ `max_position_embeddings` | `nb c40:L41-L44`, `nb c39:L27` |

**Batch sizes**
- [VERIFIED] q14 8, q8 16, ll8 16, n14b 8, q25 8 (`nb c6:L1`); complementary LoRA 8 (`nb c25:L32`); native 4 (`nb c49:L147`); GSQ 1 (`nb c43:L123`).

**Time**
- [VERIFIED] Code only logs elapsed times with `time.monotonic()`. `NTL_STARTED_MONOTONIC` is set but never compared to a limit (`nb c4:L6`; grep for budget, deadline or time limit found nothing).

**Small-input behaviour**
- [VERIFIED] Sinkhorn balancing is skipped for fewer than 18 rows (`nb c8:L75`). ATLAS covers all rows when there are fewer than 50 (`nb c43:L102`).

**dev2 T4 path**
- [VERIFIED] fp16, batch 16, length-sorted, `max_length=buckets[-1]-1` (383) plus EOS, bf16 fallback on non-finite logits — `src/infer_t4.py:13, 27-47`.

### Inferences
- [INFERRED] The hidden rerun executes, in sequence: 6 core members × up to 6 adapters, the complementary stack (including 5 UltraBERT folds), ATLAS on low-margin rows, Qwen3.8-27B at batch 1 over full texts, and 5 native folds. The 27B GSQ and native stages are likely the dominant runtime costs. The submission completed, so the total fit within Kaggle's hidden limit, but the margin is unknown.
- [INFERRED] T4 has no native bf16 tensor-core path, so bf16 redos are a slow fallback. The count of redos on hidden data is unknown.

### Gaps
- [UNKNOWN] Kaggle's hidden-rerun time limit and the actual runtime of submission 58200298. No logs are stored.
- [UNKNOWN] The hidden test row count.

## Q6. What structural limits bound the current system's error profile?

### Takeaway
The main structural bounds are these:
- dev1 is a frozen third-party ensemble the team cannot retrain or validate offline
- the blend weights were picked by single public-LB submissions
- several heads rest on two assumptions: transductive test-set grouping (18-member prompt clusters) and Sinkhorn balancing toward an assumed balanced class prior
- the output is a flat 18-way argmax with no hierarchical decision and only ad hoc calibration

dev2 is a single-view model (one truncated backbone, mean pooling, one linear head) trained only on the external pool. Truncation is probably not binding on train-like lengths, but hidden-test lengths are unknown. Per the brief, no per-class error profile can be inferred from local outputs.

### Cited Findings
- [VERIFIED] **Lengths in train.csv.** Characters p50 670, p90 1326, p99 1669, max 1900. Words p50 101, p99 248, max 274 (local diagnostic). The 3 public test rows have 321, 760 and 1077 characters (local diagnostic).
- [VERIFIED] **Group assumption.** Train has exactly 18 responses per prompt, one per model, except one prompt with 17 (local diagnostic). Test grouping caps clusters at 18 (`nb c8:L33`), and the twin, Gaussian and native-centred heads require group size 2–18 (`nb c14:L70-L79`, `nb c50:L16`).
- [VERIFIED] **Prior assumption.** The Sinkhorn target mixes the predicted marginal 50/50 with the official train prior, which is uniform (150 per class), and is applied at strength 0.30 (`nb c8:L73-L90`). The team's single changed setting moved this mix from 0.35 to 0.50 (`nb c0:L9`).
- [VERIFIED] **Label granularity.** Every head outputs a flat 18-class vector (`nb c43:L33, L151`). Family structure enters only through complementary "family gates" that rerank siblings while preserving family mass (`nb c25:L636-L641`, `nb c18:L1-L3`) and through typography marker likelihoods (`nb c8:L42-L70`).
- [VERIFIED] **Calibration.** In dev1:
  - only the native folds carry fitted temperature scales (`plain_scale`, `centered_scale`, `nb c50:L21-L26`)
  - GSQ uses a raw softmax of plain logits (`nb c43:L131`)
  - the core vote is a weighted log-prob sum divided by the active-vote count (`nb c14:L99-L100`)
  - all mixture weights are fixed constants: 0.85/0.15, 0.9/0.1 ATLAS, 0.12 GSQ and 0.4 geometric native (`nb c43:L40-L48`, `nb c51:L1-L3`)

  dev2 and `infer_t4.py` have no calibration (`src/infer_t4.py:23-47`).
- [VERIFIED] **dev2 single view.** One backbone, masked mean pooling at 7 depths, one linear head, pool-only training (`k/dev2/ntl-dev2.py:115-124, 64, 185`).
- [VERIFIED] **dev1 dependency breadth.** 20 datasets, 8 kernels and 6 models (`k/dev1/kernel-metadata.json:12-54`), and the mount shim resolves 30 sources (`out/dev1/ntl-dev1.log:9`).

### Inferences
- [INFERRED] Truncation at 384 tokens is likely rare on train-like text (max 274 words, 1900 characters). Many members use 447/512 or full-text windows. Truncation is therefore probably not the main error source for dev1, but it could matter if hidden texts are longer (UNKNOWN).
- [INFERRED] If the hidden test breaks either the prompt × 18-model structure or class balance, the group-centred heads and Sinkhorn would inject systematic bias. Those assumptions are untested locally.
- [INFERRED] The likely confusable regions are sibling models within the same vendor: 5 Claude, 4 Gemini, 5 GPT and 2 GLM variants. The pool fidelity table shows low pool-to-real fidelity for some (claude-opus-4-8 ratio 0.5394, `ext/openpool/fidelity_by_label.csv`). This is a hypothesis about where errors concentrate, **not** a measured error matrix.
- [INFERRED] Because dev1's components were all tuned on the same pool, official train and LB feedback, a new model adds diversity only if its errors are decorrelated. That cannot be measured without dev1's per-row probabilities on a held-out labelled set, and that set does not exist locally.

### Gaps
- [UNKNOWN] Per-class accuracy and the confusion matrix of dev1. The only local output is the hard-coded 3-row preview. Per instructions, nothing is inferred from it.
- [UNKNOWN] The token-level truncation rate. No Qwen tokenizer is available offline in this environment, so the 384-token impact is estimated only from character and word counts.
- [UNKNOWN] Whether the hidden test follows the train construction of 18 models per prompt, and its class balance.
