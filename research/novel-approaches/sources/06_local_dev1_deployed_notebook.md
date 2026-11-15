# 06_local_dev1_deployed_notebook
- title: k/dev1/ntl-dev1.ipynb (deployed; scored 0.682), kernel-metadata.json, out/dev1/*
- url: file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/k/dev1/ntl-dev1.ipynb ; upstream https://www.kaggle.com/code/omerfarukmerey/0-682-0-681-qwen3-8-27b-gsq-12-mix-0-50 (same title in kernel list)
- code: local
- type: primary (local artifact; deployed code)
- author(s) / date: team fork of a public notebook; header names lightningv08 v15 as parent
- accessed: 2026-11-12 (read-only)
- scores: Credibility 5/5 (it is the code) | Recency 5/5 | Bias: low
## Verbatim quotes
> "This is lightningv08's [0.681 | Qwen3-Embedding LoRA + Qwen3.8-27B](https://www.kaggle.com/code/lightningv08/0-681-qwen3-embedding-lora-qwen3-8-27b) (version 15, the one that scored), run unchanged except for 2 mixing settings; nothing is retrained and the runtime is the same."
> "Shared-prompt Gaussian group head (ported, Apache-2.0, from public Kaggle notebooks by prvsiyan / kirilltym / sangam21)."
> "Two-way model x = class mean + prompt effect + noise; exact (n<=3) or cavity-EM (n>3) injective group marginals."
> "All rows are one submission each, so read single steps of 0.001 with care: the public board scores a part of the test set."
> {"mode": "exact_public_format_preview", "rows": 3, "model_weights_loaded": false, "hidden_runtime_verified": false, "all_other_inputs": "full ensemble inference"}
(the line above is out/dev1/submission.mode.json verbatim)
## Audit notes (lead agent; FACT = read from code)
- FACT: dev1 is a mixing-only fork of a public notebook. No model in it was trained by the team. Inputs: 20 public datasets, 8 public kernels, 6 Qwen model sources (kernel-metadata.json). GPU T4×2, internet off.
- FACT: components include LoRA24 readouts of Qwen3-14B/8B-Base, Llama-3.1-8B, two-stage Qwen3-14B and Qwen2.5-14B, DeBERTa/ELECTRA/desklib/UltraBERT encoders, a 4-view TF-IDF SVC (char/word/shape/function-word) pre-fit on the Open Pool, a "stylegram" complementary stack, ATLAS (source-conditioned likelihood on a proxy LM, arXiv 2610.19583) on low-margin rows, a Qwen3.8-27B GSQ3 classifier at 12%, a 5-fold Qwen3-Embedding-8B multi-depth LoRA expert at 40% (geometric), Gaussian prompt-group heads, capped agglomerative test clustering (`GROUP_MIN, GROUP_MAX = 2, 18`), prompt-centred native heads, soft Sinkhorn toward a target mix, and typography "GUARDS".
- FACT: the local run executed only the 3-row CPU preview (hard-coded fixture predictions). No weights were loaded, and the hidden-rerun behaviour was not observed locally.
- INFERENCE: every signal family in dev1 is a pooled-representation classifier on generated pool data, a sparse n-gram model, or a test-time group/prior heuristic. There is no generator-specific likelihood under any of the 18 actual labelled models, and no team-owned data.
