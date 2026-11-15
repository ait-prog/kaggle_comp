# 10_kernel_landscape
- title: Top-60 public kernels by votes (titles, authors, best public score)
- url: https://www.kaggle.com/competitions/name-that-llm/code ; API `kernels.KernelsService/ListKernels`
- code: listing only (no notebook code read)
- type: code (public notebooks; landscape)
- author(s) / date: various; snapshot 2026-11-12
- accessed: 2026-11-12. Raw: sources/raw/kaggle_top60_kernels_2026-11-12.json
- scores: Credibility 4/5 (platform scores) | Recency 5/5 | Bias: med (titles are self-descriptions)
## Verbatim quotes (titles)
> "Name That LLM 0.684: All Public + What Failed"
> "The Prompt Is Not the Author"
> "0.681 | Qwen3-Embedding LoRA + Qwen3.8-27B"
> "Two-Stage LoRA + Gaussian + Full Hungarian [0.604+"
> "Teach the Model + 10% ATLAS"
## Families (lead-agent grouping from titles plus dev1 internals; max public score seen in the list)
- F1 sparse stylometry (char/word/shape/function-word TF-IDF + SVC/LR): 0.32–0.45 alone; a member in every top notebook.
- F2 truncated decoder LoRA + multi-layer mean pooling (Qwen3-8B/14B, Qwen2.5-14B, Llama-3.1-8B, Qwen3.5-9B), including the two-stage "pool then train" variant: single models about 0.59–0.65.
- F3 Qwen3-Embedding-8B multi-depth LoRA, 5 prompt-folds, with plain and prompt-centred heads: 0.679 as the main part of the stack.
- F4 large-model classifier (Qwen3.8-27B GSQ3): used at 6–12% inside 0.681–0.684 stacks.
- F5 proxy-LM source-conditioned likelihood (ATLAS): 0.607 notebook; 10–15% on low-margin rows.
- F6 test-time structure: content clustering, Gaussian group heads, Hungarian, centring, Sinkhorn, typography guards.
- F7 encoder members (DeBERTa/ELECTRA/ModernBERT/UltraBERT): small weights.
- Ceiling of the listed public notebooks: 0.684 (many forks). No public notebook in the list exceeds 0.684.
## What it supports
- INFERENCE: the public frontier is an ensemble of F1–F7 on the same community pool. Its members are trained on overlapping data, so errors are correlated (see sources/31).
