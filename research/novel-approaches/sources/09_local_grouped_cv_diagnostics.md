# 09_local_grouped_cv_diagnostics
- title: Lead-agent read-only diagnostics on data/train.csv (TF-IDF LinearSVC, grouped OOF, shortcut probes, prompt-thinning simulation)
- url: scripts at /private/tmp/claude-501/-Users-i2gunshaker-orca-workspaces-kaggle-comp-name-that-llm/715c8dc3-c85b-4607-80fd-cd053d781a3e/scratchpad/scripts/diag1.py, diag2.py (session scratch; logic summarised here)
- code: as above (sklearn 1.8.0)
- type: primary (local experiment, weak model)
- author(s) / date: lead agent, 2026-11-12
- accessed: 2026-11-12
- scores: Credibility 4/5 (reproducible, but a weak 0.35-BA model; magnitudes will differ for strong models) | Recency 5/5 | Bias: low
## Results (verbatim JSON values)
> "tfidf_svc_groupkfold_BA": 0.35114093959731546
> "tfidf_svc_randomkfold_BA": 0.14481978622918218
> "length_only_groupkfold_BA": 0.10666915237385038
> "punct_only_groupkfold_BA": 0.2344668158090977
> "topic_masked_char_svc_groupkfold_BA": 0.30707183693760876
- Prompt-centred (oracle groups, m=18): 0.412. Hungarian (oracle complete groups): 0.440.
- Family (lab) accuracy of the 18-way OOF: 0.692. Within-family errors = 52.5% of all errors.
- Lowest recalls: claude-fable-5 0.081, claude-opus-4-8 0.147, glm-5.3 0.167, glm-5.3-flash 0.173, gpt-5.6-sol 0.227, gpt-5.6-terra 0.233.
- Top confusions (true→pred, count/150): gemini-3.8→3.7 49; glm-5.3→claude-opus-5 38; gemini-3.7→3.8 35; gpt-5.6-sol→luna 35; claude-fable-5→claude-opus-5 34.
- Near duplicates: max cross-prompt char-5gram cosine = 0.225; 0 cross-prompt pairs > 0.5. Within-prompt mean cosine 0.169 vs cross-prompt 0.025.
- Prompt-thinning simulation (20 reps, oracle true groups, each prompt cut to m answers), BA of plain / full centring / half-weight cavity centring / Hungarian:
  - m=1: 0.344 / 0.056 / 0.344 / 0.344
  - m=2: 0.347 / 0.277 / 0.339 / 0.350
  - m=3: 0.357 / 0.345 / 0.372 / 0.363
  - m=6: 0.349 / 0.376 / 0.375 / 0.369
  - m=18: 0.351 / 0.412 / 0.384 / 0.440
## What it supports (INFERENCE unless marked)
- FACT (this data): random K-fold is badly pessimistic (0.145 vs 0.351). Content is anti-informative across same-prompt siblings, so validation must be prompt-grouped.
- Length alone is near chance (0.107 vs 0.056 chance). Punctuation alone gives 0.234. Masking topic words costs only 0.044 BA, so most n-gram signal is not topical.
- Group-based gains need many answers per prompt. With 1–2 answers per prompt they vanish, and full centring is catastrophic. These are oracle groups, so this is an upper bound.
- About half of errors are within-lab sibling confusions. The rest are cross-lab, dominated by GLM↔Claude.
