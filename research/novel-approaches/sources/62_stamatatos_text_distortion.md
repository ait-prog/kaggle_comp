# 62_stamatatos_text_distortion
- title: Authorship Attribution Using Text Distortion
- url: https://aclanthology.org/E17-1107/
- code: none found
- type: academic
- author(s) / venue / date: Stamatatos / EACL 2017 / 2017-04
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 1/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "attempts to mask topic-specific information that is not related to the personal style of authors."
> "introducing a text distortion step before extracting stylometric measures."
> "can enhance existing methods especially under cross-topic conditions."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Replace (mask) content words/characters before feature extraction so only function-word and punctuation structure remains.
- regime: SOURCE CLAIM: Cross-topic closed-set attribution and verification.
- when it fails: SOURCE CLAIM/INFERENCE: Masking discards content-word signal; for LLM attribution, vocabulary choices are themselves a model signature, so over-masking may hurt (INFERENCE).
- cost / code: Cheap preprocessing; no code found.
- relevance to Name That LLM: INFERENCE: Cheap ablation: train a char-n-gram/TF-IDF branch on distorted text (mask rare content words) and check whether leave-prompt-out gain appears. Ensemble diversity candidate.
