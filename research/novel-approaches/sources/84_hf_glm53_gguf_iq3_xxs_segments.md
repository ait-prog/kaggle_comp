# 84_hf_glm53_gguf_iq3_xxs_segments
- title: Community GGUF quantisations of GLM-5.3 / GLM-5.3-Flash (zephel01 IQ3_XXS; Scantist SLIM)
- url: https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF (also fetched: huggingface.co/Scantist/GLM-5.3-SLIM-GGUF README, HF API for both)
- code: none found
- type: code
- author(s) / venue / date: zephel01 (HF repo created 2026-11-06); Scantist (created 2026-11-02)
- accessed: 2026-11-12
- scores: Credibility 3/5 | Recency 5/5 | Bias: med (individual community uploads, unreviewed quality claims)
## Verbatim quotes
> "a mixed-type file - the experts are IQ3_XXS 89%, IQ4_XS 8%, Q2_K 1% (by bytes), 3.13 bits a weight on average, about IQ3_XXS."
> "Every expert is kept (no pruning)."
> "At 179.9 GB it is 24.6% smaller than the 256-expert UD-IQ2_M (238.6 GB)"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: zephel01 repo is GLM-5.3-Flash only: 4 GGUF shards in folder Q120d, totalFileSize 127,926,002,528 bytes (HF API gguf metadata), README targets "one 32 GB GPU plus 128 GB RAM or more", marked BETA, speed unmeasured. Scantist SLIM is GLM-5.3 (753B original) pruned to 192 experts/563B, UD-IQ2_M 179.9 GB, targets one B200 or a 192 GB Mac. The "IQ3_XXS GGUF layer segments" mentioned by the participant most plausibly match the zephel01 Flash repo (INFERENCE; the Kaggle post was not seen). SOURCE CLAIM: KL vs unquantised reported as 0.137 for this build vs 0.265 for unsloth UD-IQ2_XXS (author measurement on own KL set).
- regime: Quantised weights lower the cost but the smallest Flash build is still ~128 GB, 4x the combined VRAM+RAM of the Kaggle box.
- generalization / failure evidence: Quantisation shifts the token distribution (author-reported KL 0.137 even for the best build), so log-probs from it differ from the vendor API anyway.
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: not loadable in llama.cpp on 2xT4+30 GB RAM without disk mmap; layer-by-layer streaming (one layer resident, all passages batched through it) is the only plausible route, and requires custom code for IQ3_XXS dequant on sm_75.
- relevance to Name That LLM: Only route to an exact-family likelihood feature for glm-5.3-flash. Cost/benefit doubtful vs a small proxy LM; test only if an offline Kaggle dataset provides the shards.
