# F5: Several label models (or same-family siblings) have public weights, so exact-family likelihood is possible in principle
Status: **FACT for existence** (verified via HF API by the lead agent and S-C). **Usefulness UNTESTED; feasibility marginal.**

## Evidence
| Source | Type | Label |
|---|---|---|
| sources/81 + lead HF API check: zai-org/GLM-5.3-Flash, 321,323,031,390 params, MIT, created 2026-08-25; card: "320B total parameters and just 18B active"; config 45 layers, hidden 4096, 288 experts, 8 per token | industry (model card) | FACT |
| sources/80: zai-org/GLM-5.3, 753B params, licence "other" | industry | FACT |
| sources/84: zephel01 GLM-5.3-Flash IQ3_XXS GGUF, ≈128 GB (4 shards), BETA, KL vs original reported by uploader | code | SOURCE CLAIM |
| sources/82: moonshotai/Kimi-K3 ≈2.78T params | industry | FACT (infeasible) |
| sources/83 + HF API: Qwen/Qwen3.8-27B (27.8B, Apache-2.0); Qwen3.8-2.4T-A95B open base of Max | industry | FACT |
| sources/28: proxy-LM likelihood (GLM-4-9B, Qwen3-8B) gave −0.002; exact GLM-5.3 likelihood is untried publicly | discussion | SOURCE CLAIM |
| sources/86, 87: served API models differ from released weights; log-prob audits are fragile | academic | SOURCE CLAIM (counter-evidence) |

## Caveats
- Answers were generated with thinking on and conditioned on unknown prompts and reasoning traces (sources/99), while our scoring is unconditional. Quantisation shifts distributions.
- No IQ3_XXS dequant path on T4 (sm_75) was verified. Kaggle dataset size limits for a 128 GB upload were not verified.
