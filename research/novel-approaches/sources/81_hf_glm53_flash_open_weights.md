# 81_hf_glm53_flash_open_weights
- title: GLM-5.3-Flash model card, config and licence (Hugging Face, zai-org)
- url: https://huggingface.co/zai-org/GLM-5.3-Flash (also fetched: /raw/main/README.md, /raw/main/config.json, huggingface.co/api/models/zai-org/GLM-5.3-Flash)
- code: none found
- type: industry
- author(s) / venue / date: Z.ai (zai-org); HF repo created 2026-08-25, last modified 2026-10-15 (per HF API)
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 5/5 | Bias: low (vendor card; size/licence facts directly checkable)
## Verbatim quotes
> "With 320B total parameters and just 18B active parameters, it outperforms GLM-5.2 across benchmarks and real-world workloads at one-tenth the price"
> "We introduce GLM-5.3-Flash, the first natively multimodal model in the GLM-5 series."
> "license: mit"
> "GLM-5.3-Flash supports controlling the thinking budget through the `reasoning_effort` parameter, which accepts three levels: `low`, `high`, and `max`."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: 320B total / 18B active (card). HF API safetensors total = 321,323,031,390. Arch Glm5NextForConditionalGeneration (image-text-to-text), MIT licence. config.json: 45 layers, 288 routed experts, 8 per token, hidden 4096, vocab 154,880 (same vocab size as GLM-5.3, so tokenizer probably shared; INFERENCE). SOURCE CLAIM: reasoning_effort low/high/max default max; weights are public as BF16 and FP8, plus many community GGUF/EXL3/NVFP4 quants.
- regime: Open weights exist; the only GLM candidate where own-model likelihood might be tractable. Per-token compute = 18B active.
- generalization / failure evidence: Card says Flash approaches Claude Opus 4.8 on coding/agentic benchmarks (SOURCE CLAIM), consistent with the community-reported GLM vs Claude confusion (no direct evidence on text style here).
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: bf16 is ~640 GB; FP8 ~320 GB; community IQ3_XXS ~128 GB (see 84). None fits 32 GB VRAM + 30 GB RAM resident. Layer-streaming from disk (load one layer, push all ~4,000 passages through, cache hidden states ~10 GB in RAM) could fit: ~1.2M tokens x 36 GFLOP/token = ~4e16 FLOP, order 1-3 h on 2xT4 if a dequant kernel exists (untested estimate).
- relevance to Name That LLM: Verdict: open-weight YES (MIT). Exact-generator likelihood for glm-5.3-flash is feasible in principle but needs layer-streaming engineering; API-vs-local quant mismatch and stripped reasoning traces will blur the signal.
