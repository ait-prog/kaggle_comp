# 80_hf_glm53_open_weights
- title: GLM-5.3 model card, config and licence (Hugging Face, zai-org)
- url: https://huggingface.co/zai-org/GLM-5.3 (also fetched: /raw/main/README.md, /raw/main/config.json, /raw/main/LICENSE, huggingface.co/api/models/zai-org/GLM-5.3)
- code: none found
- type: industry
- author(s) / venue / date: Z.ai (zai-org); HF repo created 2026-08-25, last modified 2026-10-14 (per HF API)
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 5/5 | Bias: low (vendor card; benchmark claims are self-reported, but the existence of weights/config/licence is directly checkable)
## Verbatim quotes
> "GLM-5.3 uses the same base model as GLM-5.2 — every gain comes from post-training."
> "GLM-5.3 is the most capable open-weights model for coding, with a 50% improvement over GLM-5.2 on our in-house Z.ai Code Bench."
> "GLM-5.3 supports controlling the thinking budget through the `reasoning_effort` parameter, which accepts three levels: `low`, `high`, and `max`. It defaults to `max` if not passed"
> "We use sampling parameters of `temperature=1.0` and `top_p=0.95` for evaluation"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Weights are public on HF (gated=False). HF API safetensors total = 753,329,940,480 params; arch GlmMoeDsaForCausalLM; config.json: 78 layers, 256 routed experts, 8 experts per token, hidden 6144, vocab 154,880. Licence tag "other" (glm-5.3 licence: permissive, with a security-review clause for Model-as-a-Service businesses above USD 10B revenue). Active-parameter count is NOT stated on the fetched card (do not assume). SOURCE CLAIM: reasoning_effort in {low, high, max}, default max; clear_thinking defaults false; benchmarks use temperature 1.0, top_p 0.95 (or 1.0).
- regime: Open weights exist, so exact-generator likelihood is possible in principle for glm-5.3. Same base as GLM-5.2, so post-training deltas only (INFERENCE: GLM-5.2 vs 5.3 text likely very similar in token statistics).
- generalization / failure evidence: Benchmark table on card lists Opus 4.8, Fable 5, GPT-5.6 Sol, Kimi K3, Qwen3.8-Max as comparators (confirms those model names exist as of Aug 2026).
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: 753B params is ~1.5 TB in bf16; smallest community GGUF seen (Scantist SLIM E192 UD-IQ2_M) is 179.9 GB and Unsloth UD-IQ2_M 238.6 GB, far beyond 2xT4 (32 GB VRAM) + 30 GB RAM. Infeasible for full-model scoring in 9h; at best disk-streamed layer-by-layer.
- relevance to Name That LLM: Verdict: open-weight YES, but too large for the compute budget. Competition labels come from an API; the API default (reasoning max) is the card default, so reasoning-on text with traces removed is the likely regime.
