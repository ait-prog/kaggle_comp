# 82_hf_kimi_k3_open_weights
- title: Kimi K3 model card and config (Hugging Face, moonshotai)
- url: https://huggingface.co/moonshotai/Kimi-K3 (also fetched: /raw/main/README.md, /raw/main/config.json, huggingface.co/api/models/moonshotai/Kimi-K3 and /Kimi-K3.1-Pro)
- code: none found
- type: industry
- author(s) / venue / date: Moonshot AI; HF repo created 2026-06-13, last modified 2026-09-02 (per HF API)
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 5/5 | Bias: low (vendor card)
## Verbatim quotes
> "Kimi K3 is an open-weight, native multimodal agentic model and our most capable model to date."
> "It is a 2.8T-parameter model built on Kimi Delta Attention (KDA) and Attention Residuals (AttnRes)"
> "Kimi K3 always has thinking enabled, and will return `reasoning_content`."
> "All Kimi K3 results are obtained with reasoning effort set to 'max' and temperature = 1.0."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: 2.8T total; the model-summary table on the card lists Activated Parameters 104B (table cell text). HF API safetensors total = 2,779,931,837,184; compressed-tensors 8-bit; licence "Kimi K3 License" (tag other). config.json: 93 layers, 896 experts, vocab 163,840. A later repo moonshotai/Kimi-K3.1-Pro exists (created 2026-11-02), a different model from the label kimi-k3. SOURCE CLAIM: thinking cannot be turned off; reasoning_effort in {low, high, max}, default max. So the community remark that "reasoning high matches better than reasoning off" cannot refer to thinking disabled via the official K3 API (INFERENCE: it may refer to low/off-like settings from a third-party router, or to a different tool; unverified).
- regime: Open weights exist but at 2.8T they are ~2.8 TB at 8-bit; no way to run on the target hardware.
- generalization / failure evidence: No attribution evidence; card only.
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: infeasible on 2xT4 (even a 2-bit quant would be ~700 GB).
- relevance to Name That LLM: Verdict: open-weight YES (2.8T / 104B active, custom licence), infeasible for likelihood scoring. Useful only for config knowledge: the K3 data was almost surely generated with thinking on, effort low/high/max.
