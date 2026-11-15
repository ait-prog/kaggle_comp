# 83_hf_qwen38_open_weights
- title: Qwen3.8 family on Hugging Face: 2.4T-A95B (Max base), Flash-Next, 27B
- url: https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B (also fetched: README, config.json, huggingface.co/api/models?author=Qwen&search=Qwen3.8, Qwen3.8-Flash-Next and Qwen3.8-27B API/config)
- code: none found
- type: industry
- author(s) / venue / date: Qwen team (Alibaba); 2.4T-A95B created 2026-08-08; Flash-Next 2026-08-24; 27B 2026-08-05 (per HF API)
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 5/5 | Bias: low (vendor card); note naming ambiguity with the API product
## Verbatim quotes
> "For the first time, Qwen3.8 brings a Qwen-Max-class model to open release."
> "Number of Parameters: 2.4T in total and 95B activated"
> "Qwen3.8-2.4T-A95B is a text-only model that requires thinking mode for all interactions."
> "temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Open models found: Qwen/Qwen3.8-2.4T-A95B (2.4T total/95B active; licence name qwen3.8-max; HF safetensors 2,446,182,725,504; 92 layers, 512 experts, 10 per token, vocab 248,320), Qwen/Qwen3.8-Flash-Next (125B total, 6B activated plus 51B n-gram embedding and 4B MTP; licence qwen-community-1.0), Qwen/Qwen3.8-27B (dense-ish 27B, Apache-2.0). No repo named "Qwen3.8-Max" exists in the search. SOURCE CLAIM: card says "Qwen3.8-Max is the official version based on Qwen3.8-2.4T-A95B with more features, such as vision input & non-thinking support" (so the API qwen3.8-max is NOT identical to the open checkpoint; INFERENCE: its non-thinking mode is selectable via API, and the label qwen3.8-max may be either mode). Open checkpoint: thinking cannot be disabled; recommended sampling temp 1.0, top_p 0.95, top_k 20.
- regime: The qwen3.8-max label corresponds to an open-weight base (2.4T-A95B) but with product-layer differences; proxy options exist at 27B (Apache-2.0) and 125B-A6B.
- generalization / failure evidence: The API product differs from the open checkpoint, so even a local copy is a proxy rather than the exact generator.
- cost on 2×T4 (INFERENCE if estimated): INFERENCE: 2.4T infeasible. Qwen3.8-27B in 4-bit (~14-16 GB) fits one T4, ~4000 x 300 tokens scoring is minutes to under 1 h; usable as a generic same-family proxy, not exact generator.
- relevance to Name That LLM: Verdict: Qwen3.8-Max base is open-weight (2.4T/95B) but infeasible; smaller Qwen3.8 siblings are feasible as same-lab proxy LMs for feature extraction.
