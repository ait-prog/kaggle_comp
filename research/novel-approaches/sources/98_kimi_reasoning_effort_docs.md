# 98_kimi_reasoning_effort_docs
- title: Kimi API Platform: Reasoning Effort guide
- url: https://platform.kimi.ai/docs/guide/use-reasoning-effort (redirected from /use-thinking-effort)
- code: none found
- type: industry
- author(s) / venue / date: Moonshot AI platform docs; fetched 2026-11-12
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 5/5 | Bias: low (official docs)
## Verbatim quotes
> "It supports "low" , "high" , and "max" , with "max" as the default."
> "K3 requires the complete assistant message returned by the API to be passed back to messages as-is, including reasoning_content and tool_calls"
> "K3's top-level reasoning-effort field; supports "low" , "high" , and "max" , with "max" as the default."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: only reasoning_effort in {low, high, max}; no off switch in the doc excerpts I extracted; the docs example model id is kimi-k3.1-pro. INFERENCE: the community claim that "reasoning high matches official data better than reasoning-off" is compatible with the doc only if "off" was emulated otherwise; the most likely official default (max) is another candidate worth stating for generation-matching.
- regime: Config axis: reasoning_effort (default max).
- generalization / failure evidence: n/a
- cost on 2×T4 (INFERENCE if estimated): n/a
- relevance to Name That LLM: If participants regenerate data, high vs max should be tested rather than off; for our purposes only distribution of the label text matters.
