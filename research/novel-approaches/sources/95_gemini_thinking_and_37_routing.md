# 95_gemini_thinking_and_37_routing
- title: Gemini API: Thinking docs (default thinking level per model) and Deprecations page (3.7-flash routing)
- url: https://aistudio.google.com/docs/thinking ; https://ai.google.dev/gemini-api/docs/deprecations
- code: none found
- type: industry
- author(s) / venue / date: Google (Gemini API docs); fetched 2026-11-12
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 5/5 | Bias: low (official docs)
## Verbatim quotes
> "Gemini models engage in dynamic thinking by default, automatically adjusting the amount of reasoning effort based on the complexity of the request."
> "| gemini-3.8-flash | On (medium) | low, medium, high |"
> "| gemini-3.1-pro-preview | On (high) | low, medium, high |"
> "`gemini-3.7-flash` is no longer available in Google AI Studio; the endpoint is not shutting down and automatically routes API requests to `gemini-3.8-flash`"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: thinking_level param; defaults per model: 3.8-flash medium; 3.6-flash medium (levels minimal/low/medium/high); 3.1-pro-preview high. gemini-3.7-flash is absent from the thinking table. SOURCE CLAIM: deprecations page confirms 3.7-flash (dated August 13, 2026 in the table) routes to 3.8-flash. This independently verifies the community note that "gemini-3.7 requests were routed to 3.8" (for the flash model; date and wording as fetched).
- regime: Config axis = thinking_level; routing means gemini-3.7 and gemini-3.8 labels may contain near-identical text.
- generalization / failure evidence: INFERENCE: if data was collected after the routing date, gemini-3.7-flash and gemini-3.8-flash classes are partly the same generator, so a ceiling on their separability exists; merge or soft-label them.
- cost on 2×T4 (INFERENCE if estimated): n/a
- relevance to Name That LLM: Directly explains one expected confusion pair and suggests treating gemini-3.7/3.8 as nearly one class in calibration or using prior-based decisions.
