# 99_zai_thinking_mode_docs
- title: Z.AI developer docs: Thinking Mode
- url: https://docs.z.ai/guides/capabilities/thinking-mode
- code: none found
- type: industry
- author(s) / venue / date: Z.AI developer docs; fetched 2026-11-12
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 5/5 | Bias: low (official docs)
## Verbatim quotes
> "Thinking is activated by default in GLM-5.3 GLM-5.3-FLASH GLM-5.2 GLM-5.1 GLM-5 GLM-4.7 series"
> "note that GLM-5.3 and GLM-5.3-FLASH use forced thinking and cannot be disabled"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: GLM-5.3 and GLM-5.3-Flash always think via the API; interleaved and preserved thinking modes also documented; the HF cards list reasoning_effort low/high/max with default max. INFERENCE: both GLM classes are generated with thinking on and, unless set otherwise, max effort, so final answers are conditioned on long reasoning traces, a potential shared style between glm-5.3 and glm-5.3-flash (supports the high confusion between the two).
- regime: Config axes: thinking.type enabled/disabled (not for 5.3), reasoning_effort.
- generalization / failure evidence: n/a
- cost on 2×T4 (INFERENCE if estimated): n/a
- relevance to Name That LLM: Explains the GLM-5.3 vs Flash similarity (same post-training recipe, forced thinking) and helps distinguish nothing by config alone.
