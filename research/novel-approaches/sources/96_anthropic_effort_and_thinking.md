# 96_anthropic_effort_and_thinking
- title: Anthropic docs: Effort parameter and extended-thinking deprecation
- url: https://platform.claude.com/docs/en/build-with-claude/effort ; https://platform.claude.com/docs/en/build-with-claude/extended-thinking
- code: none found
- type: industry
- author(s) / venue / date: Anthropic (Claude Platform docs); fetched 2026-11-12
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 5/5 | Bias: low (official docs)
## Verbatim quotes
> "Most Claude models default to high effort, spending as many tokens as needed for excellent results; Claude Opus 5.5 and Claude Haiku 5.5 default to medium."
> "Claude 4.7 and later models do not support it and reject requests that use it, returning a 400 error."
> "Set output_config.effort on the request. With no value, the model’s default level applies."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: effort set via output_config.effort (levels up to max); adaptive thinking replaces manual budget_tokens, which is deprecated and rejected by Claude 4.7+. INFERENCE: docs name "Opus 5.5"/"Haiku 5.5", while the competition labels say claude-opus-5 and claude-sonnet-5; the mapping is not established by the fetched page.
- regime: Config axes: effort (default mostly high; medium for Opus 5.5) and adaptive thinking.
- generalization / failure evidence: Per-model default effort differs, a plausible stylistic difference between claude-opus-5 and siblings (INFERENCE).
- cost on 2×T4 (INFERENCE if estimated): n/a
- relevance to Name That LLM: Gives config axes for the Claude classes; no evidence about stylistic effect of effort.
