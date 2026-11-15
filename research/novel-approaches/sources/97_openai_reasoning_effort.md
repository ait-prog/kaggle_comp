# 97_openai_reasoning_effort
- title: OpenAI docs: Reasoning models and reasoning.effort
- url: https://developers.openai.com/api/docs/guides/reasoning
- code: none found
- type: industry
- author(s) / venue / date: OpenAI API docs; fetched 2026-11-12
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 5/5 | Bias: low (official docs)
## Verbatim quotes
> "Supported values are model-dependent and can include none , minimal , low , medium , high , xhigh , and max ."
> "If you omit reasoning.effort , GPT-5.6 defaults to medium in both modes. GPT-6.1 Sol, GPT-6 Sol, and GPT-6 Luna also default to medium reasoning effort."
> "gpt-5.5 defaults to medium reasoning effort."
> "GPT-6 Astra does not support none reasoning effort."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: reasoning.effort (Responses API) / reasoning_effort (Chat Completions); also a "mode" (e.g. pro) in the Responses API; defaults medium for gpt-5.5, GPT-5.6, GPT-6 Sol/Luna. INFERENCE: label gpt-6-astra cannot be generated with effort none; labels gpt-5.6-luna/sol/terra versus GPT-6 Luna/Sol naming in docs does not map one-to-one from the fetched page.
- regime: Config axes: effort levels and mode; defaults medium.
- generalization / failure evidence: n/a
- cost on 2×T4 (INFERENCE if estimated): n/a
- relevance to Name That LLM: Config axes for OpenAI classes; the Kaggle data likely used defaults (INFERENCE).
