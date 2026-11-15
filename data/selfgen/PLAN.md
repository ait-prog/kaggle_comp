# Self-generated data plan (free/subscription access only, no paid API)

Prompts: `prompts.csv`, 300 prompts from the open-pool prompt bank. They were chosen where the pool already has
about 11 curated labels but the target models are missing, so each new answer completes a prompt panel.
Prompts close to official train prompts (sim ≥ 0.25) are excluded. Send the `instruction` column exactly as written:
it adds a word target drawn from the official length distribution (40–250 words, median about 100) and the official
format (one paragraph, no markdown).

## Which models, in order

| # | Label | Why | Good pool rows now | Evidence |
|---|---|---|---:|---|
| 1 | kimi-k3 | Only `reasoning=high` rows match the official data (fidelity 0.81, other settings 0.11–0.37). Keeping only those rows raised recall from 0.14 to 0.40 | 2,673 | harness KGR; fidelity_by_reasoning |
| 2 | qwen3.8-max | Fidelity ≤ 0.56 under every setting in the pool, so the pool doesn't capture the host's setting | 4,566 | fidelity_by_label |
| 3 | glm-5.3 | Recall about 0.19; GLM passages get called Claude | 11,891 | harness recall; REPORT §F4 |
| 4 | glm-5.3-flash | Recall about 0.23; same confusion as glm-5.3 | 12,214 | harness recall |
| 5 | claude-fable-5 | Lowest recall (0.15); fidelity is fine (0.88), so the gap is sibling separation | 12,773 | harness recall |
| 6 | claude-opus-4-8 | Fidelity 0.54, recall about 0.30 (only 31 prompts are missing it; answer all 300 anyway) | 10,719 | fidelity_by_label |

Skip gemini-3.7-flash: the API serves 3.7 requests with 3.8, so a subscription can't produce genuine 3.7 data.
Skip the GPT-5.6 trio unless your subscription exposes those exact model names.

## Settings per model (match the host's, not your own taste)
- **kimi-k3:** thinking/reasoning ON (high). This setting is supported by the data.
- **qwen3.8-max:** unknown. Generate half with thinking ON and half OFF, and record which was used in each row. We then check which half the model trained on official data likes more.
- **glm-5.3 / glm-5.3-flash:** default (Z.ai forces thinking by default). Pool `default` rows have fidelity 0.90 for glm-5.3.
- **claude-fable-5 / claude-opus-4-8:** default. For opus-4-8, `low` effort matched slightly better (0.76 vs 0.62), so log the setting if you can choose it.

## Rules for each passage
1. Start a **new chat per prompt**. Turn off memory and custom instructions, and don't use project or system prompts.
2. Send the `instruction` text exactly as given. No follow-ups, no regenerating to get a "better" answer: the first answer is the data.
3. Copy only the passage text. Strip chat UI artifacts, but don't edit the wording.
4. Record the exact model name shown by the app and the thinking setting.
5. Don't script or scrape chat web UIs (most terms of service forbid it). CLI tools officially covered by your subscription (e.g. Claude Code `claude -p`) are fine. Note that they add their own system prompt, which can shift style, so log which tool you used.

## Output format: `data/selfgen/answers.csv`
`sg_id,model,setting,tool,text`

Example: `sg000,kimi-k3,thinking_high,kimi.com,"The Casimir effect ..."`

## Pilot first
Do 50 prompts (sg000–sg049) for kimi-k3 and qwen3.8-max, which is 150 passages counting both Qwen halves.
We add them to the training data and check recall on the official rows for those two classes. Scale up only if it improves.
