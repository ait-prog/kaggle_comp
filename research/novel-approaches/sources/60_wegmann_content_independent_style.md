# 60_wegmann_content_independent_style
- title: Same Author or Just Same Topic? Towards Content-Independent Style Representations
- url: https://arxiv.org/abs/2204.04907
- code: none found
- type: academic
- author(s) / venue / date: Wegmann, Schraagen, Nguyen / RepL4NLP @ ACL 2022 / 2022-04-11
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 3/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "representations trained on AV might also encode content information instead of style alone."
> "We introduce a variation of the AV training task that controls for content using conversation or domain labels."
> "We find that representations trained by controlling for conversation are better than"
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Authorship-verification (AV) contrastive training leaks content; the fix is to build positive/negative pairs matched on conversation/domain so topic cannot separate them.
- regime: SOURCE CLAIM: Style embeddings trained on author-verification pairs.
- when it fails: SOURCE CLAIM/INFERENCE: Abstract is cut off in fetch; magnitude of gain not captured. Needs conversation-style metadata for pair construction.
- cost / code: Training recipe only; no code found on arXiv page.
- relevance to Name That LLM: INFERENCE: Our data has a natural content-matched structure: 18 models answering the SAME prompt. Same-prompt pairs are the exact analogue of conversation-matched pairs, so contrastive/metric training on same-prompt model-vs-model pairs forces style, not topic, features. Strongest fit to unseen-prompt generalisation.
