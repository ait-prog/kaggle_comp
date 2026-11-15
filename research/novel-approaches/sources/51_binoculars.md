# 51_binoculars
- title: Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text
- url: https://arxiv.org/abs/2401.12070
- code: https://github.com/ahans30/Binoculars
- type: academic
- author(s) / venue / date: Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, Tom Goldstein; arXiv 22 Jan 2024 (v3 Oct 2024)
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 3/5 | Bias: low
## Verbatim quotes
> "called Binoculars, achieves state-of-the-art accuracy without any training data."
> "capable of spotting machine text from a range of modern LLMs without any model-specific modifications."
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM (abstract): ratio of perplexity to cross-perplexity between two related LMs (full mechanism not quoted on the fetched page; INFERENCE from general knowledge).
- regime (text length, #classes, domains): binary; abstract says over 90% of ChatGPT samples detected at 0.01% FPR.
- generalization evidence: zero-shot across LLMs.
- cost / code: two ~7B LMs forward; repo exists.
- relevance to Name That LLM: INFERENCE: perplexity/cross-perplexity scalars from two available Qwen models are cheap extra features (a few numbers per text); expected gain for 18-way low, but they encode "predictability" which differs by lab decoding style.
