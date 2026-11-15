# 47_semeval2024_task8
- title: SemEval-2024 Task 8: Multidomain, Multimodel and Multilingual Machine-Generated Text Detection
- url: https://arxiv.org/abs/2404.14183 (full text read at https://arxiv.org/html/2404.14183v1)
- code: none found
- type: academic
- author(s) / venue / date: Yuxia Wang, Jonibek Mansurov, Petar Ivanov et al., Preslav Nakov; SemEval-2024 workshop; arXiv Apr 2024
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 3/5 | Bias: low
## Verbatim quotes
> "Subtask B is to detect the exact source of a text"
> "For all subtasks, the best systems used LLMs."
> "which led to the highest accuracy of 90.85."
> "a weighted Cross-Entropy loss and an ensemble approach based on model performance per class."
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: Subtask B six-way (human, davinci-003, ChatGPT, Cohere, Dolly-v2, BLOOMz); winners used LLM-based classifiers, class-weighted CE and per-class-performance ensembles.
- regime (text length, #classes, domains): 6 classes; dev data PeerRead, test OUTFOX (domain shift between dev and test).
- generalization evidence: SOURCE CLAIM: test is a different domain from dev; top 90.85 accuracy. (fetch found no dev-vs-test comparison.)
- cost / code: no overview repo.
- relevance to Name That LLM: INFERENCE: precedent that LLM-backbone fine-tuning plus class-balanced loss and ensembling is the winning recipe on cross-domain attribution; ChatGPT/Cohere were merged by one team, a hint of sibling-style overlap.
