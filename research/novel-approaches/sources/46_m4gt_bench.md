# 46_m4gt_bench
- title: M4GT-Bench: Evaluation Benchmark for Black-Box Machine-Generated Text Detection
- url: https://arxiv.org/abs/2402.11175 (full text read at https://arxiv.org/html/2402.11175v2)
- code: https://github.com/mbzuai-nlp/M4GT-Bench
- type: academic
- author(s) / venue / date: Yuxia Wang, Jonibek Mansurov, Petar Ivanov, et al., Preslav Nakov; ACL 2024; arXiv 17 Feb 2024 (rev. 27 Jun 2024)
- accessed: 2026-11-12
- scores: Credibility 4/5 | Recency 3/5 | Bias: low
## Verbatim quotes
> "obtaining good performance in MGT detection usually requires an access to the training data from the same domain"
> "Only when training over all domains and testing over all domains, accuracy reaches 97."
> "the best distinction accuracy for humans is 27%, and the average accuracy 21% is less than random guess."
> "the distribution between davinci-003 and ChatGPT are more similar to each other, making it hard to make distinctions."
## Mechanism and transfer notes
- mechanism: SOURCE CLAIM: Task 2 is generator identification (human plus six generators) with fine-tuned detectors (e.g., RoBERTa, Longformer).
- regime (text length, #classes, domains): 7 labels, multiple domains/languages.
- generalization evidence: SOURCE CLAIM: 97 only when train and test share all domains; unseen domains and generators degrade sharply (arXiv hardest if unseen).
- cost / code: repo available.
- relevance to Name That LLM: INFERENCE: supports disconfirming warning that in-domain attribution accuracy is near-ceiling and unseen-domain is far lower; humans cannot do it, so only learned features help.
