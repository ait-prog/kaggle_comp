# 66_gulrajani_lost_domain_generalization
- title: In Search of Lost Domain Generalization
- url: https://arxiv.org/html/2007.01434
- code: https://github.com/facebookresearch/DomainBed (not on fetched page; unverified)
- type: academic
- author(s) / venue / date: Gulrajani, Lopez-Paz / ICLR 2021 / 2020-07-02
- accessed: 2026-11-12
- scores: Credibility 5/5 | Recency 3/5 | Bias: low (academic preprint/paper; abstracts fetched via summarizer, quotes are verbatim fragments)
## Verbatim quotes
> "when carefully implemented, empirical risk minimization shows state-of-the-art performance across all datasets."
> "empirical risk minimization achieves state-of-the-art performance when compared to eight popular domain generalization alternatives"
> "Given any model selection criterion, no method improves upon the average performance of ERM in more than one point."
## Mechanism and transfer notes (label each line SOURCE CLAIM / INFERENCE)
- mechanism: SOURCE CLAIM: Fair benchmark (DomainBed) with equal hyperparameter budgets and explicit model-selection rules.
- regime: SOURCE CLAIM: Image DG benchmarks, held-out-domain evaluation.
- when it fails: SOURCE CLAIM/INFERENCE: Images, not text; fixed backbone families.
- cost / code: DomainBed (repo unverified here).
- relevance to Name That LLM: INFERENCE: OPPOSITION. Prior: do not spend GPU hours on DANN/IRM/GroupDRO unless validated by leave-prompt-out; spend on ERM + augmentation + ensembling + proper model selection.
