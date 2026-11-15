# F6: Content invariance should be learned at train time (layout-agnostic), not imposed at test time
Status: **PARTIALLY SUPPORTED** (literature + local evidence). In-competition transfer is **insufficient evidence** (2 participant sources, both indirect).

## Evidence
| Source | Type | Label |
|---|---|---|
| sources/60 Wegmann: controlling content (conversation) in training pairs gives better style representations | academic | SOURCE CLAIM |
| sources/63 StyleDistance: content-matched synthetic pairs give style-over-content; LUAR still encodes content | academic | SOURCE CLAIM |
| sources/62 Stamatatos text distortion helps cross-topic | academic | SOURCE CLAIM |
| sources/64 Sawatphol: topic leakage persists in cross-topic splits | academic | SOURCE CLAIM (caution) |
| sources/66 Gulrajani: DG algorithms rarely beat ERM by more than 1 point | academic | SOURCE CLAIM (counter) |
| sources/09 local: random-KFold 0.145 vs grouped 0.351 (content is anti-informative); topic-masked char SVC loses only 0.044 | primary (local) | FACT |
| sources/31, 23: readout or data diversity adds about 2 points; correlated members add little | discussion | SOURCE CLAIM |

## Implication
INFERENCE: a member trained with prompt-panel contrastive or adversarial content removal targets the same benefit that test-time centring chases, without depending on the test layout. Its main expected value is error diversity. Public notebooks have "prompt-centred heads" and same-prompt triplet batches (sources/31), so the novelty is partial: the objective and the panel data are new; the motivation is not.
