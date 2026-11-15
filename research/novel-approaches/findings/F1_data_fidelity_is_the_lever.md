# F1: Data fidelity and coverage, not backbone capacity, is the dominant measured lever
Status: **SUPPORTED, triangulated (≥3 independent, differently-typed sources). Not proven by an equal-compute ablation.**
Hypothesis link: H1 (supported).

## Thesis
In this competition, changes to *which* generated rows a model trains on (coverage of all 18 labels, generation-config match, label cleanliness) have produced larger and more repeatable gains than adding or enlarging backbones or ensemble members.

## Evidence
| Source | Type | What it shows | Label |
|---|---|---|---|
| sources/29 (754860, 752357) | discussion (multiple independent teams) | "nearly all of the gain came from the data"; pool growth is "the biggest lever"; new large members "no" repeat | SOURCE CLAIM |
| sources/08 (Open Pool tables) | primary dataset measurement | typography-matched sampling +0.026 BA (CI 0.012–0.039) at fixed size; Vertex-only gemini-3.7 +0.018 prompt-centred | SOURCE CLAIM (measured) |
| sources/26 (752766) | discussion (dataset author) | kimi-k3 fidelity 0.46 overall; reasoning setting changes recognisability 3× within one source; 284 mislabelled rows | SOURCE CLAIM |
| sources/31 (753177) | discussion | stronger member on the same pool did not help the stack → correlated errors | SOURCE CLAIM |
| sources/23 (751871) | discussion | uncovered labels lose 6 points recall when others get external data ("coverage > volume") | SOURCE CLAIM |
| sources/90, 91 (prior Kaggle DAIGT) | code/README | winning approach relied on large diverse generated data; TF-IDF shake-up | SOURCE CLAIM |
| sources/42, 46, 48 | academic | attribution collapses cross-domain without in-domain training data | SOURCE CLAIM |

## Caveats
- All in-competition numbers are participant self-reports or single LB submissions. Public-LB SE ≈ 0.011 (sources/03).
- "Data" and "model" were never compared at equal compute in any source. Experiment 2 in REPORT.md does this.
