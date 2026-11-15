# F3: For balanced accuracy, correct for the *training* prior; do not chase the test prior
Status: **SUPPORTED** (theory + participant simulation + LB data points + literature). H4 as originally stated (EM test-prior estimation) is **REFUTED**.

## Thesis
Balanced accuracy weights every class equally. Under label shift (p(x|y) fixed), the BA-optimal rule is argmax_y p(x|y) ∝ p_train(y|x)/p_train(y). This is independent of the test prior. Two consequences follow. (a) Fixed-target Sinkhorn balancing is a bet on an even test mix: gains are a few thousandths if even, and losses run to several points if uneven. (b) Per-class offsets learned on prompt-grouped OOF (cross-fitted) capture the benign part of the gain without assuming the test mix.

## Evidence
| Source | Type | Label |
|---|---|---|
| sources/01 official metric definition | primary | FACT |
| sources/24 (752365): Dirichlet(1) mix → full Sinkhorn 0.636→0.570; EM worse than offsets | discussion with code | SOURCE CLAIM |
| sources/22 (750213): "best decision ignores the author frequencies" | discussion | SOURCE CLAIM |
| sources/72 Menon logit adjustment (post-hoc log-prior subtraction) | academic | SOURCE CLAIM |
| sources/67–70 label-shift EM/BBSE/RLLS: these estimate test priors; they are useful for accuracy, not for BA | academic | INFERENCE |

## Implication
dev1 raised the Sinkhorn target mix from 0.35 to 0.50 (sources/06). That is an increased bet on an even test, and it is not supported by the official statement. Single-submission LB evidence (+0.001) is within noise.
