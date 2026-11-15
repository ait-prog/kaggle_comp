# F4: Error mass is concentrated in a few corners: GLM↔Claude (cross-lab) and sibling versions (within-lab)
Status: **SUPPORTED, triangulated** (local OOF + several independent participant confusion matrices + docs explaining two pairs + literature on within-family confusion).

## Evidence
| Source | Type | Label |
|---|---|---|
| sources/09: lab accuracy 0.69; 52% of errors within lab; glm-5.3→claude-opus-5 38/150; gemini-3.8→3.7 49/150 | primary (local, weak model) | FACT |
| sources/28 (755314, 757199): strong ~0.68 system still has glm-5.3 recall 0.32; GLM called Claude 68 vs 34 | discussion | SOURCE CLAIM |
| sources/34 (749593): lab BA 0.587 vs model 0.369 | discussion | SOURCE CLAIM |
| sources/27, 95: gemini-3.7 endpoint routes to 3.8 (official deprecation note); host labels appear distinct | discussion + industry docs | FACT (routing) / SOURCE CLAIM (host clean) |
| sources/99: GLM-5.3 and Flash both use forced thinking | industry docs | FACT |
| sources/46, 48: majority of confusions occur within the same family | academic | SOURCE CLAIM |
| sources/11 ATLAS: likelihood can identify distillation teachers | academic | SOURCE CLAIM |

## Implication
INFERENCE: a new signal is valuable only if it moves these corners: GLM↔Claude, gemini 3.6/3.7/3.8, gpt-5.6 luna/sol/terra/6-astra, and claude-fable-5/opus-4-8. Generic capacity mostly re-solves the easy lab split.
