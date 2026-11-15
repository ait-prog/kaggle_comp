# Own-model results (official 2,699 rows; prompt-grouped 5 folds = data/folds.csv)
| run | what | BA |
|---|---|---|
| tfidf (harness4 KGR cap3000, 2 seeds) | pool-only char+word SVC | 0.472 |
| tfidf_cv cap3000 w3 | pool + official 4 folds (w=3) | 0.458 (worse) |
| dev6 | Qwen3-4B L24 LoRA r16, KGR pool cap3000, 1 epoch, pool-only | 0.536 (loss still falling) |
| dev8 | dev6 + 3-epoch FT on official rows, 5-fold OOF | 0.567 (+0.030); per-fold 0.58/0.59/0.56/0.55/0.56 |
| dev8 + pool tfidf, w=0.2 | z-score blend | 0.587 (nested-w 0.584; +balancing 0.589) |
| dev7 v1 | Qwen3-8B same recipe | NaN at step ~410 (fp16), weights all NaN; v2 rerun with clamp/skip/lr 1e-4 |
Weak cluster: glm-5.3 / glm-5.3-flash / claude-fable-5 confuse with each other (dev8 recall 0.29/0.39/0.28).
Full-fit adapter for submission: out/dev8/out/lora_head_fullft.safetensors (kernel output flashpm/ntl-dev8).
| dev7 v2 | Qwen3-8B L24 LoRA, KGR cap3000, pool-only (fp16 clamp, lr_a 1e-4) | 0.553; curve 0.403/0.457/0.521/0.551 at steps 800/1600/2400/3200 |
| dev8 + dev7v2 + pool tfidf | z-blend, nested weights | 0.592 (best fixed 0.595 at .6/.4/.1) |
| dev9 | dev7 v2 + 3-epoch official FT, 5-fold OOF | 0.585 (+0.033); folds 0.60/0.59/0.59/0.56/0.58 |
| dev9 + pool tfidf (.3) | z-blend | 0.602 |
| dev9 + dev8 + dev7v2 + tfidf | nested weights (1/.25-.5/.25/.15) | 0.608 (best fixed 0.611) |
Full-fit adapters: out/dev9/out/lora_head_fullft.safetensors (8B), out/dev8/out/lora_head_fullft.safetensors (4B).

# Public-LB runs (flashpm)
| run | what | public LB |
|---|---|---|
| dev1 | fork of public 0.682 (omerfarukmerey: lightningv08 stack, GSQ 12%, Sinkhorn mix 0.50) | 0.682 |
| dev3 | dev1, Sinkhorn balancing off | 0.676 |
| dev4 | dev1, test-time prompt/content grouping off | 0.657 |
| dev10 | verbatim fork of lightningv08 0.694 (dev1 stack + GLM-5.3 / GLM-5.3-Flash likelihood stage); short mount paths rewritten to long layout | 0.691 |
| dev11 | dev10 + dev9 (Qwen3-8B LoRA full-fit) at fixed 10% linear share of the parent, before native fusion + Sinkhorn; fail-safe fallback to dev10 | 0.691 |

Takeaways
- Same code rescored moves up to ±0.009 on public LB (alexchilton's repeat-submission table), so single-submission deltas < 0.01 are noise. Weight grids cannot be resolved on the LB.
- dev11 = dev10 to three decimals: our dev9 adds nothing on top of the public stack (or the fail-safe skipped it; the hidden rerun leaves no log). Consistent with public evidence that extra LoRA members on strong stacks move ≤ 0.004.
- Sinkhorn and test-time grouping both help (dev3 −0.006, dev4 −0.025 vs dev1).
- The step 0.682 → 0.69x came from likelihood under the real open-weight source models (GLM-5.3 / Flash via `ntl-segrun`, IQ3_XXS layer segments). Out of fold it lifts glm-5.3 recall 0.27 → 0.63 and glm-5.3-flash 0.45 → 0.65; scoring the top 20% of test by GLM mass keeps the whole gain (10%: +0.020, 20%: +0.032, 30%: +0.032).
- Runtime is the binding constraint: the stack takes ~6 h of the 9 h limit, the GLM stage ~2 h.
- Kaggle may mount inputs as `/kaggle/input/datasets/<owner>/<slug>`; public notebooks with short `/kaggle/input/<slug>` paths fail the input check in that layout.

Next directions
1. Likelihood stages for other open-weight classes, paid for by dropping low-value stack members to free runtime.
2. GLM-stage text selection that also takes texts the stack calls claude-fable-5 (GLM passages are often predicted as Claude), tested offline on train likelihood features first.
3. Retrain own models on pool rows filtered to host-matching generation settings (kimi-k3 reasoning=high: recall 0.14 → 0.40 in the CPU harness).
