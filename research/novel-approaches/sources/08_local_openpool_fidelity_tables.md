# 08_local_openpool_fidelity_tables
- title: ext/openpool/{fidelity_by_label,fidelity_by_reasoning,fidelity_by_source,dataq_results,dataq4_paired,dataq4_runs}.csv; gemini37_unverified.parquet (metadata only)
- url: file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/ext/openpool/ ; upstream https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool (CC BY-SA 4.0 per dev1 credits)
- code: none
- type: primary (community dataset tables; measurement by a participant)
- author(s) / date: crtproduction ("Carrot Production"), Open Pool v14 (2026-10-26) and later
- accessed: 2026-11-12 (read-only, `python3 -I`)
- scores: Credibility 3/5 (weak yardstick classifier, self-reported) | Recency 5/5 | Bias: med (pool author)
## Verbatim quotes (CSV rows)
> kimi-k3,9731,0.143,0.143,0.0855,0.313,0.313,0.12,0.457
> within one source,kimi-k3,willrice/name-that-llm-openrouter-harvest,high,988,0.2095,0.313,0.6694
> within one source,kimi-k3,willrice/name-that-llm-openrouter-harvest,off,1077,0.0752,0.313,0.2403
> T,typography-matched sampling,balanced_acc,0.0263,0.0119,0.0393,0.0134,0.0219
> 2,US,the previous-but-one row minus the 30% least train-like passages per label,0,45000,0.4916,0.4675,0.4778,0.3211
## Notes
- FACT (table): pool-vs-real fidelity ratios are lowest for kimi-k3 (0.457), qwen3.8-max (0.509) and claude-opus-4-8 (0.539). The GPT labels sit at 0.79–1.02.
- FACT (table): within one source, kimi-k3 "high" reasoning rows are recognised at 0.21 vs 0.075 for "off" (real 0.313).
- FACT (table): typography-matched sampling of the pool gave +0.026 BA (95% CI 0.012–0.039) over random sampling at a fixed 45k budget. Dropping the 30% least train-like rows did not help (US vs U).
- FACT: gemini37_unverified.parquet has 10,838 gemini-3.7 rows. 8,297 carry `flag_gemini37_routing_risk=True`; `flag_near_train_prompt` is False for all rows.
- INFERENCE: generation configuration (reasoning mode, route) is a measurable axis of train↔pool mismatch. Selecting data on it changes the measured fidelity by several points. This is a data-side lever, separate from model capacity.
