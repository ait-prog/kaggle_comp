# 07_local_dev2_src_state
- title: k/dev2/ntl-dev2.py + kernel-metadata.json; src/tpu_train.py, src/infer_t4.py, src/pe.py; README.md run table
- url: file:///Users/i2gunshaker/orca/workspaces/kaggle_comp/name-that-llm/k/dev2/ntl-dev2.py
- code: local
- type: primary (local artifact; unvalidated code)
- author(s) / date: team; README row dated 2026-11-10
- accessed: 2026-11-12 (read-only)
- scores: Credibility 5/5 | Recency 5/5 | Bias: low
## Verbatim quotes
> "| dev2 v1 | TPU v5e-8 | 2026-11-10 12:31 | | | smoke: Qwen3-8B-base k24 LoRA SPMD DP |"  (README.md)
> "run='dev2', smoke=True, model='8b-base', keep=24, layers=[3, 6, 9, 12, 16, 20, 24],"  (src/tpu_train.py and k/dev2/ntl-dev2.py)
> "ev = np.arange(len(T_ids)) if not CFG['smoke'] else np.arange(256)"  (smoke evaluates on 256 train rows)
## Audit notes
- FACT: dev2 is a smoke configuration: 12 steps per length bucket (`smoke_steps=12`), cap 12,000/label from `ntl_open_pool.parquet`, kimi-k3 filtered to `gen_reasoning=='high'`. It trains on the pool only and evaluates BA on official train rows (256 in smoke mode).
- FACT: src/tpu_train.py differs from k/dev2/ntl-dev2.py only by a class-weighted loss (`CW`, `cw_pow`). The local src copy is newer than the pushed dev2 script.
- FACT: README shows no end time for dev2 v1. No out/dev2 directory exists locally. Kaggle CLI was unauthenticated in this agent session, so the kernel status could not be queried. **dev2 state = UNKNOWN (no completion evidence).**
- FACT: src/infer_t4.py is an inference path for the dev2 LoRA+head (T4, fp16 with bf16 retry). It has not been exercised (no artifacts).
- FACT: src/pe.py references a Windows path (`D:/hack/Name That LLM/`) and `ntl_open_pool.parquet`, which is not present in this worktree. It is not runnable here.
- INFERENCE: even when complete, dev2 reproduces the dominant public family (truncated Qwen3-8B LoRA + multi-layer pooling on the open pool). It is a re-implementation baseline, not a new signal.
