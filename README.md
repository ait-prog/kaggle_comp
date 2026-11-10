import kagglehub

# Download latest version
path = kagglehub.competition_download('name-that-llm')

print("Path to competition files:", path)

# Kaggle runs (quota tracking; week resets Sat 00:00 UTC)
| run | acc | start UTC | end UTC | h | note |
|---|---|---|---|---|---|
| dev1 | T4x2 | 2026-11-10 12:08 | 12:09 | 0.02 | commit of public 0.682 fork; submitted 58200298 |
| dev2 v1 | TPU v5e-8 | 2026-11-10 12:31 | | | smoke: Qwen3-8B-base k24 LoRA SPMD DP |
