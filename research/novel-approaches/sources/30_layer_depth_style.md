# 30_layer_depth_style
- title: Early Qwen layers know the author better than late ones: 0.39 vs 0.31
- url: https://www.kaggle.com/competitions/name-that-llm/discussion/750462
- type: discussion
- author(s) / date: Aleksey Osipenko 2026-10-18
- accessed: 2026-11-12 via Kaggle internal JSON API (discussions.DiscussionsService/GetForumTopicById), downloaded by lead agent; raw text in sources/raw/forum_topics_txt/
- scores: Credibility 3/5 | Recency 5/5 | Bias: low
## Verbatim quotes
> "Layer 3 (8% depth): 0.391" (750462)
> "Layer 36 (last layer): 0.311" (750462)
> "In my view, deeper layers encode what the text is about, while earlier layers capture how it is written" (750462)
## What it supports
- SOURCE CLAIM: frozen early-layer features beat late ones for authorship under prompt-grouped CV. The public family already exploits this with truncated multi-depth pooling (sources/06, 07).
