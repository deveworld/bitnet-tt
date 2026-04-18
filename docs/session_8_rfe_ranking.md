# Session 8 — per-op RFE ranking

Prompts: 1. Capture layers: [0, 5, 15, 25, 29].

| rank | layer | capture | mean RFE | std | n |
|-----:|------:|:--------|---------:|----:|--:|
| 1 | 5 | post_input_norm | 0.385009 | 0.000000 | 1 |
| 2 | 5 | post_post_attn_norm | 0.373617 | 0.000000 | 1 |
| 3 | 15 | post_input_norm | 0.323942 | 0.000000 | 1 |
| 4 | 15 | post_post_attn_norm | 0.309094 | 0.000000 | 1 |
| 5 | 5 | post_self_attn | 0.293077 | 0.000000 | 1 |
| 6 | 15 | post_mlp | 0.272813 | 0.000000 | 1 |
| 7 | 0 | post_self_attn | 0.239994 | 0.000000 | 1 |
| 8 | 15 | post_self_attn | 0.222454 | 0.000000 | 1 |
| 9 | 25 | post_mlp | 0.217367 | 0.000000 | 1 |
| 10 | 25 | post_input_norm | 0.197585 | 0.000000 | 1 |
| 11 | 25 | post_post_attn_norm | 0.190333 | 0.000000 | 1 |
| 12 | 29 | post_post_attn_norm | 0.179070 | 0.000000 | 1 |
| 13 | 0 | post_post_attn_norm | 0.176977 | 0.000000 | 1 |
| 14 | 29 | post_input_norm | 0.173635 | 0.000000 | 1 |
| 15 | 0 | post_mlp | 0.139170 | 0.000000 | 1 |
| 16 | 25 | post_self_attn | 0.139021 | 0.000000 | 1 |
| 17 | 0 | block_output | 0.138873 | 0.000000 | 1 |
| 18 | 15 | block_output | 0.127001 | 0.000000 | 1 |
| 19 | 15 | block_input | 0.124947 | 0.000000 | 1 |
| 20 | 29 | post_mlp | 0.113761 | 0.000000 | 1 |
| 21 | 29 | post_self_attn | 0.111532 | 0.000000 | 1 |
| 22 | 5 | post_mlp | 0.109016 | 0.000000 | 1 |
| 23 | 25 | block_input | 0.099056 | 0.000000 | 1 |
| 24 | 25 | block_output | 0.093832 | 0.000000 | 1 |
| 25 | 29 | block_output | 0.084394 | 0.000000 | 1 |
| 26 | 29 | block_input | 0.083796 | 0.000000 | 1 |
| 27 | 5 | block_input | 0.080335 | 0.000000 | 1 |
| 28 | 5 | block_output | 0.069142 | 0.000000 | 1 |
| 29 | 0 | post_input_norm | 0.008894 | 0.000000 | 1 |
| 30 | 0 | block_input | 0.000000 | 0.000000 | 1 |
