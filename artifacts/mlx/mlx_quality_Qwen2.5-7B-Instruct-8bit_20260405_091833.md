# MLX Quality Suite — Qwen2.5-7B-Instruct-8bit

**Date:** 2026-04-05 09:18
**Model:** `mlx-community/Qwen2.5-7B-Instruct-8bit`
**Config:** turbo4_asymmetric

## NIAH (Needle In A Haystack)

| Context | Depth | Config | Result | Prompt t/s | Gen t/s |
|---------|-------|--------|--------|------------|---------|
| 1024 | 0% | baseline | PASS | 2539 | 65.1 |
| 1024 | 0% | turbo | PASS | 2974 | 50.9 |
| 1024 | 25% | baseline | PASS | 3092 | 64.7 |
| 1024 | 25% | turbo | PASS | 2986 | 50.8 |
| 1024 | 50% | baseline | PASS | 2977 | 65.2 |
| 1024 | 50% | turbo | PASS | 2891 | 50.9 |
| 1024 | 75% | baseline | PASS | 3024 | 65.4 |
| 1024 | 75% | turbo | PASS | 2951 | 50.7 |
| 1024 | 90% | baseline | PASS | 3041 | 65.6 |
| 1024 | 90% | turbo | PASS | 2929 | 50.9 |
| 2048 | 0% | baseline | PASS | 3306 | 64.5 |
| 2048 | 0% | turbo | PASS | 3227 | 49.6 |
| 2048 | 25% | baseline | PASS | 3273 | 63.4 |
| 2048 | 25% | turbo | PASS | 3216 | 49.8 |
| 2048 | 50% | baseline | PASS | 3026 | 64.4 |
| 2048 | 50% | turbo | PASS | 2990 | 49.8 |
| 2048 | 75% | baseline | PASS | 3096 | 63.3 |
| 2048 | 75% | turbo | PASS | 3051 | 49.8 |
| 2048 | 90% | baseline | PASS | 3202 | 63.6 |
| 2048 | 90% | turbo | PASS | 3169 | 49.7 |
| 4096 | 0% | baseline | PASS | 2970 | 62.9 |
| 4096 | 0% | turbo | PASS | 2977 | 47.4 |
| 4096 | 25% | baseline | PASS | 3086 | 63.1 |
| 4096 | 25% | turbo | PASS | 2992 | 47.4 |
| 4096 | 50% | baseline | PASS | 3038 | 63.7 |
| 4096 | 50% | turbo | PASS | 2975 | 47.8 |
| 4096 | 75% | baseline | PASS | 3029 | 62.7 |
| 4096 | 75% | turbo | PASS | 2977 | 47.7 |
| 4096 | 90% | baseline | PASS | 3057 | 63.7 |
| 4096 | 90% | turbo | PASS | 2959 | 47.3 |

**Baseline:** 15/15 passed  
**Turbo:** 15/15 passed
