# MLX Quality Suite — Qwen2.5-7B-Instruct-8bit

**Date:** 2026-04-05 09:07
**Model:** `mlx-community/Qwen2.5-7B-Instruct-8bit`
**Config:** turbo4

## NIAH (Needle In A Haystack)

| Context | Depth | Config | Result | Prompt t/s | Gen t/s |
|---------|-------|--------|--------|------------|---------|
| 1024 | 0% | baseline | PASS | 2749 | 64.5 |
| 1024 | 0% | turbo | FAIL | 2894 | 50.4 |
| 1024 | 25% | baseline | PASS | 3078 | 64.5 |
| 1024 | 25% | turbo | FAIL | 2939 | 51.0 |
| 1024 | 50% | baseline | PASS | 2918 | 64.6 |
| 1024 | 50% | turbo | FAIL | 2811 | 94.6 |
| 1024 | 75% | baseline | PASS | 3049 | 64.7 |
| 1024 | 75% | turbo | FAIL | 2897 | 52.1 |
| 1024 | 90% | baseline | PASS | 3022 | 64.9 |
| 1024 | 90% | turbo | FAIL | 2898 | 58.1 |
| 2048 | 0% | baseline | PASS | 2982 | 63.8 |
| 2048 | 0% | turbo | FAIL | 2729 | 56.8 |
| 2048 | 25% | baseline | PASS | 2714 | 64.7 |
| 2048 | 25% | turbo | FAIL | 2994 | 72.6 |
| 2048 | 50% | baseline | PASS | 2961 | 64.2 |
| 2048 | 50% | turbo | FAIL | 3119 | 49.4 |
| 2048 | 75% | baseline | PASS | 3172 | 64.3 |
| 2048 | 75% | turbo | FAIL | 3085 | 73.0 |
| 2048 | 90% | baseline | PASS | 3168 | 64.1 |
| 2048 | 90% | turbo | FAIL | 3062 | 49.0 |
| 4096 | 0% | baseline | PASS | 2717 | 62.9 |
| 4096 | 0% | turbo | FAIL | 2950 | 51.8 |
| 4096 | 25% | baseline | PASS | 3008 | 62.9 |
| 4096 | 25% | turbo | FAIL | 2949 | 48.0 |
| 4096 | 50% | baseline | PASS | 2995 | 62.9 |
| 4096 | 50% | turbo | FAIL | 2925 | 47.2 |
| 4096 | 75% | baseline | PASS | 3021 | 63.0 |
| 4096 | 75% | turbo | FAIL | 2922 | 55.0 |
| 4096 | 90% | baseline | PASS | 2950 | 63.3 |
| 4096 | 90% | turbo | FAIL | 2929 | 49.1 |

**Baseline:** 15/15 passed  
**Turbo:** 0/15 passed

## KL Divergence

| Metric | Value |
|--------|-------|
| Config | turbo4 |
| Positions | 467 |
| Mean KLD | 6.862924 |
| Max KLD | 17.495138 |
| Top-1 Match | 10.5% |

## Context Size Scaling

| Context | Config | Prompt t/s | Gen t/s | Peak Memory (GB) |
|---------|--------|------------|---------|-------------------|
| 128 | baseline | 1480 | 65.4 | 8.27 |
| 128 | turbo4 | 1589 | 65.3 | 8.26 |
| 512 | baseline | 2594 | 65.2 | 8.50 |
| 512 | turbo4 | 2398 | 50.3 | 8.49 |
| 1024 | baseline | 3062 | 65.2 | 8.76 |
| 1024 | turbo4 | 2922 | 56.4 | 8.74 |
| 2048 | baseline | 3253 | 63.4 | 8.74 |
| 2048 | turbo4 | 3148 | 56.0 | 8.72 |
| 4096 | baseline | 3140 | 63.2 | 8.91 |
| 4096 | turbo4 | 3010 | 49.7 | 8.91 |
