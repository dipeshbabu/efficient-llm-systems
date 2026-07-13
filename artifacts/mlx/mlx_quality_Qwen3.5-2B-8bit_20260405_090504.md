# MLX Quality Suite — Qwen3.5-2B-8bit

**Date:** 2026-04-05 09:05
**Model:** `mlx-community/Qwen3.5-2B-8bit`
**Config:** turbo4

## NIAH (Needle In A Haystack)

| Context | Depth | Config | Result | Prompt t/s | Gen t/s |
|---------|-------|--------|--------|------------|---------|
| 1024 | 0% | baseline | PASS | 7011 | 207.7 |
| 1024 | 0% | turbo | PASS | 9478 | 174.9 |
| 1024 | 25% | baseline | PASS | 10982 | 208.1 |
| 1024 | 25% | turbo | PASS | 9568 | 174.7 |
| 1024 | 50% | baseline | PASS | 10604 | 205.7 |
| 1024 | 50% | turbo | PASS | 9558 | 176.2 |
| 1024 | 75% | baseline | PASS | 10733 | 205.2 |
| 1024 | 75% | turbo | PASS | 9509 | 174.2 |
| 1024 | 90% | baseline | PASS | 10992 | 206.1 |
| 1024 | 90% | turbo | PASS | 9727 | 174.3 |
| 2048 | 0% | baseline | PASS | 11795 | 206.4 |
| 2048 | 0% | turbo | PASS | 11005 | 171.2 |
| 2048 | 25% | baseline | PASS | 11720 | 205.5 |
| 2048 | 25% | turbo | PASS | 11012 | 171.8 |
| 2048 | 50% | baseline | PASS | 11672 | 204.9 |
| 2048 | 50% | turbo | PASS | 10993 | 174.0 |
| 2048 | 75% | baseline | PASS | 11836 | 202.9 |
| 2048 | 75% | turbo | PASS | 11070 | 168.4 |
| 2048 | 90% | baseline | PASS | 11812 | 207.2 |
| 2048 | 90% | turbo | PASS | 10986 | 171.8 |
| 4096 | 0% | baseline | PASS | 12007 | 203.4 |
| 4096 | 0% | turbo | PASS | 11625 | 166.2 |
| 4096 | 25% | baseline | PASS | 11914 | 204.1 |
| 4096 | 25% | turbo | PASS | 11520 | 166.1 |
| 4096 | 50% | baseline | PASS | 11965 | 201.9 |
| 4096 | 50% | turbo | PASS | 11569 | 166.0 |
| 4096 | 75% | baseline | PASS | 12029 | 200.2 |
| 4096 | 75% | turbo | PASS | 11508 | 165.5 |
| 4096 | 90% | baseline | PASS | 11939 | 201.8 |
| 4096 | 90% | turbo | PASS | 11532 | 165.9 |

**Baseline:** 15/15 passed  
**Turbo:** 15/15 passed
