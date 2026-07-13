# MLX Quality Suite — Qwen3.5-2B-8bit

**Date:** 2026-04-05 09:05
**Model:** `mlx-community/Qwen3.5-2B-8bit`
**Config:** turbo4

## Context Size Scaling

| Context | Config | Prompt t/s | Gen t/s | Peak Memory (GB) |
|---------|--------|------------|---------|-------------------|
| 128 | baseline | 1924 | 208.9 | 2.27 |
| 128 | turbo4 | 4207 | 205.6 | 2.26 |
| 512 | baseline | 9359 | 208.6 | 2.76 |
| 512 | turbo4 | 7548 | 177.5 | 2.75 |
| 1024 | baseline | 10873 | 208.5 | 3.01 |
| 1024 | turbo4 | 9713 | 175.4 | 3.00 |
| 2048 | baseline | 11788 | 195.8 | 3.18 |
| 2048 | turbo4 | 11033 | 173.3 | 3.18 |
| 4096 | baseline | 6589 | 202.6 | 3.28 |
| 4096 | turbo4 | 11601 | 166.1 | 3.29 |
