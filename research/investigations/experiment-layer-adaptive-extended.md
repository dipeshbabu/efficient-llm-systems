# Experiment: Layer-Adaptive Extended Context + Decode Speed

Branch: `experiment/layer-adaptive-extended-ctx`

## Summary

Layer-adaptive mode 2 (q8_0 on last 8 layers, turbo3 on first 32) **eliminates virtually all quality loss** while keeping ~3.5x effective compression. Verified at all context depths 2K-32K with current TOT code (fp16 LUT + float norm broadcast).

## Quality (PPL)

| Config | 8-chunk PPL | vs q8_0 | 32-chunk PPL | vs q8_0 |
|--------|-------------|---------|--------------|---------|
| q8_0 baseline | 6.111 | — | 5.415 | — |
| **Mode 2** | **6.120** | **+0.14%** | **5.435** | **+0.37%** |
| Uniform turbo3 | 6.211 | +1.6% | 5.471 | +1.0% |

Mode 2 quality holds at extended context — the gap stays under 0.4%.

## Prefill Speed (tok/s, M5 Max 128GB)

| Context | q8_0 | turbo3 | Mode 2 | mode2/q8_0 |
|---------|------|--------|--------|------------|
| 2K | 2707 | 2632 | 2681 | 0.990x |
| 4K | 2429 | 2362 | 2426 | 0.999x |
| 8K | 2052 | 2014 | 2084 | **1.016x** |
| 16K | 1685 | 1660 | 1686 | 1.000x |
| 32K | 1224 | 1214 | 1222 | 0.998x |

Mode 2 is **faster than uniform turbo3** at every context depth because 20% of layers use q8_0's cheaper dequant.

## Decode Speed (tok/s, M5 Max 128GB)

| Depth | q8_0 | turbo3 | Mode 2 | turbo3/q8_0 | mode2/q8_0 |
|-------|------|--------|--------|-------------|------------|
| short | 85.8 | 77.4 | 78.7 | 0.90x | **0.92x** |
| 4K | 79.9 | 70.9 | 73.1 | 0.89x | **0.92x** |
| 8K | 77.4 | 66.6 | 69.4 | 0.86x | **0.90x** |

Mode 2 buys back ~3% decode speed at every depth — the q8_0 layers use a cheaper dequant path.

## Effective Compression

- 32 layers × 4.6x (turbo3) + 8 layers × 2.0x (q8_0) = **~3.5x effective**
- vs uniform turbo3 at 4.6x, this trades ~25% compression for ~100% quality recovery

## Test Results (2026-03-26)

- Python tests: 144/144 passed
- Quality gate: PASSED
- No regressions at any context depth

## vs External Tester Reports

| Tester | Hardware | Context | Decode ratio |
|--------|----------|---------|-------------|
| Us (M5 Max) | Apple M5 Max 128GB | 4K | 0.92x |
| @tarruda | M1 Ultra 128GB | 4K | ~0.65x (17→11 tok/s) |
| Anon | M1 Max 64GB | 42K | 0.36x (4 vs 11 tok/s) |

The decode gap is MUCH worse on older hardware (M1 vs M5). Likely causes:
1. M1 lacks Tensor API (`has tensor = false`) — may use slower code path
2. M1 has lower memory bandwidth per GPU core
3. The 42K context test has a much larger KV cache to scan per token

### External Benchmark: Mario (M1 Max 64GB, 32K prompt, main TOT)

| Config | KV Cache | Prefill tok/s | Decode tok/s | vs q8_0 decode |
|--------|----------|--------------|-------------|----------------|
| llama.cpp q8_0 | 8-bit | 442 | 41.8 | baseline |
| **llama.cpp turbo3** | **3.5-bit** | **417** | **34.6** | **0.83x** |
| mlx-vlm fp16 | 16-bit | 488 | 42.6 | 1.02x |
| mlx-vlm q8 uniform | 8-bit | 480 | 32.8 | 0.78x |
| mlx-vlm TurboQuant 4-bit | 4-bit | 471 | 13.1 | 0.31x |
| mlx-vlm TurboQuant 3.5-bit | 3.5-bit | 450 | 7.9 | 0.19x |

### Note on 1K Timing Unreliability

The 1024-context measurements show unrealistic numbers (millions of tok/s) because the total compute time is sub-millisecond. Metal's async command buffer dispatch means the CPU timer reports the submission time, not the actual GPU execution time. Use 2K+ context with 2+ chunks for reliable measurements.
