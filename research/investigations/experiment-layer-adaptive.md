# Experiment: Layer-Adaptive KV Cache

Branch: `experiment/layer-adaptive`

## Hypothesis
Late attention layers are more quality-sensitive. Give them higher precision (q8_0), use turbo3 for the less-sensitive early/middle layers. Trade slightly less compression for q8_0 quality.

## Implementation
Added `TURBO_LAYER_ADAPTIVE` env var to select per-layer cache type strategy:
- Mode 0: uniform turbo3 (default)
- Mode 1: q8_0 for first+last 4 layers, turbo3 for middle 32
- Mode 2: q8_0 for last 8 layers, turbo3 for first 32

## Results (Qwen3.5-35B-A3B, wikitext-2, 8 chunks)

| Config | Layers at turbo3 | PPL | vs q8_0 (6.111) | Effective compression |
|--------|-----------------|-----|-----------------|----------------------|
| Uniform turbo3 | 40/40 (100%) | 6.193 | +1.3% | 4.6x |
| Mode 1: q8_0 edges | 32/40 (80%) | 6.185 | +1.2% | ~3.5x |
| **Mode 2: q8_0 last 8** | **32/40 (80%)** | **6.110** | **-0.02%** | **~3.5x** |
| Uniform q8_0 | 0/40 | 6.111 | baseline | 2.0x |

## Key Finding

**The last 8 layers account for essentially ALL of the turbo3 quality loss.**

Mode 2 achieves PPL 6.110 — indistinguishable from q8_0 (6.111) — while keeping 80% of layers at turbo3 compression. The effective compression is ~3.5x overall:
- 32 layers × 4.6x (turbo3) + 8 layers × 2.0x (q8_0)
- Weighted: 32/40 × 4.6 + 8/40 × 2.0 = 3.68 + 0.40 = ~3.5x effective

## Implications

1. **Layer sensitivity is highly non-uniform.** Early layers are insensitive to quantization. Late layers are very sensitive.
2. **A smarter allocation could go further.** Instead of binary turbo3/q8_0, a gradient (turbo2 early → turbo3 mid → q8_0 late) could push effective compression to ~4x while matching q8_0 quality.
3. **This compounds with temporal decay.** Old tokens in early layers could use turbo2, old tokens in late layers use turbo3. Recent tokens everywhere use turbo3/q8_0.

## Status
COMPLETE — prototype working. Production version would need:
- CLI arg for adaptive mode (instead of env var)
- Per-model calibration to find the optimal layer cutoff
- Proper compression ratio calculation for the effective mix
