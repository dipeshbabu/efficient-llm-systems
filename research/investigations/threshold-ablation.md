# Sparse V Threshold Ablation (τ sweep)

**Date:** 2026-03-27
**Hardware:** Apple M5 Max 128GB
**Model:** Qwen3.5-35B-A3B Q8_0
**KV Cache:** turbo3 (3.5-bit, 4.6× compression)

## Results

| τ | PPL (8-chunk) | vs q8_0 (6.111) | Decode tok/s (short) | Decode tok/s (pp32768+tg128) |
|---|--------------|-----------------|---------------------|------------------------------|
| 1e-4 | 6.1756 | +1.06% | 76.3 | 1111.1 |
| 1e-5 | 6.1756 | +1.06% | 76.5 | 1112.7 |
| **1e-6** | **6.1756** | **+1.06%** | **76.1** | **1113.8** |
| 1e-7 | 6.1756 | +1.06% | 75.7 | 1113.8 |
| 1e-8 | 6.1756 | +1.06% | 76.4 | 1114.4 |

## Analysis

**PPL is identical across all thresholds.** Even τ=1e-4 (the most aggressive skip) produces the exact same 8-chunk perplexity as τ=1e-8 (essentially no skip). This confirms the attention sparsity hypothesis: positions below 1e-4 contribute nothing measurable to output quality.

**Short-context decode speed is flat.** At short context (~128 tokens), attention is dense — almost no positions have weights below any of these thresholds, so the skip condition rarely triggers. The ~±1 tok/s variation is within measurement noise.

**The threshold effect is context-dependent.** The sparse V benefit scales with context length because longer contexts have exponentially more near-zero attention weights. The original regression suite (in `sparse-v-dequant.md`) measured +22.8% at 32K and +1.4% at short context, which is consistent with these results.

## Conclusion

τ=1e-6 remains the right default. More aggressive thresholds (1e-4, 1e-5) are equally safe quality-wise but don't improve short-context speed. The benefit at long context is already captured by the existing threshold. There's potential headroom to raise to 1e-4 if future long-context benchmarks confirm no degradation on harder retrieval tasks (e.g., multi-needle NIAH at 128K).

## Raw Logs

See `threshold-ablation-logs/` for per-threshold benchmark output.
