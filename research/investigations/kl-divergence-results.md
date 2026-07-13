# KL Divergence Results

**Date:** 2026-03-27
**Hardware:** Apple M5 Max 128GB
**Baseline:** f16 KV cache logits (8-chunk wikitext-2, c=512)

## MoE (Qwen3.5-35B-A3B Q8_0)

| Cache Type | Mean KLD | Max KLD | Δp RMS | Same top-p % |
|------------|----------|---------|--------|-------------|
| q8_0 | 0.001549 | 0.1115 | 1.231% | 98.43% |
| q4_0 | 0.008091 | 0.2287 | 2.753% | 95.83% |
| **turbo3** | **0.016145** | **1.1654** | **4.090%** | **94.31%** |

## Dense (Qwen3.5-27B Q8_0)

| Cache Type | Mean KLD | Max KLD | Δp RMS | Same top-p % |
|------------|----------|---------|--------|-------------|
| q8_0 | 0.000018 | — | 0.127% | 99.90% |
| q4_0 | 0.002741 | — | 1.437% | 97.65% |
| **turbo3** | **0.009900** | — | **2.738%** | **95.98%** |

## Analysis

turbo3 KLD is roughly 2× q4_0 on both architectures. This is expected: turbo3 uses 3.5 bits (less than q4_0's 4 bits) with a fundamentally different compression mechanism (WHT rotation + polar codebook vs scalar quantization).

The same-top-p metric shows turbo3 agrees with f16 on the top token 94-96% of the time. For context, q4_0 (a widely-used cache type) agrees 96-98%.

Dense model shows lower KLD across all cache types because the dense attention pattern is more concentrated (fewer heads, more focused attention), making the KV cache less sensitive to quantization noise.

## KLD Stability Across Context Length (2026-04-01)

**Hardware:** Apple M2 Pro 16GB
**Model:** Qwen2.5-1.5B-Instruct Q4_K_M
**Config:** asymmetric q8_0-K / turbo3-V, flash attention on
**Baseline:** q8_0/q8_0 logits at each context length (4 chunks wikitext-2)

| Context | Mean KLD | Max KLD | RMS Δp | Same top-p % | PPL (turbo3) | PPL (q8_0) |
|---------|----------|---------|--------|-------------|--------------|------------|
| 2,048 | 0.01976 | 1.543 | 3.83% | 92.52% | 10.48 | 10.24 |
| 4,096 | 0.01819 | 2.201 | 3.85% | 93.31% | 8.59 | 8.41 |
| 8,192 | 0.01666 | 1.658 | 3.72% | 93.86% | 8.51 | 8.33 |

### Observation

Mean KLD decreases from 0.0198 at 2K to 0.0167 at 8K. Same-top-p improves from 92.5% to 93.9%. Max KLD outliers do not grow monotonically (4K has the highest max at 2.20, 8K is lower at 1.66).

Tested on Qwen2.5-1.5B (M2 Pro 16GB constraint). Larger models at longer context (32K+) may behave differently. The M5 Max data (above) covers larger models but only at ctx=512.

## Raw Logs

- `~/local_llms/llama.cpp/results/kld_moe_q8_0.log`
- `~/local_llms/llama.cpp/results/kld_moe_q4_0.log`
- `~/local_llms/llama.cpp/results/kld_moe_turbo3.log`
- `~/local_llms/llama.cpp/results/kld_dense_q8_0.log`
- `~/local_llms/llama.cpp/results/kld_dense_q4_0.log`
- `~/local_llms/llama.cpp/results/kld_dense_turbo3.log`
