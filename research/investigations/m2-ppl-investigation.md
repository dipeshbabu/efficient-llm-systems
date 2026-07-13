# TurboQuant PPL Investigation (2026-03-29)

Issue: historical `llama-cpp-turboquant#27`
([public endpoint unavailable](../../docs/reference/historical-forks.md#archived-identifiers))

## Final Status: RESOLVED

Two issues found:
1. **Missing q8_0 × turbo FA kernel instantiations** — caused NaN/undefined behavior for mixed q8_0 + turbo K/V configs. Fixed by adding 150 kernel instantiations.
2. **Quantization stacking on Q4_K_M models** — Q4_K_M weights + turbo K is too lossy for attention routing. Mitigated by asymmetric q8_0-K + turbo-V.

The original M2/Apple8 hardware bug hypothesis was retired after same-model cross-hardware tests confirmed identical behavior on M5 and M2.

## Evidence Summary

### No hardware-specific bug

| Config | M5 PPL | M2 PPL |
|--------|--------|--------|
| phi-4-Q8_0 + q8_0 KV | 4.690 | 4.691 |
| phi-4-Q8_0 + turbo4 KV | 4.770 | 4.787 |
| phi-4-Q8_0 + turbo3 KV | 4.886 | 4.956 |
| Qwen2.5-7B Q4_K_M + q8_0 KV | 6.577 | 6.579 |
| Qwen2.5-7B Q4_K_M + turbo3 KV | 3556 | 3778 |

Layer-0 attention probes (`__fattn__-0`, `kqv_out-0`) matched across M2 and M5. KV cache bytes matched byte-for-byte after SET_ROWS.

### Missing kernel bug

Asymmetric K/V support only instantiated turbo × turbo FA kernel pairs. q8_0 × turbo pairs were missing:
- `q8_0-K + turbo4-V` → NaN (no kernel, undefined dispatch)
- `q8_0-K + turbo3-V` → accidentally worked via fallback (lucky)

Fixed by adding all q8_0 × turbo instantiations for both vec and non-vec FA paths.

### Asymmetric rescue for Q4_K_M

Qwen2.5-7B-Instruct-Q4_K_M with proper kernels:

| K | V | PPL | vs q8_0 (6.58) |
|---|---|------|----------------|
| q8_0 | turbo4 | **6.64** | +1.0% |
| q8_0 | turbo3 | **6.71** | +2.0% |
| q8_0 | turbo2 | **6.91** | +5.1% |

K precision is the dominant quality factor. V tolerates aggressive compression because errors are proportional through the attention weighted sum, not exponential through softmax.

## Investigation Timeline

| Step | Finding |
|------|---------|
| 1-5 | Eliminated 4-mag LUT, FA kernel, Metal WHT precision, SET_ROWS as M2-specific culprits |
| 6-9 | CPU dequant never called with -ngl 99; CPU quantize byte-identical on M2/M5 |
| 10-13 | KV cache bytes identical; FA output identical across M2/M5 |
| 14-16 | Same model gives same (bad) PPL on both machines — not hardware-specific |
| 17-18 | Q8_0 weight models work fine; Q4_K_M is the problem |
| 19 | Asymmetric q8_0-K + turbo3-V rescues quality to +1.6% |
| 20 | turbo4-V NaN traced to missing kernel instantiations |
| 21 | 150 new kernels added; all q8_0 × turbo pairs validated |

## Files Changed (production)

- `ggml/src/ggml-metal/ggml-metal.metal` — 150 new q8_0 × turbo FA kernel instantiations
- `ggml/src/ggml-metal/ggml-metal-device.m` — gatekeeper allows q8_0 × turbo pairs
- `ggml/src/ggml-metal/ggml-metal-device.cpp` — pipeline naming (already correct from k/v prefix format)
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` — assertion allows q8_0 × turbo pairs

## Regression sweep

Pending — running A/B/C/D matrix on M5 and M2 before push.
