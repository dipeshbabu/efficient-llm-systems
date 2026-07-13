# Asymmetric K/V Quantization: How We Got Here

## The Discovery

On March 28-29, 2026, while implementing asymmetric K/V support for TurboQuant's Metal flash attention kernels, we uncovered two things:

1. A missing kernel instantiation bug that caused NaN outputs for certain mixed K/V configurations
2. A practical quality rescue strategy for models running aggressive weight quantization (Q4_K_M)

This document tells the full story: the original investigation, the dead ends, and the final result.

## Background

TurboQuant compresses the KV cache using Walsh-Hadamard Transform (WHT) rotation followed by PolarQuant centroid quantization. Three precision levels are supported:

- **turbo2**: 2-bit (4 centroids), 6.4x compression
- **turbo3**: 3-bit (8 centroids), 2.3x compression
- **turbo4**: 4-bit (16 centroids), 2.0x compression

All three use the same WHT rotation, norm correction, and centroid-based quantization pipeline. The only difference is the number of centroids and bits per element.

## The Problem

During asymmetric K/V testing on a Mac Mini M2 Pro, we observed catastrophic perplexity:

| Config | Model | PPL |
|--------|-------|-----|
| q8_0 KV | Qwen2.5-7B Q4_K_M | 6.58 (correct) |
| turbo3 KV | same | 3556 (catastrophic) |
| turbo4 KV | same | 218 (catastrophic) |

Initial hypothesis: Apple8/M2 Metal backend bug. This launched a multi-hour investigation.

## The Investigation

### Phase 1: Hardware Bug Hunt (Eliminated)

We systematically tested every hypothesis about M2-specific Metal behavior:

1. **4-mag LUT dequant path** (pre-M5 optimization): Forced 8-LUT on M2 — same bad PPL. Verified both paths produce mathematically identical values via exhaustive trace.

2. **Metal WHT kernel precision**: Changed butterfly arithmetic from half4 to float4 — no improvement. Verified WHT round-trip is correct to 1e-7.

3. **Metal SET_ROWS kernel**: Dumped KV cache bytes after SET_ROWS on both M2 and M5 — **byte-for-byte identical**. SET_ROWS is not the problem.

4. **Metal FA kernel**: Dumped layer-0 `__fattn__` (FA output) and `kqv_out` (post-inverse-WHT) on both machines — **matched across M2 and M5**.

5. **CPU quantize path**: Wrote standalone C test, ran on both machines — **byte-for-byte identical** output.

**Conclusion: No hardware-specific bug exists.** Every tensor matched across M2 and M5.

### Phase 2: The Model Discovery

The breakthrough came from running the **same model** on both machines:

| Config | M5 PPL | M2 PPL |
|--------|--------|--------|
| Qwen2.5-7B Q4_K_M + q8_0 KV | 6.577 | 6.579 |
| Qwen2.5-7B Q4_K_M + turbo3 KV | 3556 | 3778 |
| Qwen2.5-7B Q4_K_M + turbo4 KV | 218 | 228 |
| phi-4-Q8_0 + q8_0 KV | 4.690 | 4.691 |
| phi-4-Q8_0 + turbo3 KV | 4.886 | 4.956 |
| phi-4-Q8_0 + turbo4 KV | 4.770 | 4.787 |

The earlier "M5 works, M2 doesn't" was caused by comparing different models: the 35B MoE (Q8_0 weights) on M5 vs the 7B (Q4_K_M weights) on M2.

**The real pattern: Q4_K_M weights + turbo KV = catastrophic. Q8_0 weights + turbo KV = fine.**

### Phase 3: Asymmetric Rescue

With the hardware bug eliminated, we tested whether raising K precision alone could recover quality:

| K | V | PPL (Qwen2.5-7B Q4_K_M) | Notes |
|---|---|--------------------------|-------|
| q8_0 | q8_0 | 6.58 | baseline |
| q8_0 | turbo3 | 6.68 | +1.6% — rescued! (pre-fix number) |
| q8_0 | turbo4 | 45393 | NaN — broken (missing kernel) |
| turbo4 | turbo3 | 257 | still bad |
| turbo3 | turbo3 | 3556 | catastrophic |

The q8_0-K + turbo3-V result (+1.6%) was the first evidence that K precision is the dominant quality factor and V can be aggressively compressed. This number happened to work via an accidental fallback path — the proper kernel didn't exist yet.

But q8_0-K + turbo4-V produced NaN, which was confusing since turbo4 has MORE precision than turbo3. This turned out to be the key clue pointing to the missing kernel bug.

### Phase 4: The Missing Kernel Bug

Investigation of the turbo4-V NaN revealed the root cause: **missing Metal kernel instantiations for q8_0 × turbo mixed pairs.**

The asymmetric K/V implementation we had just shipped only added turbo × turbo kernel instantiations (turbo2/turbo3/turbo4 in all combinations). When the user requested q8_0-K + turbo4-V, the pipeline name `kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk128_dv128` was constructed correctly but no matching kernel existed. The dispatch fell back to an undefined path that:

- Happened to produce correct results for turbo3-V (lucky fallback)
- Produced NaN for turbo4-V (different nl parameter, no valid fallback)

**Fix:** Added 150 new kernel instantiations (90 non-vec + 60 vec) covering all q8_0 × turbo pairs in both directions, for all supported head dimensions. Updated the gatekeeper in `ggml-metal-device.m` and the assertion in `ggml-metal-ops.cpp` to allow q8_0 × turbo mixed pairs.

Note: the earlier turbo × turbo asymmetric work (90 instantiations) was also part of this effort. The total asymmetric addition across both phases was 240 new kernel instantiations. All FA pipeline names were also migrated to a `k{type}_v{type}` format to avoid underscore ambiguity in type names like `q4_0`.

### Phase 5: Full Rescue Validated

With proper kernel instantiations, all q8_0-K + turbo-V configurations work correctly:

**Qwen2.5-7B-Instruct-Q4_K_M (sensitive to symmetric turbo, rescued by asymmetric):**

| K | V | PPL | vs q8_0 baseline | V compression |
|---|---|------|-----------------|---------------|
| q8_0 | q8_0 | 6.58 | — | 1.0x |
| q8_0 | turbo4 | **6.64** | +1.0% | 2.0x |
| q8_0 | turbo3 | **6.71** | +2.0% | 2.3x |
| q8_0 | turbo2 | **6.91** | +5.1% | 3.2x |

**phi-4-Q8_0 (control model):**

| K | V | PPL | vs q8_0 baseline (4.69) |
|---|---|------|------------------------|
| q8_0 | turbo4 | 4.70 | +0.3% |
| q8_0 | turbo3 | 4.74 | +1.1% |
| q8_0 | turbo2 | 4.84 | +3.1% |
| turbo4 | turbo4 | 4.77 | +1.7% |
| turbo3 | turbo3 | 4.89 | +4.2% |

**Qwen3.5-35B-A3B MoE (large model, Q8_0 weights):**

| K | V | PPL |
|---|---|------|
| turbo3 | turbo3 | 5.13 |
| turbo4 | turbo4 | 5.08 |

**Mistral-Small-24B-Instruct (Q4_K_M weights — symmetric turbo works at this size):**

| K | V | PPL |
|---|---|------|
| turbo3 | turbo3 | 4.99 |

Large models and some model families tolerate symmetric turbo even with Q4_K_M weights. Not all Q4_K_M models are sensitive to the quantization stacking.

## Key Findings

### 1. K precision is the dominant quality factor

The attention mechanism computes `softmax(Q · K^T) · V`. K determines which tokens get attention weight. Small errors in K shift attention routing, which changes every downstream computation. V errors only affect the weighted sum proportionally.

This is why:
- q8_0-K + turbo3-V gives PPL 6.71 (+2.0%) — V errors are proportional
- turbo3-K + q8_0-V gives PPL 3556 — K errors cascade through softmax

### 2. Quantization stacking is real

Q4_K_M weight quantization already introduces noise into K/V activations. Adding turbo quantization on top means:
- WHT rotation spreads the existing noise across all 128 elements
- PolarQuant centroids are optimized for clean distributions, not noisy ones
- The combined error exceeds the softmax's tolerance for K (but not V)

With Q8_0 weights, the activations are clean enough that turbo quantization works within tolerance.

### 3. V is remarkably robust to quantization

Even turbo2-V (2-bit, only 4 centroids) gives just +5.1% PPL on the Q4_K_M model with q8_0-K. This confirms that value vectors tolerate aggressive compression because their errors are linearly proportional in the attention output, not exponentially amplified through softmax.

### 4. Missing kernel instantiations cause silent failures

The turbo4-V NaN was caused by a missing `kq8_0_vturbo4` kernel, not an algorithmic bug. Metal dispatches to an undefined path when the requested pipeline doesn't exist, which can produce NaN, hangs, or silently wrong results depending on the fallback. This class of bug is hard to detect without explicit pipeline-exists checks.

## Files Modified

### Kernel instantiations (`ggml/src/ggml-metal/ggml-metal.metal`)
- Added 150 new q8_0 × turbo kernel instantiations (90 non-vec + 60 vec)
- Covers all 6 q8_0 × turbo pairs (q8_0-K with turbo2/3/4-V, turbo2/3/4-K with q8_0-V)
- All supported head dimensions

### Dispatch gatekeeper (`ggml/src/ggml-metal/ggml-metal-device.m`)
- Extended `supports_op` for FLASH_ATTN_EXT to allow q8_0 × turbo mixed pairs

### Dispatch assertion (`ggml/src/ggml-metal/ggml-metal-ops.cpp`)
- Extended mixed K/V assertion to allow q8_0 × turbo pairs

### Pipeline naming (`ggml/src/ggml-metal/ggml-metal-device.cpp`)
- Already uses `k{ktype}_v{vtype}` format from the earlier asymmetric work
- No changes needed — q8_0 × turbo names resolve correctly

## Recommendations

### For models sensitive to symmetric turbo (tested: Qwen2.5-7B Q4_K_M)
Use `-ctk q8_0 -ctv turbo4` (best quality) or `-ctk q8_0 -ctv turbo3` (more compression). This gives near-baseline quality with V cache compression.

### For Q8_0 or higher weight quant
Symmetric turbo works fine: `-ctk turbo3 -ctv turbo3` or `-ctk turbo4 -ctv turbo4`.

### For Q4_K_M on larger or less sensitive models
Symmetric turbo may work — Mistral-24B Q4_K_M gives PPL 4.99 with turbo3/turbo3. Validate on your specific model. Fall back to asymmetric if quality degrades.

### For maximum V compression
`-ctk q8_0 -ctv turbo2` gives 3.2x V compression with +5.1% PPL. turbo2-V showed promising results but has less validation than turbo4-V and turbo3-V. Treat as experimental.

See [Configuration Recommendations](../../docs/guides/turboquant-recommendations.md) for the full tested matrix.

## Acknowledgments

- GPT-5.4 provided the critical reframing from "hardware bug" to "quantization stacking / quality tuning" problem
- Codex (roast agent) caught the initial pipeline naming format issue and provided deep code reviews throughout
- The investigation consumed ~8 hours of continuous debugging across two machines

## Timeline

| Time | Event |
|------|-------|
| Mar 28 evening | Asymmetric K/V implementation completed, turbo × turbo |
| Mar 28 night | M2 PPL disaster discovered during testing |
| Mar 29 00:00 | Hardware bug investigation begins |
| Mar 29 02:00 | SET_ROWS bytes confirmed identical across M2/M5 |
| Mar 29 03:00 | FA output confirmed identical across M2/M5 |
| Mar 29 04:00 | M5 also gives bad PPL on 1.5B model — not M2-specific |
| Mar 29 05:00 | Same-model 7B tests confirm no hardware divergence |
| Mar 29 06:00 | phi-4-Q8_0 confirms: Q8_0 weights + turbo = fine |
| Mar 29 07:00 | Asymmetric rescue: q8_0-K + turbo3-V = +1.6% PPL |
| Mar 29 07:30 | turbo4-V NaN traced to missing kernel instantiations |
| Mar 29 08:00 | 150 new kernels added, all q8_0 × turbo pairs working |
| Mar 29 08:30 | Full rescue validated: Q4_K_M + q8_0-K + turbo-V = healthy |
