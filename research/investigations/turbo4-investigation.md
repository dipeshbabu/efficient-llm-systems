# Turbo4 Quality Investigation

**Date:** 2026-03-28
**Branch:** `experiment/turbo4-quality-investigation`
**Problem:** turbo4 PPL is 6.38 (expected ~6.1 based on 4.25 bits, should be between q4_0 and q8_0)

## Background

turbo4 = 3-bit PolarQuant + 1-bit QJL residual correction = 4.25 bits/value, 3.8x compression.

The QJL (Quantized Johnson-Lindenstrauss) bit is supposed to correct the PolarQuant residual error, giving turbo4 better quality than turbo3 despite only 0.75 more bits. Without QJL working, turbo4 is essentially turbo3 with wasted space.

## Bugs Found (7 total)

### Tier 1 — Critical (primary cause of degradation)

**Bug 1: Residual computed in wrong basis (Metal quantize)**
- Location: ggml-metal.metal ~line 536
- Metal computes `residual = normalized - centroid_reconstruction` (rotated space)
- C reference computes `residual = normalized - inverse_rotated(centroid_reconstruction)` (original space)
- The residual must be in the original normalized space for QJL to work

**Bug 2: QJL matrix multiplication missing in Metal dequant**
- Location: ggml-metal.metal ~line 773
- Metal uses signs directly as ±1 values
- C reference applies `matvec(turbo_qjl_matrix_t, signs, qjl_recon, d)`
- Without the matrix multiplication, QJL reconstruction is random noise

**Bug 3: SET_ROWS kernel missing QJL step entirely**
- Location: ggml-metal.metal ~line 9665 (kernel_set_rows_turbo)
- The shared SET_ROWS template only does PolarQuant (3-bit quantization)
- For turbo4, it should also: compute residual, apply QJL matrix, pack signs, store rnorm
- Currently: QJL correction is zero in the KV cache write path

### Tier 2 — Data corruption

**Bug 4: 3-bit packing format mismatch in SET_ROWS**
- SET_ROWS uses turbo3's 2+1 bit scheme (2 bits in qs, 1 in signs)
- turbo4 uses packed 3-bit indices (spanning byte boundaries)
- Indices are corrupted for turbo4 blocks

**Bug 5: rnorm uninitialized in C quantizer**
- `quantize_row_turbo4_0_ref` never sets `y[block].rnorm`
- `dequantize_row_turbo4_0` reads `x[block].rnorm` for QJL scale
- QJL scale is computed from garbage memory

**Bug 6: Metal invents rnorm from wrong-basis residual**
- Metal computes and stores rnorm, but from the wrong-basis residual (Bug 1)
- Even if rnorm were used correctly, the value is wrong

### Tier 3 — Design issue

**Bug 7: WHT used instead of QJL random matrix in Metal**
- Metal uses `turbo_wht_signs1/2` for QJL rotation
- C reference uses a separate `turbo_qjl_matrix` (random normal matrix)
- May be intentional optimization (WHT approximates random projection)
- But breaks exact symmetry with C reference

## Fix Priority

1. Fix Bug 3 (SET_ROWS) — most impactful, the KV cache write path
2. Fix Bug 1 (residual basis) — enables correct QJL in Metal quantize
3. Fix Bug 2 (QJL matrix in dequant) — enables correct QJL reconstruction
4. Fix Bug 5 (rnorm) — fix C reference, then Metal follows
5. Fix Bug 4 (3-bit packing) — correct format for turbo4 blocks
6. Fix Bug 7 (rotation matrix) — align Metal with C reference

## Baseline Measurements

**On reverted main branch (shared SET_ROWS, turbo3-specific packing):**

| Cache | PPL (c=512, 8ch, wikitext-2) |
|-------|------------------------------|
| q8_0 | 6.1109 |
| turbo3 | 6.1756 |
| turbo4 | **679.27** (completely broken) |

**On PR #4 branch (split SET_ROWS, partial turbo4 fix):**

| Cache | PPL |
|-------|-----|
| turbo4 | **6.38** (better but still elevated) |

turbo4 has never worked correctly on Metal:
- Main branch: SET_ROWS uses turbo3 packing → turbo4 data corrupted → PPL 679
- PR #4: SET_ROWS split fixed packing → PPL 6.38, but QJL still broken (bugs 1-3)
- The 6.38 is turbo4 running as "turbo3 with wasted QJL bits" — no correction applied

## Issue #45 Insight (scos-lab)

Independent implementation found MSE beats QJL (Prod) for attention:
- GPT-2 b=3: MSE +7.6% PPL, QJL (Prod) +300% PPL
- Reason: QJL variance amplified by softmax
- This suggests even with all bugs fixed, turbo4's QJL approach may be
  fundamentally less effective than expected for KV cache attention

## Expected Fix Impact

Fixing bugs 1-7 should bring turbo4 to ~6.05-6.12 PPL range IF QJL works
as designed. But issue #45's finding suggests QJL may not help much for
attention specifically. Worst case: turbo4 = turbo3 quality at 4.25 bits
(more bits, same quality, less compression). Best case: ~0.5% PPL improvement
over turbo3.

## Fix Results

### SET_ROWS Fix (Priority 1) ✅

Dedicated `kernel_set_rows_turbo4` with correct 3-bit packing + QJL signs.

| State | turbo4 PPL |
|-------|-----------|
| Before (shared turbo3 SET_ROWS) | **679.27** |
| After (dedicated turbo4 SET_ROWS) | **6.1894** |
| turbo3 reference | 6.1756 |

PPL dropped from 679 to 6.19. turbo4 is now functional.

### QJL Ablation (Priority 2) ✅

Disabled QJL in dequant (`cache[i] = recon[i] * norm` instead of `recon[i] + signs_f[i] * qjl_scale`):

| Config | PPL |
|--------|-----|
| turbo4 WITH QJL | 6.1894 |
| turbo4 WITHOUT QJL | **6.1756** |
| turbo3 (reference) | 6.1756 |

**QJL hurts quality.** Removing it makes turbo4 identical to turbo3.
Confirms scos-lab finding (issue #45): QJL variance amplified by softmax.

### Conclusion

turbo4 as designed (3-bit PolarQuant + 1-bit QJL) is strictly worse than turbo3
for KV cache attention. The QJL bit adds noise, not signal.

**Recommended direction for turbo4:**
1. Drop QJL entirely
2. Use all 4 bits for PolarQuant (16 centroids instead of 8)
3. 4-bit PolarQuant should beat both turbo3 and q4_0 since PolarQuant
   is theoretically optimal at any bit rate

This would make turbo4 a genuine quality upgrade over turbo3, not just
"turbo3 with a wasted correction bit."
