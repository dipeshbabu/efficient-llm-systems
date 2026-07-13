# Speed Experiments Log

Branch: `experiment/speed-optimization` (both repos)
Goal: prefill speed closer to q8_0 (currently 1074 vs 2694 tok/s) while PPL stays at 6.19 +/- 0.1

## Baseline (before experiments)

| Config | Prefill tok/s | PPL | Notes |
|--------|-------------|-----|-------|
| q8_0 | 2694 | 5.41 | target |
| turbo3 fp16 WHT | 1074 | 5.47 | current top-of-tree (32 chunks) |
| turbo3 fp16 WHT | — | 6.195 | 8-chunk PPL reference |
| turbo3 no rotation | 1577 | — | speed ceiling (wrong quality) |

## Experiment 1: Vectorized half4 WHT + packed centroid lookup

**Hypothesis:** The WHT butterfly and centroid unpacking can be vectorized with half4 operations for 4x wider SIMD throughput. Also optimizes memory access patterns for qs/signs bytes.

**Changes:**
- `turbo_fwht_128_half4()`: WHT butterfly on 32 x half4 vectors instead of 128 x half scalars
  - h=1,2: intra-vector swizzle (no loop over pairs)
  - h=4..64: inter-vector butterfly with computed stride
- Centroid lookup: process 4 elements per qs byte (natural byte boundary)
- Sign application: vectorized half4 multiply
- Final conversion: float4 output with fused norm scale

**Results:**

| Config | Prefill tok/s | PPL (32-chunk) | PPL (8-chunk) |
|--------|-------------|----------------|---------------|
| Baseline (scalar fp16 WHT) | 1074 | 5.47 | 6.195 |
| **half4 vectorized WHT** | **1411** | **5.47** | **6.195** |
| q8_0 | 2694 | 5.41 | — |

**+31% speedup, PPL unchanged.** Gap to q8_0: 1.91x (was 2.51x).

**Codex review:** No correctness bugs found. Butterfly pairing, centroid unpacking, and sign application all verified correct.

**Status:** COMPLETE — committed

---

## Experiment 2: Pre-packed half4 sign arrays

**Changes:** Pre-computed `turbo_wht_signs1_h4[32]` and `turbo_wht_signs2_h4[32]` as constant half4 arrays, eliminating per-element float→half conversion in the dequant.

**Results:** 1411 → 1424 tok/s (+1%). Marginal — Metal compiler already optimized constant reads.

**Status:** COMPLETE — committed (minor win)

---

## Experiment 3: RoPE-aware pre-rotate-queries (THE BIG WIN)

**Hypothesis:** Earlier pre-rotate-queries failed (PPL 23.5) because it was placed in build_attn_mha (after permute). The fix: apply WHT in build_attn, after RoPE is already applied to Q, before build_attn_mha. This matches the K pipeline: K gets WHT during quantize (SET_ROWS), which happens after RoPE.

**Changes:**
- Q rotation: `ggml_mul_mat(R, q)` in build_attn, after cpy_k/cpy_v, before build_attn_mha
- V un-rotation: `ggml_mul_mat(R^T, cur)` after build_attn_mha, before wo projection
- Stripped WHT from turbo3_dequantize_full_block (returns centroid * norm, no rotation)

**Results:**

| Config | Prefill tok/s | PPL (32-chunk) | PPL (8-chunk) | vs q8_0 |
|--------|-------------|----------------|---------------|---------|
| turbo3 dequant WHT (Exp1+2) | 1424 | 5.47 | 6.195 | 0.53x |
| **turbo3 graph WHT** | **2095** | **5.46** | **6.201** | **0.78x** |
| q8_0 baseline | 2694 | 5.41 | — | 1.00x |

**+47% speedup over Exp1+2. 4.9x compression at 78% of q8_0 speed.**

PPL 6.201 — within 0.01 of 6.195 baseline. Quality target MET.

**Codex review findings:**
1. Q rotation gate `q->ne[0] % 128 == 0` would skip non-256 head dims — add assert
2. V un-rotation keyed off k->type not v->type — acceptable for turbo3 (always both)
3. Only covers the `llm_graph_input_attn_kv` build_attn overload — other paths need same treatment

**Why it works now (vs PPL 23.5 earlier):**
The earlier attempt applied WHT in build_attn_mha AFTER the ggml_permute. But the permute doesn't change values, so the pipeline point is equivalent. The real fix was the ggml column-major storage correction (swapping TURBO_ROTATION_R and TURBO_ROTATION_RT). The Gemini RoPE/WHT commutativity explanation was wrong — the issue was purely the matrix orientation in ggml.

**Status:** COMPLETE — committed

---

## Experiment 4: Block-32 with graph WHT (THE BREAKTHROUGH)

**Hypothesis:** With graph-side WHT, dequant no longer needs the 128-element butterfly. Block-128 forces the flash attention kernel to process 128 elements per block (nl=8 non-vec, nl=32 vec). Block-32 gives nl=2/8 — matching q4_0's parallelism.

**Changes:**
- QK_TURBO3: 128 → 32
- Dequant functions: simple centroid lookup + norm scale (16 lines total)
- Flash attention nl: 8→2 (non-vec), 32→8 (vec)

**Results:**

| Config | Prefill tok/s | PPL (32-chunk) | PPL (8-chunk) | vs q8_0 |
|--------|-------------|----------------|---------------|---------|
| turbo3 block-128 graph WHT | 2095 | 5.46 | 6.201 | 0.78x |
| **turbo3 block-32 graph WHT** | **2747** | **5.46** | **6.193** | **1.02x** |
| q8_0 baseline | 2694 | 5.41 | — | 1.00x |

**Q8_0 PARITY ACHIEVED.** 4.6x compression at 102% of q8_0 prefill speed.

**Status:** COMPLETE — committed

---

## Full Optimization Journey

| Step | Prefill tok/s | vs q8_0 | Commit |
|------|-------------|---------|--------|
| fp32 WHT (start) | 739 | 0.27x | feature branch |
| + fp16 WHT | 1074 | 0.40x | feature branch |
| + half4 vectorized butterfly | 1411 | 0.52x | e4e0bde |
| + pre-packed half4 signs | 1424 | 0.53x | 640e10e |
| + graph-side WHT rotation | 2095 | 0.78x | 676f929 |
| + block-32 | **2747** | **1.02x** | c84e124 |
| q8_0 baseline | 2694 | 1.00x | — |

**3.72x total speedup from first to last optimization.**

---

## Experiment 5 (was 4): Reduced centroid lookup overhead

**Hypothesis:** The 3-bit index unpacking does 3 loads + 2 shifts + 1 OR per element. Pre-combining indices during quantize into a single packed array would reduce dequant to 1 load + mask.

**Status:** PENDING

---

## Experiment 3: RoPE-aware pre-rotate-queries

**Hypothesis:** The earlier pre-rotate-queries failed because WHT and RoPE don't commute. Fix: apply WHT immediately AFTER RoPE in the model code (not in build_attn_mha). This eliminates the WHT from dequant entirely.

**Status:** PENDING
