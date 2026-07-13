# Experiment: Context Scaling Fix

Branch: `experiment/context-scaling-fix`
Issue: https://github.com/dipeshbabu/turboquant_plus/issues/32

## Problem
turbo3 prefill speed degrades relative to q8_0 as context grows. ~2% gap per context doubling.

| Context | turbo3 tok/s | q8_0 tok/s | ratio |
|---------|-------------|-----------|-------|
| 1024 | 4795 | 4856 | 0.99x |
| 2048 | 3009 | 3156 | 0.95x |
| 4096 | 2415 | 2584 | 0.93x |

## Root Cause
Two `ggml_mul_mat(128x128, 128xN)` per attention layer for Q/V rotation. N scales with context x heads.

**Key finding:** Generation (tg) overhead is <0.1% — the issue is ONLY prefill (pp).

## Fix Approaches Investigated

### Approach 1: Skip ggml_cont if contiguous
**Result:** +1% (negligible). Tensors were already contiguous.

### Approach 2: Reduce rotation group size (128 → 32)
**Hypothesis:** 32x32 matmul is 16x less compute than 128x128. Python test shows kurtosis 3.06 (better than 128's 3.41).

**Changes needed:**
- QK_TURBO3_GROUP: 128 → 32
- Metal SET_ROWS kernel: 32-element WHT rotation
- Metal WHT signs: new arrays for d=32
- turbo-rotation-data.h: 32x32 R and R^T (generated, saved as turbo-rotation-data-32.h)
- Graph rotation: 32x32 matmul (automatic from tensor size)
- KV cache: 32x32 rotation tensor allocation

**Expected speedup:** 16x less matmul compute. The 2% per-doubling gap should shrink to ~0.1%.

**Result:** FAILED. PPL = 7.06 (target 6.19). Real KV tensors from Qwen3.5 need 128-element groups for proper Gaussianization. Python random data showed good kurtosis at d=32 but real model data is different. Speed was also worse. Dead end.

**Status:** FAILED

### Approach 3: Custom GGML_OP_TURBO_WHT
**Codex analysis:** 10+ files to modify. `ggml_map_custom1` exists for CPU prototyping but no Metal support. Full custom op is the "right" solution but high effort.

**Status:** DEFERRED — approach 2 may be sufficient

### Approach 4: Hybrid dequant/graph rotation
Put WHT back in dequant for prefill, keep graph-side for generation. Not feasible — can't switch strategy at runtime without architectural changes.

## External Data
@tarruda benchmark on M1 Ultra with 397B model: q8_0 baseline itself degrades 4.7x from 0→250k context (natural attention scaling). Need turbo3 comparison at same depths.
