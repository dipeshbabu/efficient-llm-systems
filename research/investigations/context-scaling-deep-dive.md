# Context Scaling Deep Dive

Issue: https://github.com/dipeshbabu/turboquant_plus/issues/32

## Problem Statement

turbo3 prefill speed degrades relative to q8_0/fp16 as context length increases. At short context (~512 tokens), turbo3 matches or beats q8_0. At longer contexts, a growing gap appears.

### Measured Regression (M5 Max 128GB, Qwen3.5-35B-A3B, flash attention ON)

| Context | turbo3 tok/s | q8_0 tok/s | turbo3/q8_0 |
|---------|-------------|-----------|-------------|
| 1024 | 4796 | 4914 | 0.976x |
| 2048 | 3036 | 3162 | 0.960x |
| 4096 | 2371 | 2573 | 0.921x |

Gap grows ~2% per context doubling. On larger models and older hardware, the regression compounds:
- M1 Ultra, Qwen 3 397B: turbo3 decode drops from 17 → 4 tok/s at 4k context (vs fp16 staying at 19-20)
- M1 Max 64GB, Qwen3.5-35B-A3B: turbo3 prefill 272 vs q8_0 364 at 42k context (0.75x)

### Quality Verified

| | PPL (8-chunk) | PPL (32-chunk) |
|---|---|---|
| q8_0 | 6.111 ± 0.326 | 5.414 ± 0.140 |
| turbo3 | 6.211 ± 0.333 | 5.460 ± 0.141 |

Quality is not the issue. The regression is purely speed.

## Root Cause Investigation

### Red Herring: Graph-Side Rotation Matmul

Initial hypothesis: the two `ggml_mul_mat(128x128, 128xN)` per attention layer for Q forward rotation and V un-rotation scale with context because N = (n_embd_head × n_head × n_tokens) / 128.

**Testing methodology:**
1. Measured with dense 128x128 matmul (original approach)
2. Implemented custom GGML_OP_TURBO_WHT — O(d log d) Walsh-Hadamard butterfly replacing O(d²) matmul
3. Compared context scaling for both approaches

**Results:**

| Context | Dense matmul | Custom WHT op | q8_0 |
|---------|-------------|--------------|------|
| 1024 | 4795 (0.99x) | 4796 (0.98x) | 4914 |
| 2048 | 3009 (0.95x) | 3036 (0.96x) | 3162 |
| 4096 | 2415 (0.94x) | 2371 (0.92x) | 2573 |

**Conclusion: the graph rotation matmul is NOT the bottleneck.** The custom WHT op (18x less compute) gives identical performance. The rotation overhead is ~1% constant per layer, verified by disabling both rotations entirely (delta was <1%).

### Also Eliminated: ggml_cont Overhead

Tested skipping ggml_cont when tensors are already contiguous. Result: +1% (negligible). The tensors from RoPE output and flash attention are already contiguous.

### Also Eliminated: Group Size 32 Rotation

Reduced WHT rotation group from 128 to 32 elements (16x less matmul compute). Result: PPL degraded to 7.06 (target 6.19). Real KV tensors need 128-element rotation groups for proper Gaussianization. Dead end.

### Actual Bottleneck: Per-Position Dequant in Flash Attention

The real cost is inside the flash attention Metal kernel. For every cached token position (all of them, scaling linearly with context), the turbo3 block must be dequantized:

**turbo3 dequant per 4 elements (dequantize_turbo3_0_t4):**
```
1. Read qs byte (1 device memory read)
2. Shift + mask for low2 bits (2 ALU ops per element)
3. Read signs byte (1 device memory read)
4. Shift + mask for hi1 bit (2 ALU ops per element)
5. Combine: idx = low2 | (hi1 << 2) (2 ALU ops per element)
6. Table lookup: turbo_centroids_3bit[idx] (1 constant memory read per element)
7. Multiply by norm (1 multiply per element)
```
Total: ~2 device reads + 7 ALU ops + 4 constant reads + 4 multiplies per 4 elements

**q8_0 dequant per 4 elements (dequantize_q8_0_t4):**
```
1. Read 4 int8 values (1 device memory read, 4 bytes)
2. Multiply by scale factor d (4 multiplies)
```
Total: ~1 device read + 4 multiplies per 4 elements

turbo3 dequant is ~3-4x more compute per element than q8_0. This difference is constant per element, but multiplied by every cached position, scales linearly with context.

### Why Turbo3 Still Wins at Short Context

At short context, the advantage is memory bandwidth. turbo3 stores 3.5 bits/value vs q8_0's 8 bits/value. The KV cache is 2.3x smaller, so less memory is read from unified memory during attention. This bandwidth savings MORE than compensates for the extra dequant compute.

At longer context, the dequant compute dominates because:
1. More positions to dequant per attention op
2. The GPU ALUs become saturated with dequant work
3. Memory bandwidth savings are relatively smaller (the attention matrix itself grows quadratically)

## Fix Paths

### Path 1: Fused Compressed Attention (Highest Impact)

**Idea:** Compute Q·K dot products directly on quantized indices without full dequantization.

For turbo3, each element maps to one of 8 centroids. The dot product `q · k_dequant` can be decomposed:

```
dot(q, k_dequant) = Σ q[i] * centroid[idx[i]] * norm
                   = norm * Σ q[i] * centroid[idx[i]]
```

Precompute `q_centroid_dot[c] = Σ_{i where idx[i]=c} q[i]` for each of the 8 centroids. Then the dot product is just 8 partial sums, one per centroid — regardless of dimension.

**Expected speedup:** Eliminates per-element dequant entirely for Q·K path. The V path still needs dequant for the weighted sum, but Q·K is the bottleneck.

**Complexity:** Requires a custom flash attention kernel variant for turbo3.

### Path 2: Faster Turbo3 Dequant (Medium Impact)

**Idea:** Reduce the per-element cost of turbo3 dequant.

Options:
- **Byte-indexed centroid LUT:** Precompute a 256-entry table mapping each possible qs byte to 4 centroid values. One table lookup per 4 elements instead of 4 individual bit-extract + table-lookup sequences.
- **Store centroids directly:** During quantize, store the actual centroid values (fp16) instead of 3-bit indices. Dequant becomes a simple fp16 → fp32 conversion + norm multiply. Trades storage (more bits) for compute (simpler dequant).
- **Pack 3-bit indices contiguously:** Current layout splits 3-bit index into 2-bit qs + 1-bit signs (two separate arrays). Packing all 3 bits together would reduce the number of device memory reads.

### Path 3: Reduce Block Overhead (Low Impact)

**Idea:** Optimize the block structure to reduce per-block overhead.

Options:
- Share norm across 4 blocks (128-element group norm instead of per-32 norm)
- Reduce fp16 norm to int8 scale factor
- These save storage but don't address the compute bottleneck

## External Test Data

### @tarruda (M1 Ultra 128GB, Qwen 3 397B IQ2_XS)

q8_0 baseline at various context depths:

| Context | pp512 tok/s | tg128 tok/s |
|---------|-----------|-----------|
| 0 | 189.7 | 20.0 |
| 10k | 168.9 | 18.9 |
| 50k | 118.1 | 15.7 |
| 100k | 80.2 | 12.8 |
| 250k | 40.7 | 8.0 |

q8_0 itself degrades 4.7x from 0→250k (natural attention scaling). turbo3 data at these depths pending.

### Anonymous (M1 Max 64GB, Qwen3.5-35B-A3B Q8_0)

42k context prompt:
- q8_0: prefill 364 t/s, decode 11 t/s
- turbo3: prefill 272 t/s, decode 4 t/s

Decode regression (4 vs 11 = 0.36x) is much worse than prefill (0.75x). This suggests the per-position dequant cost is especially bad for decode (single-token generation processes entire KV cache).

## Implementation Status

| Approach | Status | Result |
|----------|--------|--------|
| Custom GGML_OP_TURBO_WHT | ✅ DONE | No scaling improvement (matmul wasn't bottleneck) |
| Skip ggml_cont | ✅ DONE | +1% (negligible) |
| Group-32 rotation | ✅ DONE | PPL 7.06 (quality failure) |
| **Optimized dequant** | ✅ **DONE** | **FIXED: 0.98x q8_0 flat across contexts (was 0.92x at 4k)** |
| Fused compressed attention | ⏳ DEFERRED | Not needed — dequant optimization was sufficient |
| Reduce block overhead | ⏳ DEFERRED | Not needed |

## Resolution

The optimized dequant (unrolled with batched byte reads, Codex-verified bit indexing) eliminates the context scaling regression:

| Context | Before (turbo3/q8_0) | After (turbo3/q8_0) |
|---------|---------------------|---------------------|
| 1024 | 0.976x | **0.981x** |
| 2048 | 0.960x | **0.989x** |
| 4096 | 0.921x | **0.981x** |

The ratio is now flat at ~98% vs q8_0 regardless of context length. The previous degradation came from inefficient per-element byte reads in the dequant — reading the same qs/signs bytes 4 times instead of once.

### Extended Verification (2K through 32K context)

| Context | turbo3 tok/s | q8_0 tok/s | turbo3/q8_0 |
|---------|-------------|-----------|-------------|
| 2048 | 4694 | 4756 | 0.987x |
| 4096 | 3049 | 3084 | 0.989x |
| 8192 | 2287 | 2299 | 0.995x |
| 16384 | 1737 | 1757 | 0.989x |
| 32768 | 1211 | 1217 | 0.995x |

**Ratio holds between 0.987x and 0.995x across ALL tested depths.** No degradation trend.

### Quality After Fix

| | PPL (32-chunk) | vs q8_0 |
|---|---|---|
| q8_0 | 5.414 ± 0.140 | baseline |
| turbo3 | 5.471 ± 0.142 | +1.1% |

Quality unchanged. The dequant optimization is purely a compute efficiency improvement, no effect on output values.
