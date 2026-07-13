# SMEM Pre-Dequant Experiment — NEGATIVE RESULT

## TL;DR

Pre-dequantizing K/V tiles into threadgroup memory (SMEM) before the FA dot product loop is **2x slower** than the baseline at 8K context on M2 Pro. The threadgroup store/load overhead exceeds any benefit from avoiding constant cache stalls.

## What Was Changed

Branch: `experiment/smem-pre-dequant`

Added a new code path in the FA vec kernel gated behind `TURBO_USE_SMEM_DEQUANT`:

1. **Pre-dequant phase (K)**: Before the Q*K^T block, all 32 threads cooperatively dequant C=32 cache positions' K vectors into a threadgroup `half4` buffer. Thread `tiisg` dequants cache position `tiisg` (all DK4=32 float4s).

2. **Compute phase (K)**: The dot product reads from threadgroup memory instead of calling `deq_k_t4`.

3. **Pre-dequant phase (V)**: Before the O accumulation block, same pattern for V data. Reuses the same SMEM buffer.

4. **Compute phase (V)**: Weighted accumulation reads from threadgroup memory instead of calling `deq_v_t4`.

Threadgroup memory budget: C * max(DK, DV) * sizeof(half) = 32 * 128 * 2 = **8,192 bytes** extra. Total SMEM ~9.5KB (well within 32KB limit).

### Files Modified

- `ggml/src/ggml-metal/ggml-metal.metal` — SMEM buffer declaration + pre-dequant + compute paths
- `ggml/src/ggml-metal/ggml-metal-device.m` — `TURBO_SMEM_DEQUANT=1` env var → compile flag
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` — increased SMEM allocation when enabled

## Results (M2 Pro, Qwen2.5-7B-Instruct-Q4_K_M)

| Test | Baseline (4-mag LUT) | SMEM Pre-Dequant | Delta |
|------|---------------------|------------------|-------|
| Short decode | 25.93 ± 0.09 | 26.39 ± 0.11 | +1.8% |
| 8K decode | 20.99 ± 4.87 | 10.17 ± 0.35 | **-51.5%** |

## Why It Failed

### The kernel's parallelism pattern doesn't benefit from SMEM caching

In the FA vec kernel with turbo3 dk128 (NE=1):
- 32 threads, each handles a different part of the DK=128 K vector
- Each thread dequants only DK4/NL = 1 float4 per cache position
- The dequanted value is used EXACTLY ONCE by the same thread

Adding SMEM means each dequanted value is:
1. Written to threadgroup memory (extra store)
2. Synchronized via barrier (pipeline stall)
3. Read back from threadgroup memory (extra load)

For data that's only used once by its producer, this is pure overhead.

### Total dequant calls unchanged

| | Per thread per outer iteration | Total |
|---|---|---|
| Baseline | 32 dequants (interleaved with 32 dots) | Same |
| SMEM | 32 dequants (bunched), then 32 dots | Same |

The SMEM approach adds 64 threadgroup memory ops (32 stores + 32 loads) plus a barrier, for zero reduction in constant LUT reads.

### ILP destruction — THE ROOT CAUSE of the 2x slowdown

The baseline interleaves dequant + dot product, enabling instruction-level parallelism:
```
deq_k → dot(k, q) → deq_k → dot(k, q) → ...
         ^-- GPU overlaps this with next constant read
```

The SMEM approach forces sequential phases:
```
deq_k → deq_k → ... → BARRIER → dot(k,q) → dot(k,q) → ...
^-- all constant reads       ^-- all ALU
```

**Quantitatively**: if constant read latency ≈ dot product ALU latency (call both X cycles):
- Interleaved: 32 × max(X, X) = 32X per outer iteration
- Separated: 32X + 32X = 64X per outer iteration (2x slower)

This perfectly explains the 2x slowdown. The per-phase SMEM overhead (64 writes + barrier + 64 reads) is negligible (~300 extra cycles per iteration). The ILP loss is ~32X extra cycles per iteration, where X ≈ 5-10 cycles = 160-320 extra cycles per iteration per phase. Over 2 phases: 320-640 extra cycles × 256 iterations = 82K-164K extra cycles total.

**Key insight**: The 4-mag LUT works BECAUSE constant reads interleave with ALU. The per-element `* norm` multiply provides ALU work that overlaps with the next constant read. Any approach that separates the memory and compute phases loses this overlap.

This also explains why "deferred norm multiply" (approach #5, 12.9 tok/s) was slower — deferring norm to the end reduced ILP in the same way.

### Short context was neutral (not beneficial)

At short context, the outer loop runs ~1 iteration, so the overhead is negligible relative to model weight loading. The +1.8% is noise.

## Predicted Occupancy vs Reality

SMEM usage went from ~1.5KB to ~9.5KB (well within 32KB). Occupancy was NOT the bottleneck — the threadgroup memory access pattern was.

## How to Build and Test

```bash
# Build (Mac Mini M2 Pro)
cd ~/dev/turbo_test/llama-cpp-turbo
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Baseline (no SMEM)
./build/bin/llama-bench -m ~/dev/turbo_test/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -ngl 99 -fa 1 -ctk turbo3 -ctv turbo3 -t 1 -p 8192 -n 128

# SMEM enabled
TURBO_SMEM_DEQUANT=1 ./build/bin/llama-bench -m ~/dev/turbo_test/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -ngl 99 -fa 1 -ctk turbo3 -ctv turbo3 -t 1 -p 8192 -n 128
```

## Lessons Learned

1. **SMEM only helps when data is shared between threads.** The FA vec kernel's parallelism distributes work so each thread operates on unique data — caching in SMEM adds overhead without benefit.

2. **Don't separate what the hardware pipelines together.** The constant LUT read interleaved with ALU provides ILP that the GPU exploits. Batching all reads first destroys this.

3. **The 4-mag LUT IS the dequant-level ceiling.** After 15 approaches tested (14 from decode-speed-hardware-analysis.md + this one), the conclusion is firm: the remaining 38% gap requires kernel-structural changes, not just dequant changes.

## What Would Actually Work (Hypotheses)

### 1. Device-memory centroid×norm table (MOST PROMISING)

**Idea**: Add 8 pre-computed `centroid[i] × norm` values (fp16) per 128-element rotation group, stored alongside the block data in device memory.

**Current dequant**: `index → constant_LUT[index] × norm` (divergent constant read + ALU)
**Proposed dequant**: `index → device_table[group_offset + index]` (sequential device read)

**Cost**: 16 extra bytes per 128 elements (8 centroids × fp16). Block size goes from 56 bytes → 72 bytes per 128 elements. Bits/value: 3.5 → 4.5 (still 1.78× compression vs q8_0).

**Why it should work**: Device memory reads are what the GPU is optimized for. The 16-byte table is contiguous and fits in a single cache line. Eliminates ALL constant memory from dequant. The sign can be applied via ALU (1 multiply), which the GPU pipelines efficiently.

**Format change required**: Yes — on-disk block format changes. Needs quantize-side update + all dequant paths. Breaking change to existing turbo3 GGUF files.

### 2. Sparse K attention (K-side SPARSE_V equivalent)

**Idea**: For K dot products, skip positions with zero or near-zero attention contribution. Currently SPARSE_V skips V dequant for negligible weights, but K dequant always runs for all C positions.

**Challenge**: K scores aren't known until after Q*K^T — you need the K dequant to compute the score. Would need a cheap approximation (e.g., using just the norm to estimate whether a position's score will be significant).

### 3. Fused Q·centroid precompute

**Idea**: Before the K loop, precompute Q·centroid for all 8 (or 4) centroids: `q_dot_c[i] = dot(Q_chunk, centroid[i])`. Then each K element contributes `q_dot_c[index] × norm × sign` — a per-element table index into a LOCAL (register) table of 4-8 values.

**Why it might work**: The q_dot_c table is tiny (4-8 floats), computed ONCE per Q chunk, and reused across all C cache positions. The per-element dequant becomes: read 3-bit index → index into register table → multiply by norm and sign. Zero constant memory reads in the inner loop.

**Challenge**: The Q chunk varies across threads (each handles different K dimensions). Need per-thread precomputation. With DK4/NL=1 (128 heads), only 4 Q elements per thread — so 4 or 8 q_dot_c values fit in registers.

## Also Implemented: Q·Centroid Precompute (#16)

Branch: same (`experiment/smem-pre-dequant`)
Env var: `TURBO_QC_PRECOMPUTE=1`

Precomputes `mag[c] * Q[j]` for c=0..3, j=0..3 (4 float4 named variables, NOT arrays to avoid register spill). Done ONCE before the cache position loop. Inner loop uses `select()` (2-level, 2 selects per element) to index into the precomputed values.

**Constant reads**: 4 total (vs 4 × C = 128 in baseline). O(1) vs O(C).
**Branches**: 8 selects per float4 (2 per element × 4 elements). On Apple8, likely worse than 4 constant reads.

**Result**: 10.10 tok/s at 8K = **-33% vs 4-mag baseline** (15.1). Even worse than predicted.
The 8 select() calls per float4 are catastrophically expensive on Apple8 — each likely compiles to conditional moves or branches. Total per float4: ~40-60 ALU/branch operations vs 4 constant reads in the baseline.

Note: QC precompute still has the ILP problem — the precomputation happens BEFORE the cc loop, so the constant reads (for mag values) don't overlap with the inner loop ALU.

## Approach Count

This is approach **#15** (SMEM) and **#16** (QC precompute) in the M2 decode speed experiments. The 4-mag LUT at 15.1 tok/s (62% of ceiling) remains the best result.
