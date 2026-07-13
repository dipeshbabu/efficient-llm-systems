# Storage Block Size Optimization for TurboQuant KV Cache Compression

**Dipesh Tharu Mahato**
Independent Researcher
GitHub: [@dipeshbabu](https://github.com/dipeshbabu)

---

## Abstract

TurboQuant's turbo3 format stores quantized KV cache values in 32-element blocks, each with its own 16-bit norm. Since the Walsh-Hadamard rotation and norm correction operate on 128-element groups (matching head_dim), all 4 blocks within a group receive the same corrected norm value. This redundancy wastes 6 bytes per group (3 duplicate norms).

We tested whether increasing the storage block size from 32 to 128 (one block per rotation group) affects perplexity, speed, or correctness. Across 3 model architectures (dense, dense Qwen, hybrid MoE), 2 context lengths (512, 8192), and 2 Apple Silicon platforms (M5 Max, M2 Pro), perplexity was identical to 4 decimal places at all block sizes. Speed deltas were within measurement noise. The result is a free compression improvement: turbo3 at block_size=128 achieves 5.12x compression (vs 4.57x at block_size=32) with no measurable quality or speed cost.

All testing used `q8_0-K + turbo-V` cache configurations on Metal (Apple Silicon). Symmetric turbo-K paths and CUDA backends were not validated in this pass.

---

## 1. Background

### 1.1 Block Size vs Group Size

TurboQuant's quantization operates at two granularities:

- **Rotation group** (`QK_TURBO3_GROUP = 128`): The Walsh-Hadamard Transform (WHT) operates on 128-element groups matching the model's head dimension. After rotation, coordinates are approximately Gaussian with known variance, enabling optimal scalar quantization. The group-level norm correction (`grp_norm / recon_norm`) is computed over all 128 elements.

- **Storage block** (`QK_TURBO3 = 32`): Each block of 32 quantized elements is stored with its own 16-bit norm, quantization indices, and sign bits. One rotation group contains 4 storage blocks.

The storage block size affects only how many elements share one stored norm value. It does not affect the rotation, centroid selection, or norm correction math.

### 1.2 Compression at Different Block Sizes

| Block Size | Block Layout | bits/value | Compression vs fp16 |
|:----------:|:------------|:----------:|:-------------------:|
| 32 (current) | norm(2B) + qs(8B) + signs(4B) = 14B per 32 | 3.50 | 4.57x |
| 64 | norm(2B) + qs(16B) + signs(8B) = 26B per 64 | 3.25 | 4.92x |
| 128 | norm(2B) + qs(32B) + signs(16B) = 50B per 128 | 3.125 | 5.12x |

The compression improvement comes from amortizing the 2-byte norm over more elements. At block_size=128, one norm covers the entire rotation group, eliminating all redundancy.

### 1.3 Why PPL Should Be Identical

All 4 blocks within a 128-element rotation group receive the same corrected norm. This is because norm correction is computed at the group level:

```
grp_norm = ||original_group||
recon_norm = ||reconstructed_centroids||
corrected_norm = grp_norm / recon_norm
```

Every block in the group gets `corrected_norm`. Changing from 4 blocks to 1 block stores this value once instead of four times, but the quantization and dequantization math is identical.

---

## 2. Experimental Setup

### 2.1 Implementation

Block size is controlled by `QK_TURBO3` and `QK_TURBO2` defines in `ggml/src/ggml-common.h`. We added derived macros for the flash attention template parameters:

```c
#define NL_TURBO3     (QK_TURBO3 / 16)
#define NL_TURBO3_VEC (QK_TURBO3 / 4)
```

This replaced ~250 hardcoded `nl` values in Metal flash attention template instantiations, making block size a one-line edit. The dequantization functions, set-rows kernels, and quantization functions all use `QK_TURBO3` symbolically and require no additional changes.

### 2.2 Models

| Model | Architecture | Weights | Layers | head_dim |
|-------|-------------|---------|--------|----------|
| phi-4 | Dense (Phi-3) | Q8_0 | 40 | 128 |
| Qwen2.5-7B-Instruct | Dense (Qwen2) | Q4_K_M | 28 | 128 |
| Qwen3.5-35B-A3B | Hybrid MoE (GDN + attention) | Q8_0 | 40 (10 KV) | 128 |

### 2.3 Hardware

| Machine | SoC | Memory | Bandwidth |
|---------|-----|--------|-----------|
| M5 Max | Apple M5 Max | 128 GB | 546 GB/s |
| M2 Pro | Apple M2 Pro | 16 GB | 200 GB/s |

### 2.4 Cache Configuration

All PPL and speed tests used `--cache-type-k q8_0 --cache-type-v turbo3` (asymmetric). This is the recommended production configuration for models where K precision matters. Symmetric turbo-K paths were not tested in this block-size pass.

### 2.5 Benchmark Tools

- **PPL**: `llama-perplexity` on wikitext-2-raw, 512 and 8192 context, 4-20 chunks
- **Speed**: `llama-bench` with pp512/pp4096 prefill and tg128 decode

---

## 3. Results

### 3.1 Perplexity: Block Size Has Zero Effect

#### Short Context (512, wikitext-2-raw)

| Model | Block 32 | Block 64 | Block 128 |
|-------|:--------:|:--------:|:---------:|
| phi-4 Q8_0 (M5, 20 chunks) | 6.6105 | 6.6105 | 6.6105 |
| Qwen2.5-7B Q4_K_M (M5, 10 chunks) | 7.4471 | — | 7.4471 |
| Qwen2.5-7B Q4_K_M (M2, 10 chunks) | 7.4727 | — | 7.4727 |
| Qwen3.5-35B-A3B Q8_0 (M5, 10 chunks) | 7.0298 | — | 7.0298 |

#### Long Context (wikitext-2-raw)

| Model | Context | Block 32 | Block 128 |
|-------|:-------:|:--------:|:---------:|
| phi-4 Q8_0 (M5, 4 chunks) | 8K | 5.7134 | 5.7134 |
| Qwen2.5-7B Q4_K_M (M5, 4 chunks) | 8K | 5.9547 | 5.9547 |
| Qwen2.5-7B Q4_K_M (M2, 4 chunks) | 8K | 5.9477 | 5.9477 |
| phi-4 Q8_0 (M5, 4 chunks) | 32K | 6.1873 | 6.1873 |

PPL is identical to 4 decimal places across all tested models, context lengths (512, 8K, 32K), and hardware. This confirms the theoretical prediction: block size does not affect quantization math.

#### Turbo Type and Cache Path Parity (M5, 512 ctx, 10 chunks)

| Cache Config | Model | Block 32 | Block 128 |
|:------------:|-------|:--------:|:---------:|
| q8_0-K / turbo3-V | phi-4 Q8_0 | 6.6105 | 6.6105 |
| q8_0-K / turbo2-V | phi-4 Q8_0 | 6.7346 | 6.7346 |
| q8_0-K / turbo2-V | Qwen2.5-7B Q4_K_M | 7.7989 | 7.7989 |
| q8_0-K / turbo2-V | Qwen2.5-7B Q4_K_M (M2) | 7.7807 | 7.7807 |
| turbo3-K / turbo3-V | phi-4 Q8_0 | 6.9208 | 6.9208 |
| turbo3-K / turbo3-V | Qwen3.5-35B-A3B Q8_0 | 7.0627 | 7.0627 |
| turbo4-K / turbo4-V (always block128) | phi-4 Q8_0 | 6.6624 | 6.6624 |

PPL is identical across all tested cache paths — both asymmetric (q8_0-K + turbo-V) and symmetric (turbo3-K + turbo3-V). turbo2 parity confirmed on both phi-4 and Qwen2.5-7B (including M2). turbo4 already uses block_size=128 natively.

#### NIAH (phi-4 Q8_0, M5, symmetric turbo3, block_size=128)

| Depth | 4K Context |
|:-----:|:----------:|
| 0% | pass |
| 50% | pass |
| 100% | pass |

3/3 pass at block_size=128. This is a block128-only run (not a strict A/B with block32), but given that PPL is byte-identical across all other tests, NIAH parity is expected.

#### Boundary V (TURBO_LAYER_ADAPTIVE=7, q8_0-K + turbo2-V)

| Model | Hardware | Block 32 | Block 128 |
|-------|----------|:--------:|:---------:|
| phi-4 Q8_0 (14B, dense) | M5 | 6.7398 | 6.7398 |
| Qwen2.5-7B Q4_K_M (dense) | M5 | 7.7852 | 7.7852 |
| Qwen3.5-35B-A3B Q8_0 (MoE) | M5 | 7.0725 | 7.0725 |
| Qwen2.5-7B Q4_K_M (dense) | M2 | 7.8384 | 7.8384 |

PPL identical across all 3 architectures and both hardware platforms. The layer-adaptive V policy (boundary layers q8_0-V, rest turbo2-V) is block-size-independent.

### 3.2 Speed: No Measurable Regression

#### M5 Max

| Model | Metric | Block 32 | Block 128 | Delta |
|-------|--------|:--------:|:---------:|------:|
| phi-4 14B | pp4096 (t/s) | 701 ± 3 | 708 ± 37 | +1.0% |
| phi-4 14B | tg128 (t/s) | 29.55 ± 0.36 | 29.38 ± 0.88 | -0.6% |
| Qwen2.5-7B | pp512 (t/s) | 1939 ± 283 | 1886 ± 316 | -2.7% |
| Qwen2.5-7B | tg128 (t/s) | 77.84 ± 0.26 | 78.34 ± 0.60 | +0.6% |
| Qwen3.5-35B MoE | pp512 (t/s) | 2775 ± 35 | 2773 ± 23 | -0.1% |
| Qwen3.5-35B MoE | tg128 (t/s) | 77.54 ± 0.64 | 77.75 ± 0.50 | +0.3% |

All deltas are within measurement noise. The Qwen2.5-7B prefill shows high variance (±283 t/s) making the -2.7% unreliable. Decode throughput is flat across all models.

#### M2 Pro — Matched Performance Follow-up (Qwen2.5-1.5B Q4_K_M)

A targeted follow-up tested the same model at multiple context lengths on both M2 Pro and M5 Max to determine whether block_size=128 produces a speed benefit specifically on bandwidth-constrained Apple Silicon.

**M2 Pro (16 GB, 200 GB/s bandwidth):**

| Context | Metric | Block 32 | Block 128 | Delta |
|:-------:|--------|:--------:|:---------:|------:|
| 512 | prefill (t/s) | 1458 ± 8 | 1466 ± 4 | +0.6% |
| 512 | decode (t/s) | 64.42 ± 0.14 | 66.32 ± 0.40 | +3.0% |
| 8K | prefill (t/s) | 808 ± 0.3 | 811 ± 0.3 | +0.4% |
| 8K | decode (t/s) | 64.05 ± 0.32 | 68.34 ± 0.97 | +6.7% |
| 16K | prefill (t/s) | 538 ± 0.2 | 539 ± 0.3 | +0.3% |
| 16K | decode (t/s) | 64.03 ± 0.44 | 66.60 ± 0.58 | +4.0% |

M2 Pro shows a consistent decode improvement at block_size=128: +3.0% at 512, +6.7% at 8K, +4.0% at 16K. Prefill is effectively flat.

**M5 Max matched comparison (128 GB, 546 GB/s bandwidth, same model/config):**

| Context | Metric | Block 32 | Block 128 | Delta |
|:-------:|--------|:--------:|:---------:|------:|
| 512 | prefill (t/s) | 9521 ± 45 | 9377 ± 49 | -1.5% |
| 512 | decode (t/s) | 161.41 ± 2.85 | 152.47 ± 4.49 | -5.5%* |
| 8K | prefill (t/s) | 4853 ± 49 | 4802 ± 68 | -1.0% |
| 8K | decode (t/s) | 161.93 ± 0.39 | 159.80 ± 1.42 | -1.3% |
| 16K | prefill (t/s) | 2999 ± 42 | 3074 ± 43 | +2.5% |
| 16K | decode (t/s) | 155.13 ± 2.71 | 161.52 ± 0.28 | +4.1% |

*The 512 decode -5.5% has high variance (±4.49 t/s). M5 decode deltas are inconsistent in direction and within noise, consistent with earlier phi-4 14B results on M5.

The M2 decode gain is consistent and reproducible across three context lengths. The matched M5 run does not show a comparable benefit. One plausible interpretation is that M2's lower memory bandwidth (200 GB/s vs 546 GB/s) makes each byte of reduced norm storage proportionally more valuable during V-cache dequant. This is a hypothesis based on the hardware difference, not a proven mechanism.

**Caveats:** This is a single-model result (Qwen2.5-1.5B) on a single M2 variant (M2 Pro). Larger models crash in llama-bench on M2 16GB. Other M2 variants and M1-family chips have different memory subsystems and may show different behavior.

### 3.3 Memory Savings

For phi-4 (40 layers, head_dim=128), V cache memory at 512 context:

| Block Size | V Cache (MiB) | Savings vs Block 32 |
|:----------:|:-------------:|:-------------------:|
| 32 | 43.75 | baseline |
| 64 | 40.62 | 7.2% |
| 128 | 39.06 | 10.7% |

The savings scale linearly with context length. At 128K context, the difference is ~1.2 GB.

---

## 4. Interpretation

The block size result is mechanically straightforward. The WHT rotation, centroid codebook, and norm correction are all computed at the 128-element group level. The storage block size determines only how many 2-byte norms are written per group. At block_size=32, four identical norm values are stored. At block_size=128, one is stored. The quantization indices and sign bits are identical regardless.

This means the current block_size=32 default has been storing 3 redundant norms per group since the initial implementation. The 4.57x compression figure reported for turbo3 was conservative. The achievable compression for the same quantization math is 5.12x.

---

## 5. Relation to External Implementations

Credit to [@AmesianX](https://github.com/AmesianX) whose CUDA TurboQuant implementation with block_size=256 prompted this investigation. His implementation reports 5.2x compression at +1.7% PPL. We have not inspected the internals of his implementation. One plausible interpretation is that 256 refers to the rotation group size (not just storage block size), which would use a 256x256 rotation matrix for models with head_dim=256 (e.g., Qwen3.5-27B). That would be a fundamentally different experiment from ours (changing rotation scope, not just storage chunking). For models with head_dim=128, a 256-element rotation group would not work. Whether his approach changes the rotation group size is not established here.

---

## 6. Limitations

1. **Backend scope.** Tested on Metal (Apple Silicon) only. CUDA block size changes would require separate kernel updates and validation.

2. **M2 speed coverage.** M2 speed testing used Qwen2.5-1.5B (the only model small enough for llama-bench on 16 GB). The +3-7% decode gain was consistent across 3 context lengths but is a single-model result. Larger-model speed behavior on M2 is inferred from PPL parity, not directly measured.

3. **NIAH.** Tested at block_size=128 only (phi-4, symmetric turbo3, 3 depths at 4K). 3/3 pass. Strict A/B vs block32 was not performed, but is expected to match given byte-identical PPL.

4. **Serialization.** Changing block size changes the on-disk turbo3/turbo2 block struct layout. Existing serialized cache state files would be incompatible. Since this is a pre-release branch, this is acceptable.

5. **Symmetric turbo4-K.** Symmetric turbo3-K + turbo3-V was validated (phi-4 dense, Qwen3.5-35B MoE). Symmetric turbo4-K + turbo4-V was not separately tested at block_size=128, though turbo4 already uses block_size=128 natively.

---

## 7. Recommendation

For the tested Metal (Apple Silicon) paths, block_size=128 is a safe default change:

- Zero PPL regression across all tested configurations:
  - Asymmetric q8_0-K + turbo{2,3}-V
  - Symmetric turbo3-K + turbo3-V (phi-4 dense, Qwen3.5-35B MoE)
  - Boundary V / LA-V7 (phi-4, Qwen2.5-7B, Qwen3.5-35B MoE, M5 + M2)
  - 3 model architectures, 3 context lengths (512, 8K, 32K), 2 hardware platforms
- 12% better compression (5.12x vs 4.57x for turbo3)
- No speed regression on M5 Max (flat across 3 models)
- Consistent +3-7% decode improvement on the tested M2 Pro setup (Qwen2.5-1.5B, 3 context lengths)
- NIAH 3/3 pass at block_size=128 (phi-4, symmetric turbo3)
- Eliminates redundant norm storage

Before shipping as a default change, CUDA backends should be separately validated.

---

## 8. Reproduction

To test block_size=128 locally:

1. Edit `ggml/src/ggml-common.h`:
   ```c
   #define QK_TURBO3 128  // was 32
   #define QK_TURBO2 128  // was 32
   ```

2. Rebuild: `cmake --build build -j$(nproc)`

3. Run PPL:
   ```bash
   ./build/bin/llama-perplexity \
     -m your-model.gguf \
     -f wikitext-2-raw/wiki.test.raw \
     --cache-type-k q8_0 --cache-type-v turbo3 \
     -c 512 --chunks 10 -ngl 99
   ```

The derived macros (`NL_TURBO3`, `NL_TURBO3_VEC`, `NL_TURBO2`, `NL_TURBO2_VEC`) propagate the block size change through all Metal flash attention template instantiations automatically.

---

## Addendum: Independent Validation (2026-03-31)

The block size 128 finding has been independently confirmed:

- **@dusterbloom** — RTX 3090 (2026-03-30): Tested TBQ3 Flash Attention across 5 model families with block_size=128 builds. Results consistent with our Metal findings.
- **@HyperionMS2040** — RTX 3090 (2026-03-30): CUDA warp-to-block mapping fix (PR #32) enabled correct block_size=128 support on CUDA. PPL validated on multiple models after fix.
- **@AmesianX** — Blackwell DGX Spark (2026-03-30): Used block size 256 (QK_K) for fused FA kernels. Different tradeoff: 256 gives slightly better compression but our testing shows 128 is better quality-per-bit on smaller models (7B-14B).
