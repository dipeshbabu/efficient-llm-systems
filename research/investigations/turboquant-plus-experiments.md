# TurboQuant+ Experiments — Status and Findings

## Overview

TurboQuant v1 is complete: 4.6x KV cache compression at 99% of q8_0 speed across all context depths, with 1.1% PPL loss. These experiments explore improvements beyond the base paper.

## Experiment Status

### MERGED (in production)

| Experiment | Branch | Result |
|-----------|--------|--------|
| **Speed optimization** | `experiment/speed-optimization` | 739 → 2747 tok/s (3.72x speedup). fp16 WHT, half4 butterfly, graph-side rotation, block-32. |
| **Context scaling fix** | `experiment/context-scaling-fix` | Custom GGML_OP_TURBO_WHT + optimized dequant. Flat 99% of q8_0 through 32K context. |

### ACTIVE (promising, not yet merged)

#### Layer-Adaptive KV Cache
**Branch:** `experiment/layer-adaptive-extended-ctx`
**Finding:** The last 8 of 40 layers account for essentially ALL of turbo3's quality loss. **Verified 2K-32K context.**

| Config | 8-chunk PPL | vs q8_0 | 32-chunk PPL | vs q8_0 |
|--------|-------------|---------|--------------|---------|
| Uniform turbo3 | 6.211 | +1.6% | 5.471 | +1.0% |
| **Mode 2: q8_0 last 8** | **6.120** | **+0.14%** | **5.435** | **+0.37%** |

**Prefill:** Mode 2 matches or beats q8_0 at every context depth (0.99-1.02x). Faster than uniform turbo3 because 20% of layers use q8_0's cheaper dequant.

**Decode:** Mode 2 is 0.90-0.92x q8_0 (vs 0.86-0.90x for uniform turbo3). ~3% decode improvement from the q8_0 layers.

**Effective compression:** ~3.5x (vs 4.6x uniform, 2.0x q8_0). Trades ~25% compression for ~100% quality recovery.

**Status:** VERIFIED — ready to merge. See [extended benchmarks](experiment-layer-adaptive-extended.md) for full data.

#### Temporal Decay
**Branch:** `experiment/temporal-decay`
**Status:** IMPLEMENTED AND VALIDATED. Zero decode overhead. NIAH preserved. GitHub issue #38.

Proper 3→2 requantization for old tokens (not the broken zero-signs approach — Codex caught that).
- Trigger: `init_update()` every 64 decode steps (amortized, invisible to user)
- Method: dequant turbo3 → nearest 2-bit centroid → map to 3-bit → repack with norm correction
- Batch limit: 64 cells/cycle (eliminates CPU↔GPU transfer overhead)
- Layer-aware: layers 0 to n_layer-8 decay, last 8 preserved
- Attention sinks (pos 0-3) always exempt

**Results (38K-token PDF, Qwen3.5-35B-A3B, M5 Max):**

| Metric | No Decay | Decay (64/cycle) |
|--------|----------|-----------------|
| Prefill | 1064 tok/s | 1071 tok/s |
| Decode | 42.6 tok/s | 43.0 tok/s (±0%) |
| NIAH | 7/9 (78%) | 7/9 (78%) — same misses |

**Memory savings:** 30-34% at 32K-256K context.
**Quality:** Python prototype cos=0.940 synthetic, cos=0.949 real Qwen3-1.7B.

**Key finding:** Unbatched decay was -54% decode speed (CPU tensor transfers block GPU). Batching at 64 cells/cycle eliminates the overhead entirely.

**Next:** GPU-side Metal compute kernel (no CPU↔GPU transfers), multi-tier decay (turbo3→turbo2→1-bit).

### BLOCKED (needs engineering work)

#### Asymmetric K/V Compression
**Branch:** `experiment/asymmetric-kv`
**Idea:** K at lower precision, V at higher — V carries content, K carries direction.

**Blocker:** Flash attention kernel template only has same-type instantiations. Mixed K/V types (e.g., turbo3 K + q8_0 V) need new template instantiations. Significant work for unclear payoff since uniform turbo3 already achieves 99% speed parity.

**Shipped fix:** V un-rotation now correctly checks `v->type` instead of `k->type`, which is needed for any future asymmetric support.

### DEAD (invalidated)

#### MoE-Aware Expert Gating
**Branch:** `experiment/moe-aware-gating`
**Idea:** Track expert firing frequency, give frequently-activated experts' KV higher precision.
**Why it's dead:** Expert routing is an FFN signal. KV cache stores attention projections computed BEFORE expert routing. K and V values are identical regardless of which experts fire. The premise doesn't apply to KV cache compression.

Codex independently reached the same conclusion: "Expert routing is an FFN signal; KV cache importance is an attention signal. The premise may not map cleanly."

### NOT YET STARTED

#### Rotation-Free via Outlier Channeling (Idea E from notes)
**Status:** DEAD — tested 2026-03-26, does not work.
**Test result:** Even removing 32 of 128 channels (25%), kurtosis stays at 8-50. WHT rotation gets it to 2.9. Some layers get WORSE after removing top channels (55.6 → 156.98). Heavy tails are a structural property of attention, not concentrated in outlier channels.
**Why it fails:** WUSH paper was right — Hadamard is optimal among data-agnostic transforms. Outlier removal is model-dependent AND doesn't work. Would also need per-model calibration, killing TurboQuant's "no calibration" pitch.

#### Fused Compressed Attention (Priority 6 from notes)
**Potential:** Compute Q·K dot products directly on quantized indices without full dequant. Precompute Q·centroid table (8 values), then each K element is a table lookup instead of centroid lookup + multiply.
**Status:** Deferred. The optimized dequant closed most of the gap. This would help decode speed at very long context but is complex (custom flash attention kernel).

#### Speculative Cache Warming (Idea B from notes)
**Status:** Skip. Codex and the Obsidian notes analysis both recommend against — complexity doesn't justify marginal gains.

#### Codebook Interpolation (Idea C from notes)
**Status:** Skip. Soft assignment (storing interpolation weights) costs more bits than the quality improvement justifies. Compression theory agrees.

## Key Learnings

1. **Layer sensitivity is extremely non-uniform.** The last 20% of layers account for ~100% of turbo3's quality loss. This is the single most important finding for future work.

2. **Dequant compute, not rotation, is the real bottleneck.** The graph-side WHT rotation (whether dense matmul or custom O(d log d) op) adds <1% overhead. The per-position centroid lookup in flash attention is what scales with context.

3. **Byte-level optimization matters on GPU.** Reading the same qs/signs bytes once per 4 elements instead of per-element eliminated the context scaling regression. GPU constant memory access patterns dominate.

4. **WHT and RoPE don't commute — but it doesn't matter.** Graph-side WHT applied after RoPE (same pipeline point as KV quantize) works correctly. The earlier failure was a matrix orientation bug, not a commutativity issue.

5. **MoE expert routing is irrelevant for KV cache.** KV vectors are computed in shared attention before expert routing. Don't waste time on MoE-aware KV compression.

6. **Always run perplexity.** "Coherent text" evaluation caught nothing when PPL was 165. Speed numbers are meaningless without quality validation.

## Key Learnings (added 2026-03-26)

7. **Register LUT spills on Metal but not CUDA.** float cn[8] works on CUDA (255 regs/thread) but spills to device memory on Metal. Constant half[8] + float norm broadcast is optimal for Apple Silicon. Fundamental architecture difference.

8. **Outlier channeling doesn't work.** Even removing 32/128 channels (25%), kurtosis stays 8-50. WHT rotation gets it to 2.9. Heavy tails are structural to attention, not concentrated in outlier channels. WUSH paper confirmed.

9. **Temporal decay must be batched.** Unbatched CPU→GPU tensor transfers for 34K cells = -54% decode. Batching at 64 cells/cycle eliminates overhead entirely.

10. **Norm correction is free quality.** Store `norm / ||centroid_vector||` instead of raw norm. PPL +1.6% → +1.1% at zero cost. Ported from @spiritbuun's CUDA implementation.

## Recommended Next Steps (Priority Order)

1. ~~**Test layer-adaptive at extended contexts**~~ — ✅ DONE. Mode 2 holds 2K-32K, +0.14-0.37% PPL.
2. ~~**Rotation-free kurtosis test**~~ — ✅ DEAD. Kurtosis stays 8-50 even removing 32/128 channels.
3. ~~**Temporal decay MVP**~~ — ✅ IMPLEMENTED. Zero overhead, NIAH preserved, 30-34% savings. GitHub #38.
4. **Fused Q·centroid decode** — eliminate constant memory from FA inner loop. GitHub #39.
5. **Upstream PR preparation** — llama.cpp CONTRIBUTING.md requires perplexity, KL divergence, and CPU perf baselines

## Overall TurboQuant+ Status (2026-03-26)

**What ships today:**
- 4.6x KV cache compression at 99% prefill speed
- PPL +1.1% vs q8_0 (with norm correction)
- NIAH retrieval: 80% single needle, 100% multi-key distractors through 32K
- Layer-adaptive mode 2: q8_0 quality at 3.5x effective compression
- Temporal decay: 30-34% additional memory savings at zero decode cost
- Diagnostic tool: 2385 lines, 253 tests, 100% coverage, cross-platform
- NIAH benchmark: Kamradt + RULER methodology, industry-standard
- 511 total tests across all modules

**What doesn't work (yet):**
- M1 decode speed (0.39x at 32K — constant cache thrashing, needs fused centroid fix)
- Prefill dequant-then-attend on Metal (blocked on turbo3→f16 cast)
- Register LUT on Metal (register spill — CUDA-only optimization)
- Rotation-free outlier channeling (dead — kurtosis doesn't drop)
- CUDA port (use @spiritbuun's fork for now)
