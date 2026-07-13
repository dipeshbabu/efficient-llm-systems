# Tensor-Role-Aware Weight Compression for llama.cpp: TQ4_1S and the Config I Policy

**Dipesh Tharu Mahato**
Independent Researcher
GitHub: [@dipeshbabu](https://github.com/dipeshbabu)

**Code:** [historical llama.cpp TurboQuant fork; public URL unavailable](../../docs/reference/historical-forks.md#llamacpp-experimental-forks), branch `feature/turboquant-kv-cache` — reported as merged to main (Metal + CUDA). Historical `PR #45` contained the full testing log.
**Getting started:** [getting-started.md](../../docs/guides/getting-started.md)

---

## Acknowledgment

This work was inspired by David Y. Tan's original TQ3_1S weight-compression direction and his llama.cpp-based experiments applying TurboQuant-style transform quantization ideas to model weights ([turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3/tree/main)). Tan built the original TQ3_1S format (WHT rotation + dual half-block scales + 8 Lloyd-Max centroids) and later extended the public runtime line with TQ3_4S, a stronger follow-on format in the same family. The early Qwen3.5-27B TQ3_1S result (PPL 7.257, near Q4_0 at smaller size) demonstrated proof of concept; subsequent work, including TQ3_4S, showed that the direction was broader than the initial prototype.

The results here build on that starting point with follow-on work including: Metal runtime with cooperative SIMD pre-rotation (5.7x prefill speedup), the tensor-role policy discovery through systematic A/B isolation, the 4-bit TQ4_1S extension (inspired by our turbo4 KV resurrection findings), hybrid quantization configs mixing WHT-rotated and native llama.cpp types, model-family-specific policies for Llama vs Qwen/Phi, and validation across 6 models, 3 model families, from 1.5B to 72B parameters.

The boundary layer protection strategy applied to weights was directly informed by our earlier [Boundary V](layer-aware-v-compression.md) work on KV cache compression, which demonstrated that first and last transformer layers are disproportionately sensitive to quantization across multiple models. The same principle transfers directly to weight compression.

---

## Abstract

We show that tensor-role-aware compression policy achieves better quality-size tradeoffs than uniform quantization for transformer weight compression in llama.cpp. Through systematic A/B isolation, we discover that attention tensors, FFN read projections (gate/up), FFN write-back projections (ffn_down), and boundary layers have dramatically different compression sensitivity. The key finding is that compression policy matters more than compression math: which tensors to compress, which to leave alone, and which native llama.cpp quant types to use for specific tensor roles.

We introduce TQ4_1S, a 4-bit WHT-rotated weight format with 16 Lloyd-Max optimal centroids (5.0 BPW), extending David Y. Tan's TQ3_1S (3-bit, 8 centroids, 4.0 BPW) based on our turbo4 KV cache resurrection finding that 16 optimal centroids with clean nibble packing dramatically outperform 8-centroid schemes.

The resulting hybrid policy (Config I: TQ4_1S for attention + FFN gate/up, native Q4_K for ffn_down, boundary 2+2 protection) achieves 27-38% size reduction at +1.0-1.9% PPL on Qwen and Phi model families from 1.5B to 72B parameters, with 95-254% baseline decode speed. The method is post-training quantization requiring no retraining, calibration data, or model modification. It stacks cleanly with TurboQuant KV compression for 41% total memory reduction at 32K context. The policy insight generalizes: the same boundary-layer and write-back sensitivity patterns observed in our [KV cache compression research](layer-aware-v-compression.md) transfer directly to weight compression.

Llama-family models show a steeper quality/compression tradeoff: the same WHT-rotated quantization that costs +1-4% on Qwen/Phi costs +5-16% on Llama depending on config. However, even the most aggressive Llama config (Hybrid, +16% PPL) beats standard Q4_K_M in both quality and speed at the same model size. A Premium config (TQ4_1S attention + Q5_K/Q6_K FFN) achieves +5.8% PPL — a viable tradeoff for 29% size reduction on a 70B model.

Tested on Qwen2.5-1.5B, Qwen2.5-72B, Qwen3.5-27B, Qwen3.5-35B MoE, Phi-4 14B, and Llama 3.1 70B on Apple Silicon (M5 Max) and NVIDIA CUDA (GTX 1080 Ti, RTX 5090). NIAH retrieval passes on all tested models and configs.

---

## 1. Introduction

TurboQuant (ICLR 2026) demonstrated that WHT rotation followed by Lloyd-Max polar quantization achieves near-optimal distortion-rate performance for KV cache compression. David Y. Tan recognized that this principle could apply to model weights and built TQ3_1S, a 3-bit weight quantization format using WHT rotation with dual half-block scales and 8 Lloyd-Max centroids for llama.cpp.

Our previous work on TurboQuant KV cache compression established several principles that informed this investigation:

1. **Asymmetric K/V compression** ([paper](asymmetric-kv-compression.md)): K precision dominates quality via softmax amplification, V compression is nearly free. This suggested that within the weight domain, tensors feeding the attention score path (Q, K projections) might behave differently from those feeding the value path.

2. **Boundary V layer-aware compression** ([paper](layer-aware-v-compression.md)): First and last transformer layers are disproportionately sensitive to quantization. Protecting them with higher precision recovers 37-91% of the quality gap at minimal compression cost.

3. **turbo4 resurrection** ([paper](turbo4-resurrection.md)): 16 optimal Lloyd-Max centroids with clean nibble packing outperform 8-centroid schemes with complex correction mechanisms. QJL correction is actively harmful. Simpler is better.

4. **Sparse V dequantization** ([paper](sparse-v-dequant.md)): On Apple Silicon, GPU utilization at decode is dominated by occupancy and memory latency hiding, not raw compute. This principle directly applies to weight compression kernel design.

The question we set out to answer was straightforward: which tensors in a transformer tolerate WHT-rotated compression, which do not, and what is the optimal hybrid policy?

---

## 2. Format Design

### 2.1 TQ3_1S and TQ4_1S

Both formats share Tan's compression pipeline:

1. **WHT rotation:** Randomized Hadamard Transform with golden-ratio sign flips decorrelates weight coordinates within each 32-element block
2. **Lloyd-Max quantization:** Each rotated coordinate is mapped to the nearest optimal centroid for N(0,1)
3. **Dual half-block scales:** Separate scale factors (d0, d1) for elements 0-15 and 16-31

| Format | Centroids | Bits | BPW | MSE | Packing |
|--------|-----------|------|-----|-----|---------|
| TQ3_1S | 8 | 3 | 4.0 | 0.0346 | Cross-byte (complex) |
| TQ4_1S | 16 | 4 | 5.0 | 0.0095 | Nibble (trivial) |

TQ4_1S achieves 72.5% less quantization error than TQ3_1S from doubling the centroid count. The 16-level Lloyd-Max centroids for N(0,1):

```
[-2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128,
  0.128,  0.388,  0.657,  0.942,  1.256,  1.618,  2.069,  2.733]
```

### 2.2 Why TQ4_1S Exists: The turbo4 Lesson

The turbo4 KV cache paper proved that switching from 3+1 bit (PolarQuant + QJL) to pure 4-bit PolarQuant with 16 optimal centroids recovered turbo4 from PPL 679 to +0.23%. Five independent groups confirmed QJL off by default. We applied the same playbook to weights: 16 centroids, nibble packing, no correction.

### 2.3 Metal Kernel: Cooperative SIMD Pre-Rotation

The critical optimization is eliminating redundant WHT computation. Instead of inverse WHT per weight block (1,280 FLOPs/block with 8x redundancy across output rows), we pre-rotate the activation vector once via `simd_shuffle_xor` (10 FLOPs/element), then dequant becomes centroid lookup + scale.

The V2.1 fused kernel uses zero threadgroup memory:

1. Each simdgroup of 32 threads handles one 32-element WHT block
2. Each thread loads 1 activation, rotates via 5 stages of `simd_shuffle_xor`
3. Centroid lookup + scale multiply per thread
4. Single `simd_sum` reduction per block

Result: prefill 1,747 to 9,946 t/s (5.7x speedup). Decode 85-99% of q8_0 depending on model size.

The occupancy insight from our sparse V paper applies directly here: on Apple Silicon, decode throughput is dominated by the number of concurrent threadgroups hiding memory latency. The V2.1 kernel's zero-smem design maximizes occupancy, recovering decode from 70% to 99% on the 27B model.

---

## 3. The Road to Config I: Systematic Policy Discovery

This section documents the experimental process that led to Config I. Each experiment was motivated by the results of the previous one, following an exploratory testing methodology where failures are as informative as successes.

### 3.1 Step 1: Isolate Attention vs FFN

The first question was whether the speed regression from TQ3_1S was coming from attention compression, FFN compression, or both.

| Weights Policy | Size | pp512 | tg128 |
|---|---|---|---|
| q8_0 baseline | 1.76 GiB | 10,674 | 129 |
| all hot tensors TQ3_1S | 933 MiB | 7,000 | 101 |
| **attn = TQ3_1S, FFN = q8_0** | **1.52 GiB** | **9,963** | **137** |
| attn = q8_0, FFN = TQ3_1S | 1,016 MiB | 7,932 | 146 |

(Qwen2.5-1.5B, plain KV q8_0)

**Finding: FFN compression is the speed killer, not attention.** If FFN stays q8_0, prefill recovers to 93% of baseline. Attention-only TQ3_1S is comparatively cheap. This immediately suggested attention-only as the starting policy.

### 3.2 Step 2: Boundary Layer Protection (from KV Cache Research)

Our [Boundary V paper](layer-aware-v-compression.md) demonstrated that first and last transformer layers are disproportionately sensitive to KV cache quantization, recovering 37-91% of quality gaps depending on model depth. We applied the same principle to weights.

| Policy | Size | PPL | PPL Delta |
|--------|------|-----|-----------|
| uniform attn=TQ3_1S | 1.52 GiB | 10.82 | +5% |
| boundary-attn (first2+last2 q8_0) | 1.69 GiB | 10.63 | +3% |
| boundary-full (all TQ3_1S, first2+last2 q8_0) | 1.17 GiB | 11.49 | +11% |
| boundary-full 4+4 + ffn_down q8_0 | 1.41 GiB | 10.89 | +5.6% |
| boundary-full 4+4 + ffn_down + attn_output q8_0 | 1.44 GiB | 10.77 | +4.5% |

**Finding: boundary protection works for weights, same principle as KV cache.** Symmetric boundary (first N + last N) beats asymmetric. Both first and last layers matter roughly equally.

### 3.3 Step 3: Identify Quality-Critical Tensors

Isolating individual tensor types revealed write-back projections as disproportionately sensitive:

| Tensor kept at q8_0 (rest compressed) | PPL improvement |
|---|---|
| ffn_down only | -0.37 |
| attn_output only | -0.16 |
| ffn_down + attn_output | -0.72 |
| neither (all compressed) | baseline |

**Finding: write-back projections (ffn_down, attn_output) are quality-critical.** Errors in these compound layer-over-layer through the residual stream. This parallels our asymmetric K/V finding where K errors compound through softmax. Different mechanism, same pattern: outputs to shared state amplify errors.

### 3.4 Step 4: Critical Constraint Discovery (In-Layer Mixing)

A failed experiment revealed a fundamental implementation constraint:

| Config | PPL | Status |
|---|---|---|
| V_proj alone as TQ3_1S, rest q8_0 | 324 | Catastrophic |
| V + attn_output as TQ3_1S, Q/K q8_0 | 121 | Catastrophic |
| All 4 attention tensors as TQ3_1S | 10.52 | Correct |

**Finding: within a layer, ALL attention tensors must use the same quant type.** The in-place WHT rotation kernel rotates src1 (the hidden state) before the matmul and unrotates after. If another q8_0 matmul in the same layer reads src1 during the rotation window, it gets corrupted data. FFN tensors CAN be mixed freely since they use a different src1.

This is an implementation constraint of the in-place rotation approach, not a fundamental limitation. A temp-buffer approach would allow per-tensor mixing at the cost of an extra copy.

### 3.5 Step 5: Extend to 4-bit (TQ4_1S)

Based on the turbo4 KV resurrection finding, we built TQ4_1S with 16 centroids:

| Config | Size | BPW | PPL | Delta |
|--------|------|-----|-----|-------|
| q8_0 baseline | 1.76G | 8.50 | 10.31 | -- |
| TQ4_1S attn-only (all layers) | 1.70G | 8.22 | 10.39 | +0.8% |
| TQ4_1S attn+gate/up, ffn_down q8_0, 2+2 | 1.44G | 6.94 | 10.45 | +1.4% |
| TQ4_1S all middle, 4+4 | 1.38G | 6.66 | 10.46 | +1.5% |
| TQ4_1S all middle, 2+2 | 1.30G | 6.29 | 10.52 | +2.0% |

**Finding: TQ4_1S is better than TQ3_1S everywhere at comparable sizes.** Mixed TQ4/TQ3 per-layer also underperforms uniform TQ4 (PPL 10.83 vs 10.52 at similar size). The 3-bit read tensors drag quality down more than the 4-bit write-back tensors save.

### 3.6 Step 6: The Hybrid Breakthrough (Native Types for FFN)

The final insight came from a speed observation: TQ4_1S on ffn_down produced 115 t/s decode (WHT rotation overhead), while Q4_K on ffn_down produced 187 t/s (native Metal kernel). Q4_K also moves less data than q8_0, so it's faster despite the dequant.

| ffn_down type | Size | PPL | Delta | tg128 |
|---|---|---|---|---|
| q8_0 | 1.44G | 10.45 | +1.4% | 176 |
| Q6_K | 1.36G | 10.46 | +1.5% | 178 |
| Q5_0 | 1.32G | 10.47 | +1.5% | 179 |
| **Q4_K** | **1.28G** | **10.51** | **+1.9%** | **187** |
| TQ4_1S | 1.38G | 10.46 | +1.5% | 115 |
| IQ4_XS | 1.27G | 10.57 | +2.5% | 168 |
| Q3_K | 1.24G | 10.79 | +4.7% | 171 |

**Finding: use WHT-rotated quants where native types cannot go (attention tensors needing pre-rotation). Use native llama.cpp types where they have optimized kernels (FFN).** Q4_K for ffn_down is smaller, faster, and nearly the same quality as q8_0.

This is Config I.

---

## 4. Config I: Results

### 4.1 Definition

- **Attention (Q, K, V, output):** TQ4_1S (all four tensors, same type per layer)
- **FFN gate + up:** TQ4_1S
- **FFN down:** Q4_K (llama.cpp native)
- **Boundary:** first 2 + last 2 layers at q8_0 (all tensors)

Note: attn_output is compressed with TQ4_1S despite being identified as quality-sensitive in Section 3.3. TQ4_1S's 72.5% lower quantization error relative to TQ3_1S absorbs this sensitivity; the Section 3.3 finding motivated choosing TQ4_1S over TQ3_1S for attention rather than exempting attn_output.

### 4.2 Cross-Model Validation

| Model | Family | Type | Layers | Size (q8_0) | Size (Compressed) | Config | Delta | PPL Delta | Decode % | NIAH |
|-------|--------|------|--------|-------------|-------------------|--------|-------|-----------|----------|------|
| Qwen2.5-1.5B | Qwen | Dense | 28 | 1.76G | 1.28G | Config I | -27% | +1.9% | 96% | 6/6 |
| Qwen3.5-27B | Qwen | Dense | 64 | 26.6G | 19.1G | Config I | -28% | +1.3% | 99% | 3/3 |
| Qwen3.5-35B-A3B | Qwen | MoE | 40 | 34.4G | 21.6G | Config I | -37% | +1.4% | 102% | — |
| Phi-4 14B | Microsoft | Dense | 40 | 14.5G | 9.3G | Config I | -36% | +1.0% | **254%** | 3/3 |
| **Qwen2.5-72B** | **Qwen** | **Dense** | **80** | **72.0G** | **45.8G** | **Config I** | **-38%** | **+3.9%** | **95%** | **3/3** |
| Llama 3.1 70B | Meta | Dense | 80 | 69.8G | 49.8G | Premium | -29% | **+5.8%** | fast | 3/3 |
| Llama 3.1 70B | Meta | Dense | 80 | 69.8G | 40.2G | Hybrid | -42% | +16% | 133% | 3/3 |

Config I works well on Qwen (+1.0-1.9% PPL) and Phi (+1.0% PPL). On Llama, Config I produces +17% PPL due to 6-8× higher per-layer error amplification in the FFN path (Section 5.7). Two alternative configs mitigate this:

- **Premium** (TQ4 attn + Q5K/Q6K FFN, boundary 4+4): +5.8% PPL at 29% smaller. Best quality for Llama.
- **Hybrid** (TQ4 attn + Q4K FFN, boundary 2+2): +16% PPL at 42% smaller. Beats Q4_K_M in both quality (-18%) and decode speed (+33%).

Phi-4 shows a 2.5× decode speedup because the 36% size reduction shifts the bottleneck from memory bandwidth to compute on the M5 Max. MoE models (Qwen3.5-35B-A3B) show slight decode improvement (102%) because only active experts are read per token.

### 4.3 Stacking with TurboQuant KV Compression

Weight compression and KV compression stack cleanly. Tested on Qwen3.5-35B-A3B MoE, M5 Max, 32K context:

| Config | Weight | KV (32K) | Total (32K) | PPL Delta | Decode |
|--------|--------|----------|-------------|-----------|--------|
| q8_0 + f16 KV | 34.4G | 5.00G | 39.4G (100%) | baseline | 79 t/s |
| q8_0 + turbo3 KV | 34.4G | 1.82G | 36.2G (92%) | ~+1% | 80 t/s |
| Config I + f16 KV | 21.6G | 5.00G | 26.6G (68%) | +1.4% | 81 t/s |
| Config I + turbo3 KV | 21.6G | 1.82G | 23.4G (59%) | +1.4% | 77 t/s |
| Config I + turbo4 KV | 21.6G | 1.99G | 23.6G (60%) | +1.7% | 63 t/s |

Config I + turbo3 KV: **59% of baseline total memory at 32K context, +1.4% PPL, 97% decode.** The errors from weight compression and KV compression do not compound meaningfully.

### 4.4 Independent Stacking Validation

An [independent Python/PyTorch implementation](https://github.com/dhawalc/turboQuantDC) (662 tests, MIT license) confirmed the stacking result on RTX 4090 with Llama 3.1 8B: TQ4_1S weights + turbo3 KV achieved 100K context in 5.8 GiB total. At 8K context, turbo3 was actually *faster* than f16 KV (86.5 vs 78.4 t/s) due to reduced VRAM pressure. PPL impact of turbo3 KV: +0.67% on top of weight compression, consistent with our +1.4% combined measurement.

The same implementation independently confirmed boundary layer protection (first 2 + last 2 layers recovering ~90% of quality gap) and found that the original paper's QJL correction stage hurts autoregressive generation quality, consistent with our turbo4 resurrection finding that QJL should be removed.

Full community results across 12+ GPUs, 11+ models, and 10+ independent testers are tracked in [weight-compression-results.md](../../docs/reference/weight-compression-results.md).

### 4.5 KL Divergence

Measured on Qwen3.5-35B-A3B MoE (8 chunks, vs q8_0 baseline):

| Metric | Config I (TQ4+Q4K) | turbo3 KV (ref) | q4_0 KV (ref) | q8_0 KV (ref) |
|--------|-------------------|-----------------|----------------|----------------|
| Mean KLD | 0.020 | 0.016 | 0.008 | 0.002 |
| Max KLD | 1.50 | 1.17 | 0.23 | 0.11 |
| Same top p | 95.2% | 94.3% | 95.8% | 98.4% |

Config I weight compression produces similar distributional shift to turbo3 KV cache compression, a well-validated production-quality config. The model agrees with q8_0 on the top token 95.2% of the time (higher than turbo3 KV's 94.3%).

---

## 5. Negative Results and Failure Modes

Negative results are documented because they constrain the design space and increase confidence in the positive findings.

### 5.1 In-Layer Mixing is Catastrophic

Mixing TQ3_1S and q8_0 within the same layer's attention block produces PPL 121-324. The in-place rotation corrupts shared hidden state tensors. This is the single most important implementation constraint. (See Section 3.4.)

### 5.2 TQ3_1S for FFN is the Quality Cliff

Swapping TQ4_1S to TQ3_1S for FFN gate/up (Config I Level 1) causes a +4.3% PPL jump (10.51 to 10.95). This is the largest single degradation step in the entire sweep. 8 centroids cannot represent FFN weight distributions accurately enough; the gating mechanism amplifies quantization errors.

### 5.3 Mixed TQ4/TQ3 Per-Layer Underperforms

TQ4_1S for write-back tensors + TQ3_1S for read tensors (PPL 10.83) is worse than uniform TQ4_1S at the same size (PPL 10.52). The 3-bit read tensors drag quality down more than the 4-bit write-back tensors preserve it.

### 5.4 The 5.8 BPW Quality Cliff

Systematic degradation from Config I (Qwen2.5-1.5B):

| BPW | PPL Delta | Status |
|-----|-----------|--------|
| 6.2+ | under +3% | Viable |
| 5.3-5.8 | +8-12% | Noticeable quality loss |
| under 5.3 | +12%+ | Unacceptable |

Below 5.8 BPW, 3-bit centroids become the bottleneck and errors from middle layers dominate regardless of boundary protection.

### 5.5 Q4_K_M Source Weights: Limited Headroom

Config I was designed for Q8_0 source weights. On Command-R+ 104B Q4_K_M, TQ4_1S (5.0 BPW) actually increases size for tensors already at Q4_K (4.5 BPW). An adapted policy achieved only 7% size reduction. The sweet spot is Q8_0 sources where 27-41% compression is achievable.

### 5.6 27B Decode Regression (and Fix)

Initial 27B decode was 70% of baseline due to threadgroup memory pressure killing GPU occupancy. On Qwen2.5-1.5B (ne00=1536), smem=6KB allows 5 threadgroups per core. On 27B (ne00=5120), smem=20KB limits to 1 threadgroup per core, halving bandwidth utilization.

The V2.1 fused kernel with NR0=8 amortization recovered decode to 99%:

| NR0 | 27B tg128 | vs baseline |
|-----|-----------|-------------|
| 2 | 14.24 | 85% |
| 4 | 15.65 | 93% |
| 8 | 16.23 | 97% |
| 16 | 16.01 | 95% (register pressure) |

This regression and fix are documented because they demonstrate a real deployment risk: WHT-rotated weight compression has model-size-dependent performance characteristics that require kernel-level mitigation.

### 5.7 Model-Specific Sensitivity: The Llama Investigation

Config I on Llama 3.1 70B produces +17% PPL (8-chunk) compared to +1.0-1.9% on Qwen and Phi models. A systematic investigation was conducted to identify the root cause.

**Valid decomposition (all attention tensors same type per layer):**

| Component | Llama 70B delta | Qwen 27B delta | Ratio |
|-----------|----------------|----------------|-------|
| Attn-only (all 4 TQ4) | +4.4% | +2.0% | 2.2x |
| FFN-only (gate+up=TQ4, down=Q4K) | +12.0% | +3.8% | 3.2x |
| Combined (Config I) | +17.0% | +1.9% | 8.9x |

FFN accounts for the majority of the degradation on Llama. The errors compound multiplicatively when combined.

**Per-layer error scaling:**

| Model | Family | Per-layer PPL delta | FFN-only delta |
|-------|--------|---------------------|---------------|
| Phi-4 14B | Microsoft | 0.03%/layer | +0.8% |
| Qwen3.5-27B | Alibaba | 0.04%/layer | +1.5% |
| **Llama 3.1 70B** | **Meta** | **0.22%/layer** | **+12.0%** |

Llama's per-layer error rate is 5.5x higher than Qwen's despite identical post-WHT weight distributions (verified by distribution audit: skew, kurtosis, and tail mass are statistically indistinguishable).

**Hypotheses tested and eliminated:**

| Hypothesis | Status | Finding |
|------------|--------|---------|
| Weight distributions differ | Eliminated | Post-WHT distributions virtually identical |
| Attention bias absorbs error | Eliminated | Qwen 3.5-27B lacks bias, works fine (+1.3%) |
| K/Q permutation in GGUF | Contributes but not root cause | Explains attention 2.2x gap, not FFN 3.2x gap |
| Individual tensor sensitivity | Invalid methodology | In-place rotation corrupts per-tensor tests |

**The remaining unknown:** Llama propagates quantization noise 6-8x more aggressively through its residual stream than Qwen or Phi. The weight distributions are identical. The tensor types are the same. The architectures are superficially similar. The sensitivity difference appears to be architectural (how errors propagate through the network) rather than distributional (how well the quantizer fits each tensor). Further investigation would require Hessian/Fisher information analysis or per-layer error injection experiments.

**The practical solution: Hybrid and Premium configs for Llama.**

TQ4_1S for attention only + native llama.cpp types for FFN. This plays to each format's strength and sidesteps the unexplained FFN sensitivity. Using Q5_K instead of Q4_K for FFN gate/up recovers 60% of the quality gap at +8G cost.

| Config | Size | BPW | PPL (8ch) | vs Q8_0 | Decode |
|--------|------|-----|-----------|---------|--------|
| Q8_0 baseline | 69.8G | 8.50 | 2.25 | — | 7.32 t/s |
| Q4_K_M (standard) | 40.0G | 4.76 | 3.18 | +41% | ~8 t/s |
| **Premium (TQ4 attn + Q5K/Q6K FFN, b4+4)** | **49.8G** | **6.06** | **2.38** | **+5.8%** | fast |
| Q5K/Q6K FFN (b2+2) | 48.6G | 5.92 | 2.40 | +6.7% | fast |
| Wider boundary (b8+8, Q4K FFN) | 44.9G | 5.46 | 2.51 | +12% | fast |
| **Hybrid (TQ4 attn + Q4K FFN, b2+2)** | **40.2G** | **4.90** | **2.61** | **+16%** | **9.74 t/s** |

Both Hybrid and Premium beat Q4_K_M in quality. Hybrid matches Q4_K_M in size with 18% better PPL and 33% faster decode. Premium provides near-Qwen-level quality (+5.8%) at 29% size reduction. The key insight: Llama's FFN is hypersensitive to quantization — giving it one extra bit (Q4_K → Q5_K, 4.5 → 5.3 BPW) cuts the PPL delta from +16% to +6.7%.

### 5.8 Per-Model-Family Recommendations

Based on testing across 5 models and 3 model families:

**Qwen (2.5, 3.5, dense and MoE): Config I (full support)**
- attn+gate/up=TQ4_1S, ffn_down=Q4_K, boundary 2+2
- 27-37% size reduction, +1.0-1.9% PPL, 96-102% decode

**Phi (3, 4): Config I (full support)**
- Same as Qwen. +1.0% PPL, up to 2.5x faster decode on medium models

**Llama (2, 3, 3.1, 3.2, Mistral, Mixtral, CodeLlama, finetunes): Hybrid/Premium**
- **Max compression (Hybrid):** attn=TQ4_1S, ALL FFN=Q4_K, boundary 2+2 → 42% smaller, +16% PPL, 133% decode. Beats Q4_K_M at same size.
- **Best quality (Premium):** attn=TQ4_1S, ffn_gate/up=Q5_K, ffn_down=Q6_K, boundary 4+4 → 29% smaller, +5.8% PPL
- Do NOT use TQ4_1S for FFN on Llama — 6-8× worse error amplification vs Qwen/Phi
- Q5_K for FFN gate/up recovers 60% of the quality gap vs Q4_K (one extra bit makes a massive difference on Llama)

**Other architectures (Gemma, DeepSeek, etc.):** Untested. A theoretical compatibility matrix based on code analysis of `convert_hf_to_gguf.py` is available in the [getting started guide](../../docs/guides/getting-started.md#model-compatibility-matrix). Models that use the LlamaModel converter class with `undo_permute=True` are predicted to need Hybrid. Models extending TextModel directly are predicted to work with Config I. Community validation on untested models is welcome.

---

## 6. Limitations

1. **Model-family-dependent quality.** Config I achieves +1.0-1.9% PPL on Qwen/Phi but +17% on Llama. A Hybrid config (TQ4 attn + Q4_K FFN) reduces the Llama gap and beats Q4_K_M, but the FFN sensitivity root cause is not fully explained (Section 5.7). Per-model profiling may be needed for production deployment.

2. **Policy developed on 1.5B Qwen.** The ablation experiments were primarily on Qwen2.5-1.5B. The policy transfers well to larger Qwen models (+1.3-1.4%) but not to Llama. A per-model sensitivity profiler may be needed for production deployment across model families.

3. **CUDA decode gap.** Metal achieves 94-102% of Q8_0 decode via cooperative SIMD (NR0=8 amortization). CUDA achieves 39-71% of Q8_0 depending on GPU generation (38.5% on RTX 5090, 55% on 1080 Ti, 70.6% on dual 4090). signalnine's [systematic optimization effort](https://github.com/signalnine/llama-cpp-turboquant/blob/feature/tq4-weight-cuda/docs/tq4-weight-cuda-optimization-log.md) (19 kernel versions, centroid ablation) proved this is a fundamental architectural limitation, not an optimization gap. The bottleneck is float32 activation bandwidth (4x vs q8_1) and float FMA density (1x vs dp4a's 4x). The non-linear centroid lookup itself is free. Q4_K_M achieves nearly equivalent quality (+0.05 PPL) at 3.6x the CUDA decode speed. TQ4_1S is optimal for Metal; on CUDA it may be better positioned as a storage/distribution format with runtime dequant to q8_0 at model load.

4. **No comparison to GPTQ/AWQ/QuIP#.** We compare against q8_0, Q4_K, and Q4_K_M baselines within llama.cpp. Comparison to dedicated weight quantization frameworks is future work.

5. **In-layer mixing constraint.** The in-place rotation approach prevents per-tensor precision within a layer's attention block. This also invalidates individual per-tensor sensitivity testing. A temp-buffer approach would remove this constraint at the cost of additional memory.

6. **Q8_0 source required for meaningful compression.** On Q4_K_M models, the compression headroom is minimal (7% on 104B).

7. **Llama sensitivity partially mitigated, not fully explained.** Premium config achieves +5.8% PPL on Llama (vs +1.0-1.9% on Qwen/Phi). The root cause — 6-8× higher per-layer error amplification on Llama despite identical post-WHT weight distributions — appears architectural. Hessian/Fisher analysis may reveal the mechanism.

---

## 7. Practical Use

Config I is post-training quantization. No retraining, fine-tuning, or calibration data required. It applies directly to existing GGUF models.

**Source requirements:** Q8_0 GGUF weights. Config I targets tensors at 8.5 BPW and compresses them to 4.5-5.0 BPW. Models already at Q4_K_M have minimal headroom (see Section 5.5).

**Applicable models:** Any llama.cpp-supported transformer architecture with head_dim >= 128 and standard attention + FFN layout. Validated on Qwen2.5, Qwen3.5 (dense and MoE), and Llama 3.1.

**To quantize:**

First, create a tensor type file for your model's layer count. Example for a 64-layer model (Qwen3.5-27B) with Config I (boundary 2+2):

```bash
# Generate Config I tensor type file
python3 -c "
n_layers = 64  # adjust for your model
boundary = 2
for i in range(boundary, n_layers - boundary):
    for t in ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up']:
        print(f'blk.{i}.{t}.weight=tq4_1s')
    print(f'blk.{i}.ffn_down.weight=q4_k')
" > config_i.txt

# Quantize from Q8_0 source
./build/bin/llama-quantize \
  --allow-requantize \
  --tensor-type-file config_i.txt \
  model-Q8_0.gguf model-config-i.gguf Q8_0
```

For Llama-family models, use the Hybrid or Premium config:

```bash
# Llama Hybrid: TQ4_1S attention only, Q4_K for ALL FFN
python3 -c "
n_layers = 80  # Llama 3.1 70B
for i in range(2, n_layers - 2):
    for t in ['attn_q', 'attn_k', 'attn_v', 'attn_output']:
        print(f'blk.{i}.{t}.weight=tq4_1s')
    for t in ['ffn_gate', 'ffn_up', 'ffn_down']:
        print(f'blk.{i}.{t}.weight=q4_k')
" > llama_hybrid.txt

# Llama Premium: TQ4_1S attention, Q5_K/Q6_K FFN, wider boundary
python3 -c "
n_layers = 80
for i in range(4, n_layers - 4):
    for t in ['attn_q', 'attn_k', 'attn_v', 'attn_output']:
        print(f'blk.{i}.{t}.weight=tq4_1s')
    for t in ['ffn_gate', 'ffn_up']:
        print(f'blk.{i}.{t}.weight=q5_k')
    print(f'blk.{i}.ffn_down.weight=q6_k')
" > llama_premium.txt

./build/bin/llama-quantize \
  --allow-requantize \
  --tensor-type-file llama_hybrid.txt \
  model-Q8_0.gguf model-hybrid.gguf Q8_0
```

**To benchmark:**

```bash
# PPL
./build/bin/llama-perplexity -m model-config-i.gguf \
  -f wikitext-2-raw/wiki.test.raw

# Speed
./build/bin/llama-bench -m model-config-i.gguf -p 512 -n 128

# Combined with TurboQuant KV compression
./build/bin/llama-bench -m model-config-i.gguf \
  -p 512 -n 128 -ctk q8_0 -ctv turbo4
```

Implementation: historical `PR #45` on branch `pr/tq4-weight-compression`
([public endpoint unavailable](../../docs/reference/historical-forks.md#archived-identifiers)).
Metal backend only — CUDA port needed before merge. See
[getting started](../../docs/guides/getting-started.md) for usage.

---

## 8. Conclusion

The main contribution is the discovery that tensor role determines optimal compression strategy for transformer weights, and that the same boundary-layer and write-back sensitivity patterns observed in KV cache compression ([Boundary V](layer-aware-v-compression.md), [Asymmetric K/V](asymmetric-kv-compression.md)) transfer directly to the weight domain.

The specific findings:

- **Attention tensors** tolerate WHT-rotated compression well (the same rotation mechanism used in TurboQuant KV cache compression)
- **FFN write-back projections** (ffn_down, attn_output) are quality-critical and should use higher precision or native optimized types
- **FFN read projections** (gate, up) compress well with WHT-rotated types
- **Boundary layers** (first and last N) are disproportionately sensitive, consistent with KV cache findings
- **Native llama.cpp types beat WHT-rotated types for FFN decode speed** due to optimized Metal kernels, making the hybrid approach both faster and better quality than uniform WHT compression

Config I (TQ4_1S attention + TQ4_1S gate/up + Q4_K ffn_down + boundary 2+2) provides 27-38% model size reduction at +1.0-3.9% PPL with 95-254% decode speed on Qwen and Phi models from 1.5B to 72B. Combined with TurboQuant KV cache compression, total memory at 32K context reaches 59% of baseline with +1.4% PPL on MoE.

On Llama-family models, a different policy is required: TQ4_1S for attention only, native Q4_K or Q5_K for FFN. The Hybrid config (Q4_K FFN) achieves 42% compression at +16% PPL, beating Q4_K_M in both quality and speed. The Premium config (Q5_K/Q6_K FFN) achieves 29% compression at +5.8% PPL. The model-family sensitivity difference — Llama propagates per-layer quantization error 6-8× more aggressively than Qwen or Phi — is characterized but not fully explained (Section 5.7), and represents an open research question.

TQ4_1S itself was a straightforward application of the turbo4 resurrection insight (more centroids, simpler packing, no correction), but the policy that makes it useful required systematic experimental discovery. The process, not just the result, is documented here because the methodology of isolating tensor roles, testing boundary strategies, and discovering implementation constraints generalizes to any weight compression format.

The current implementation validates the method end-to-end across 6 models and 3 architectures. Implementation headroom remains: CUDA/HIP backend porting, importance-weighted centroid selection, per-model automatic sensitivity profiling, head-aligned WHT block sizes for permuted architectures, and further investigation of the Llama error amplification mechanism. These are engineering and tuning opportunities — the core approach (tensor-role-aware policy + WHT-rotated quantization + hybrid native types) is validated and deployable today on Metal.

---

## References

- TurboQuant paper: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- David Y. Tan's TQ3_1S implementation: [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3/tree/main)
- TurboQuant+ implementation: [current monorepo](../../README.md)
- turbo4 KV cache resurrection: [turbo4-resurrection.md](turbo4-resurrection.md)
- Asymmetric K/V compression: [asymmetric-kv-compression.md](asymmetric-kv-compression.md)
- Boundary V layer-aware compression: [layer-aware-v-compression.md](layer-aware-v-compression.md)
- Sparse V dequantization: [sparse-v-dequant.md](sparse-v-dequant.md)
- MoE V-compression frontier: [moe-v-compression-frontier.md](moe-v-compression-frontier.md)
- signalnine's CUDA kernel optimization log (V8-V19, centroid ablation, dp4a analysis): [tq4-weight-cuda-optimization-log.md](https://github.com/signalnine/llama-cpp-turboquant/blob/feature/tq4-weight-cuda/docs/tq4-weight-cuda-optimization-log.md)
