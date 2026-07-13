# MoE V-Compression Frontier: Aggressive V Quantization with Boundary V on MoE Architectures

**Dipesh Tharu Mahato**
Independent Researcher
GitHub: [@dipeshbabu](https://github.com/dipeshbabu)

---

## Abstract

We investigate the optimal V cache compression config for MoE (Mixture-of-Experts) architectures on Apple Silicon. Starting from the established `q8_0-K + turbo3-V` asymmetric recommendation, we tested whether `q8_0-K + turbo2-V` with Boundary V (auto-enabled) could provide stronger compression without quality loss on MoE models.

On Qwen3.5-35B-A3B MoE Q8_0 (M5 Max), `q8_0/turbo2` with Boundary V delivers:
- **7.53x V compression** (vs 5.12x for turbo3-V) — 47% more compression
- **PPL within 0.4–1.0% of q8_0 baseline** across 512, 8K, and 32K context
- **Quality equal to or better than `q8_0/turbo3`** at every tested context length
- **32K decode 2–3% faster than `q8_0/turbo3`** (reproducible across 2 runs)

This investigation grew out of a systematic decode-speed study that ruled out micro-optimization paths and pivoted to config-level exploration. Several negative results are documented below.

---

## 1. Background and Motivation

### 1.1 Prior Work

Previous TurboQuant investigation established:
- K precision dominates quality (softmax amplification); V tolerates aggressive compression
- Asymmetric `q8_0-K + turbo-V` rescues sensitive Q4_K_M models where symmetric turbo fails
- Boundary V (Layer-Aware V Compression, mode LA-V7) protects first/last 2 layers with q8_0-V while compressing middle layers with turbo2-V
- Sparse V skips V dequant for near-zero attention weights (+22.8% combined pp+tg metric on MoE)

### 1.2 The Question

The established MoE recommendation was `q8_0-K + turbo3-V` (2.1% decode gap, 5.12x V compression). Could we push V compression further to turbo2 (7.53x) without quality loss, now that Boundary V auto-enables on turbo2?

### 1.3 What Led Here

This paper documents the endpoint of a broader decode-speed investigation. The investigation started with kernel-level optimization (20+ dequant approaches), progressed through structural analysis (register pressure, kernel comparison), eliminated time-adaptive KV and Sparse K as speed paths, and pivoted to config-level exploration when the data showed that MoE decode gaps are primarily K-side-dominated and the V side is nearly free thanks to Sparse V.

---

## 2. Decode Speed Investigation (Negative Results)

The following directions were tested and ruled out before reaching the V-compression frontier. These are documented to prevent re-exploration.

### 2.1 Exhausted Decode Optimization Paths

| Direction | Result | Why ruled out |
|-----------|--------|---------------|
| LUT restructuring (8 approaches) | Neutral/negative | LUT is free on M5 constant cache |
| Register pressure rewrites | -2.6% regression | Metal compiler already optimal |
| Fused compressed-domain K·Q | -5.1% regression | Comparisons more expensive than LUT on M5 |
| Time-adaptive KV for speed | +1–4% theoretical max | Implementation cost far exceeds benefit |
| Specialized turbo-only kernels | Neutral | Compiler already fully specializes templates |

### 2.2 Dense vs MoE Bottleneck Split

Profiling on phi-4 14B (dense) and Qwen3.5-35B (MoE) revealed opposite bottleneck structures:

| Metric | phi-4 14B (dense) | Qwen3.5-35B (MoE) |
|--------|:-----------------:|:------------------:|
| Kernel structural overhead | 9.3% | 1.2% |
| Dequant cost (short context) | 2.4% | 7.2% |
| Dominant bottleneck | Register pressure | Dequant math |
| FFN % of decode | ~92% | ~88% |

On dense models, the turbo3 decode gap is structural and unfixable. On MoE, the gap is dequant-dominated but still bounded by the small attention fraction of total decode.

### 2.3 Key Insight: K-Side Dominates MoE Decode Cost

Asymmetric measurement isolated K vs V contributions:

| Config | Short decode (t/s) | Gap vs q8_0 |
|--------|:-----------------:|:-----------:|
| q8_0/q8_0 | 76.06 | baseline |
| q8_0/turbo3 | 74.43 | -2.1% (V-only cost) |
| turbo3/turbo3 | 69.73 | -8.3% (K+V cost) |

V compression costs only 2.1% of MoE decode. K compression costs 6.3%. This means:
- V can be compressed aggressively with minimal decode penalty
- K should stay at q8_0 for decode-sensitive workloads

This directly motivated testing turbo2-V (more aggressive than turbo3-V).

---

## 3. Setup

### 3.1 Hardware

| Machine | Chip | RAM | Role |
|---------|------|-----|------|
| M5 Max MacBook Pro | Apple M5 Max | 128 GB | Primary testing |
| M2 Pro Mac mini | Apple M2 Pro | 16 GB | Cross-machine validation |

### 3.2 Software

- llama.cpp branch: `feature/turboquant-kv-cache`
- Block size: QK_TURBO3=128, QK_TURBO2=128 (shipped default)
- Sparse V: enabled by default
- Boundary V: auto-enabled when `-ctv turbo2` on models with ≥8 layers
- Flash attention: on
- `iogpu.wired_limit_mb=122880` set on M5 Max

### 3.3 Models

| Model | Params | Arch | Weights | KV Layers | Machine |
|-------|--------|------|---------|-----------|---------|
| Qwen3.5-35B-A3B | 35B (3B active) | MoE + GDN hybrid | Q8_0 | 16 of 64 (attention every 4th) | M5 Max |
| phi-4 | 14B | Dense pure-attention | Q8_0 | 40 of 40 | M2 Pro |

### 3.4 PPL Methodology

Wikitext-2-raw, flash attention on, all layers GPU-offloaded.
- 512 context: 20 chunks (M5 and M2)
- 8K context: 4 chunks (M5 only)
- 32K context: 2 chunks (M5 only)

---

## 4. Results

### 4.1 MoE Quality (Qwen3.5-35B-A3B Q8_0, M5 Max)

| Config | PPL @ 512c | PPL @ 8K | PPL @ 32K | V compress |
|--------|:----------:|:--------:|:---------:|:----------:|
| q8_0/q8_0 | 6.568 | 5.399 | 6.015 | 1.0x |
| q8_0/turbo3 | 6.629 (+0.9%) | 5.443 (+0.8%) | 6.073 (+1.0%) | 5.12x |
| **q8_0/turbo2+BV** | **6.629 (+0.9%)** | **5.422 (+0.4%)** | **6.073 (+1.0%)** | **7.53x** |

turbo2+BV matches turbo3 at 512c and 32K. At 8K, turbo2+BV is **better** (+0.4% vs +0.8%). Quality is stable across all tested context lengths.

### 4.2 MoE Decode Speed (Qwen3.5-35B-A3B Q8_0, M5 Max)

| Config | Short (t/s) | 32K Run 1 (t/s) | 32K Run 2 (t/s) |
|--------|:-----------:|:---------------:|:---------------:|
| q8_0/q8_0 | 76.06 | 79.62 | — |
| q8_0/turbo3 | 74.43 | 74.80 | 78.45 |
| q8_0/turbo2+BV | 71.89 | 77.26 | 80.12 |

At 32K context, turbo2+BV is consistently faster than turbo3 (2 runs, +2–3% advantage). At short context, turbo2+BV is slower (-3.4%). The crossover is somewhere between short and 32K context.

**Interpretation (hypothesis):** At long context, Sparse V skips 80%+ of V positions. For the ~20% that are not skipped, turbo2's smaller block size (fewer bytes per position) results in less bandwidth per dequant. At short context, fewer positions are skipped, making turbo2's more complex dequant a net cost. *This mechanism is not directly proven.*

### 4.3 Cross-Machine Validation (phi-4 14B Q8_0, M2 Pro)

| Config | M2 PPL @ 512c | vs q8_0 |
|--------|:-------------:|:-------:|
| q8_0/q8_0 | 6.571 | baseline |
| q8_0/turbo3 | 6.609 | +0.6% |
| q8_0/turbo2+BV | 6.657 | +1.3% |

turbo2+BV runs correctly on M2 Pro with reasonable quality. On this pure-attention model, turbo2+BV is expectedly worse than turbo3 (Boundary V was designed to narrow the turbo2→turbo3 gap, not eliminate it on non-MoE architectures).

> **M2 limitation:** Qwen3.5-35B MoE does not fit on M2 Pro (16 GB). The M2 validation confirms correctness and general quality behavior but does NOT validate the MoE-specific decode speed finding.

---

## 5. TTFT / Prefill Scaling (Corrective Finding)

During this investigation, we also measured time-to-first-token across prompt lengths. This corrected an earlier overclaim.

| Model | turbo3 vs q8_0 prefill |
|-------|------------------------|
| phi-4 14B (M5 Max) | -1 to -17% (turbo3 SLOWER at all lengths) |
| Qwen3.5-35B MoE (M5 Max) | -1 to -11% (turbo3 SLOWER at all lengths) |
| Llama-70B (M5 Max, earlier test) | +7% at 32K (turbo3 FASTER) |

**Revised claim:** turbo3 prefill advantage is model-size-dependent. It only appears on 70B+ bandwidth-saturated models where KV write savings free bandwidth for weight reads. On smaller models, turbo3 prefill is slower. Previous +7% claim applies specifically to 70B+.

---

## 6. Recommendation

### 6.1 MoE Long-Context V Compression

For MoE models where decode speed matters and V compression is desired:

```bash
# Maximum V compression with near-q8_0 quality and decode speed
llama-server -m model-Q8_0.gguf -ctk q8_0 -ctv turbo2 -fa 1
# Boundary V auto-enables, protecting first/last 2 layers
```

| Metric | q8_0/turbo3 | q8_0/turbo2+BV |
|--------|:-----------:|:--------------:|
| V compression | 5.12x | **7.53x** |
| PPL vs baseline | +0.8–1.0% | **+0.4–1.0%** |
| 32K decode vs turbo3 | baseline | **~2–3% faster (n=2)** |

### 6.2 Scope and Limitations

This recommendation is based on:
- **One MoE model** (Qwen3.5-35B-A3B Q8_0)
- **One hardware platform** for speed (M5 Max; a limited cross-machine quality sanity check was run on M2 Pro using phi-4 14B dense)
- **Context lengths 512–32K**

Not validated:
- NIAH retrieval accuracy (tooling limitation — NIAH script does not support asymmetric K/V configs)
- Other MoE model families
- CUDA backend
- Context lengths beyond 32K

### 6.3 When to Use Which Config

| Situation | Recommended config |
|-----------|-------------------|
| MoE, decode-sensitive, need V compression | `q8_0/turbo2` (Boundary V auto) |
| MoE, maximum compression (K+V) | `turbo3/turbo3` |
| Dense model, any workload | `turbo4/turbo4` or `turbo3/turbo3` |
| Unknown model, safe default | `q8_0/turbo4` |

---

## 7. Independent Validation

**@sztlink (Felipe Sztutman)** — [Qwen3-30B-A3B Q4_K_M, RTX 4090, AmesianX v1.2.0](https://github.com/ggml-org/llama.cpp/discussions/20969#discussioncomment-16403244) (2026-04-01):
- First independent PPL validation of the MoE V-compression finding on CUDA hardware (our results were Metal-only)
- q8_0/tbq3 (asymmetric): PPL 7.5910 (+0.57% vs f16 7.5477) — confirms V compression is nearly free on MoE, consistent with our +0.8-1.0% on Qwen3.5-35B-A3B
- Symmetric tbq3/tbq3: PPL 9.5221 (+26.16%) — catastrophic. Validates asymmetric as the only safe path on Qwen MoE
- Different model (Qwen3-30B-A3B vs our Qwen3.5-35B-A3B), different hardware (RTX 4090 vs M5 Max), different implementation (AmesianX v1.2.0 vs our fork) — same conclusion: compress V aggressively, keep K at q8_0

**@Madreag** — [Optimized CUDA fork](https://github.com/Madreag/turbo3-cuda/tree/release/cuda-optimized), RTX 5090, Qwen3.5-27B Q6_K (2026-04-01):
- turbo2 beats q8_0 by 5.4% at 32K decode (58.61 vs 55.60 t/s) at 7.53x compression. Same crossover pattern as our Metal findings: smaller cache = less bandwidth = faster at long context
- turbo2 at 256K: 42.57 t/s on consumer 5090 where q8_0/f16 OOM. First 256K turbo2 data point on CUDA
- Kernel optimizations yield +13-69% decode improvement at 32K across 4 GPUs vs base TurboQuant implementation
- Confirms V compression dominance from asymmetric K/V matrix: V type varies PPL more than K type

**@sjoerdmaessen (Sjoerd Maessen)** — [Qwen3.5-122B-A10B Q5_K_S, 2x NVIDIA L40S 48GB, 82K context](https://github.com/ggml-org/llama.cpp/discussions/20969#discussioncomment-16412852) (2026-04-01):
- turbo2 at 82K on this hybrid MoE (12 attention layers + 36 recurrent): TG 57.7 t/s vs q8_0 61.1 t/s (-5.6%). **No crossover** where turbo2 beats q8_0, unlike our Qwen3.5-35B finding at 32K on Metal
- Suggests the turbo2 decode crossover is architecture-dependent: hybrid MoE with only 12 KV attention layers may not generate enough cache bandwidth savings for turbo2 to win, unlike pure-transformer MoE with more KV layers
- Asymmetric q8_0/turbo3: 100% decode recovery, confirming V compression is free on this MoE architecture

**@mudler (Ettore Di Giacinto)** — [APEX](https://github.com/mudler/apex-quant) + TurboQuant integration, [LocalAI](https://github.com/mudler/LocalAI) (44.7k stars) (2026-04-01):
- Tested TurboQuant KV cache compression on top of APEX MoE weight quantization for Qwen3.5-35B-A3B at 8K context
- +14% prompt processing speedup across all APEX tiers (I-Quality: 1,752 to 2,003 t/s, I-Compact: 1,714 to 1,959 t/s, Mini: 1,696 to 1,938 t/s)
- Zero quality loss from TurboQuant KV on top of APEX weights, 4.6x KV cache compression
- APEX Mini (12.2 GB) + TurboQuant = 35B MoE at 8K context on a 16GB consumer GPU
- First external validation of TurboQuant KV compression as a complementary layer on top of advanced weight quantization for MoE models

---

## 8. Open Questions

1. **Does the turbo2+BV advantage generalize to other MoE architectures?** Only tested on Qwen3.5 (GDN+attention hybrid, 16 KV layers out of 64). Models with different attention-to-expert ratios may behave differently.

2. **What is the mechanism for the 32K decode advantage?** The hypothesis (smaller blocks = less bandwidth per non-skipped V position under Sparse V) is plausible but not directly proven. Could also be a Metal caching effect.

3. **Does NIAH retrieval hold?** PPL is strong but retrieval accuracy is a distinct quality signal. Needs NIAH script update for asymmetric configs.

4. **Where is the short-vs-long context crossover?** turbo2+BV is slower at short context but faster at 32K. The crossover point (likely 8K–16K) was not measured.
