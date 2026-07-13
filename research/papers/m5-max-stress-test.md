# TurboQuant M5 Max 128GB Stress Test — Large Model Validation

**Dipesh Tharu Mahato**
Independent Researcher
GitHub: [@dipeshbabu](https://github.com/dipeshbabu)

---

## Abstract

We stress-test TurboQuant KV cache compression on Meta's Llama-3.1-70B-Instruct and CohereForAI Command-R+ 104B (both Q4_K_M) running on a single Apple M5 Max with 128GB unified memory. These are the largest models tested with TurboQuant to date.

Key findings:

1. **104B at full 128K native context on a laptop.** Command-R+ 104B with turbo3/turbo3 achieves PPL 4.024 at 128K context, using ~74GB of 128GB. 5.12x KV compression.

2. **Larger models tolerate symmetric turbo better.** 104B turbo3/turbo3 is +3.6% PPL (vs +11.4% on 70B). turbo4/turbo4 is +1.9%, nearly lossless.

3. **turbo3 prefill is faster than q8_0 at 32K context.** Consistent on both 70B (80.8 vs 75.2 t/s, +7.4%) and 104B (64.5 vs 62.3 t/s, +3.5%). Smaller KV cache = less memory bandwidth.

4. **NIAH retrieval is perfect.** 30/30 on 70B (q8_0 and turbo3). 10/10 on 104B turbo3 at 4K/8K.

5. **macOS context wall identified and fixed.** The default `iogpu.wired_limit_mb` caps GPU memory at ~75% of RAM. On 128GB, this causes Metal to stall at ~49K context on 70B+ models. Fix: `sudo sysctl iogpu.wired_limit_mb=122880`. One command, no reboot.

All tests used Metal flash attention with full GPU offload. Block size 128 throughout (5.12x compression for turbo3).

---

## 1. Setup

### 1.1 Hardware

| Component | Spec |
|-----------|------|
| SoC | Apple M5 Max |
| Memory | 128GB unified (LPDDR5X) |
| GPU Cores | 40-core |
| Backend | Metal with flash attention |
| OS | macOS |

### 1.2 Model

| Property | Value |
|----------|-------|
| Model | Meta-Llama-3.1-70B-Instruct |
| Weight quantization | Q4_K_M (GGUF, 40GB) |
| Layers | 80 |
| Attention heads | 64 |
| KV heads | 8 (GQA 8:1) |
| Head dimension | 128 |
| Native context | 128K |

### 1.3 Build

- Branch: `feature/turboquant-kv-cache`
- Block size: `QK_TURBO3=128`, `QK_TURBO2=128`
- Sparse V: enabled (default on M5+)
- Boundary V: not active (test predates auto-enable)
- Full GPU offload: `-ngl 99`

---

## 2. Perplexity

### 2.1 Short Context (512 tokens, 20 chunks, wikitext-2-raw)

| K | V | PPL | vs q8_0 | Status |
|---|---|-----|---------|--------|
| q8_0 | q8_0 | 3.257 | baseline | healthy |
| q8_0 | turbo4 | 3.301 | +1.3% | healthy |
| q8_0 | turbo3 | 3.325 | +2.1% | healthy |
| turbo4 | turbo4 | 3.461 | +6.3% | healthy |
| turbo3 | turbo3 | 3.629 | +11.4% | usable |
| turbo2 | turbo2 | 5.161 | +58.5% | degraded |

**Finding:** Llama-70B Q4_K_M tolerates symmetric turbo quantization across all formats. This contrasts with Qwen2.5-7B Q4_K_M, where symmetric turbo3/turbo3 produces catastrophic PPL (3556). The 70B model has sufficient capacity to absorb the quantization stacking that breaks smaller models.

turbo2/turbo2 shows significant degradation (+58.5%) but is not catastrophic — the model remains coherent unlike the Qwen2.5-7B case.

### 2.2 Long Context

| K | V | Context | Chunks | PPL |
|---|---|---------|--------|-----|
| q8_0 | q8_0 | 8K | 4 | 3.617 |
| turbo4 | turbo4 | 8K | 4 | 3.770 |
| turbo3 | turbo3 | 8K | 4 | 3.937 |
| turbo3 | turbo3 | 32K | 2 | 4.839 |
| q8_0 | q8_0 | 48K | 1 | 3.575 |
| turbo3 | turbo3 | 48K | 1 | 4.019 |

PPL remains healthy at all tested context lengths. The turbo3 PPL at 48K (4.019) is higher than q8_0 at 48K (3.575), consistent with the +11.4% gap observed at 512 context.

---

## 3. Speed

### 3.1 Short Context (512 tokens)

| K | V | Prefill (t/s) | Decode (t/s) |
|---|---|:-------------:|:------------:|
| q8_0 | q8_0 | 166.8 | 10.9 |
| q8_0 | turbo4 | 174.4 | 10.1 |
| q8_0 | turbo3 | 174.8 | 10.2 |
| turbo4 | turbo4 | 173.8 | 10.3 |
| turbo3 | turbo3 | 165.0 | 9.9 |
| turbo2 | turbo2 | 170.5 | 9.9 |

Speed is flat across all configs at short context. The 40GB model weights dominate memory bandwidth; the KV cache at 512 tokens is negligible.

### 3.2 8K Context

| K | V | Prefill (t/s) | Decode (t/s) |
|---|---|:-------------:|:------------:|
| q8_0 | q8_0 | 139.2 | 11.9 |
| turbo4 | turbo4 | 134.5 | 10.6 |
| turbo3 | turbo3 | 136.2 | 10.1 |

Still flat. KV cache at 8K is ~1.25GB (q8_0) or ~250MB (turbo3), both negligible vs 40GB weights.

### 3.3 32K Context

| K | V | Prefill (t/s) | Decode (t/s) |
|---|---|:-------------:|:------------:|
| q8_0 | q8_0 | 75.2 | 10.4 |
| turbo4 | turbo4 | 72.5 | 10.5 |
| turbo3 | turbo3 | 80.8 | 10.2 |

**turbo3 prefill is 7.4% faster than q8_0** (80.8 vs 75.2 t/s). At 32K context, the KV cache is large enough (~5GB for q8_0 vs ~1GB for turbo3) that reduced memory bandwidth during attention outweighs dequantization cost. This crossover point is consistent with the observation on smaller models (Qwen3.5-35B MoE on M1 Max, where turbo2 beats q8_0 at 65K prefill).

Decode remains flat at ~10 t/s across all configs. On a 70B model, decode is dominated by the 40GB weight read per token, not the KV cache.

> **Note:** llama-bench with default 5 repetitions hangs on 70B at 32K+ context. All 32K measurements use `-r 1`. Root cause unclear; appears to be Metal resource contention across reps.

---

## 4. Needle-In-A-Haystack (NIAH)

Kamradt single-needle methodology. 5 depths (0%, 25%, 50%, 75%, 100%) × 3 context lengths (4K, 8K, 16K) × 2 cache types.

### q8_0 (baseline)

| Depth | 4K | 8K | 16K |
|-------|:--:|:--:|:---:|
| 0% | PASS | PASS | PASS |
| 25% | PASS | PASS | PASS |
| 50% | PASS | PASS | PASS |
| 75% | PASS | PASS | PASS |
| 100% | PASS | PASS | PASS |

### turbo3 (5.12x compression)

| Depth | 4K | 8K | 16K |
|-------|:--:|:--:|:---:|
| 0% | PASS | PASS | PASS |
| 25% | PASS | PASS | PASS |
| 50% | PASS | PASS | PASS |
| 75% | PASS | PASS | PASS |
| 100% | PASS | PASS | PASS |

**30/30 pass. Zero difference between q8_0 and turbo3.** TurboQuant preserves retrieval accuracy at 5.12x KV cache compression on a 70B model.

---

## 5. Maximum Context

### 5.1 Memory at 48K Context

| Config | KV Cache (MiB) | Model + Context (MiB) | Free (MiB) |
|--------|:--------------:|:---------------------:|:----------:|
| q8_0/q8_0 | 8,160 | 48,991 | 61,108 |
| turbo3/turbo3 | ~3,000 | ~44,000 | ~66,000 |

With turbo3, the KV cache at 48K is 3GB instead of 8GB. Both fit comfortably in 128GB with 60+ GB to spare.

### 5.2 Context Wall (Identified and Fixed)

Without the fix, both q8_0 and turbo3 hang at 50K+ context:

| K | V | Context | Status |
|---|---|---------|--------|
| turbo3 | turbo3 | 48K | works (PPL 4.019) |
| q8_0 | q8_0 | 48K | works (PPL 3.575) |
| turbo3 | turbo3 | 50K | **HANGS** |
| turbo3 | turbo3 | 56K | **HANGS** |
| q8_0 | q8_0 | 64K | **HANGS** |

**Root cause:** macOS sets `recommendedMaxWorkingSetSize` to ~75% of physical RAM by default. On 128GB, this is ~96GB. When total GPU allocations (model + KV + compute buffers) exceed this limit, Metal's buffer allocation blocks indefinitely.

**Fix:** One command, no reboot required:

```bash
# Recommended: 90% of physical RAM (safe for sustained inference)
# Setting above 90% risks kernel panics under sustained load.

# 128GB Mac
sudo sysctl iogpu.wired_limit_mb=117964

# 96GB Mac
sudo sysctl iogpu.wired_limit_mb=88474

# 64GB Mac
sudo sysctl iogpu.wired_limit_mb=58982
```

With the fix applied, 70B runs at 64K context (PPL 4.135). See Section 8 for 104B results at 128K.

Isolation experiments confirmed `iogpu.wired_limit_mb` is the sole fix needed. `GGML_METAL_NO_RESIDENCY=1` and custom ubatch sizes had no measurable effect.

> **Note:** The original stress tests used 122880 (96%) without issues, but community testing (@treblewoe) reports kernel panics at sustained load above 90%. 90% is the recommended safe default.

---

## 6. GQA Impact on Compression Savings

Llama-3.1-70B uses GQA 8:1 (8 KV heads for 64 attention heads). This means the KV cache is already 1/8th the size it would be without GQA. TurboQuant's compression ratios are the same (5.12x for turbo3 at block_size=128), but the absolute memory savings are smaller:

| Config | KV at 48K | vs fp16 KV |
|--------|:---------:|:----------:|
| fp16 | ~15,360 MiB | 1.0x |
| q8_0 | 8,160 MiB | 1.9x |
| turbo3 | ~3,000 MiB | 5.1x |

On models without GQA (e.g., GPT-class with n_kv_heads = n_heads), TurboQuant's savings scale 8× larger in absolute terms.

---

## 7. Comparison with Smaller Models

| Model | Weights | Symmetric turbo3 PPL | Status |
|-------|---------|:--------------------:|--------|
| Qwen2.5-7B | Q4_K_M | 3,556 | catastrophic |
| Qwen2.5-1.5B | Q4_K_M | 8,641 | catastrophic |
| Mistral-24B | Q4_K_M | 4.987 | healthy |
| **Llama-70B** | **Q4_K_M** | **3.629** | **healthy** |
| **Command-R+ 104B** | **Q4_K_M** | **6.415** | **healthy** |

The Q4_K_M symmetric turbo sensitivity appears to be model-family-dependent, not purely size-dependent. Qwen2.5 is sensitive at all sizes. Mistral, Llama, and Command-R+ tolerate it. Larger models show better tolerance (104B +3.6% vs 70B +11.4%). For sensitive models, asymmetric q8_0-K + turbo-V is the recommended path.

---

## 8. Command-R+ 104B Q4_K_M

### 8.1 Model

| Property | Value |
|----------|-------|
| Model | CohereForAI Command-R+ 104B |
| Weight quantization | Q4_K_M (~59GB on disk, 58.4 GiB loaded) |
| Layers | 64 |
| Attention heads | 96 |
| KV heads | 8 (GQA 12:1) |
| Head dimension | 128 |
| Native context | 128K |

Metal configuration: `iogpu.wired_limit_mb=122880` applied before testing.

### 8.2 Perplexity (512 tokens, 20 chunks, wikitext-2-raw)

| K | V | PPL | vs q8_0 | Status |
|---|---|-----|---------|--------|
| q8_0 | q8_0 | 6.192 | baseline | healthy |
| q8_0 | turbo4 | 6.211 | +0.3% | healthy |
| q8_0 | turbo3 | 6.296 | +1.7% | healthy |
| q8_0 | turbo2 | 6.678 | +7.9% | healthy |
| turbo4 | turbo4 | 6.312 | +1.9% | healthy |
| turbo3 | turbo3 | 6.415 | +3.6% | healthy |
| turbo2 | turbo2 | 7.049 | +13.8% | usable |

**104B tolerates symmetric turbo even better than 70B.** turbo3/turbo3 is +3.6% (vs +11.4% on 70B). turbo4/turbo4 is nearly lossless at +1.9%. Bigger models = more headroom for quantization stacking.

### 8.3 Speed

#### 512 Context

| K | V | Prefill (t/s) | Decode (t/s) |
|---|---|:-------------:|:------------:|
| q8_0 | q8_0 | 125.0 | 8.5 |
| q8_0 | turbo4 | 128.5 | 7.6 |
| q8_0 | turbo3 | 129.4 | 7.3 |
| q8_0 | turbo2 | 127.3 | 7.8 |
| turbo4 | turbo4 | 118.0 | 7.9 |
| turbo3 | turbo3 | 125.7 | 6.8 |
| turbo2 | turbo2 | 126.5 | 7.7 |

#### 8K Context

| K | V | Prefill (t/s) | Decode (t/s) |
|---|---|:-------------:|:------------:|
| q8_0 | q8_0 | 101.6 | 7.6 |
| q8_0 | turbo3 | 100.1 | 8.1 |
| turbo3 | turbo3 | 98.4 | 7.6 |

#### 32K Context

| K | V | Prefill (t/s) | Decode (t/s) |
|---|---|:-------------:|:------------:|
| q8_0 | q8_0 | 62.3 | 8.3 |
| turbo3 | turbo3 | 64.5 | 7.8 |

turbo3 prefill faster than q8_0 again at 32K (64.5 vs 62.3 t/s, +3.5%). Same crossover pattern as 70B.

### 8.4 Context Scaling (the big test)

| K | V | Context | PPL | Pass time | Status |
|---|---|---------|-----|-----------|--------|
| turbo3 | turbo3 | 48K | 3.672 | 931s | works |
| turbo3 | turbo3 | 64K | 4.321 | 1481s | works |
| turbo3 | turbo3 | 96K | 4.170 | 2966s | works |
| turbo3 | turbo3 | 128K | 4.024 | 4996s | **works** |

**104 billion parameters. 128K full native context. On a MacBook. PPL 4.024.** turbo3 (5.12x KV compression). Peak memory ~74GB of 128GB. Pass time 83 minutes.

The context wall that blocked 70B at 49K is fully resolved with `iogpu.wired_limit_mb=122880`.

### 8.5 NIAH (Needle-In-A-Haystack)

Kamradt single-needle methodology. 104B decode is slow (~8 t/s), so 16K tests timed out.

#### turbo3 (5.12x compression)

| Depth | 4K | 8K | 16K |
|-------|:--:|:--:|:---:|
| 0% | PASS | PASS | timeout |
| 25% | PASS | PASS | — |
| 50% | PASS | PASS | — |
| 75% | PASS | PASS | — |
| 100% | PASS | PASS | — |

**10/10 pass at 4K and 8K.** 16K timed out due to slow decode, not a retrieval failure.

---

## 9. Limitations

1. **Two models tested.** Results are for Llama-3.1-70B-Instruct and Command-R+ 104B, both Q4_K_M. Other 70B+ models (Qwen-72B, DeepSeek-67B) may behave differently.

2. **Q4_K_M weights only.** Q8_0 weights on 70B would likely show even better turbo PPL, but require ~70GB for weights alone, leaving limited room for KV cache on 128GB.

3. **Metal only.** CUDA and HIP backends were not tested at this model size.

4. **Single run per config.** PPL measurements are single-run, not averaged across multiple seeds. Error bars are from the wikitext-2 chunk variance, not run-to-run variance.

5. **Boundary V not tested.** The auto-enable for turbo2-V was added after this test. Boundary V on 70B turbo2 would likely recover significant quality from the +58.5% degradation.

6. **104B NIAH limited.** 16K context timed out due to slow decode (~8 t/s). Not a retrieval failure. Requires longer query timeout for full coverage.

7. **macOS GPU memory fix required for 70B+ at long context.** The `iogpu.wired_limit_mb` sysctl must be set to ~90% of physical RAM (community testing confirmed >90% risks kernel panics under sustained load). Resets on reboot.

---

## Addendum: Independent Validation (2026-03-31)

The key findings from this stress test have been independently confirmed:

**Asymmetric K/V rescue (Sections 2, 8):**
- **@HyperionMS2040** — RTX 3090, 10-model CUDA sweep (2026-03-30): q8_0/turbo4 "lossless across all tested architectures" (4 architectures validated). Confirmed model-family-dependent sensitivity: Qwen2.5-7B symmetric turbo3 catastrophic (PPL 3,472) vs Llama 3.1 8B symmetric turbo3 +6.4%.
- **@sztlink** — RTX 4090, Qwen3-4B (2026-03-31): Full asymmetric matrix confirms V compression is completely free (1.000 cosine similarity with fp16-K + 2bit-V). All degradation from K.
- **AMD HIP** — RX 9070 XT, gfx1201 (2026-03-29): Asymmetric q8_0/turbo4 confirmed at +1.0% PPL. (Author's own testing on Windows AMD hardware.)

**turbo3 prefill faster than q8_0 at long context (Sections 3, 8):**
- **@spiritbuun** — RTX 3090 (X collaborator, ~2026-03-28): Reported near-parity prefill speed with dequant-then-MMA path on CUDA.
- **@AmesianX** — Blackwell DGX Spark (2026-03-30): turbo decode 63.5 t/s, faster than q8_0 (50.1 t/s) at 8K context.
- **@dusterbloom** — RTX 3090 (2026-03-30): Decode faster on 4/5 models with TBQ3 Flash Attention.
- **@HyperionMS2040** — RTX 3090 (2026-03-30): Asymmetric q8_0-K/turbo3-V 14% faster decode than symmetric turbo3/turbo3 (120.9 vs 106 t/s).

**macOS GPU memory wall (Section 5):**
- **@treblewoe** (X collaborator, 2026-03-31): Confirmed the wall exists. Reported kernel panics at >90% allocation under sustained load (Minimax M2.5 3-bit starting at 100GB). Recommended 90% as safe ceiling. Docs updated accordingly.

---

## References

- TurboQuant paper: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- TurboQuant+ implementation: [github.com/dipeshbabu/turboquant_plus](https://github.com/dipeshbabu/turboquant_plus)
- llama.cpp fork: [github.com/dipeshbabu/llama-cpp-turboquant](https://github.com/dipeshbabu/llama-cpp-turboquant)
- Block size optimization: [block-size-experiment.md](block-size-experiment.md)
- Sparse V dequant: [sparse-v-dequant.md](sparse-v-dequant.md)
- Configuration recommendations: [turboquant-recommendations.md](../../docs/guides/turboquant-recommendations.md)
