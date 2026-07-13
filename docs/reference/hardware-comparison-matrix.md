# TurboQuant Hardware Comparison Matrix

Tracking performance across all tested hardware configurations. Updated as new diagnostic results come in.

## Baseline: M5 Max 128GB (our reference)

| Metric | q8_0 | turbo3 | Mode 2 | turbo3/q8_0 |
|--------|------|--------|--------|-------------|
| **Prefill 2K** | 2707 | 2632 | 2681 | 0.972x |
| **Prefill 4K** | 2429 | 2362 | 2426 | 0.972x |
| **Prefill 8K** | 2052 | 2014 | 2084 | 0.981x |
| **Prefill 16K** | 1685 | 1660 | 1686 | 0.985x |
| **Prefill 32K** | 1224 | 1214 | 1222 | 0.992x |
| **Decode short** | 85.8 | 77.4 | 78.7 | 0.902x |
| **Decode 4K** | 79.9 | 70.9 | 73.1 | 0.887x |
| **Decode 8K** | 77.4 | 66.6 | 69.4 | 0.860x |
| **PPL 8-chunk** | 6.111 | 6.211 | 6.120 | +1.6% / +0.1% |
| **PPL 32-chunk** | 5.415 | 5.471 | 5.435 | +1.0% / +0.4% |

- GPU Family: MTLGPUFamilyApple10 (1010)
- Tensor API: **true**
- Model: Qwen3.5-35B-A3B-Q8_0.gguf
- Build: dfc1097 (feature/turboquant-kv-cache TOT)

## Mario: M1 Max 64GB

| Metric | q8_0 | turbo3 | turbo3/q8_0 | Notes |
|--------|------|--------|-------------|-------|
| **Prefill 42K** | ~500* | 324.8 | ~0.65x | 70-page PDF, 42K tokens |
| **Decode 42K** | 41.8* | 3.79 | **0.091x** | Catastrophic — constant cache thrashing |

- GPU Family: MTLGPUFamilyApple7 (1007)
- Tensor API: **false**
- Build: dfc1097 (our TOT)
- *q8_0 numbers from earlier test at 32K context

### Mario: Dual 4090 (NVIDIA)

| Metric | q8_0 | unixsysdev turbo | ratio | Notes |
|--------|------|------------------|-------|-------|
| **Prefill (PDF)** | 5116.9 | 3182.7 | 0.622x | Third-party impl |
| **Decode (PDF)** | 103.6 | 39.9 | **0.385x** | github.com/unixsysdev/llama-turboquant |

- NOT our implementation — this is unixsysdev's CUDA port
- Missing our optimizations: graph-side WHT, block-32, fp16 LUT, split LUT
- Useful as comparison floor for our eventual CUDA port

## @tarruda: M1 Ultra 128GB

| Metric | q8_0 | turbo3 | turbo3/q8_0 | Notes |
|--------|------|--------|-------------|-------|
| **Decode 4K** | 17* | 11 | ~0.65x | Qwen 3.5 397B |

- GPU Family: MTLGPUFamilyApple7 (1007)
- Tensor API: **false**
- Older build (pre-decode-optimization)

## Anonymous: M1 Max 64GB

| Metric | q8_0 | turbo3 | turbo3/q8_0 | Notes |
|--------|------|--------|-------------|-------|
| **Prefill 42K** | 364 | 272 | 0.75x | |
| **Decode 42K** | 11 | 4 | **0.36x** | Likely older code version |

---

## Decode Ratio Scaling (turbo3/q8_0 by context depth)

| Depth | M5 Max | M1 Max (Mario) | M1 Ultra (@tarruda) | Expected |
|-------|--------|-----------------|---------------------|----------|
| short | 0.90x | — | — | 0.88-0.92x |
| 4K | 0.89x | — | ~0.65x | 0.85-0.90x |
| 8K | 0.86x | — | — | 0.82-0.88x |
| 16K | — | — | — | 0.78-0.85x |
| 32K | — | — | — | 0.72-0.80x |
| 42K | 0.72x* | **0.09x** | — | 0.68-0.75x |

*M5 Max 48K number from server benchmark with Mario's PDF

## Key Findings

1. **M5 Max (Apple10, has_tensor=true)**: Decode degrades gradually, ~4% per context doubling. Usable through 48K+.
2. **M1 Max/Ultra (Apple7, has_tensor=false)**: Decode falls off a cliff at long context. Constant cache thrashing causes 10x+ slowdown.
3. **NVIDIA (unixsysdev impl)**: Even worse than M1 Metal. Their implementation lacks our optimizations.
4. **Layer-adaptive mode 2**: Free quality win — q8_0 PPL at 3.5x compression.

## Pending Diagnostic Runs

- [ ] Mario M1 Max 64GB — full diagnostic with turbo-diag v5
- [ ] Dipesh RX 9070 XT — first AMD RDNA 4 test (Ubuntu native or WSL2)
- [ ] @spiritbuun RTX 3090 — CUDA, fixed dequant independently
- [ ] @jxwalker 2x 3090 NVLink + DGX Spark — multi-GPU
- [ ] @vaiduakhu 1-2x RTX 5090 — Blackwell consumer GPU
- [ ] @MyopicRaccoon 3090 via Thunderbolt eGPU — bandwidth edge case
- [ ] @JamesNumb3rs AMD 9070 — RDNA 4
- [ ] @DarkSmak812 AMD 6800 XT — RDNA 2, ROCm (FA broken on ROCm >7.0.1)
- [ ] @tarruda M1 Ultra 128GB — retest with current TOT + diagnostic
- [ ] @MgkMshrmBrkfst M1 Max 64GB — retest (turbo3 decode 3.96 tok/s confirmed)
