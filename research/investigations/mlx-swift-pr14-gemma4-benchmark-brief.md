# MLX Swift PR14 Benchmark Brief

This note summarizes the current Gemma 4 benchmark picture after `mlx-swift-lm` PR14. The goal is not to argue for one framework in the abstract, but to present what the measured data currently says on Apple Silicon.

## Scope

- **Primary comparison:** `mlx-swift-lm` PR14 vs `llama.cpp` TurboQuant fork
- **Primary model:** Gemma 4 E2B 4-bit on M5 Max 128GB
- **Secondary hardware:** M2 Pro 16GB
- **Secondary model:** Gemma 4 31B dense as a larger-model sanity check

## Test configurations

### MLX Swift

- Repo: [`ekryski/mlx-swift-lm`](https://github.com/ekryski/mlx-swift-lm)
- PR: [#14](https://github.com/ekryski/mlx-swift-lm/pull/14) (`session/all-perf-fixes`)
- Model: `mlx-community/gemma-4-e2b-it-4bit`
- KV modes: `none`, `turbo4v2`

### llama.cpp

- Repo: [historical llama.cpp TurboQuant fork](../../docs/reference/historical-forks.md#llamacpp-experimental-forks) (public URL unavailable)
- Branch at test time: `feature/turboquant-kv-cache`
- Model: `bartowski/google_gemma-4-E2B-it-Q4_K_L.gguf`
- KV modes: no-turbo baseline, `turbo4v2`

## Executive summary

On the measured M5 Max setup, `mlx-swift-lm` PR14 is faster than `llama.cpp` on Gemma 4 E2B decode both with and without TurboQuant. The largest difference is under `turbo4v2`: `mlx-swift` is materially faster while also maintaining a small compressed KV footprint. The most important practical result is that TurboQuant is effectively free in `mlx-swift` after the PR14 fixes, while it remains a noticeable decode penalty in the tested `llama.cpp` setup.

At the same time, this is not a universal “mlx-swift wins everything” claim. The larger-model comparison is not perfectly apples-to-apples because the 31B runs use different quantizations, and long-context prefill still shows mixed leadership depending on context depth.

## M5 Max: Gemma 4 E2B decode

### Short-context decode

| Framework | No-turbo | Turbo4v2 | Turbo overhead |
|-----------|----------|----------|----------------|
| llama.cpp | 158 t/s | 107 t/s | -32% |
| mlx-swift | 187 t/s | 194 t/s | +4% |

### Context-scaled decode

| Context | llama no-turbo | llama turbo4v2 | mlx no-turbo | mlx turbo4v2 |
|---------|----------------|----------------|--------------|--------------|
| ~1K | 158 | 107 | 187 | 194 |
| 16K | 157 | 106 | 168 | 168 |
| 32K | 157 | 106 | 145 | 146 |

### What this means

- `mlx-swift` leads short-context decode both with and without TurboQuant.
- The biggest gap is the compressed path: `194 t/s` vs `107 t/s` at ~1K context.
- In `mlx-swift`, `turbo4v2` is faster than `none` on decode. This is not noise — it is expected behavior on Apple Silicon because the compressed (4-bit) KV cache dequantizes through the same matmul path that already favors smaller types. 4-bit is faster than bf16 for decode, and 8-bit is faster than 4-bit.
- In the measured `llama.cpp` build, `turbo4v2` still carries a real decode penalty.

## M5 Max: Gemma 4 E2B prefill

| Context | llama no-turbo | llama turbo4v2 | mlx no-turbo | mlx turbo4v2 |
|---------|----------------|----------------|--------------|--------------|
| 128 | 4974 | 4527 | 4132 | 3805 |
| 512 | 6656 | 6171 | — | — |
| 1024 | 6557 | 5812 | 8282 | 8027 |
| 4096 | 5910 | 4807 | 6992 | 6667 |
| 8192 | 5083 | 3791 | 5375 | 5910 |
| 16384 | 4708 | 2757 | 3819 | 3615 |
| 32768 | 3523 | 1836 | 2196 | 2235 |

### What this means

- `mlx-swift` takes the lead at 1K, 4K, and 8K on the measured M5 Max runs.
- `llama.cpp` remains ahead at the deepest measured prefill lengths, especially no-turbo at 16K and 32K.
- TurboQuant prefill is materially more expensive in the tested `llama.cpp` build than in the tested `mlx-swift` build.

## Memory and compression

### Measured PR14 takeaway

- TurboQuant on `mlx-swift` keeps decode speed essentially intact while cutting KV cache footprint by roughly `73-80%`.
- At 4K `turbo4v2`, the measured `mlx-swift` KV cache footprint is about `184-191 MB`.
- At 4K `turbo4v2`, measured GPU peak during NIAH / summarization stayed near `3.32-3.33 GB` on M5 Max.

### Quality and retrieval checks

No speed story is complete without checking whether the compressed path still retrieves and generates sanely. For `mlx-swift` PR14 on Gemma 4 E2B `4bit + turbo4v2`, we have both a retrieval check and a baseline-relative generation-quality check.

#### NIAH 4096 (`turbo4v2`)

| Depth | Result | Prefill | Decode | TTFT | GPU Peak | KV Cache |
|-------|--------|---------|--------|------|----------|----------|
| 10% | PASS | 6136.7 tok/s | 161.9 tok/s | 676 ms | 3.33 GB | 184 MB |
| 25% | PASS | 6159.0 tok/s | 158.8 tok/s | 673 ms | 3.33 GB | 184 MB |
| 50% | PASS | 6131.5 tok/s | 160.0 tok/s | 676 ms | 3.33 GB | 184 MB |
| 75% | PASS | 6159.7 tok/s | 161.6 tok/s | 673 ms | 3.33 GB | 184 MB |
| 90% | PASS | 6156.9 tok/s | 161.8 tok/s | 673 ms | 3.33 GB | 184 MB |

- Retrieval output was correct at every tested insertion depth: `BLUE TIGER 42`
- This is a strong sign that the compressed KV path is not just fast on synthetic decode, but still preserves long-prompt retrieval behavior at 4K context

#### KLD 4096 summarization (`turbo4v2`)

| Metric | Result |
|--------|--------|
| Gen PPL | 1.4698 |
| Gen KLD vs bf16 baseline | 1.205454 |
| Prefill | 6564.4 tok/s |
| Decode | 144.3 tok/s |
| TTFT | 623 ms |
| GPU Peak | 3.32 GB |
| KV Cache | 191 MB |

- The harness computes KLD by generating with the target config, then forced-decoding the generated tokens through the highest-fidelity baseline for the model family without KV quantization
- This is useful as a regression guardrail, but not a final quality verdict from a single run

#### What the quality data supports

- `turbo4v2` on `mlx-swift` PR14 passes the 4K NIAH retrieval sweep cleanly
- The compressed path remains coherent under summarization and can be compared numerically against a bf16 baseline
- Quality is not unmeasured anymore; it is part of the benchmark story

#### What is still missing

- Multi-run KLD averaging if you want a less noisy divergence estimate
- 64K and 128K quality checks if the goal is to make stronger long-context claims

## Lower-model PR14 story

The smaller Gemma 4 E2B model is the model that made PR14 legible quickly because it was cheap enough to iterate on while still exposing the relevant decode pathologies.

### Before vs after on M5 Max

| Metric | Start | End | Delta |
|--------|-------|-----|-------|
| Decode @ 1024 | 131 t/s | 195 t/s | +49% |
| Prefill @ 1024 | 2323 t/s | 8340 t/s | +259% |
| Memory @ 4096 | 5.72 GB | 3.32 GB | -42% |

### Why this matters

PR14 was not one giant kernel win. It was mostly a cleanup of correctness and dtype issues that were quietly forcing bad execution:

- first-token async race fix
- prefill loop fix
- removal of a harmful compiled residual path
- bf16 / fp32 promotion fixes
- TurboQuant dtype cleanup
- symbolic sliding-window mask optimization

The result is that the compressed path on Gemma E2B stopped behaving like a “research tax” and started behaving like a real deployment option.

## M2 Pro results

Gemma 4 E2B 4-bit on M2 Pro 16GB, `mlx-swift` PR14:

| Context | Prefill t/s | Decode t/s | GPU Peak |
|---------|-------------|------------|----------|
| 128 | 559 | 74.8 | 2.62 GB |
| 1024 | 787 | 81.9 | 3.22 GB |
| 4096 | 967 | 77.9 | 3.33 GB |

### What this means

- M2 Pro lands at roughly `40%` of M5 Max decode throughput on this model.
- Memory pressure is very similar to M5 Max because the model and KV geometry are the same.
- This is a useful deployment signal: the same PR14 path remains workable on smaller Apple Silicon, but the experience is clearly in a lower throughput tier.

## Larger-model checks

### Gemma 4 26B-A4B MoE (apples-to-apples, both 4-bit)

| Framework | Config | Decode | Peak memory |
|-----------|--------|--------|-------------|
| Python `mlx-lm` | 4bit, no turbo | 123 t/s | 14.5 GB |
| mlx-swift PR14 | 4bit, no turbo | 119.5 t/s | 13.6 GB |
| mlx-swift PR14 | 4bit, turbo4v2 | 107 t/s | 13.6 GB |

This is a cleaner framework comparison: same model, same quantization, same hardware. mlx-swift is within 3% of Python `mlx-lm` and uses 6% less peak memory. Turbo carries a modest overhead on MoE (the Router fp32 fix in PR14 helps but MoE routing adds quant/dequant work per expert).

### Gemma 4 31B dense (different quants, not directly comparable)

| Framework | Quant | Decode | Peak memory |
|-----------|-------|--------|-------------|
| llama.cpp | Q8_0 (no turbo) | 10.0 t/s | — |
| llama.cpp | Q8_0 (turbo4v2) | 9.5 t/s | — |
| mlx-swift | 4bit (no turbo) | 27.1 t/s | 16.4 GB |

### Caveat

The 31B comparison is not apples-to-apples: `llama.cpp` used `Q8_0` (8-bit, 18.9 GB model) while `mlx-swift` used `4bit` (smaller model). The 26B-A4B comparison above is a better framework benchmark. The 31B data is included as a large-model sanity check showing both frameworks handle Gemma 4 dense architecture, but the quantization difference means the decode gap is not purely framework performance.

## Objective comparison: what the data supports

### Strong claims supported by the data

- On Gemma 4 E2B for the tested M5 Max setup, `mlx-swift` PR14 is faster than the tested `llama.cpp` build on decode.
- On the same setup, `mlx-swift` handles `turbo4v2` much better: the compressed path stays near no-turbo speed.
- `mlx-swift` now has a credible compressed deployment story on Apple Silicon because speed, memory footprint, and retrieval behavior all look good in the measured runs.

### Claims the data does not support yet

- It does not prove `mlx-swift` is universally faster than `llama.cpp` across all model families.
- It does not prove the 31B comparison is framework-pure, because the quantization setups differ.
- It does not prove quality parity from a single KLD run; KLD is useful here as a guardrail, not a final verdict.

## Bottom line

PR14 changes the story from “MLX Swift is behind and still debugging basics” to “MLX Swift is now a serious Apple Silicon baseline, especially for compressed Gemma inference.”

For this benchmark set:

- If the priority is **Gemma 4 E2B decode speed on Apple Silicon**, `mlx-swift` PR14 is the best measured result here.
- If the priority is **compressed KV with minimal decode penalty**, `mlx-swift` is clearly ahead in this snapshot.
- If the priority is **longest-context prefill regardless of compression path**, the picture is mixed and `llama.cpp` still has competitive territory at the deepest measured prompt lengths.
