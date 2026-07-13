# TurboQuant+ Benchmark Results

**Date**: 2026-03-25
**Hardware**: Apple M5 Max 128GB, Metal GPU
**Build**: llama.cpp feature/turboquant-kv-cache
**Test**: 512 context, max_tokens=50, temperature=0
**Method**: llama-server API `/v1/chat/completions`, timing from response `timings` field
**Runs**: Single run per config (no warm-up separation, no stddev)
**Note**: Compression ratios are theoretical (bits/val vs 16-bit fp16), not empirically measured KV memory

## Qwen 3.5 35B-A3B MoE (Q8_0)

| Cache Type | Bits/val | Prompt tok/s | Gen tok/s | KV Compression |
|------------|----------|-------------|-----------|----------------|
| q8_0       | 8.0      | **225.4**   | **85.0**  | 2.0×           |
| q4_0       | 4.0      | 221.5       | 84.5      | 4.0×           |
| turbo4     | 4.25     | 7.1         | 2.4       | 3.8×           |
| turbo3     | 3.25     | 4.2         | 2.4       | **4.9×**       |

## Qwopus v2 27B Dense (Q8_0)

| Cache Type | Bits/val | Prompt tok/s | Gen tok/s | KV Compression |
|------------|----------|-------------|-----------|----------------|
| q8_0       | 8.0      | **91.3**    | **17.6**  | 2.0×           |
| q4_0       | 4.0      | 90.8        | 17.6      | 4.0×           |
| turbo4     | 4.25     | 5.5         | 1.3       | 3.8×           |
| turbo3     | 3.25     | 5.3         | 1.3       | **4.9×**       |

## Key Findings

### Compression
- turbo3: **4.9× compression** — matches paper's claim for ~3-bit KV cache
- turbo4: **3.8× compression** — better quality at slightly less compression

### Speed Regression
- turbo3/turbo4 are **13-35× slower on generation** than q8_0 (varies by model)
  - MoE: 85.0 → 2.4 tok/s (35×)
  - Dense 27B: 17.6 → 1.3 tok/s (13.5×)
- Likely cause (unverified via profiling): Metal dequantize performs full 128×128
  matrix-vector multiply (rotation) per 4/16-element chunk — O(d²) vs O(d)
- This is the unoptimized initial implementation — tracked in #23

### Quality (Qualitative Only)
- Both models produce coherent text with turbo3/turbo4
- No obvious quality degradation in short generations (subjective, not quantified)
- **Missing**: perplexity comparison, NIAH test, pass@k metrics
- Python prototype validated cosine similarity 0.91 (turbo3) / 0.97 (turbo4) on real KV tensors

### Speed Optimization Path
1. Cache full-block dequantize (eliminate 8× redundant rotation per block)
2. Use threadgroup memory for rotation scratch arrays
3. Implement fast Walsh-Hadamard rotation — O(d log d) vs O(d²)
4. Custom flash attention kernel with integrated dequantize

## Gotchas

1. Qwen 3.5 thinking tokens inflate output — use API max_tokens, not -n
2. Model takes 30-45s to load on M5 Max — benchmark scripts must wait
3. Metal library compilation takes ~6s on first run
4. llama-bench crashes with our build — used server API instead
5. Don't leave background llama-cli processes — they starve GPU memory
