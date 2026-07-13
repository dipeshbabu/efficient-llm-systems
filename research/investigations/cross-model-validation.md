# Cross-Model Validation Results

TurboQuant quick bench across multiple model families, architectures, and sizes.
Hardware: Apple M5 Max 128GB. All tests with sparse V enabled.

## The ISWA Bug — Cross-Model Testing Catches What Single-Model Testing Misses

After turbo4 was working on Qwen, we built `turbo-quick-bench.sh` — a rapid smoke test (~5 min per model) that runs PPL, decode speed, and NIAH across q8_0/turbo3/turbo4. Then we downloaded 7 models from 5 families and ran the bench on each.

Gemma 2 27B showed PPL 13.7 trillion. Everything else passed.

Initial investigation went down the wrong path — we assumed head_dim=256 (Gemma's n_embd/n_head = 4608/32 = 144). But parsing GGUF metadata directly showed K/V heads are explicitly 128. q4_0 worked fine on Gemma. Disabling flash attention didn't help (same PPL). The issue was turbo-specific but not head_dim-specific.

The root cause was found by reading all 5 overloads of `build_attn` in `llama-graph.cpp`. The regular `build_attn` (used by Qwen, Llama, Phi, Mistral) includes turbo WHT Q rotation and V inverse rotation. The ISWA overload (used by Gemma 2, Cohere2, OLMo2, Gemma3N) didn't. K/V were WHT-rotated in SET_ROWS but Q was unrotated. Attention computed `rotated_K × unrotated_Q` — complete garbage, 13.7 trillion PPL garbage.

16 lines fixed it. PPL 13.7 trillion → 3.80.

**Key lesson:** This bug would never have been found by testing on Qwen alone. The cross-model smoke testing methodology — downloading diverse models and running a quick bench on each — is what caught it. ISWA models were silently broken, producing normal-looking inference speed but nonsensical output. Always test across model families, not just the model you developed on.

## Test Matrix

| # | Model | Type | Family | Size | Head dim | Quant |
|---|-------|------|--------|------|----------|-------|
| 1 | Qwen3.5-35B-A3B | MoE | Qwen | 34GB | 128 | Q8_0 |
| 2 | Qwen3.5-27B | Dense | Qwen | 27GB | 128 | Q8_0 |
| 3 | Mixtral 8x7B Instruct | MoE | Mistral | ~26GB | 128 | Q4_K_M |
| 4 | Gemma 2 27B IT | Dense | Google | ~16GB | 256 | Q4_K_M |
| 5 | Phi-4 | Dense | Microsoft | ~14GB | 128 | Q8_0 |
| 6 | Llama 3.1 70B Instruct | Dense | Meta | ~40GB | 128 | Q4_K_M |
| 7 | Mistral Small 24B | Dense | Mistral | ~14GB | 128 | Q4_K_M |

## Summary

| Model | hd | q8_0 PPL | turbo4 PPL | turbo4 vs q8_0 | turbo3 PPL | Decode | NIAH |
|-------|-----|---------|-----------|---------------|-----------|--------|------|
| Qwen 35B MoE | 128 | 6.11 | 6.13 | +0.23% | 6.18 | 0.93x | ✅ |
| Qwen 27B Dense | 128 | 6.89 | 6.94 | +0.72% | 7.01 | 0.99x | ✅ |
| Phi-4 | 128 | 6.00 | 6.10 | +1.68% | 6.23 | 0.91x | ✅ |
| Mistral Small 24B | 128 | 6.09 | 6.12 | +0.46% | 6.28 | 0.86x | ✅ |
| Gemma 2 27B (ISWA) | 128 | 7.06 | 7.18 | +1.7% | 7.15 | 0.88x | ✅ |
| Llama 3.1 70B | 128 | 2.44 | 2.64 | +8.3% | 2.79 | 0.94x | ✅ |
| Mixtral 8x7B MoE | 128 | — | — | — | — | — | skipped (bad GGUF) |

**Key findings:**
- All hd128 models work across 4 families (Qwen, Meta, Google, Microsoft, Mistral)
- turbo4 consistently closer to q8_0 than turbo3
- Gemma 2 (ISWA) **FIXED** — was missing WHT rotation in ISWA build_attn overload (PPL 13.7T → 3.80)
- Llama 70B shows higher PPL gap (+8.3%) — likely Q4_K_M weight quant stacking with KV quant

## Results

### Model 1: Qwen3.5-35B-A3B Q8_0 (MoE, reference)

Already validated extensively. See README and turbo4-resurrection.md.

| Cache | PPL | Decode tok/s | vs q8_0 |
|-------|-----|-------------|---------|
| q8_0 | 6.1109 | 85.71 | — |
| turbo4 | 6.1250 | 79.87 | 0.93x |
| turbo3 | 6.1756 | 76.84 | 0.90x |

### Model 2: Qwen3.5-27B Q8_0 (Dense)

| Cache | PPL | Decode tok/s | vs q8_0 |
|-------|-----|-------------|---------|
| q8_0 | 6.8884 | 17.17 | — |
| turbo4 | 6.9378 | 17.25 | 1.00x |

### Model 3: Mixtral 8x7B Instruct Q4_K_M (MoE)

⚠️ **GGUF incompatible** — `second-state` repo uses old format, missing `ffn_down_exps` tensor. Need bartowski or official Mixtral GGUF. Skipped.

### Model 4: Gemma 2 27B IT Q4_K_M (Dense, ISWA)

✅ **FIXED** — was PPL 13.7 trillion, now 3.80.

**Root cause:** The ISWA (interleaved sliding window attention) `build_attn` overload was missing turbo WHT Q rotation and V inverse rotation. K/V were rotated in SET_ROWS but Q was unrotated → `rotated_K × unrotated_Q = garbage`. One fix in `llama-graph.cpp`.

**Note:** Gemma 2 K/V head_dim IS 128 (not 256 as initially assumed — GGUF metadata confirms `key_length=128`).

| Cache | PPL (8-chunk) | vs q8_0 | Decode tok/s | NIAH |
|-------|--------------|---------|-------------|------|
| q8_0 | 7.0590 | — | 28.28 | 3/3 |
| turbo3 | 7.1489 | +1.3% | 24.08 | — |
| turbo4 | 7.1794 | +1.7% | 24.79 | — |

Note: turbo4 slightly worse than turbo3 on Gemma 2 — the only model showing this pattern. May be related to ISWA dual-cache interaction or Gemma's logit softcapping. Earlier 1-chunk estimates (3.75-3.80) were unreliable — use 8-chunk numbers above.

This fix also applies to any other ISWA model (Cohere2, OLMo2, Gemma3N).

### Model 5: Phi-4 Q8_0 (Dense)

| Cache | PPL | Decode tok/s | NIAH |
|-------|-----|-------------|------|
| q8_0 | 6.0014 | 33.66 | 3/3 |
| turbo3 | 6.2336 (+3.9%) | 29.82 | 3/3 |
| turbo4 | 6.1024 (+1.7%) | 30.54 | 3/3 |

turbo4 quality advantage holds on Phi-4. +1.7% vs q8_0.

### Model 6: Llama 3.1 70B Instruct Q4_K_M (Dense, large)

| Cache | PPL | vs q8_0 | Decode tok/s | NIAH |
|-------|-----|---------|-------------|------|
| q8_0 | 2.4397 | — | 11.50 | 3/3 |
| turbo4 | 2.6431 | +8.3% | 10.78 | 3/3 |
| turbo3 | 2.7878 | +14.3% | 10.68 | 3/3 |

Larger PPL gap than smaller models. Likely due to Q4_K_M weight quantization stacking with KV quantization — the model weights are already quantized, adding KV quantization compounds the error. turbo4 still significantly better than turbo3.

### Model 7: Mistral Small 24B Q4_K_M (Dense)

| Cache | PPL | Decode tok/s | NIAH |
|-------|-----|-------------|------|
| q8_0 | 6.0946 | 34.52 | 3/3 |
| turbo3 | 6.2792 (+3.0%) | 28.73 | 3/3 |
| turbo4 | 6.1224 (+0.46%) | 29.54 | 3/3 |

Excellent results — turbo4 within 0.5% of q8_0 on Mistral.

## Notes

- Results collected via `scripts/turbo-quick-bench.sh --no-ref`
- PPL: wikitext-2, c=512, 8 chunks
- Decode: llama-bench tg128
- NIAH: 3 positions at 8K (if supported)
- turbo3/turbo4 may fail on models with non-128 head_dim (known limitation, #13)

## TODO

- [ ] Mixtral 8x7B quick bench (model downloaded + verified loading, PPL 3.31 on q8_0)
- [ ] Decode scaling tests: Llama 70B, Gemma 2, Mixtral at 4K/8K/16K/32K context lengths
- [ ] 50-chunk 32K sparse V ON/OFF on Llama 70B (long run, overnight)
- [ ] Investigate Gemma 2 turbo4 outlier (turbo4 slightly worse than turbo3, only model showing this)
- [ ] Non-128 head_dim padding fix (prefer padding over q8_0 fallback, see signalnine #13)
