# Sparse V Upstream Validation Note

## What changed
A single conditional added to the Metal flash attention vec kernel's V accumulation loop. When the post-softmax attention weight for a position is below 1e-6, the V dequantization and accumulation for that position is skipped.

Gated by compile-time preprocessor define `GGML_METAL_FA_SKIP_V`, controlled via environment variable. Default: off.

## Why safe
- The conditional only fires in the quantized V path (dequant branch)
- Threshold 1e-6 is far below meaningful contribution to output
- When disabled (`GGML_METAL_FA_SKIP_V=0`), code path is identical to unpatched
- No changes to attention scores, softmax, normalization, or K computation

## What was tested
- PPL: 32K context, 50 chunks, wikitext-103, q8_0/q4_0/turbo3 — ON/OFF delta 0.0000 in all formats
- PPL: 8K/16K/32K multi-context sweep — ON/OFF delta 0.0000 at all depths
- Decode speed: q8_0 +5% short, turbo3 +22.8% at 32K, q4_0 neutral
- NIAH retrieval: identical or improved across formats
- KL divergence: measured, consistent with underlying quantization
- Threshold ablation: 1e-4 through 1e-8, PPL identical at all values
- Dense model: no regression
- Hardware: Apple M5 Max 128GB

## Supported claims (safe for PR)
- No measurable PPL change when enabled (across tested formats and contexts)
- Decode speed improvement scales with V dequant cost
- Safe to enable for quantized KV cache formats on Metal

## Unsupported claims (do NOT put in PR)
- Universal improvement across all hardware
- CUDA validation (not tested on upstream CUDA path)
- "Improves quality" (NIAH improvement is secondary signal, not primary claim)
- Specific percentage claims beyond tested setup

## Limitations
- Validated on Apple Metal only (M5 Max)
- Threshold is hardcoded (not adaptive)
- Skip rate directly measured on smaller model (1.7B), estimated on larger
- Short context benefit is minimal (attention is dense)
- Dense models see negligible benefit (FFN dominates)
