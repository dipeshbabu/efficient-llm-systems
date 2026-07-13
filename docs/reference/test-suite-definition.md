# TurboQuant Full Test Suite Definition

Standard validation suite for any KV cache format or optimization change. All tests must pass before results are published or code is merged.

## Suite A: Quality (must pass for any change)

1. **PPL — Short context baseline**
   - wikitext-2, c=512, 8 chunks
   - Compare: ON vs OFF vs q8_0 baseline
   - Pass: delta < 0.5% vs baseline

2. **PPL — Long context (primary validation)**
   - wikitext-103, c=32768, 50 chunks
   - Compare: ON vs OFF vs q8_0 baseline vs q4_0 baseline
   - Pass: ON/OFF delta = 0.000 (within CI)

3. **PPL — Multi-context sweep**
   - wikitext-2, c=8192 (20ch), c=16384 (10ch), c=32768 (5ch)
   - Compare: ON vs OFF with q8_0 baseline at each depth
   - Pass: ON/OFF delta = 0.000 at all depths

4. **KL Divergence vs f16**
   - wikitext-2, c=512, 8 chunks
   - Report: mean KLD, delta-p RMS, same-top-p %
   - Compare against q8_0 and q4_0

5. **NIAH — Retrieval accuracy**
   - Single needle: 3 depths (0%, 50%, 100%) × 3 contexts (4K, 8K, 16K)
   - Multi-key with distractors: 4K through 32K
   - Compare: ON vs OFF
   - Pass: no regression (identical or improved)

## Suite B: Performance (must pass for any change)

6. **Decode speed — Short context**
   - llama-bench, tg128
   - Compare: ON vs OFF vs q8_0 baseline

7. **Decode speed — Long context**
   - llama-bench, pp8192+tg128, pp16384+tg128, pp32768+tg128
   - Compare: ON vs OFF vs q8_0 baseline

8. **Prefill speed — Context scaling**
   - llama-bench or llama-perplexity at 2K, 4K, 8K, 16K, 32K
   - Compare: ON vs OFF vs q8_0

## Suite C: Robustness (run for new optimizations)

9. **Dense model validation**
   - Repeat decode speed tests on dense model (Qwen3.5-27B)
   - Confirm no regression on non-MoE architectures

10. **Threshold ablation** (sparse V specific)
    - Sweep τ = 1e-4, 1e-5, 1e-6, 1e-7, 1e-8
    - Confirm PPL identical across all values

11. **Cross-format validation**
    - Run full Suite A + B on each KV format: q8_0, q4_0, turbo3
    - Confirm format-independence

## Suite C+: Ad-Hoc Experiments (formalized)

These tests were developed during research and are now part of the standard suite:

12. **Skip rate direct measurement** (sparse V specific)
    - Run `tools/validation/measure_skip_rate.py` on Qwen3-1.7B with eager attention
    - Report: overall skip rate, per-layer min/max/median, at 512/2K/4K context
    - Validates that sparse V is actually active at target context lengths

13. **Real-world server benchmark**
    - 70-page PDF via llama-server chat completions API
    - Compare turbo3 vs q8_0 prefill + decode at real-world context (~24K)
    - Documents gap between synthetic bench and server workload

14. **Failed experiment log**
    - Track all attempted optimizations (tile skip, f16 sparse V, etc.)
    - Document methodology, results, and reason for rejection
    - Stored in `docs/attention-gated-optimizations.md`

## Suite D: Regression (run before merging external PRs)

12. **turbo3 PPL smoke test**
    - wikitext-2, c=512, 8 chunks
    - Must match known good value (6.1756 for current model)
    - Catches silent quantization path corruption (e.g., PR #4 regression)

13. **Build + basic inference**
    - cmake clean build
    - Short generate test (coherent output)
    - Server start + health check

## Methodology requirements

- Run q8_0 baselines FIRST to confirm corpus/settings are sane before testing variants
- Run all tests sequentially (no GPU contention)
- Kill stale llama processes before starting
- Record: date, hardware, model, branch/commit, exact commands
- Save raw logs to docs/threshold-ablation-logs/

## Hardware

- Primary: Apple M5 Max 128GB
- Secondary: Mac Mini M2 Pro 32GB (via SSH to toms-mac-mini.local)
- Community: CUDA testers (buun RTX 3090, Mario dual 4090, etc.)

## Models

- MoE: ~/local_llms/models/Qwen3.5-35B-A3B-Q8_0.gguf
- Dense: ~/local_llms/models/Qwen3.5-27B-Q8_0.gguf
- Attention inspection: Qwen/Qwen3-1.7B (via transformers)

## Corpora

- wikitext-2: ~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw (1.3MB)
- wikitext-103: ~/local_llms/llama.cpp/wikitext-103-raw/wiki.train.raw (516MB)
