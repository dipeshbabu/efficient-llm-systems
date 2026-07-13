# Turbo4 Systematic Profiling Plan

Goal: Explain exactly WHY turbo4 is worse than turbo3, with numbers at each level. Same rigor as the 14 decode experiments that led to sparse V.

## Phase 1: Component Isolation (Metal, M5 Max)

Strip turbo4 dequant layer by layer, measure each:

| Experiment | What's disabled | Measures |
|-----------|----------------|----------|
| A. Full turbo4 (baseline) | Nothing | Total cost |
| B. QJL disabled | `signs_f[i] * qjl_scale` zeroed | QJL correction cost |
| C. QJL + rotation disabled | Skip inverse WHT in dequant | Rotation cost |
| D. No-op dequant | Return zeros | Dequant floor (memory only) |

For each: PPL + decode tok/s + prefill tok/s

Expected: B should match turbo3 speed. C should be faster. D gives the floor.

## Phase 2: Block Size Effect

| Experiment | Config | Measures |
|-----------|--------|----------|
| E. turbo3 (4x32 blocks) | Current turbo3 | Reference |
| F. turbo4 PolarQuant-only (1x128 blocks) | turbo4 with QJL disabled | Block size overhead |

Same compression (3-bit PolarQuant), different block layout. Isolates the 128 vs 32 block size effect on FA tiling and cache locality.

## Phase 3: QJL Noise Accumulation Curve

PPL at increasing context lengths to chart degradation:

| Context | turbo3 PPL | turbo4 PPL | turbo4 no-QJL PPL | QJL noise |
|---------|-----------|-----------|-------------------|-----------|
| 512 | ? | ? | ? | ? |
| 2K | ? | ? | ? | ? |
| 8K | ? | ? | ? | ? |
| 16K | ? | ? | ? | ? |
| 32K | ? | ? | ? | ? |

buun's CUDA data shows: +0.19% at 2K, -0.16% at 32K, +3.69% at 64K.
We need the Metal curve to compare.

## Phase 4: Prefill Deep Dive

turbo4 prefill is catastrophically slow (buun: 51.9% of q8_0 on CUDA).

Root cause candidates:
1. **nl=8 vs nl=2** — non-vec FA kernel calls dequant 8x per block (turbo4) vs 2x (turbo3). 4x more dequant invocations.
2. **128-element dequant** — each dequant call processes 16 elements from a 128-element block, requiring full block read every time.
3. **Full inverse WHT per dequant call** — the block cache should prevent this, but verify.
4. **QJL matrix multiply in dequant** — even the broken version adds overhead.

Test: profile prefill at 8K with turbo4 vs turbo3, log time per operation.

## Phase 5: Recommendations

Based on profiling results, determine:
1. Is turbo4-without-QJL worth keeping? (Same quality as turbo3, potentially different speed characteristics)
2. Should turbo4 become 4-bit PolarQuant? (16 centroids, no QJL, should beat turbo3 quality)
3. Should turbo4 be deprecated entirely? (buun's recommendation)
4. Is asymmetric K/V (turbo3 K, turbo4 V) worth pursuing?

## Cross-Reference

- buun's CUDA data: `threshold-ablation-logs/turbo4_cuda_buun_data.txt`
- scos-lab QJL finding: turboquant_plus issue #45
- Our QJL ablation: `threshold-ablation-logs/turbo4_qjl_ablation.txt`
- Our SET_ROWS fix: `threshold-ablation-logs/turbo4_setrows_fix_results.txt`
- 7 bugs found: `turbo4-investigation.md`

## Hardware

- Primary: Apple M5 Max 128GB
- Cross-reference: buun's RTX 3090 24GB data
- Model: Qwen3.5-35B-A3B Q8_0 (MoE) — same as all prior benchmarks
