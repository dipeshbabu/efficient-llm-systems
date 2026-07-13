# Block-Selector Sparse Attention on Apple Silicon — A Work-in-Progress Log

**Dipesh Tharu Mahato**
Independent Researcher
GitHub: [@dipeshbabu](https://github.com/dipeshbabu)

---

## Abstract

This is an in-progress log of an ongoing sprint on a second lever for the long-context dequant bottleneck described in our companion paper [sparse-v-dequant.md](sparse-v-dequant.md). Where that work skipped value dequantization *after* softmax for negligible attention weights, this work skips entire key positions *before* the SDPA call by means of a learned per-(slot, KV-head) block selector. The selected block list is rendered into a fused mask and handed to MLX's tuned `sdpa_vector_2pass` kernel, so the only custom code in the hot path is the selector and a Metal mask builder; everything else rides upstream MLX.

The work-in-progress here lives on the `feature/retrieval-attention` branches of the historical `mlx-swift-lm` and `vllm-swift` prototypes ([public source URLs unavailable](../../docs/reference/historical-forks.md#swift-and-long-context-prototypes)). It is **not merged** to the upstream ekryski-alpha master, and the four relevant pull requests — `ekryski/mlx#34` (MetalAllocator zero-on-recycle; the `MLX_SDPA_BLOCKS` env override was independently merged into upstream alpha during the sprint and was dropped from this PR during rebase), `ekryski/mlx-swift#36` (submodule bump to mlx#34's cleaned tip), `ekryski/mlx-swift-lm#224` (batched sparse decode), and `ekryski/mlx-swift-lm#225` (sparse prefill, stacked on #224) — are still open. The numbers below were measured locally on a single M5 Max in May 2026; they have not been independently validated and the cliff-mitigation work is still being characterized cell-by-cell.

What the sprint has produced so far: a per-(B, KV-head) selector index that scales linearly with batch size, a batched sparse decode path that beats dense by 1.16-1.73× on Qwen2.5-14B-Instruct-1M-4bit across a 19-cell long-context grid (B=4-8 × ctx=16K-48K), a sparse prefill path that lands 2.9× faster prefill at 128K context on the same model, and an extension pattern by which the same plumbing was ported across ten model families (Qwen2, Qwen3, Qwen3MoE, Llama, Gemma3, Gemma4, Phi3, Mistral3, and the two hybrid attention+SSM families). The sprint also produced a longer list of failed paths than wins: a custom Metal sparse SDPA kernel that lost 14× at B=8, a compose-gather batched path that lost 13.7×, a bool-mask kernel variant that regressed 25%, and a per-query top-K decode design that never beat union top-K. The Apple-Silicon-specific findings — a mis-tuned MLX block heuristic, a pass-2 reducer modulo bug, and an allocator non-determinism cliff at long context — are the substantive output of the sprint and may be the most useful pieces to other practitioners working on the same hardware.

The framing of this document is a sprint journal, not a results paper. Numbers are preliminary, the design space is still being explored, and the recommendations at the end are "current best guess" rather than settled defaults.

---

## Status

This section is intentionally near the top because the rest of the document only makes sense given which pieces are merged and which are not.

| Component | Branch | Upstream PR | State |
|---|---|---|---|
| MetalAllocator zero-on-recycle | `feature/retrieval-attention` | `ekryski/mlx#34` | open, not merged |
| `MLX_SDPA_BLOCKS` env override | independently merged into `ekryski/mlx:alpha` during sprint | (was bundled in `ekryski/mlx#34`, dropped during rebase as redundant) | upstream-native in alpha |
| Pass-2 reducer modulo finding | local — worked around by constraining `MLX_SDPA_BLOCKS` to multiples of 32 | not yet PR'd | finding only; no upstream fix |
| Submodule bump to mlx#34 cleaned tip | `feature/retrieval-attention` | `ekryski/mlx-swift#36` | open, not merged |
| Batched sparse decode (per-family ports) | `feature/retrieval-attention` | `ekryski/mlx-swift-lm#224` | open, not merged |
| Sparse prefill (stacked on #224) | `feature/retrieval-attention` | `ekryski/mlx-swift-lm#225` | open, not merged |
| Bridge auto-`setenv` at engine create | `feature/retrieval-attention` (vllm-swift) | included in #224 | open, not merged |
| Per-family `+Sparse.swift` extensions | `feature/retrieval-attention` | included in #224 + #225 | open, not merged |
| `BatchedRetrievalAttentionKVCache` + `rawKeyMode` compose with TurboQuant+ | partial | deferred from #224 | TODO — see §8 |

Nothing in this document has been validated against the post-merge state of the alpha master; reviewers should expect the merge to surface integration issues the local branch has not yet hit.

---

## 1. Why two levers, not one

Our prior paper, [Attention-Gated Value Dequantization for Quantized KV Cache Inference](sparse-v-dequant.md), targeted the dequant bottleneck from the inside: a single conditional in the V dequant loop that skips positions whose attention weight has already been computed to be below a threshold. That work was a wholly post-softmax optimization. It needed the K dequant and full Q·Kᵀ computation to know which V positions were skippable.

This work targets the same bottleneck from the other side. If we knew, *before* SDPA, which K positions a query was going to attend to, we could skip the K dequant and the Q·Kᵀ work for those positions entirely. We cannot know the exact distribution that cheaply — that would be the SDPA call itself — but we can build a coarse selector that picks the right K *blocks* at low cost, and we can then hand the selected block list to a fused mask kernel and ride MLX's tuned `sdpa_vector_2pass` kernel for the actual attention. The two levers compose: a query-aware block-selector skips bulk K positions before SDPA; the post-softmax sparse-V skip catches the per-position residue inside SDPA's V phase.

Whether the compose actually realizes the additive saving at the kernel level is still open. The companion paper measured sparse-V at +22.8% decode at 32K on llama.cpp/Metal; the block-selector work in this document was measured on MLX-Swift, which is a different stack. The kernel-level integration for the two-lever stack is deferred — see §8.

---

## 2. Background

### 2.1 Block-selector sparse attention literature

The architectural pattern is well-trodden on the CUDA side. NSA (Native Sparse Attention)[^nsa] introduced GQA-group co-loading where one threadgroup serves all Q heads in a KV group, sharing one block fetch from K/V. SeerAttention-R[^seer] and DuoAttention[^duo] explored learned vs heuristic block selectors. Quest[^quest] is the closest ancestor for what this work does at decode time: a query-aware block selector that picks K blocks at decode-step granularity, with the selector itself a low-rank projection. All of these papers report wall-clock wins on NVIDIA hardware, where the FlashAttention sparse kernel is mature.

Our finding (preliminary, see §4) is that the *algorithmic* pattern transfers cleanly to Apple Silicon, but the *kernel pattern* does not. Custom Metal sparse SDPA kernels lose to MLX's tuned `sdpa_vector_2pass` with a float mask, by a wide margin. The kernel choice is the load-bearing Apple-specific decision.

### 2.2 MLX-Swift KV cache layout

The relevant cache surface in MLX-Swift is `BatchedKVCache`, which stores K and V as rectangular `[B, nKVHeads, T_max, dHead]` tensors with a per-slot `offset` array tracking how much of the rectangular region is populated. Decode at L=1 writes one new K row and one new V row per slot, advances `offset[b]`, and then runs SDPA over the populated prefix `[B, nKVHeads, ..<offset, dHead]`. This is the shape MLX `sdpa_vector_2pass` is tuned for.

The sparse cache, `BatchedRetrievalAttentionKVCache` (file: `/Users/dipesh/dev/mlx-swift-lm/Libraries/MLXLMCommon/BatchedRetrievalAttentionKVCache.swift`), wraps a `BatchedKVCache` via composition rather than inheritance — the wrapper holds the inner cache (`let inner: BatchedKVCache`) plus the selector index (`let index: BatchedRetrievalAttentionIndexB`). All writes pass through the inner cache unchanged; the wrapper only adds index updates and a `sparseAttend(queries:scale:)` entry point that the model can call instead of running dense SDPA.

### 2.3 TurboQuant+ rawKeyMode contract

The longer-running TurboQuant+ stack (see `sparse-v-dequant.md` §2 for the WHT + polar quantization recipe) compresses K and V to 3.5-bit per element. For block-selector sparse decode to compose, the selector needs raw K to compute block features. The `rawKeyMode` contract — implemented in single-stream form at `RetrievalAttentionKVCache.swift` lines 97-253 — keeps K stored as raw fp16 while V is independently quantized. The selector reads raw K to compute block scores; the SDPA call dequantizes only the *selected* V blocks. This combines the two compression axes in principle. The implementation is currently single-stream only; the batched cache has not yet wired in the rawKeyMode compose.

---

## 3. Architecture (current state)

### 3.1 Selector index with explicit batch dimension

The selector index lives in `BatchedRetrievalAttentionIndexB.swift`. Internally:

- A Johnson-Lindenstrauss projection matrix `jlMatrixT: [dHead, contentDim]`, shared across all `(B, KV-head)` since the JL projection is a model-architecture property, not a per-slot property.
- Per-(B, KV-head) block-pooled features, stored as `[B, nKVH, blockCap, contentDim]`. Buffer is grown geometrically; valid prefix is `[0..<currentFineBlocks]`.
- Per-(B, KV-head) running sums for partial-block accumulation, so adding new tokens just bumps the running sum + count rather than re-scanning the full T.

The memory decision here was load-bearing. The single-batch `BatchedRetrievalAttentionIndex` (the prior version, which batched across KV heads only) kept a `perTokenFeatures: [nKVH, T, D_eff]` buffer for re-pooling on every block boundary. Scaled to B=8 at T=128K, that buffer alone is `[8, 8, 131072, 32] × int16 × 40 layers ≈ 20 GB`, which is unusable on a 128 GB M5 Max running a 14B model. The B-dim index drops `perTokenFeatures` entirely and maintains block features incrementally — the running-mean update is:

$$\text{mean}_{\text{after}} = \frac{\text{mean}_{\text{old}} \cdot n_{\text{old}} + \sum \text{new}}{n_{\text{new}}}$$

Same arithmetic result, no T-sized intermediate, ~20 GB recovered.

### 3.2 Selector projection

For each decode step, queries `[B, nQH, 1, D]` get reduced to per-(B, KV-head) representatives by taking head index `kvh * groupSize` from each KV group (file: `BatchedRetrievalAttentionKVCache.swift:128-134`). Then:

$$\hat{q}_{b, kvh} = W_{JL}^T \cdot q_{b, kvh}, \quad \hat{q} \in \mathbb{R}^{B \times nKVH \times d_{\text{eff}}}$$

The same JL projection was applied earlier to every block's pooled K features. Top-K block selection is a small argpartition over `<projectedQ, blockFeatures>` per (B, KV-head). For typical config (`contentDim=32`, fine block size 64), the per-step selector cost is small relative to SDPA.

The selector is intentionally not learned per-task or per-prompt; the JL matrix is initialized once and reused. F-46 (in our internal log) found that adding a positional term (`lambdaPos > 0`) hurt cross-task generality, so the ship config is `lambdaPos = 0.0`, content-only.

### 3.3 Fused mask kernel + MLX float-mask SDPA

The default kernel choice (`VSM_SPARSE_BATCHED_KERNEL=f73`) is the one we have currently found least-broken on Apple Silicon. Given per-slot per-KV-head block start indices `[B, nKVH, K_blocks]`, a Metal kernel renders an fp16 additive mask `[B, 1, 1, T]` where positions outside the selected blocks (and outside the static-prefix + sliding-window region) get `-∞`. The mask is broadcast over Q heads inside MLX. The actual SDPA call is:

```
MLXFast.scaledDotProductAttention(
    queries:  q,    // [B, nQH, 1, D]
    keys:     k,    // [B, nKVH, T, D]
    values:   v,    // [B, nKVH, T, D]
    scale:    1/sqrt(D),
    mask:     .array(fp16Mask)
)
```

which dispatches into MLX's `sdpa_vector_2pass` `_floatmask` kernel variant.

The kernel is *not* doing what naive sparse-attention literature calls "sparse SDPA" — there is no actual K skipping at the load level. The float mask still streams the full K/V buffer; `-∞` positions just kill the softmax contribution. The savings are compute-side (softmax + output projection on a smaller effective set) and dispatch-side (one large mask launch amortizes across nQH Q-heads via broadcast). This is exactly the architectural lever the F-71b custom kernel was trying to replace with real K-skipping; that experiment failed (§5).

### 3.4 Bridge wiring

`vllm-swift/swift/Sources/VLLMBridge/Bridge.swift` has a `buildBatchedSparseCaches` entry point that the runtime calls at engine creation time when the env path is set (`VSM_SPARSE=1 VSM_SPARSE_BATCHED=1`). It walks each per-layer cache, builds the wrapping `BatchedRetrievalAttentionKVCache`, migrates the existing K into the inner cache (one `eval` per layer — see §5.4 for why that one-line discipline matters), and sets the selector index up for the post-prefill state.

The Bridge also `setenv("MLX_SDPA_BLOCKS", "128", 1)` at engine create when the cliff conditions are met. This is the auto-mitigation discussed in §4.1. There is a B*T threshold gate (`B * commonT > 224_000` → fall back to dense) so that the catastrophic cliff cells don't ship — see §4.6.

---

## 4. Apple Silicon Findings

This section is the substantive part of the document. The findings are the things that would have been obvious if we had access to the right kernel-level documentation, and were not obvious from outside.

### 4.1 The MLX block heuristic is mis-tuned for the float-mask path

MLX's `sdpa_vector_2pass` kernel splits the K dimension into `blocks` partitions and runs a parallel reduction. The number of partitions is picked by a hardcoded heuristic in `mlx/backend/metal/scaled_dot_product_attention.cpp` (lines 444-477):

```cpp
char devc = d.get_architecture().back();
int N = k.shape(2);
int blocks;
if (devc == 's') {
  blocks = 64;
  if (N > 1024 && n_simds > 4) {
    if (N <= 8192)       blocks = 128;
    else if (N <= 32768) blocks = 256;
    else if (N <= 65536) blocks = 512;
    else                 blocks = 1024;
  }
}
```

`devc == 's'` is the M-series ARM (`applegpu_g17s` on M5 Max). The heuristic was tuned in upstream PR #3023 — but the tuning was done against the `_nomask` kernel variant, not the `_floatmask` variant we hit on the sparse path. The `_floatmask` kernel has different occupancy characteristics: it does an extra mask load per K row and an extra softmax-killing comparison, which changes the optimal block count.

Concretely, at our F-85 sparse cell `B=8 nKVH=8 gqa=5 N=30720`:

- M5 Max has 40 GPU cores × ~24 resident simdgroups = ~960 concurrent simdgroups.
- At `blocks=256` the kernel demands `8 × 8 × 256 × 5 = 81,920` simdgroups = 85× oversubscription.
- A reasonable target is `blocks ≈ ceil(960 × 1.1 / (B × nKVH × gqa))` = `ceil(2.6)` → snap to the next valid value (see §4.2 for which values are valid). For this cell that's `blocks=128`.

Measured at B=8 ctx=30K, Qwen2.5-14B-Instruct-1M-4bit (preliminary, local):

| `MLX_SDPA_BLOCKS` | Sparse tok/s | Ratio vs dense (50.6 tok/s) |
|---:|---:|---:|
| 32 | 44.7 | 0.88× |
| 64 | 39.1 | 0.77× |
| 128 | **46.6** | **0.92×** (peak) |
| 192 | 34.0 | 0.67× |
| 256 (default) | 32.8 | 0.65× (oversub) |

The env override `MLX_SDPA_BLOCKS=128` recovers +42% (32.8 → 46.6 tok/s) at this cell. The upstream PR that added the env override is `ml-explore/mlx#3455`; we backported it locally to the mlx-swift fork at `scaled_dot_product_attention.cpp:478-480` and have the Bridge auto-`setenv` at engine create time to land the auto-mitigation transparently.

Dense at the same cell is largely unaffected: `MLX_SDPA_BLOCKS=128` gives dense `51.5 tok/s` vs default `50.6 tok/s` — within noise. The `blocks=128` choice is safe globally; it benefits sparse and is a wash for dense.

### 4.2 Pass-2 reducer has a modulo bug

While bisecting the `MLX_SDPA_BLOCKS` value, we observed that *not all values produce correct output*. Setting `blocks=8` produced garbage (cosine dropped to ~0 vs reference); `blocks=88` also produced corrupt output even though it ran without crashing. Walking the `sdpa_vector_2pass_2` kernel exposed the cause: the pass-2 reducer loops

```
for (int b = 0; b < blocks / BN; ++b) { ... }
```

with `BN = 32`. Any `blocks` value not a multiple of 32 silently drops `blocks % 32` partials. At `blocks=88`, 24 partials get dropped; the output is plausible-looking but quantitatively wrong.

This means the *only* valid values are multiples of 32: `{32, 64, 96, 128, 160, 192, 224, 256, ...}`. The PR #3455 patch as-shipped advertised `blocks=88` as the M4 Ultra default, which would also be silently corrupting output on the float-mask path.

The fix is one line — round `blocks` up to the next multiple of `BN`. The fix is in our local fork. The upstream PR (`ekryski/mlx#36`) is open; we have not pushed to mainline MLX per the local-only constraint on this stack.

### 4.3 Custom Metal sparse SDPA kernels lose at B=8

The most expensive negative result of the sprint. The F-71b kernel (`retrievalAttentionGroupSparseSDPA`) is a hand-rolled Metal kernel that does real K skipping: given a per-(B, KV-head) gather list, it loads only the selected K rows. On paper this is the right kernel for batched sparse — it eliminates the bandwidth penalty of the float-mask path.

Measured at B=8 ctx=32K Qwen2.5-14B-Instruct-1M-4bit (local, preliminary):

| Path | tok/s | per-step ms |
|---|---:|---:|
| Dense B=8 | 49.3 | 162 |
| **F-71b batched sparse** | **3.5** | **2260** |
| Serial B=1 sparse (F-73 mask) | 29.9 | 33.5 |

The F-71b path was 14× slower than dense and 67× slower per-slot than serial B=1 sparse.

Root cause analysis: F-71b's grid is `B × nQH` threadgroups × 32 simdgroups per threadgroup. At B=1, nQH=40: 40 TGs × 32 simdgroups = 1280 simdgroups requested. At B=8: 320 TGs × 32 simdgroups = 10240 simdgroups requested. M5 Max has ~960 resident simdgroups. So:

- B=1: kernel saturates the GPU, 1.3× under-saturation.
- B=8: 21× more work but only 3× more concurrency headroom → 7× serialization on top of the work increase.

The literature wins (NSA, SeerAttention-R, DuoAttention) all target CUDA, where FlashAttention's sparse kernel is mature and grid co-location has been tuned across NVIDIA generations. On Apple Silicon, the kernel shape that wins (`per (B, KV-head)` threadgroup with GQA co-load, akin to F-71c if we wrote it) is at minimum a multi-day Metal kernel rewrite with no guarantee of a clean win after PSO tuning.

The honest recommendation from this experiment: **don't write custom Metal sparse SDPA kernels on Apple Silicon.** Ride MLX's `sdpa_vector_2pass` with a float mask. The float-mask path doesn't actually save K bandwidth, but it amortizes the per-Q-head dispatch and rides the heavily-tuned upstream kernel; that's enough to win at the cells we tested.

### 4.4 Bool mask loses to fp16 additive mask

A natural-looking optimization is the `_boolmask` MLX kernel variant: instead of building an fp16 `-∞` additive mask, build a uint8 valid mask. Per-row in the kernel, that's a 1-byte load + branch instead of a 2-byte fp16 load + fp-add.

Predicted +3-25% from the per-row-cost model. Measured **-25%** at B=8 ctx=30K with `MLX_SDPA_BLOCKS=128`:

| Mask variant | tok/s |
|---|---:|
| fp16 additive mask | 46.1 |
| bool valid mask (F-86 experiment) | 34.6 |

Likely reasons: M5 Max's `_boolmask` pipeline is less PSO-tuned than `_floatmask` (PR #3023 was almost certainly benched on `_floatmask`); the bool branch in the inner loop may stall simd execution; PSO compile-time and warmup differ. We reverted the change.

The lesson: don't switch MLX kernel variants based on per-row-cost analysis alone. Apple Metal pipelines have hidden PSO-specific tuning that does not transfer across kernel-variant boundaries. End-to-end measurement is the only honest path.

### 4.5 MetalAllocator non-determinism at long context

This finding belongs to the F-80 thread, which is upstream of the F-85 sparse decode work but worth surfacing here because it shows up in any long-context measurement: at ≥32K context, two identical model forwards (same seed, same prefill, same decode prompt) produce divergent logits. On Qwen3-0.6B-4bit the cliff is at exactly T=32768, cosine 1.0 → 0.

Definitive isolation traced this to `MetalAllocator::malloc` calling `BufferCache::reuse_from_cache` and returning recycled pool buffers without zeroing their contents. Some MLX kernel reads from an uninitialized region of a recycled buffer at long-context decode; the stale data differs between runs depending on prior allocation patterns, producing cross-run logit divergence.

The two-line fix lives at `mlx/backend/metal/allocator.cpp:118-131`:

```cpp
std::unique_lock lk(mutex_);
MTL::Buffer* buf = buffer_cache_.reuse_from_cache(size);
// Zero recycled buffers before returning them to the caller. Stale
// contents from prior allocations can produce cross-run non-determinism
// at long-context decode when downstream kernels read regions they did
// not explicitly initialize.
if (buf) {
  memset(buf->contents(), 0, size);
}
```

Result: 49K dense-vs-dense decode cosine **0.34 → 1.0** (bit-exact). F-79 ship regression unchanged (8/8 argmax match, cosine 0.99882).

This is the most upstream-friendly piece of the work — a small targeted memset on a definitively-traced bug. The PR is `ekryski/mlx#34` (bundled with the `MLX_SDPA_BLOCKS` env override); it is still open.

### 4.6 The B*T cliff

Putting §4.1 and §4.2 together explains a sharp asymmetric cliff we observe in the F-73 batched mask path. At cells where `B × T ≤ 224K`, F-73 batched beats dense by 1.14-1.41×. At cells where `B × T ≥ 256K`, F-73 collapses while dense degrades smoothly.

Preliminary measurements:

| B | T | B*T | Dense tok/s | F-73 tok/s | Ratio |
|---:|---:|---:|---:|---:|---:|
| 4 | 16K | 64K | 71.7 | 82.1 | 1.14× |
| 8 | 16K | 128K | 73.4 | 95.0 | 1.29× |
| 2 | 64K | 128K | 26.5 | 35.2 | 1.33× |
| 4 | 32K | 128K | 48.0 | 62.5 | 1.30× |
| 8 | 20K | 160K | 63.3 | 89.0 | **1.41×** |
| 8 | 24K | 192K | 56.5 | 74.9 | 1.33× |
| 8 | 28K | 224K | 49.9 | 64.6 | 1.29× |
| 8 | 32K | 256K | 42.5 | 27.8 | **0.65×** (cliff) |
| 2 | 128K | 256K | 15.6 | 1.4 | **0.09×** (catastrophic) |

The pattern is `B × T`, not `B` alone and not `T` alone — same cliff at B=2 T=128K and B=8 T=32K. We have not fully diagnosed the cliff; the working hypothesis is that the `MLXFast.SDPA(mask=.array)` dispatch path switches kernel variants at some mask-buffer-size threshold (mask shape `[B, 1, 1, T]` = `B*T` elements), or that threadgroup memory pressure crosses an SM cap. The shipping mitigation is a `B * commonT > 224_000` gate in Bridge.swift that falls back to dense decode when above the threshold.

Auto-tuned per-shape blocks (rather than a flat `blocks=128`) likely recovers some of the cliff. Preliminary: at B=2 ctx=128K with `MLX_SDPA_BLOCKS=128` we recover from 1.4 tok/s to 8.4 tok/s (0.09× → 0.54×), still a loss but a 6× cliff recovery. The formula `blocks ≈ ceil(960×1.1/(B × nKVH × gqa))` predicts `blocks=16` at that cell, but 16 is invalid (§4.2); the nearest valid value is 32. We have not finished the bisect at all cliff cells.

---

## 5. Bench grid (preliminary)

All numbers in this section are local measurements on a single Apple M5 Max (128 GB unified memory). The model is `Qwen2.5-14B-Instruct-1M-4bit`. The sparse path is the F-73 batched mask + MLXFast SDPA path described in §3.3 with `MLX_SDPA_BLOCKS=128` set via the Bridge auto-mitigation. The bench harness is `vllm-swift/scripts/bench_throughput.py`. Numbers below are decode tok/s after the prefill completes; prefill timing is reported separately.

### 5.1 Decode + prefill grid (19 cells)

| B | ctx | Dense prefill | Dense decode | Sparse prefill | Sparse decode | **Decode ratio** | **Prefill ratio** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16K | 14.3s | 38.8 | 12.9s | 39.4 | 1.02× | 0.90× |
| 2 | 16K | 29.0s | 56.4 | 27.1s | 58.4 | 1.04× | 0.94× |
| 2 | 20K | 48.9s | 51.5 | 36.3s | 55.2 | 1.07× | 0.74× |
| 4 | 16K | 58.4s | 65.8 | 55.7s | 78.5 | **1.19×** | 0.95× |
| 4 | 24K | 123.2s | 53.5 | 99.0s | 62.3 | **1.16×** | 0.80× |
| 4 | 32K | 192.0s | 39.0 | 145.8s | 53.5 | **1.37×** | 0.76× |
| 4 | 48K | 347.1s | 29.6 | 267.7s | 43.7 | **1.48×** | 0.77× |
| 8 | 16K | 126.0s | 59.2 | 102.1s | 97.7 | **1.65×** | 0.81× |
| 8 | 20K | 166.4s | 56.0 | 138.5s | 89.0 | **1.59×** | 0.83× |
| 8 | 28K | 342.2s | 44.0 | 220.2s | 76.2 | **1.73×** | 0.64× |
| 8 | 32K | 431.1s | 35.2 | 267.3s | 45.1 | 1.28× | 0.62× |

The peaks (and what we currently anchor on):

- **Best decode ratio**: B=8 ctx=28K = 1.73× (44.0 → 76.2 tok/s).
- **Best prefill saving**: B=8 ctx=32K = 1.62× faster (164 seconds saved, ~38% of total prefill wall time).
- **Long-context multi-stream sweet spot**: B=4 ctx=48K = 1.48× decode + 1.30× prefill.
- **Single-stream (B=1)**: parity. The selector cost is real and is amortized over the batch dimension; at B=1 there is no batch to amortize over.

These are preliminary numbers on one machine with one model. They have not been independently validated and they should not be cited as established results yet.

### 5.2 Sparse prefill alone

The prefill column above shows the combined effect; the standalone sparse prefill numbers (B=1) are:

| ctx | Dense prefill | Sparse prefill | Ratio | E2E speedup (20 decode tokens) |
|---:|---:|---:|---:|---:|
| 16K | 13.5s | 12.1s | 1.12× | 1.13× |
| 32K | 36.0s | 25.5s | 1.41× | 1.39× |
| 64K | 137.1s | 55.4s | **2.49×** | **2.49×** |
| 128K | 382.2s | 131.6s | **2.90×** | **2.87×** |

The 128K prefill win is the largest single result of the sprint. The sparse prefill design (described in §6) was implemented but disabled by default until 2026-05-16 when we wired the `VSM_SPARSE_PREFILL=1` env hook through Bridge. The decode-crossover at 128K (when the prefill saving outweighs the per-step decode cost) is ~17K output tokens, well within typical agentic workflows.

The decode per-step cost at the same long-context cells is -3% to -16% slower than dense — the F-73 mask path overhead is unchanged at B=1; the prefill is the lever there.

### 5.3 Mapping to realistic serving regimes

| Workload pattern | Cell | Sparse vs dense |
|---|---|---|
| Single-turn user, moderate context | B=1 ctx=16K | parity (1.02×) |
| 2 MCP clients (Claude + OpenCode) | B=2 ctx=16-20K | 1.04-1.07× |
| 4 concurrent sessions | B=4 ctx=16-24K | 1.16-1.19× |
| Multi-turn rescue pile-up | B=4 ctx=32-48K | **1.37-1.48×** |
| 8-agent farm concurrency | B=8 ctx=16-28K | **1.59-1.73×** |
| Heavy long-ctx batch | B=8 ctx=32K | 1.28× (post-cliff fix) |

The "where this helps" envelope is the moderate-batch / moderate-to-long context cell. The B=1 single-stream case is where TriAttention V3 (see companion paper [triattention-v3.md](triattention-v3.md)) is the right tool — different lever (eviction, not selection); see §7 on how they compose.

### 5.4 One leak fix that gates everything else

A precondition for any of the above numbers shipping: at `Bridge.swift`'s `buildBatchedSparseCaches` entry point, the K/V migration into the inner cache must `eval(batched.keys, batched.values)` *per layer*. Without that one line, MLX's lazy graph accumulates all 48 layers' K/V migration + fp32 cast into a single deferred computation. Peak RSS at engine create blew up ~10× and triggered a system-level crash mid-bisect on 2026-05-17. The fix is in `vllm-swift` commit `8a5bf88`. Calling it out here because the symptom looked like a memory leak in the sparse cache (it was not — it was a forcing-evaluation discipline in the migration path).

### 5.5 Quality validation status

Unit-level equivalence is clean:

- F-85 v2 mask-build kernel byte-equal across batched vs per-slot loop paths (max diff 0.0).
- SDPA-with-mask output numerically equal across batched vs loop paths (max diff < 1e-4 at fp32 reference).
- Selector populates with real top-K rather than placeholder `blocks[0..k-1]` after index migration.

End-to-end retrieval validation is still open. Our existing needle-in-a-haystack harness `scripts/test_f85v2_batched_quality.py` is broken as an oracle: it fails the dense baseline at B=2 across ctx 1K-16K (model hallucinates regardless of sparse path on the synthetic prompt). We have not yet built a clean end-to-end quality validation for the batched sparse path — currently we lean on unit-level numerical equivalence to the dense+mask reference plus the broader RA stack's existing 32K cosine validation (0.99995 at adaptive top-K vs dense).

The sparse prefill quality is validated at 32K via `RetrievalAttentionTests.swift::f83_sparsePrefillQuality32K_14B1M`: random-token prompt, chunked feed at `chunkSize=1024`, sparse vs chunked-dense cosine = 0.999.

---

## 6. Per-Family Generalization

The selector index, batched cache, and Bridge wiring are model-agnostic. The model-specific work is a per-family extension file that adds `fullyBatchedSparseDecode` / `fullyBatchedSparsePrefill` plus a per-layer `Block.fullyBatchedSparseForward` hook. The pattern is additive — extending the existing model class, no surgery on the dense path.

Ten model families currently have ports on the `feature/retrieval-attention` branch:

| Family | Notes | Commit |
|---|---|---|
| Qwen2 | Reference port; the F-85 work was done here first | mlx-swift-lm `8a6a231` |
| Qwen3 | Adds Q/K RMSNorm pre-RoPE preserving the Qwen3-specific ordering + bf16 upcast for unquantized | `3186262` |
| Qwen3MoE | Direct mirror of Qwen3 dense attention; the MoE FFN routing is orthogonal to sparse decode | `4908346` + `a0606b0` |
| Llama | Qwen2-pattern port (no Q/K norm) + `LlamaConfiguration.resolvedHeadDimensions` for Llama-2/Mistral-v0.1 derived head_dim | `793f40c` + `ca04df2` |
| Gemma3 | Inherits Gemma4 plumbing; partial-RoPE handled at the rope init | (bundled w/ Gemma4) |
| Gemma4 | Per-layer dispatch: global → sparseAttend, sliding → dense via inner BatchedKVCache + sliding mask; KV-shared layers (e2b `num_kv_shared_layers=20`) route to `fullyBatchedSharedKVForward`; `attentionKEqV` + v_norm handled | `955c085` + `34a6f68` + `1d8d290` |
| Phi3 | Fused qkv_proj split; `partialRotaryFactor` baked into ropeDim; LongRoPE (SuScaledRoPE) same callable surface as standard; `tieWordEmbeddings` handled | `6768376` + `b5ca942` |
| Mistral3 | Llama-4 attnScale threaded post-RoPE; sliding/full mask dispatch per `layer.useSliding`; defensive hetero-layer skip for V3 | `6768376` + `ab1ae4c` |
| Qwen3.5 (hybrid attention+GDN) | Per-layer-type dispatch — sparse attention layers, leave GDN/Mamba alone; current state partial — see project memory `project_f85_hybrid_family_scope.md` | (WIP) |
| NemotronH (hybrid attention+Mamba2) | Same per-layer-type dispatch shape as Qwen3.5; deferred | (WIP) |

The hybrid families are the open work. Pure-attention families share enough plumbing that the per-family port is in the ~half-day range (the long pole is the per-family `+Sparse.swift` extension test + live smoke). Hybrid families need a layer-type-aware dispatcher (sparse only fires on attention layers; GDN/Mamba layers run their normal path) and a Bridge composition that constructs the right mix of `BatchedKVCache` and SSM state per layer. Multi-day estimate; deferred per Dipesh's call.

The `BatchedSparseLLM` protocol (`vllm-swift` commit `2915679`) makes this composable: a model conforms to `BatchedSparseLLM` by exposing `fullyBatchedSparseDecode(_:caches:)`; Bridge's sparse paths cast to the protocol rather than to a concrete model class, so new families slot in by conformance.

---

## 7. Compose with sliding-window eviction (TriAttention V3)

Our prior paper [TriAttention V3](triattention-v3.md) describes a separate KV-memory mechanism: per-token eviction scored by a trigonometric phase-alignment formula. The two mechanisms target overlapping but not identical use cases — V3 reduces KV memory by *evicting* old tokens; block-selector sparse reduces compute by *skipping K positions before SDPA*.

We audited compose semantics on 2026-05-17 (memory: `project_v3_f85_compose_option_e.md`). Within a single forward pass, the two cannot share a cache instance:

- `TriAttentionKVCache` is a `StandardKVCache` subclass with a hard `B==1` guard (file: `TriAttentionV3.swift:343`). Its storage is `[1, nKVH, T, D]` with dynamic concat.
- `BatchedRetrievalAttentionKVCache` wraps `BatchedKVCache` via composition (file: `BatchedRetrievalAttentionKVCache.swift:60-78`). Storage is `[B, nKVH, T_max, D]` pre-allocated rectangular; decode assumes `allSameOffset`.
- V3's eviction breaks rectangular-T (slot-dependent compaction); the batched sparse fast path doesn't know how to walk ragged-after-eviction.

The shipping path is mutual-exclusion within a forward, compose across turns:

| Scenario | Path |
|---|---|
| B=1 long single-stream session | V3 (KV-memory savings via eviction) |
| B ≥ 2 concurrent serving | Block-selector batched sparse |
| Multi-turn with concurrency | V3 single-stream first turn; switch to batched sparse once N>1 |
| longctx-svc proxy (see longctx paper) | Manages turn-over-turn rescue regardless |

Intra-forward composition is plausible (B=1 V3 evicts; F-85 ragged-T variant handles the post-eviction cache) but estimated at 3-5 days of work and requires a v2 batched sparse cache that handles non-rectangular T. Deferred.

---

## 8. Failed paths (worth documenting)

A non-trivial fraction of the sprint was negative results. Documenting them so they don't get rediscovered.

### 8.1 F-71b custom Metal sparse SDPA — 14× slower at B=8

Already discussed in §4.3. The hand-rolled Metal kernel with real K skipping lost to the float-mask path because the threadgroup grid `B × nQH × 32 simdgroups` oversubscribes M5 Max by 21×.

### 8.2 F-76 implicit-sparse SDPA — slower than dense at 128K

The F-76 path is the per-query implicit-position sparse kernel from the earlier single-stream RA stack. At 128K B=1 it was 2.6× slower than dense (176 ms/step vs 67 ms/step). Same root cause class as F-71b: a hand-rolled Metal kernel can't outperform `sdpa_vector_2pass` even when it does less arithmetic work, because the tuned upstream kernel is at the hardware's bandwidth floor.

### 8.3 F-84 blockGather decode — 1.3× slower at B=1

Cross-head 1D union take + `mask:.none` SDPA per the recipe in the NSA / SeerAttention literature. At 128K B=1 on Qwen2.5-14B-1M: 86 ms vs dense 67 ms = 1.3× slower. Failure mode: `k_padded` landed at 10-35K (5-27% of T) not the 2K target. The selector's adaptive top-K policy doesn't converge tightly enough at long context to make gather competitive.

### 8.4 F-85 v3 compose-gather batched — 13.7× slower at B=8

The compose-gather batched path materializes per-slot gathered K/V slabs `[B, nKVH, K_padded, D]` and runs dense SDPA on the slab. Theoretically at B=8 ctx=32K K_padded≈2K this saves K/V bandwidth by ~16× vs the float-mask path. Measured: 4.5 tok/s = 13.7× slower than dense. Per-slot `take` materializes B independent gathered slabs (~3 GB/step at B=8), no slab reuse across slots, bandwidth blowup. Validates R3's external-literature claim that "compose-only is fundamentally wrong at B≥8" on the production stack.

### 8.5 Bool mask — 25% regression

Discussed in §4.4. R4's per-row-cost analysis predicted +3-25% from `_boolmask`; measured -25%. PSO-specific tuning matters more than per-row arithmetic.

### 8.6 MLX manual kernel fusion in Swift — 0-16% regression

This finding belongs to the F-83 fused-ops thread but is worth noting here because it bears on what "the right hot path" looks like in MLX-Swift. We tested five fused-activation paths: `MLXFast.fusedGateActivation` (+10.9% slower), custom F83FusedSwiGLU v2 (+15.9%), and several others. All regressed. Root cause: MLX's `compile()` already coalesces split + silu * mul into one kernel via the lazy-graph optimizer; manual `MLXFast.fusedX(...)` calls bypass that optimizer and run raw. The corresponding ZMLX experiment in Python wins (+3.91%) because Python has per-op overhead that fusion bypasses — Swift does not. Real Swift wins come from eliminating pathological memory paths, not from microkernel fusion.

### 8.7 Common thread

The negative-result column is dominated by "things that win on CUDA / discrete GPUs but lose on Apple Silicon." NSA's per-query top-K, FlashAttention-style sparse kernels, compose-gather with materialized slabs, bool-mask kernel variants — all win in the CUDA literature. None won here. The Apple-Silicon-specific recipe that emerges is: ride MLX `sdpa_vector_2pass` with a float mask, tune the upstream block heuristic per-shape, do not write custom Metal sparse kernels unless you are prepared to also write a per-shape PSO tuning sweep.

---

## 9. Composing with TurboQuant+ (currently deferred)

The selector reads raw K to compute block features. For the two-compression-axes compose to work — TurboQuant'd V (3.5-bit) and block-selector sparse on top — the K side needs to stay raw or quasi-raw, which is exactly the `rawKeyMode` contract from our prior asymmetric-KV work.

Status:

- Single-stream wrapper (`RetrievalAttentionKVCache.swift:97-253`) already implements rawKeyMode compose. K stays fp16; V is independently quantized; the selector reads raw K and the SDPA call only dequantizes selected V blocks.
- Batched wrapper (`BatchedRetrievalAttentionKVCache.swift`) does NOT yet wire rawKeyMode compose.
- ekryski-alpha has concurrent specs (041 / 042 / 043) in the flash quantized SDPA + dequant-then-SDPA workstream. Integration is deferred to a follow-up after that work settles.

When the compose lands, the predicted saving is multiplicative: TurboQuant+ shrinks V bytes-per-position by ~4.6×; block-selector skips ~50-70% of V positions at moderate ctx (preliminary). The kernel-level integration is the open question — whether the saving is realized depends on the dequant-then-SDPA primitive's cost structure.

The companion paper's post-softmax sparse-V skip (file: companion paper §3) is also a compose candidate: skip V dequant for negligible attention weights, *and* skip the K bandwidth for entire selected-out blocks. The unified path is "block-selector picks K blocks; SDPA computes attention on selected blocks; V dequant skips negligible-weight positions within the selected blocks." Three sparsity levers at three different stages of the attention pipeline. Whether they compose without overhead is open.

---

## 10. What's still WIP

Not a "future work" section — this is a list of things that are open *right now* and gate the work from being called "shipped":

- **Cliff cells fully bisected.** §4.6 has a partial story. The B=2 ctx=128K cell still loses badly even with `MLX_SDPA_BLOCKS=128`. Auto-tuned per-shape blocks (formula in §4.1) is a clear next step but not finished.
- **Quality validation for batched sparse decode.** §5.5 leans on unit-level equivalence; we don't yet have a clean end-to-end retrieval benchmark on the batched path. The needle-in-haystack harness is currently a broken oracle on the synthetic prompt at B>1.
- **Hybrid family ports.** Qwen3.5 attention+GDN and NemotronH attention+Mamba2 are deferred (multi-day each, requires per-layer-type dispatch).
- **rawKeyMode compose on the batched cache.** Deferred behind concurrent alpha specs 041/042/043. §9 explains.
- **B=1 single-stream sparse decode.** Currently at parity with dense; selector cost not amortized. F-71c custom Metal kernel (one threadgroup per (B, KV-head) with GQA co-load) is the spec; estimated 2-3 days; not started.
- **Continuous-batching ragged-T.** Current batched cache assumes rectangular T (all slots same offset). Real serving has ragged T. v2 territory.
- **Full quant K + V on the batched path.** Both K and V quantized; the selector pays the dequant cost on its K reads. Worth measuring at long context where the bandwidth saving may exceed the dequant overhead.
- **M6 / M7 generation revisit.** Custom Metal sparse kernels lost on M5 Max in part because of occupancy math (§4.3). Next-gen Apple Silicon may change the math; the F-71c kernel-rewrite question is worth re-asking on new hardware.
- **Upstream merge.** PRs #34, #36, #224, #225 are all open. The work has not been merged to alpha master. Reviewers should expect the merge to surface integration issues the local branch has not yet hit.

---

## 11. How to repro (as of the current branch state)

If you want to reproduce any of the numbers in §5, the recipe is:

```bash
# In mlx-swift-lm:
git fetch origin feature/retrieval-attention
git checkout feature/retrieval-attention

# In vllm-swift:
git fetch origin feature/retrieval-attention
git checkout feature/retrieval-attention

# Build vllm-swift (which pulls mlx-swift-lm as a submodule):
swift build -c release --package-path swift

# Run the longctx bench grid (requires Qwen2.5-14B-Instruct-1M-4bit locally):
VSM_SPARSE=1 \
VSM_SPARSE_BATCHED=1 \
VSM_SPARSE_BATCHED_KERNEL=f73 \
VSM_SPARSE_NO_ADAPTIVE=1 \
VSM_SPARSE_PREFILL=1 \
python3 scripts/bench_throughput.py \
    --model ~/models/Qwen2.5-14B-Instruct-1M-4bit \
    --B 8 --ctx 28000 --tokens 30
```

The Bridge auto-sets `MLX_SDPA_BLOCKS=128` at engine create time. The `B * commonT > 224_000` gate auto-falls back to dense above the cliff (see §4.6). The `VSM_SPARSE_BATCHED_KERNEL` env knob exposes the kernel-choice A/B; valid values are `f73` (default), `f73loop` (per-slot loop fallback), `f71b` (the failed custom kernel, retained for regression check), `composegather` (the failed v3 path).

Sparse prefill specifically (B=1, 128K context):

```bash
VSM_SPARSE=1 VSM_SPARSE_PREFILL=1 \
python3 scripts/bench_throughput.py \
    --model ~/models/Qwen2.5-14B-Instruct-1M-4bit \
    --B 1 --ctx 131072 --tokens 30
```

Expected (preliminary, M5 Max, local): prefill ~132s vs dense ~382s = 2.9× faster.

---

## 12. Acknowledgments

- **Awni Hannun** for the underlying MLX kernel and the original `sdpa_vector_2pass` PR #3023, even though the heuristic mis-tune at `_floatmask` cost us a week.
- **Jagrit Digani** for the PR #3211 template that informed how to file a narrow single-file tuning patch upstream (we have not yet filed; the local fork is the current state).
- **The NSA / SeerAttention / DuoAttention / Quest authors** for the algorithmic patterns we ported. The fact that their kernels do not transfer cleanly to Apple Silicon is a kernel-engineering finding, not an algorithmic one.
- **ProbioticFarmer** for the deterministic-SDPA work referenced in §4.5 even though we ended up with the simpler allocator-zero fix.
- **The ekryski alpha team** for the concurrent specs 041/042/043 work on quantized SDPA + dequant-then-SDPA, which §9's deferred compose will plug into.

---

## References

Internal — sprint logs and dev memory (cite-by-codename here):

- `mlx-swift-lm/research/retrieval_attention/F85_BATCHED_SPARSE_DESIGN.md` — initial design.
- `mlx-swift-lm/research/retrieval_attention/F85_BENCH_RESULTS.md` — F-71b 14× regression.
- `mlx-swift-lm/research/retrieval_attention/F85_V2_BENCH_RESULTS.md` — F-73 mask migration.
- `mlx-swift-lm/research/retrieval_attention/F85_V2_M1_MASK_AUDIT.md` — mask kernel B>1 audit.
- `mlx-swift-lm/research/retrieval_attention/F85_V3_BENCH_RESULTS.md` — compose-gather failure.
- `mlx-swift-lm/research/retrieval_attention/F85_CROSSOVER_BISECT.md` — crossover bisect (16K-32K).
- `mlx-swift-lm/research/retrieval_attention/F85_LONGCTX_GRID.md` — 19-cell decode + prefill grid.
- `mlx-swift-lm/research/retrieval_attention/F83_SPARSE_PREFILL_PRD.md` — sparse prefill design.
- `mlx-swift-lm/research/retrieval_attention/F83_PERF_256K_RESULTS.md` — long-context prefill results.
- `mlx-swift-lm/research/retrieval_attention/F80_CLIFF_DIAGNOSIS.md` — MetalAllocator non-determinism RCA.
- `mlx-swift-lm/research/retrieval_attention/FINDINGS.md` — cross-cutting findings summary.
- `mlx-swift-lm/research/retrieval_attention/F83_NIGHT_SUMMARY.md` — 100+ iter overnight distillation.

Code (file:line citations in main text):

- `Libraries/MLXLMCommon/BatchedRetrievalAttentionIndexB.swift` — per-(B, KV-head) selector index.
- `Libraries/MLXLMCommon/BatchedRetrievalAttentionKVCache.swift` — batched sparse cache wrapper.
- `Libraries/MLXLMCommon/RetrievalAttentionKVCache.swift` (lines 97-253) — single-stream rawKeyMode compose with TurboQuant.
- `Libraries/MLXLMCommon/BatchedKVCache.swift` (lines 55-130) — TurboQuant integration surface.
- `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp` (lines 444-486) — MLX SDPA dispatch + blocks heuristic + env override.
- `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/allocator.cpp` (lines 118-131) — MetalAllocator zero-on-recycle.
- `vllm-swift/swift/Sources/VLLMBridge/Bridge.swift` — `buildBatchedSparseCaches`, auto-`setenv`, B*T gate.

External literature:

[^nsa]: Yuan et al. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.* arXiv:2502.11089, 2025.

[^seer]: Gao et al. *SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs.* arXiv:2410.13276, 2024.

[^duo]: Xiao et al. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.* arXiv:2410.10819, 2024.

[^quest]: Tang et al. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* arXiv:2406.10774, 2024.

Companion papers in this corpus:

- [Attention-Gated Value Dequantization for Quantized KV Cache Inference](sparse-v-dequant.md) — the post-softmax sparse-V dequant lever; the lever this work pairs with.
- [TriAttention V3: A Minimal Hybrid Memory Policy for Long-Context KV Cache Eviction](triattention-v3.md) — the KV-eviction lever for single-stream long-context; the compose semantics are discussed in §7.
- [longctx — Open Long-Context Retrieval for Million-Token Inference and TriAttention Rescue](longctx-1m-and-triattention.md) — the cross-turn rescue layer that catches what eviction throws away.

Upstream MLX PRs cited (preliminary; these are the upstream MLX numbers we relate to):

- ml-explore/mlx#3023 — original `sdpa_vector_2pass` blocks heuristic (the heuristic we found mis-tuned for `_floatmask`).
- ml-explore/mlx#3455 — `MLX_SDPA_BLOCKS` env override (we backported into the local fork).
- ml-explore/mlx#3211 — M5 GEMM tuning template (the per-shape patch template we would mirror if we filed).
- ml-explore/mlx#878 — Awni's note on simd-reduction non-determinism (referenced in §4.5).

Local PRs (all open, not merged):

- `ekryski/mlx#34` — allocator zero-on-recycle + `MLX_SDPA_BLOCKS` env override backport.
- `ekryski/mlx#36` — pass-2 reducer modulo fix.
- `ekryski/mlx-swift-lm#224` — batched sparse decode + per-family ports + Bridge wiring.
- `ekryski/mlx-swift-lm#225` — sparse prefill.

---

*This document is a sprint journal, not a results paper. Reviewers should treat the numbers as preliminary, the recommendations as current-best-guess, and the merge status as the gate on any production claim. If you have data from a different Apple Silicon generation (M3 Ultra, M4 Max, future M6) that bears on the `MLX_SDPA_BLOCKS` formula in §4.1, or a clean repro of the B*T cliff at §4.6 with a different model, please drop a note in the PR discussion.*
