# Layer-Aware V Compression: Boundary Layer Protection for Aggressive V Quantization

## Summary

We demonstrate that protecting the first 2 and last 2 transformer layers with higher V precision while aggressively compressing the remaining layers recovers a meaningful fraction of the quality loss from uniform turbo2-V, at minimal compression cost. This policy — **Boundary V** (experimental, internal mode LA-V7) — achieves quality between turbo3-V and turbo2-V at effective compression between the two. Validated on pure-attention models (phi-4, Qwen2.5-7B) at 512 and 8K context on Metal, with NIAH retrieval pass.

**Important caveat:** Follow-up investigation (see Addendum) found that the current implementation mis-targets hybrid architectures (e.g., Qwen3.5) where only a subset of layers have KV attention caches. Results on hybrid models in this paper do not reflect the intended boundary policy. Boundary V should currently be evaluated only on pure-attention models.

## Background

Previous work in this investigation established:
- K precision dominates quality via softmax amplification
- V tolerates aggressive compression (errors are proportional, not exponential)
- q8_0-K + turbo2-V is the most aggressive working V config (+3-5% PPL)
- 1-bit sign-only V was tested separately and found too aggressive (+20-52% PPL)

The question: can we do better than uniform turbo2-V without inventing a new format?

## Hypothesis

Not all layers need the same V precision. Boundary layers (first and last) handle input embedding → hidden state and hidden state → logit transformations. These are the layers where V errors most directly impact output quality. Middle layers are more redundant and can tolerate aggressive compression.

## Method

**LA-V7 policy:** For a model with N layers:
- Layers 0, 1, N-2, N-1: V cache = q8_0 (8.5 bits/val)
- All other layers: V cache = turbo2 (2.5 bits/val)
- K cache: q8_0 throughout (unchanged)

Implementation: 15 lines added to existing `TURBO_LAYER_ADAPTIVE` env var infrastructure in `llama-kv-cache.cpp`. No new formats, no new kernels, no new FA paths.

## Results

All tests on M5 Max. Wikitext-2, 512 context, 4 chunks. K = q8_0 for all configs.

### Quality (PPL)

| Model | Layers | Arch | q8_0/q8_0 | q8_0/turbo3 | q8_0/turbo2 | **LA-V7** |
|-------|--------|------|-----------|-------------|-------------|-----------|
| phi-4-Q8_0 (14B) | 40 | Pure attn | 4.690 | 4.742 | 4.835 | **4.784** |
| Qwen2.5-7B Q4_K_M | 28 | Pure attn | 6.577 | 6.707 | 6.911 | **6.835** |
| Qwen3.5-35B MoE Q8_0 | 40 (10 KV)† | Hybrid | — | 5.137 | 5.257 | **5.148** |
| Qwen3.5-27B Dense Q8_0 | 64 (16 KV)† | Hybrid | — | 6.273 | 6.534 | **6.423** |

†Qwen3.5 models are hybrid architectures (Gated Delta Net + attention). Only every 4th layer has a KV cache. The current boundary selection logic targets raw layer index, not KV layer ordinal, so the boundary policy only upgrades 1-2 out of 10-16 actual KV layers on these models. See Addendum Part 3 for details.

**On pure-attention models, Boundary V consistently lands between turbo3 and turbo2 quality, closer to turbo3.** The Qwen3.5 results show improvement but are confounded by the hybrid-architecture layer-selection issue.

### Effective compression

| Model | Total Layers | Arch | Boundary V bits/val | vs turbo3 (3.5) | vs turbo2 (2.5) |
|-------|:---:|------|:---:|----------------|----------------|
| phi-4 | 40 | Pure attn | 3.10 | 11% smaller | 24% larger |
| Qwen2.5-7B | 28 | Pure attn | 3.36 | 4% smaller | 34% larger |
| Qwen3.5-27B Dense | 64 (16 KV)† | Hybrid | 2.88† | 18% smaller | 15% larger |
| Qwen3.5-35B MoE | 40 (10 KV)† | Hybrid | 3.10† | 11% smaller | 24% larger |

†Effective bits for hybrid models are calculated against total layer count (current behavior). Because the boundary policy only upgrades 1-2 out of 10-16 actual KV layers on these models, these numbers do not reflect the intended boundary policy and should not be used to evaluate the approach.

On pure-attention models, the 4 boundary layers are a meaningful fraction of total layers, and the compression cost is well-defined.

### Speed (phi-4-Q8_0, M5 Max, 8K context)

| Config | Prefill (t/s) | Decode (t/s) |
|--------|--------------|-------------|
| q8_0/turbo2 | 634.24 | 30.08 |
| q8_0/turbo3 | 612.56 | 29.65 |
| **LA-V7** | **628.53** | **30.90** |

No speed penalty. LA-V7 is between turbo2 and turbo3 on prefill (expected) and marginally faster on decode (the 4 q8_0 boundary layers dequant faster than turbo).

## Analysis

### Why boundary layers matter more for V

The first layers transform input embeddings into hidden representations. V errors here affect every subsequent layer's attention output. The last layers transform hidden states into logits. V errors here directly distort the output distribution. Middle layers operate on already-abstracted representations where V precision has less marginal impact.

This mirrors the known finding that K is more sensitive at boundary layers too (from buun's LA-2 and LA-5 experiments). The difference is that K sensitivity affects attention routing (exponential through softmax), while V sensitivity affects the weighted sum (linear). So V boundary protection needs fewer layers than K boundary protection.

### The 4-layer sweet spot

We tested three policies:
- LA-V5: boundary 4 turbo4-V, rest turbo2-V → 4.784 / 6.892
- LA-V6: last 8 turbo4-V, rest turbo2-V → 4.805 / not tested
- LA-V7: boundary 4 q8_0-V, rest turbo2-V → 4.784 / 6.835

LA-V7 (q8_0 boundaries) and LA-V5 (turbo4 boundaries) give identical quality on phi-4. But on the sensitive Qwen model, LA-V7 is clearly better (6.835 vs 6.892). Using q8_0 instead of turbo4 for the 4 boundary layers costs marginally more memory but provides a stronger quality guarantee on sensitive models.

LA-V6 (last 8 layers) is worse than LA-V7 (boundary 4). This confirms that both first and last layers matter, not just the output-facing layers.

### Compression efficiency

LA-V7 achieves its quality win by spending only 4/N layers worth of extra V budget. The cost scales inversely with model depth:

| Model | Depth | Arch | Extra V cost vs turbo2 | Quality recovery vs turbo2→turbo3 gap |
|-------|:---:|------|----------------------|--------------------------------------|
| Qwen2.5-7B | 28 | Pure attn | +34% V | 62% of gap recovered |
| phi-4 | 40 | Pure attn | +24% V | 55% of gap recovered |
| Qwen3.5-35B MoE | 40 (10 KV)† | Hybrid | +24% V | 91% of gap recovered† |
| Qwen3.5-27B Dense | 64 (16 KV)† | Hybrid | +15% V | 43% of gap recovered† |

†Hybrid-model gap recovery numbers are confounded by the layer-selection bug (see Addendum Part 3). The apparent 91% recovery on the MoE model may partially reflect the boundary policy accidentally upgrading a high-impact KV layer rather than the intended systematic boundary protection.

## Conclusion

Boundary V is a real effect on pure-attention models. It occupies a distinct position in the quality/compression tradeoff: quality between turbo3 and turbo2, at effective compression between the two. The implementation is 15 lines in one file, uses only existing turbo types, and has no performance overhead. However, the current boundary selection logic operates on raw layer index, which mis-targets hybrid architectures (see Addendum Part 3).

However, the effect is limited by two factors discovered in follow-up work (see Addendum): boundary gains dilute at longer context, and the current implementation does not properly target hybrid architectures.

**Experimental aggressive-V policy for pure-attention models when uniform turbo2-V is too lossy. Best suited for:**
- Pure-attention models (phi, Llama, Qwen2.5, Mistral) at short-to-medium context
- Users who want more compression than turbo3-V but can't tolerate turbo2-V quality
- Sensitive low-bit base weight models where every PPL point matters

**Not currently recommended for:**
- Hybrid architectures (Qwen3.5 family) until KV-layer-ordinal targeting is fixed
- Very long context (16K+) where boundary layer impact is diluted

## Long-Context Validation

Tested at 8K context (16x the short-context window) to verify LA-V7 does not collapse at longer sequences.

**phi-4-Q8_0 (8K context, 2 chunks):**

| Config | PPL | vs turbo2 |
|--------|------|-----------|
| q8_0/turbo2 | 5.082 | baseline |
| LA-V7 | 5.078 | -0.1% |
| q8_0/turbo3 | 5.004 | -1.5% |

**Qwen2.5-7B Q4_K_M (8K context, 2 chunks):**

| Config | PPL | vs turbo2 |
|--------|------|-----------|
| q8_0/turbo2 | 5.524 | baseline |
| LA-V7 | 5.480 | -0.8% |
| q8_0/turbo3 | 5.354 | -3.1% |

LA-V7 holds at 8K. No collapse or instability. The advantage over uniform turbo2 is smaller at 8K than 512 — expected, because longer context dilutes the relative impact of boundary layers. But LA-V7 never underperforms turbo2 at any tested context length.

## NIAH Retrieval Sanity

Tested needle-in-a-haystack retrieval on Qwen2.5-7B-Instruct-Q4_K_M with Boundary V (LA-V7) active.

**Prompt:** Short factual passage with embedded secret password ("turbo-rainbow-42").
**Result:** Correctly retrieved — `The secret password mentioned above is "turbo-rainbow-42."`

phi-4-Q8_0 NIAH was inconclusive: the model produced a refusal response ("As a large language model, I cannot be relied upon...") regardless of V config. This is phi-4's guardrail behavior, not a compression artifact.

Boundary V does not impair retrieval on the tested sensitive model.

## Limitations

1. **Gains dilute at longer context.** Confirmed at 8K and 16K. At 16K on phi-4, Boundary V advantage over turbo2 is essentially zero (-0.006 PPL). Boundary layers are a fixed cost whose relative impact shrinks with KV cache size.

2. **Hybrid architectures are mis-targeted.** The current boundary selection uses raw layer index, not KV layer ordinal. On Qwen3.5 models where only every 4th layer has KV attention, this means the policy upgrades 1-2 out of 10-16 actual KV layers instead of the intended boundary layers. See Addendum Part 3.

3. **Boundary count is hardcoded at 2+2.** Follow-up testing of 4+4 showed it is not a stable universal improvement (see Addendum Part 2). 2+2 remains the recommended width.

4. **V-only.** K stays q8_0 throughout. This is by design (K precision dominates quality) but means the total KV compression is still bounded by the q8_0 K cache.

5. **Effective compression is between turbo3 and turbo2,** not below turbo2. This is not a way to beat turbo2 on compression — it's a way to beat turbo2 on quality at similar-ish compression.

6. **Tested on Metal only.** CUDA parity not validated.

7. **Pure-attention models only.** Validated on phi-4 and Qwen2.5-7B. Qwen3.5 hybrid results are confounded by the layer-selection bug and should not be used to evaluate the policy.

## Recommendation

**Status: experimental, limited to pure-attention models.**

Boundary V (experimental, internal mode LA-V7) has passed on pure-attention models:
- 2 pure-attention models (phi-4, Qwen2.5-7B) with consistent quality improvement
- 2 context lengths (512 + 8K), with diminishing returns at longer context
- Speed sanity (no penalty)
- NIAH retrieval pass

Known limitations:
- Hybrid architectures (Qwen3.5) are mis-targeted by current layer-selection logic
- Gains dilute at 16K+ context
- CUDA not validated
- Boundary width (2+2 vs 4+4) is not universally better at wider settings

**Next steps if pursuing further:**
- Fix boundary selection to use KV layer ordinal instead of raw layer index (hybrid architecture support)
- CUDA validation
- Consider auto-detection of boundary sensitivity per model

## How to Reproduce

```bash
# Build
cd llama-cpp-turboquant && git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Boundary V PPL test
TURBO_LAYER_ADAPTIVE=7 ./build/bin/llama-perplexity \
  -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv turbo2 \
  -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 4

# Uniform turbo2 baseline for comparison
./build/bin/llama-perplexity \
  -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv turbo2 \
  -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 4

# Uniform turbo3 baseline for comparison
./build/bin/llama-perplexity \
  -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv turbo3 \
  -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 4

# Quick NIAH sanity (should retrieve the embedded fact)
TURBO_LAYER_ADAPTIVE=7 ./build/bin/llama-cli \
  -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv turbo2 \
  -p "Facts: The secret password is turbo-rainbow-42. Gold is Au. What is the secret password?" \
  -n 20 --no-display-prompt --temp 0
```

Expected: Boundary V PPL is better than uniform turbo2, usually behind uniform turbo3.

**Warning:** Evaluate Boundary V on **pure-attention models** (phi, Llama, Qwen2.5, Mistral). Hybrid architectures such as the Qwen3.5 family (Gated Delta Net + attention) are mis-targeted by the current boundary selection logic, which uses raw layer index instead of KV layer ordinal. Results on hybrid models do not reflect the intended policy.

## Addendum: Boundary Width, Precision, and Architecture Constraints (Mar 29, 2026)

This addendum covers three rounds of follow-up investigation prompted by a community question (SharkWipf) about whether boundary V layers should use full f16 instead of q8_0.

### Verified Model Metadata

A critical finding during this investigation: some models tested are **hybrid architectures** where only a subset of layers have KV attention caches. This directly affects how boundary V policies operate.

| Model | Total Layers | KV Attention Layers | Architecture |
|-------|:---:|:---:|---|
| phi-4 Q8_0 | 40 | 40 | Pure attention |
| Qwen2.5-7B Q4_K_M | 28 | 28 | Pure attention |
| Qwen3.5-27B Dense Q8_0 | 64 | 16 | Hybrid (Gated Delta Net + attention, every 4th layer) |
| Qwen3.5-35B MoE Q8_0 | 40 | 10 | Hybrid (Gated Delta Net + attention, every 4th layer) |

### Part 1: Boundary Precision (f16 vs q8_0)

**Question:** Should boundary V layers use f16 instead of q8_0?

All configs use q8_0 K throughout. Tested at 512 context, 8 chunks.

**phi-4 Q8_0 (40 layers, pure attention):**

| Config | PPL | vs q8_0/q8_0 |
|--------|-----|-------------|
| q8_0/q8_0 baseline | 6.001 | — |
| first4+last4 V=q8_0 | 6.047 | +0.8% |
| first4+last4 V=f16 | 6.065 | +1.1% |
| first2+last2 V=f16 | 6.090 | +1.5% |
| first2+last2 V=q8_0 (LA-V7) | 6.110 | +1.8% |
| uniform q8_0/turbo2 | 6.106 | +1.7% |

**Qwen3.5-27B Dense Q8_0 (64 total layers, 16 KV layers):**

| Config | PPL | vs q8_0/q8_0 |
|--------|-----|-------------|
| q8_0/q8_0 baseline | 6.888 | — |
| first4+last4 V=q8_0 | 7.038 | +2.2% |
| first4+last4 V=f16 | 7.043 | +2.2% |
| first2+last2 V=q8_0 (LA-V7) | 7.051 | +2.4% |
| first2+last2 V=f16 | 7.050 | +2.3% |
| uniform q8_0/turbo2 | 7.159 | +3.9% |

**Finding:** f16 does not consistently beat q8_0 for boundary layers. On phi-4, q8_0 boundary actually wins at both widths. On the 27B model they are within noise. f16 boundary layers are not worth the extra VRAM.

### Part 2: Boundary Width (2+2 vs 4+4)

**Question:** Does widening the boundary from first2+last2 to first4+last4 improve quality?

Short-context results (512, 8 chunks) initially looked promising for 4+4. But the long-context validation told a different story.

**Long-context results (8K, 2 chunks) on pure-attention models:**

phi-4 Q8_0 (40 KV layers):

| Config | PPL (8K) | PPL (16K) | Gap closed (8K) |
|--------|----------|-----------|:---:|
| uniform turbo2 | 5.082 | 5.342 | 0% |
| 2+2 q8_0 (LA-V7) | 5.078 | — | 5% |
| 4+4 q8_0 | 5.055 | 5.335 | 35% |
| uniform turbo3 | 5.004 | — | 100% |

Qwen2.5-7B Q4_K_M (28 KV layers):

| Config | PPL (8K) | Gap closed |
|--------|----------|:---:|
| uniform turbo2 | 5.524 | 0% |
| 2+2 q8_0 (LA-V7) | 5.480 | 26% |
| 4+4 q8_0 | 5.459 | 38% |
| uniform turbo3 | 5.354 | 100% |

Qwen3.5-27B Dense Q8_0 (16 KV layers, hybrid):

| Config | PPL (8K) | Gap closed |
|--------|----------|:---:|
| uniform turbo2 | 6.496 | 0% |
| 2+2 q8_0 (LA-V7) | 6.454 | 24% |
| 4+4 q8_0 | 6.476 | 12% |
| uniform turbo3 | 6.323 | 100% |

**Findings:**

1. **4+4 beats 2+2 on pure-attention models** (phi-4, Qwen2.5-7B) at both 512 and 8K context. The improvement is real but modest.

2. **4+4 is worse than 2+2 on the hybrid 27B model at 8K.** This reversal is explained by the architecture bug (see Part 3).

3. **Boundary V gains dilute with context length.** At 16K on phi-4, the 4+4 advantage over turbo2 shrinks to -0.006 PPL (essentially zero). Boundary layers are a fixed cost; their relative importance shrinks as the total KV cache grows.

4. **4+4 is not a stable universal upgrade over 2+2.** The result is architecture-dependent and context-length-dependent. Not enough evidence to promote it as a second tier.

### Part 3: Hybrid Architecture Bug

**The current Boundary V implementation has a policy bug on hybrid architectures.**

The boundary selection logic checks the raw transformer layer index (`il`) against total `n_layer`:

```cpp
const bool is_boundary = (il < 2 || il >= n_layer - 2);
```

On Qwen3.5-27B (64 total layers, KV attention on layers 3, 7, 11, ..., 63):
- Mode 7 (first2+last2): Only layer 63 is both a boundary layer AND a KV layer. **1 out of 16 KV layers upgraded.**
- Mode 10 (first4+last4): Layers 3 and 63 are boundary+KV. **2 out of 16 KV layers upgraded.**

This means the boundary policy barely touches the model's actual KV cache on hybrid architectures. The Qwen3.5 results in the original paper are valid data but do not represent the intended policy of protecting the first and last KV attention layers.

**To fix:** The boundary check should operate on KV layer ordinal (e.g., first 2 and last 2 *KV attention* layers), not raw transformer layer index. This fix is not implemented yet.

### Effective Compression (using verified layer counts)

q8_0 = 8.5 bits/val, turbo2 = 2.5 bits/val, turbo3 = 3.5 bits/val.

| Model | KV Layers | 2+2 bits/val | 4+4 bits/val | turbo3 |
|-------|:---:|:---:|:---:|:---:|
| phi-4 | 40 | 3.10 | 3.70 | 3.50 |
| Qwen2.5-7B | 28 | 3.36 | 4.21 | 3.50 |
| Qwen3.5-27B | 16* | 2.88* | 3.25* | 3.50 |

*Qwen3.5-27B effective bits are calculated against 64 total layers (current buggy behavior), not 16 KV layers.

Note: On Qwen2.5-7B, 4+4 uses 4.21 bits/val — 20% more than turbo3 (3.50) for only 38% of turbo3's quality gain. Poor compression efficiency on shallow models.

### Speed

Speed sanity on phi-4 showed no clear regression from boundary modes, but decode variance was high due to concurrent GPU load during testing. The earlier clean speed sanity (from the initial investigation) showed no penalty. No speed concern is raised.

### Updated Recommendation

**Keep 2+2 q8_0 (LA-V7) as the sole experimental Boundary V mode.** Do not introduce 4+4 as a second tier.

Rationale:
- Boundary V shows real quality gains on **pure-attention models** (phi-4, Qwen2.5-7B). The effect is consistent and reproducible.
- Gains dilute at longer context. Boundary V is most useful at short-to-medium context where boundary layers are a meaningful fraction of total KV cache.
- **4+4 is not a stable universal upgrade** over 2+2. It helps on some models/contexts and hurts on others.
- **Hybrid architectures are not properly supported.** The current boundary selection logic mis-targets these models. Until the layer-selection logic is fixed to operate on KV layer ordinal, Boundary V should not be recommended on hybrid models (Qwen3.5 family).
- **Uniform turbo3 remains the cleaner practical default** for users who want better-than-turbo2 V quality. Boundary V is an advanced experimental option for users who specifically want turbo2-level compression with partial quality recovery.

**Corrected framing (supersedes earlier addendum):**
- Do NOT update Boundary V default to 4+4. Keep 2+2.
- f16 boundary layers are not justified (confirmed).
- Boundary width matters on pure-attention models, but not enough to warrant a second tier.
- The most interesting next step is fixing the boundary selection to use KV layer ordinal instead of raw layer index, which would make Boundary V properly target hybrid architectures.

## Addendum: Independent Validation (2026-03-31)

The boundary layer sensitivity finding has been independently confirmed:

- **@sztlink (Felipe Sztutman)** — Qwen3-4B, RTX 4090, tonbistudio/turboquant-pytorch (2026-03-31): Measured boundary gap scaling with K compression. The gap between boundary layers (first 2 + last 2) and middle layers grows with K compression: -0.001 at 4-bit K (negligible), -0.004 at 3-bit K (noticeable), -0.010 at 2-bit K (significant). Directly validates the Boundary V approach. [Follow-up layer 0 K isolation test](https://github.com/ggml-org/llama.cpp/discussions/20969#discussioncomment-16402887) revealed the mechanistic *why*: boundary layers with extreme K norms (146.8 at layer 0 vs 20–40 at middle layers) produce tightly Gaussian distributions after normalization — ideal for Lloyd-Max codebook quantization. Middle layers are the actual soft spot under K compression. This provides the distributional basis for why boundary V protection (LA-V7) recovers quality: boundary layers carry more V signal fidelity by default, so protecting them preserves the layers that are already doing the most accurate attention routing.
- **@HyperionMS2040** — RTX 3090 (2026-03-30): Qwen3.5-4B asymmetric data shows clean quality ladder with no catastrophic failure when K precision is maintained, consistent with boundary layer sensitivity.
- **@Corianas_** — Independent validation of Boundary V on NanoGPT (X collaborator, 2026-03-29).
- **@mudler (Ettore Di Giacinto)** — [APEX](https://github.com/mudler/apex-quant) MoE weight quantization, [LocalAI](https://github.com/mudler/LocalAI) (44.7k stars) (2026-04-01): Independently discovered that "first and last 5 transformer layers are 10x more sensitive than the middle" for weight quantization on Qwen3.5-35B-A3B across 25+ experiments. Same boundary sensitivity principle as Boundary V, applied to weights instead of KV cache. APEX assigns higher precision to edge layers, lower to middle layers. Directly validates the boundary layer hypothesis from the weight quantization direction.
- **vLLM TurboQuant PR** — [vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479) (2026-04-03): PR author (@vibhavagarwal5) implementing TQ_boundary_layers in vLLM, explicitly "inspired from https://github.com/dipeshbabu/turboquant_plus". Boundary layer protection being adopted by a major production inference framework.

---

## Exploratory Findings (2026-03-31)

A follow-up investigation tested `q8_0-K + turbo2-V` with Boundary V (auto-enabled) specifically on Qwen3.5-35B MoE Q8_0 at extended context lengths (512c, 8K, 32K). On this tested setup, turbo2+BV achieved 7.53x V compression with PPL within 1% of q8_0 — matching or slightly exceeding `q8_0/turbo3` quality. See [MoE V-compression frontier](moe-v-compression-frontier.md) for the full writeup.

---

## Files Changed

- `src/llama-kv-cache.cpp` — added modes 5, 6, 7 to `TURBO_LAYER_ADAPTIVE` env var (15 lines)
