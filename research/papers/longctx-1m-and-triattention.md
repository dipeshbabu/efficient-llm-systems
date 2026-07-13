# longctx — Open Long-Context Retrieval for Million-Token Inference and TriAttention Rescue

**Dipesh Tharu Mahato**
Independent Researcher
GitHub: [@dipeshbabu](https://github.com/dipeshbabu)

---

## Abstract

We present **longctx**, an open-source retrieval companion service for inference engines, and document its two principal modes: (1) standalone long-context retrieval scored against the MRCR v2 benchmark on a single AMD MI300X with Qwen2.5-32B-Instruct, and (2) rescue layer for our TriAttention V3 — an independent implementation and minimal hybrid extension of Mao et al., "TriAttention: Efficient Long Reasoning with Trigonometric KV Compression" (arXiv:2604.04921, 2026) — query-aware KV-cache eviction, validated end-to-end via a 256K planted-fact NIAH on Apple M5 Max with Qwen3.5-2B-4bit. Full V3 design and per-architecture results: see [TriAttention V3](triattention-v3.md).

On MRCR v2, longctx's directional best (n=30 MultiQ Selector + bge-rerank + det copy) reaches **0.688** at 1M context, above SubQ's 0.659; mass-validation at n=60 single-query Selector lands at 0.601. On the M5 NIAH ramp, V3+longctx via `ChatSession` passes recall at every context rung from 32K through 256K (✓HIT × 4) while V3-alone fails recall at every rung (✗miss × 4) — a clean dependency: V3 evicts faster than recall can survive without a rescue layer, and longctx is that layer.

The paper documents the architecture, the V3 write/read wiring, the failure mode of running TriAttention without longctx, the latency overhead of the rescue path (~22% at 256K), and the prescribed defaults for using the pair in production. We argue that open long-context inference at million-token scale on commodity hardware is now a question of pipeline glue, not of algorithms — the algorithms exist (V3 eviction, semantic retrieval, prefix caching), and longctx is the glue.

---

## 1. Introduction

Two production paths have emerged for handling million-token context windows on a single GPU. Closed providers like SubQ ship proprietary stacks combining bespoke retrieval with model-side caching, reporting 0.659 on MRCR v2 8-needle at 1M. Open inference engines (vLLM, llama.cpp, mlx-swift-lm) ship the model and the KV cache, but the question of "what to do when the prompt outgrows GPU memory" is left to the application.

longctx fills that gap. It is a single-purpose FastAPI service that does two things:
1. **Index** — chunk and embed text spans into a faiss index, scoped per session or per repo.
2. **Retrieve** — return top-K spans for a query, with optional rerank.

Wired in front of any OpenAI-compatible engine, longctx behaves as a code-aware RAG layer. Wired *behind* an engine that ships a query-aware eviction policy — specifically, our TriAttention V3 (an independent Swift port and hybrid extension of Mao et al.'s "TriAttention" trig-scoring approach, see [TriAttention V3](triattention-v3.md) for the full design) — longctx becomes the rescue layer that catches what eviction throws away and serves it back to the next prefill.

This paper reports two measurements:

- **§3** — longctx as standalone retrieval, scored on MRCR v2 8-needle on AMD MI300X with Qwen2.5-32B-Instruct.
- **§4** — longctx as TriAttention V3 rescue, scored via a 256K planted-fact NIAH on Apple M5 Max with Qwen3.5-2B-4bit.

Both rounds are reproducible from the open-source code and the listed test harnesses.

---

## 2. Architecture

### 2.0 Full stack — top-level

```
                      ┌──────────────────────────────────────────────────┐
                      │  CLIENTS  (OpenCode / Hermes / curl / your app)  │
                      └────────────────────┬─────────────────────────────┘
                                           │  OpenAI HTTP /v1/chat/completions
                                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   vllm-swift  (Swift HTTP server, --enable-longctx)    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                       ChatSession                                │  │
│  │   • Tier-3 auto-rehydrate hook (commit fe1a3b0)                  │  │
│  │   • Tokenizer auto-bind into TriAttentionRescue                  │  │
│  │   • Multi-turn cache persistence                                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    mlx-swift-lm (Swift)                          │  │
│  │  ┌─────────────────────┐    ┌─────────────────────────────────┐  │  │
│  │  │  TokenIterator      │ OR │  MTPSpeculativeTokenIterator    │  │  │
│  │  │  (standard decode)  │    │  (Gemma4Assistant drafter loop) │  │  │
│  │  └──────────┬──────────┘    └──────────┬──────────────────────┘  │  │
│  │             │                          │                         │  │
│  │  ┌──────────▼──────────────────────────▼──────────────────┐      │  │
│  │  │              Model (Gemma4 / Qwen3 / Qwen3.5 / ...)    │      │  │
│  │  │                  • newCache(parameters)                 │      │  │
│  │  │                  • forward(inputs, cache)               │      │  │
│  │  └────┬───────────────────────────────────────────────┬────┘      │  │
│  │       │                                               │           │  │
│  │  ┌────▼─────────────┐                  ┌──────────────▼──────┐    │  │
│  │  │  KVCache (per    │  V3 enabled      │ TriAttentionV3      │    │  │
│  │  │  layer)          │◀─────────────────│ Engine              │    │  │
│  │  │  • Standard      │  per-token       │ • score / evict     │    │  │
│  │  │  • Rotating      │  scoring         │ • prefix-protect    │    │  │
│  │  │  • TurboQuant    │                  │ • per-segment quota │    │  │
│  │  │  • TriAttention  │                  └──────────┬──────────┘    │  │
│  │  └──────────────────┘                             │               │  │
│  │                                                   │ evict_pos[]   │  │
│  │                              ┌────────────────────▼──────────┐    │  │
│  │                              │  TriAttentionRescue (Tier 2)  │    │  │
│  │                              │  • decode evicted token IDs   │    │  │
│  │                              │  • POST /evict/write          │    │  │
│  │                              └────────────────┬──────────────┘    │  │
│  └───────────────────────────────────────────────┼───────────────────┘  │
└──────────────────────────────────────────────────┼──────────────────────┘
                                                   │  HTTP
                                                   ▼
                      ┌──────────────────────────────────────────┐
                      │   longctx-svc  (FastAPI, port 5054)      │
                      │   • /evict/write  ← evicted text spans   │
                      │   • /evict/retrieve  → top-K chunks      │
                      │   • /index, /retrieve  ← code-aware mode │
                      │   • per-session faiss + MiniLM embedder  │
                      └──────────────────────────────────────────┘
```

The stack is symmetric: the same `ChatSession` path drives both standard decoding and MTP speculative decoding, both can run with V3 eviction enabled, and both surface the rescue bridge to longctx via the same `/evict/write` and `/evict/retrieve` endpoints.

### 2.1 Two operating modes

**Code-aware retrieval (Mode A/B).** Every chat completion's prompt is parsed for absolute file paths. The first path's project root is detected via sentinel (`package.json`, `.git`, `pyproject.toml`, etc.). longctx-svc indexes the scope (Hot first → Package on demand), retrieves top-K chunks for the user's query, splices them into a system message, and forwards the request. The model sees a normal chat completion with a `## Retrieved code context` block at the top.

Mode A wraps any OpenAI-compatible engine (proxy in front). Mode B uses an engine flag (`--enable-longctx`) to spawn longctx-svc as a sidecar. Both compose with dipeshbabu/vllm-swift, dipeshbabu/llama-cpp-turboquant, dipeshbabu/vllm-turboquant, and any upstream engine that speaks OpenAI HTTP.

**TriAttention rescue.** When the inference engine has our TriAttention V3 enabled (env: `VLLM_TRIATT_ENABLED=1`, `LONGCTX_ENDPOINT=http://...`), V3 fires per-token eviction during prefill. V3 here refers to our independent implementation; the underlying trigonometric scoring approach is from Mao et al., arXiv:2604.04921. The hybrid prefix-protect + per-segment-quota extensions, the Tier 2 evict-callback and Tier 3 rehydrate plumbing, and the Swift port are ours. The engine's TriAttention rescue bridge POSTs every evicted span (decoded back to text via the bound tokenizer) to `/evict/write`. On the next user turn, `ChatSession` auto-fires `rescue.rehydratePrompt(query: <user_msg>)` against `/evict/retrieve`, prepending recovered chunks as a system message before that turn's prefill.

### 2.2 Why the two modes share an index

Both modes are top-K retrieval over text spans embedded into a vector index. The difference is the source of the spans:

- Code-aware mode: spans come from the on-disk repo (one-shot index per scope, refreshed via file watcher).
- Rescue mode: spans come from KV-cache evictions (streamed during inference, scoped to the session).

Same plumbing serves both. The inference-time path adds a per-session faiss store keyed by `session_id`; the code-aware path uses scope-keyed stores.

### 2.3 Engine compatibility

| Engine | Code-aware | Rescue (V3+longctx) |
|---|:-:|:-:|
| dipeshbabu/mlx-swift-lm (M-series Apple Silicon) | ✅ | ✅ ChatSession auto-Tier-3 |
| dipeshbabu/vllm-swift | ✅ `--enable-longctx` | ✅ same ChatSession path |
| dipeshbabu/llama-cpp-turboquant | ✅ via `llama-server-longctx` wrapper | – |
| dipeshbabu/vllm-turboquant (AMD MI300X) | ✅ wired in `serve.py` | ✅ |
| upstream vLLM / llama.cpp / SGLang / Ollama | ✅ Mode A proxy | – |

---

## 3. MRCR v2 — longctx as Standalone Retrieval

### 3.1 Setup

- 1× AMD Instinct MI300X (192 GB HBM3, gfx942), DigitalOcean dev cloud droplet, ROCm 7.2, Ubuntu 24.04
- Model: Qwen2.5-32B-Instruct via dipeshbabu/vllm-turboquant
- Benchmark: MRCR v2 8-needle (8 needles inserted at randomized positions in a synthetic haystack), per-bin scoring across 8K → 1M context

### 3.2 Recipes

Three configurations spanning the cost/quality frontier:

- **plain RAG** — single-query retrieval, no rerank. Cheapest, lowest quality.
- **single-query Selector + bge-rerank + det copy** — standard recipe with a deterministic copy step that re-anchors retrieved spans to the original prompt frame.
- **MultiQ Selector + bge-rerank + det copy** — fans the query out to multiple semantic variants, unions the top-K, reranks. Most expensive, highest quality.

### 3.3 Results

| bin | recipe | n | longctx | SubQ |
|---|---|---:|---:|---:|
| 8K | plain RAG | 30 | **0.822** | — |
| 32K | plain RAG | 30 | 0.697 | — |
| 64K | plain RAG | 30 | 0.641 | — |
| 64K | chunked (cs=2000) | 30 | 0.670 | — |
| 1M | plain RAG (baseline) | 30 | 0.440 | — |
| 1M | Selector + bge-rerank + det copy (single-query) | **60** | **0.601** *(mass-val)* | 0.659 |
| 1M | **MultiQ** Selector + bge-rerank + det copy | 30 | **0.688** *(directional)* | 0.659 |

The headline:
- **n=60 single-query mass-val: 0.601** — below SubQ's 0.659 by 0.058 absolute.
- **n=30 MultiQ directional: 0.688** — above SubQ's 0.659 by 0.029. Mass-validation rerun is pending; an n=80 priority run OOM'd before completing on the droplet.

The MultiQ recipe trades query latency for accuracy. The win at 1M is real but provisional until the mass-val number lands. Single-query at 1M is the conservative claim.

### 3.4 Latency

`longctx-svc` warm `/retrieve` p95: 63.2 ms. Cold scope build (20-file project): 12.7 s. Cache reload from disk: 8.9 s (mostly embedder cold-load). Test coverage: 173 tests, all green.

---

## 4. TriAttention V3 Rescue — 256K NIAH on Apple Silicon

> "TriAttention V3" throughout this section refers to **our** V3 — an independent Swift port and minimal hybrid extension of the trigonometric scoring formula introduced in Mao et al., "TriAttention: Efficient Long Reasoning with Trigonometric KV Compression" (arXiv:2604.04921, 2026). The hybrid prefix-protect + per-segment-quota policy, the Tier 2 evict-callback, the Tier 3 rehydrate hook, and the Apple Silicon plumbing are ours. Full V3 design and per-architecture results in [TriAttention V3](triattention-v3.md).

### 4.1 Setup

- 1× Apple M5 Max, 128 GB unified memory, macOS 26.4
- Model: `mlx-community/Qwen3.5-2B-4bit` (qwen3_5 hybrid Mamba+Attention, 4-bit quantized)
- Engine: dipeshbabu/mlx-swift-lm at `feature/triattention-v3`
- Test harness: `Tests/Benchmarks/V3ChatSessionRamp.swift`
- Eviction config: V3 default rate (10%), window=128, prefix=32, warmup=256, hybrid=2

### 4.2 The wiring

Three arms per context rung (32K / 64K / 128K / 256K):

- **baseline-tq8v4** — single-turn prompt with planted fact + question, turbo8v4 KV codec, no V3, no longctx. Establishes recall under the model's native context window.
- **v3-only** — two-turn structure, V3 enabled, longctx disabled. Tests whether V3 alone preserves recall.
- **v3+longctx** — two-turn structure, V3 enabled, `LONGCTX_ENDPOINT=http://127.0.0.1:5054`. Tests the full rescue stack.

The two-turn structure is necessary because `ChatSession`'s auto-Tier-3 rehydrate hook fires *before* a turn's prefill using that turn's user-message text as the query. Single-turn prompts give the rehydrate hook nothing to query against until after eviction has already happened. We split the long blob and the question into separate turns so turn 1's eviction populates the longctx index before turn 2's question fires the rescue retrieval.

### 4.3 Turn-by-turn flow

```
   USER TURN N  (long blob: filler + planted fact)
   │
   ▼
   ChatSession.respond(prompt)
   │
   ▼
   Build messages list, run prefill
   │
   ▼
   Model forward, layer-by-layer:
   ┌─────────────────────────────────────────────────────────┐
   │ for each layer:                                         │
   │     attention(Q, K, V)  →  cache.update(K, V)           │
   │     V3.accumulateLayerScore(K, layerIdx)                │
   │ when cumulative cells > budget + divideLength:          │
   │     evict_pos = V3.finalizeEvictRound()                 │
   │     for c in cache:                                     │
   │         c.removePositions(evict_pos)                    │
   │     ┌─────────────────────────────────────────────┐     │
   │     │  TriAttentionRescue.shared.onEvict(spans)   │     │
   │     │   ├─ tokenizer.decode(evicted_ids)          │     │
   │     │   └─ POST /evict/write?session_id=...       │ ────┼──▶ longctx-svc
   │     └─────────────────────────────────────────────┘     │      ingests, embeds,
   └─────────────────────────────────────────────────────────┘      indexes in faiss
                          │
                          ▼
                  Generate "OK" ack (turn N output)


   USER TURN N+1  (the question — "What's the access code?")
   │
   ▼
   ChatSession.respond(question)
   │
   ▼
   ┌─────────────────────────────────────────────────────────┐
   │  TIER-3 AUTO-REHYDRATE (only fires through ChatSession) │
   │  if any cache is TriAttentionKVCache:                   │
   │      recovered = rescue.rehydratePrompt(query=question) │ ────▶ POST /evict/retrieve
   │      if recovered:                                      │      ◀── top-K chunks
   │          messages.insert(.system(recovered),            │
   │                          before user_msg)               │
   └─────────────────────────────────────────────────────────┘
   │
   ▼
   Prefill turn N+1 (now sees recovered chunks containing planted fact)
   │
   ▼
   Generate answer  ─────▶  "481729"  ✓HIT
```

### 4.4 Results

```
ctx     arm           t1       t2     v3%    rounds  recall  total
32K     baseline-tq8v4  5.6s   0.0s   0.00%  0       ✓HIT     5.6s
32K     v3-only         6.8s   0.2s   3.72%  12      ✗miss    6.9s
32K     v3+longctx      7.7s   0.6s   3.72%  12      ✓HIT     8.3s
64K     baseline-tq8v4 16.7s   0.0s   0.00%  0       ✓HIT    16.7s
64K     v3-only        19.5s   0.2s   2.17%  18      ✗miss   19.7s
64K     v3+longctx     20.9s   0.9s   2.17%  18      ✓HIT    21.8s
128K    baseline-tq8v4 76.3s   0.0s   0.00%  0       ✓HIT    76.3s
128K    v3-only        66.9s   0.8s   1.42%  24      ✗miss   67.6s
128K    v3+longctx     69.5s   1.3s   1.42%  24      ✓HIT    70.9s
256K    baseline-tq8v4 186.7s  0.0s   0.00%  0       ✓HIT   186.7s
256K    v3-only        220.9s  1.0s   1.32%  30      ✗miss  221.9s
256K    v3+longctx     226.9s  2.4s   1.32%  30      ✓HIT   229.3s
```

### 4.5 Findings

- **V3+longctx passes recall at every rung 32K → 256K.**
- **V3-alone fails recall at every rung.** V3 evicts the planted fact under default policy regardless of context size; nothing recovers it.
- **Baseline turbo8v4 passes recall at every rung.** The model's native context window handles 256K cleanly without V3 or longctx — V3 only becomes necessary when prompt length exceeds GPU memory budget.
- **Stack overhead vs baseline at 256K: ~22%** (229s vs 187s). The cost is two prefill turns instead of one, plus the rehydrate query latency. The payoff is unbounded effective context above the model's native window.
- **V3 eviction rate decreases with context size**: 3.72% @ 32K → 1.32% @ 256K. The engine's policy is conservative at long context, presumably because the warmup/prefix/window protected regions become a smaller fraction of total evictable cells.

### 4.6 Standalone confirmation of the write path

A separate single-cell test at 256K with default (10%) eviction rate confirmed that the rescue write callback fires reliably:

```
prompt_tok=247753, budget=230400
prefill=224.55s, decode=6.22s, tps=10.1
v3_rounds=30, v3%=1.32%
longctx_session_total=19427    ← chunks ingested by longctx-svc
recall=✗miss                   ← bare container.generate() doesn't rehydrate
```

19,427 evicted text chunks were ingested by longctx-svc during turn 1's prefill. Recall ✗miss in this specific test because the harness called bare `container.generate()` instead of `ChatSession.respond()`. The auto-Tier-3 rehydrate hook is wired only on the `ChatSession` path (per commit `fe1a3b0` in mlx-swift-lm). Bare drivers must call `TriAttentionRescue.shared.rehydratePrompt(query:)` manually before each prefill.

---

## 5. The Failure Mode of V3 Without longctx

The cleanest takeaway from §4: **our TriAttention V3 alone is unsafe for long-context retrieval workloads.**

V3's eviction policy is query-aware in the sense that it scores cells by attention salience to recent queries, but the planted-fact NIAH puts the question *after* the fact in the prompt. By the time the question's queries are scoring cells, V3 has already evicted the cells that contained the fact. Eviction is one-way; cells gone from the KV are gone.

longctx makes eviction reversible. The evicted span is captured at write time, embedded, and indexed. When the next turn's query comes in, semantic retrieval finds the relevant span and prepends it back. The model gets the same fact in the prompt that V3 took out of the cache.

This dependency is now documented in mlx-swift-lm's README under "Critical: don't enable V3 without longctx", with the empirical receipt table from §4.

---

## 6. Production Defaults

For a Gemma 4 / Qwen3 / Llama-class model on M-series Apple Silicon at long context, run:

```bash
# 1. longctx service
longctx-svc serve --host 127.0.0.1 --port 5054

# 2. inference engine env
export VLLM_TRIATT_ENABLED=1
export VLLM_TRIATT_BUDGET=$((CTX_TARGET * 9 / 10))   # 10% eviction headroom
export VLLM_TRIATT_WINDOW=128
export VLLM_TRIATT_PREFIX=32
export VLLM_TRIATT_WARMUP=256
export VLLM_TRIATT_HYBRID=2
export LONGCTX_ENDPOINT=http://127.0.0.1:5054
```

Drive through `ChatSession` (mlx-swift-lm) or via vllm-swift's `--enable-longctx` flag (vllm-swift). Custom drivers must wire the rehydrate hook manually.

For AMD MI300X with dipeshbabu/vllm-turboquant, longctx-svc runs on the same droplet; the engine's `--enable-longctx` flag handles sidecar spawn.

---

## 7. Limitations and Open Items

- **V3 + TQ+ stacking is gated.** `TriAttentionKVCache` extends `KVCacheSimple` (FP16-only). Stacking V3 with dipeshbabu's TurboQuant codecs (turbo8v4, turbo4v2) requires a `TriAttentionTurboKVCache` variant. Tracked but not yet shipped.
- **V3 hooks are per-model-family.** Currently wired on Qwen3 / Qwen3.5 / Qwen3-MoE / Llama / Mistral3 / Phi / Phi3 / Gemma3 / GLM4. Other model families fall back to non-V3 caches.
- **MRCR v2 1M MultiQ at n=60 mass-val pending.** Directional n=30 result above SubQ; conservative single-query mass-val below.
- **Bare `container.generate()` skips rescue.** Auto-Tier-3 rehydrate is `ChatSession`-only. Custom drivers need explicit `rehydratePrompt(query:)` calls.
- **Latency cost of two-turn structure.** Single-turn long-context with V3+longctx requires either (a) the model's question to come from a separate API turn after the haystack, or (b) an explicit pre-prefill rehydrate using the question text. The two-turn pattern is natural for chat; document/single-shot inference patterns need integration work.

---

## 8. Conclusion

longctx ships two things at once: a code-aware retrieval companion that wraps any OpenAI-compatible engine via a CLI flag, and a rescue layer for query-aware KV-cache eviction policies that lets engines run effectively unbounded context without losing recall.

The MRCR v2 numbers position longctx within reach of SubQ at 1M context with the MultiQ recipe (0.688 directional vs 0.659), pending mass-validation. The TriAttention rescue numbers are unambiguous: V3 alone fails recall at every rung; V3+longctx passes recall at every rung. The pair is the unit of feature, not either piece alone.

Open inference engines now have a path to million-token context on a single GPU that doesn't require closed proprietary stacks. The algorithms exist; the glue exists. The remaining work is integration coverage and stack-level optimization (V3+TQ+, drafter-paired MTP, etc.) — not algorithmic invention.

---

## Reproduction

```bash
# longctx-svc
git clone https://github.com/dipeshbabu/longctx
cd longctx && pip install -e .
longctx-svc serve --host 127.0.0.1 --port 5054

# mlx-swift-lm V3+longctx ramp (Apple Silicon)
git clone -b feature/triattention-v3 https://github.com/dipeshbabu/mlx-swift-lm
cd mlx-swift-lm
RUN_V3_CHAT_RAMP=1 swift test --filter "V3ChatSessionRamp"

# vllm-turboquant on AMD MI300X (1M MRCR)
# see longctx/docs/results.md for the full recipe
```

Findings doc with raw logs and per-cell receipts is available alongside the test sources.
Test source: [Tests/Benchmarks/V3ChatSessionRamp.swift](https://github.com/dipeshbabu/mlx-swift-lm/blob/main/Tests/Benchmarks/V3ChatSessionRamp.swift).

---

## Acknowledgments

Eric Kryski (mlx-swift-lm), the MLX team at Apple, the SubQ team for the MRCR v2 reference benchmark and the AMD/ROCm team for MI300X hardware enablement.

---

## Appendix A — Module map (mlx-swift-lm side)

```
mlx-swift-lm/
├── Libraries/
│   ├── MLXLMCommon/
│   │   ├── ChatSession.swift            ← Tier-3 auto-rehydrate hook (commit fe1a3b0)
│   │   ├── Evaluate.swift               ← TokenIterator + SpeculativeTokenIterator
│   │   ├── KVCache.swift                ← Standard / Rotating / TurboQuant base
│   │   └── TriAttention/
│   │       ├── TriAttentionV3.swift     ← engine, scoring, eviction policy
│   │       ├── TriAttentionKVCache.swift← V3 cache (extends KVCacheSimple, FP16)
│   │       └── TriAttentionRescue.swift ← Tier 2 evict-write + Tier 3 rehydrate
│   │
│   ├── MLXLLM/
│   │   ├── LLMModelFactory.swift        ← model_type registry (incl. gemma4_assistant)
│   │   └── Models/
│   │       ├── Gemma4.swift             ← parent (V3 hook in newCache)
│   │       ├── Gemma4Assistant.swift    ← MTP drafter (4-layer Q-only)
│   │       ├── MTPSpec.swift            ← MTP iterator (parent verify + drafter loop)
│   │       ├── Qwen3.swift / Qwen35.swift / Qwen3MoE.swift
│   │       ├── Llama.swift / Mistral3.swift / Phi.swift / Phi3.swift
│   │       └── Gemma3.swift / GLM4.swift / ...
│   │
│   └── MLXVLM/                          ← VLM models (separate factory)
│
└── Tests/Benchmarks/
    ├── V3ChatSessionRamp.swift          ← 12-cell V3+longctx ramp (§4 receipts)
    ├── V3ChatSessionNIAH.swift          ← 256K NIAH end-to-end via ChatSession
    ├── V3CtxRamp.swift                  ← bare container.generate (no rehydrate)
    ├── V3DefaultTest.swift              ← single 256K w/ default 10% rate
    ├── TurboCtxRamp.swift               ← turbo8v4 ramp 32K → 256K
    └── MTPSpecDecode.swift              ← Gemma 4 31B MTP bench (separate from V3)
```

## Appendix B — longctx-svc endpoint surface

```
POST /evict/write?session_id=<id>
  body: {"chunks": [{"text": "...", "tokens": [...], "round": N, "layer": L}]}
  → MiniLM embed → faiss store keyed by session_id

POST /evict/retrieve?session_id=<id>
  body: {"query": "<user message text>", "top_k": 8}
  → faiss search → top-K chunks formatted as system message text

GET  /evict/dump?session_id=<id>
  → {"session_total": int, "chunk_summary": [...]}    (debug)

POST /index?scope=<path>                              (code-aware mode)
  → walk repo, chunk, embed, faiss store keyed by scope

POST /retrieve?scope=<path>&query=<text>&top_k=K       (code-aware mode)
  → top-K spans formatted as system message text

GET  /healthz   → {"status": "ok", "version": "0.3.0a3"}
GET  /longctx/status → human-readable session/scope state
```

## Appendix C — V3 environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `VLLM_TRIATT_ENABLED` | unset (off) | master switch |
| `VLLM_TRIATT_BUDGET` | required | KV cells to keep before eviction triggers |
| `VLLM_TRIATT_WINDOW` | 128 | always-keep recent window |
| `VLLM_TRIATT_PREFIX` | 32 | always-keep prompt prefix |
| `VLLM_TRIATT_WARMUP` | 256 | tokens before first eviction round |
| `VLLM_TRIATT_HYBRID` | 2 | eviction policy mode |
| `VLLM_TRIATT_COMPRESSION_LOG` | 0 | per-round log verbosity |
| `LONGCTX_ENDPOINT` | unset | URL of longctx-svc — required for rescue path |
