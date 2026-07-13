# Cross-Engine KV Cache Fidelity on AMD MI300X (Qwen3.6-35B-A3B)

**Dipesh Tharu Mahato**
Independent Researcher
GitHub: [@dipeshbabu](https://github.com/dipeshbabu)

---

## Abstract

We benchmark three production inference engines (vLLM, SGLang, llama.cpp) on a single AMD MI300X GPU running the same hybrid GDN + attention MoE model (Qwen3.6-35B-A3B). Two rounds: (1) BF16 baseline with no KV compression, measuring load time, perplexity, prefill, and decode; (2) REFRACT 4-axis fidelity scoring with each engine's native 8-bit KV cache against its own fp/bf16 reference. No advanced compression schemes (TurboQuant or otherwise) are tested. Baseline-only.

The BF16 round shows no single engine wins all axes. Model load: llama.cpp 31.8 s vs vLLM 188.9 s and SGLang ~210 s (first launch, includes AITER kernel JIT compile). PPL at 32K: 5.49 / 5.74 / 6.01 (vLLM / SGLang / llama.cpp), measured at non-uniform chunk sizes (vLLM 3 × 32K, SGLang 9 × 8K due to logprob OOM at 32K, llama.cpp 9 × 32K sliding) — comparison is loose. Prefill at 32K: SGLang 32,428 tok/s, vLLM 11,690 tok/s, llama.cpp 1,298 tok/s. Decode at 256 output tokens: llama.cpp 133.2 tok/s (clean isolated `tg256` measurement), SGLang 90.2 tok/s, vLLM 25.6 tok/s (last two derived as `total - prefill` after a 32K prefill — methodology differs from llama.cpp's isolated decode).

The REFRACT round: all three engines hit 100.0 R-NIAH (long-context retrieval at 32K) under their respective 8-bit KV. Trajectory and KLD axes diverge. llama.cpp `q8_0` (block-quantized int8) produces 0.0025 nats of mean KL drift from fp16 KV. vLLM `fp8_e4m3` produces 0.037 nats. SGLang `fp8_e4m3` (forced through Triton attention because AITER's fp8 prefill kernel rejected the hybrid model) produces 0.021 nats. The two engines that both label their compression `fp8_e4m3` produce a 1.8× difference in mean KL drift.

Composite scores (harmonic mean of all four REFRACT axes): llama.cpp 89.39 PASS, SGLang 86.97 PASS, vLLM 84.31 DEGRADED.

The bench documents 8 nontrivial engine-side bugs encountered during bring-up. Total wall time: ~12 hours.

---

## 1. Introduction

KV cache compression is one lever for fitting long context into GPU memory at inference time. Three production engines support 8-bit KV cache compression on AMD MI300X via different paths: llama.cpp via `q8_0` (block-quantized int8), vLLM via `fp8_e4m3` (AMD's fnuz fp8 variant through ROCm flash-attention), and SGLang via `fp8_e4m3` (intended through AITER's fp8 prefill kernel, in this bench forced through Triton attention).

This paper reports a controlled cross-engine measurement on a single MI300X. Methodology matches context length, tokenization, and reference anchoring across engines where the engine APIs permit. Fidelity scoring uses [REFRACT](../../components/refract/README.md), a 4-axis evaluation framework anchored to each engine's own fp/bf16 reference.

The bench is baseline-only. No advanced KV compression schemes are tested.

---

## 2. Setup

### Hardware

- 1× AMD Instinct MI300X (192 GB HBM3, gfx942)
- DigitalOcean dev cloud droplet, ROCm 7.2, Ubuntu 24.04

### Model

- `Qwen/Qwen3.6-35B-A3B` (BF16 safetensors, ~67 GB across 26 shards)
- Architecture: hybrid GDN + attention MoE, 256K native context
- vLLM and SGLang load HF safetensors directly; llama.cpp loads BF16 GGUF converted from the same source

### Engines

| Engine | Source | KV-cache option used |
|---|---|---|
| **vLLM** | [Historical TurboQuant vLLM fork](../../docs/reference/historical-forks.md#vllm-experimental-forks), branch `pr/tq-prebaked-centroids`. BF16 attention path identical to upstream main. | `kv_cache_dtype="fp8_e4m3"` (candidate); `"auto"` = bfloat16 (reference) |
| **SGLang** | `lmsysorg/sglang:v0.5.10.post1-rocm720-mi30x` Docker image (stock, not a fork) | `--kv-cache-dtype fp8_e4m3` (candidate); `auto` = bfloat16 (reference); forced `--attention-backend triton` |
| **llama.cpp** | [Historical TurboQuant llama.cpp fork](../../docs/reference/historical-forks.md#llamacpp-experimental-forks), branch `feature/turboquant-kv-cache`. Hybrid model support hardened over preceding months. | `-ctk q8_0 -ctv q8_0` (candidate); `f16/f16` (reference) |

The vLLM and llama.cpp branches are the author's forks. The SGLang configuration is the published Docker image. This is a fork-vs-stock asymmetry; readers should weight bring-up smoothness accordingly.

### Methodology

- Eval corpus: `wikitext-2-raw/wiki.test.raw` (~1.3 MB, ~250K tokens)
- Same `prompts/v0.1.jsonl` (30 prompts) across all three engines
- BF16 baseline measured at 32K context
- REFRACT axes measured at 4096 ctx (Trajectory, KLD, PLAD) and 32768 ctx_max (R-NIAH)
- All REFRACT scores anchored to each engine's own fp/bf16 reference (not a global "fp16 truth")
- Single-trial measurements; no multi-run variance bounds

---

## 3. BF16 Baseline Results

No quantization on weights or KV. All engines configured for BF16.

| Metric | vLLM | SGLang | llama.cpp |
|---|---|---|---|
| Model load (s) | 188.9 | ~210 (first launch, incl. AITER JIT) | 31.8 |
| PPL @ 32K | 5.49 (3 × 32K chunks) | 5.74 (9 × 8K chunks)¹ | 6.01 (9 × 32K sliding) |
| Prefill tok/s @ 32K | 11,690 | 32,428 | 1,298 |
| Decode tok/s @ 256 out | 25.6² | 90.2² | 133.2³ |
| KV / state footprint @ 32K | 92.1 GiB KV pool reserved | 51 GB KV + 46 GB GDN/Mamba state | 702 MiB context block |

¹ SGLang's 32K logprob query OOM'd; PPL fell back to 8K chunks. Numbers are not directly comparable to vLLM and llama.cpp's 32K-chunk PPL.

² vLLM and SGLang decode tok/s computed as `(total_dt - prefill_dt) / n_decoded_tokens` for 256 output tokens after a 32K prefill in a single `generate()` call. This subtraction approach can underestimate steady-state decode if any post-prefill warmup is captured in `total_dt`.

³ llama.cpp decode tok/s measured via isolated `llama-bench tg256` (clean steady-state). Methodology differs from vLLM/SGLang.

KV / state footprint numbers are not directly comparable: vLLM reserves a KV pool sized to `gpu_memory_utilization`; llama.cpp reports per-context allocation; SGLang separately accounts for the Mamba state of this hybrid model.

---

## 4. REFRACT 4-Axis Fidelity Results

[REFRACT](../../components/refract/README.md) scores how much fidelity each engine retains when 8-bit KV compression is enabled, anchored to that engine's own fp/bf16 reference. Four axes:

- **Trajectory (gtm):** greedy-decode N tokens per prompt under both KV configs. Score = fraction of candidate tokens that match the reference token-by-token.
- **KLD:** per-token KL divergence between candidate and reference next-token distributions on a natural-text corpus. Score = `100 * exp(-mean_kld)`.
- **R-NIAH:** insert a sentinel ("APRICOT-7-BLUE is the rare paint color") at fractional positions of long context, score retrieval accuracy at lengths up to 32K.
- **PLAD:** per-token edit distance under prompt perturbations (typos, case changes, paraphrases). Score reflects how much extra drift the candidate introduces vs the reference under the same perturbations.

Composite is the harmonic mean of the four. Bands: ≥95 EXCELLENT, ≥85 PASS, ≥70 DEGRADED, <70 FAIL.

### Cross-engine REFRACT scores

| Engine | Cand KV | gtm | kld | rniah | plad | Composite | Band |
|---|---|---:|---:|---:|---:|---:|---|
| llama.cpp | `q8_0` (int8) | 69.62 | 99.75 | 100.0 | 96.54 | 89.39 | PASS |
| SGLang | `fp8_e4m3` (triton attn) | 67.71 | 97.95 | 100.0 | 90.77 | 86.97 | PASS |
| vLLM | `fp8_e4m3` | 62.40 | 96.35 | 100.0 | 90.58 | 84.31 | DEGRADED |

### Mean KL divergence

| Engine | Cand KV | mean_kld nats | top-1 % |
|---|---|---:|---:|
| llama.cpp | `q8_0` | 0.0025 | — |
| SGLang | `fp8_e4m3` | 0.021 | 97.67 |
| vLLM | `fp8_e4m3` | 0.037 | 97.48 |

### Direct observations

1. All three engines score 100.0 on R-NIAH at 32K under their respective 8-bit KV configurations.
2. llama.cpp's `q8_0` produces the lowest mean KL drift (0.0025 nats) and highest score on every axis.
3. vLLM `fp8_e4m3` produces the highest mean KL drift (0.037 nats), 14.8× higher than llama.cpp's int8.
4. SGLang `fp8_e4m3` produces 0.021 nats of mean KL drift, falling between llama.cpp and vLLM. SGLang's path was forced through Triton attention; the AITER path that SGLang would have used by default rejected the hybrid model (see §6).
5. vLLM and SGLang both label their candidate KV `fp8_e4m3`. They produce a 1.8× difference in mean KL drift on the same model and same hardware.
6. Composite bands: llama.cpp PASS (89.39), SGLang PASS (86.97), vLLM DEGRADED (84.31).
7. Cross-engine differences are dominated by the trajectory axis (range 62.40–69.62). KLD score range: 96.35–99.75. R-NIAH: identical at 100.0. PLAD range: 90.58–96.54.

---

## 5. Same Dtype Label, Different Kernel Paths

vLLM and SGLang both label their 8-bit KV `fp8_e4m3`. The implementations downstream of that label differ:

- **vLLM** routes through ROCm flash-attention. K and V are quantized at write time, fp8 attention math is applied through the rotary path, and dequantization happens at read.

- **SGLang** would normally route through AITER's `mha_batch_prefill_fp8bf16` kernel. On this hybrid Qwen3.6 model, that kernel raises `RuntimeError: invalid argument for batch_prefill` at request time on both `fp8_e4m3` and `fp8_e5m2`. Workaround for this bench: `--attention-backend triton` re-routes prefill through Triton's attention implementation, which has its own fp8 quantization logic.

Two engines, same dtype string, two different runtime kernel paths, two different KL-divergence outcomes.

### Open question on SGLang's number

SGLang's mean_kld of 0.021 nats sits below vLLM's fp8 (0.037) and above llama.cpp's int8 (0.0025). Two non-exclusive possibilities for why:

1. The Triton attention path applies fp8 quantization with finer granularity (per-channel or per-token) than the AITER/ROCm flash-attn fp8 path.
2. The Triton fp8 path may not be fully fp8 end-to-end on this model — some operations may accept fp8 inputs but compute internally at higher precision.

R-NIAH at 100% does not disambiguate these (llama.cpp also gets 100% with bona fide int8). A stress test (32K retrieval under adversarial distractors) would help separate the cases. Not run in this bench.

---

## 6. Engine-Side Issues Encountered

This section documents bring-up failures and the workaround for each. Reproducibility-oriented; readers attempting this bench will likely hit the same.

### llama.cpp (1 issue)

REFRACT's R-NIAH axis tokenizes the haystack via `runner.tokenize_to_ids`, which previously shelled out to a `llama-tokenize` binary. On hosts where the local llama.cpp checkout had drifted from the loaded library, this failed with `Symbol not found: _llama_memory_breakdown_print`. Fix: dispatch `tokenize_to_ids` to the active backend's own tokenizer when the backend is not llamacpp. This is a REFRACT framework change, not a llama.cpp engine issue.

### vLLM (5 issues)

1. **Missing flash-attn ROCm wheel.** Qwen3.6 instantiates a `Qwen3_VisionTransformer` subcomponent at model load even for text-only use. Its RoPE imports `flash_attn.ops.triton.rotary`. There is no pre-built flash-attn wheel for ROCm. Built from source via `git+https://github.com/ROCm/flash-attention.git@main_perf` with `--no-build-isolation`. ~5,800 .hip object files. ~70 minutes wall time on the droplet.

2. **`max_num_seqs` default vs Mamba blocks.** vLLM defaults to `max_num_seqs=1024`. Hybrid Qwen3.6's Mamba state allocator at `gpu_memory_utilization=0.45` produced only 784 cache blocks. Engine init crashed: `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (784). Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.` Fixed via `REFRACT_VLLM_MAX_NUM_SEQS=32` env knob.

3. **`prompt_logprobs` cap.** REFRACT's KLD axis sent `prompt_logprobs=64`. vLLM caps this at 20: `VLLMValidationError: Requested prompt logprobs of 64, which is greater than max allowed: 20`. Fixed via `REFRACT_VLLM_KLD_TOPK=20`.

4. **Trajectory axis interleaving forces N model loads.** REFRACT's trajectory axis originally interleaved `ref` and `cand` calls per prompt. With our eviction-on-key-change cache (necessary because two LLM instances of this hybrid model don't fit 192 GB at high `gpu_memory_utilization`), interleaving meant ~60 model evictions per axis. Refactored axis to batch all-ref then all-cand. Two model loads total per axis.

5. **vLLM v1 engine subprocess holds GPU memory across `del LLM()`.** vLLM's v1 architecture runs the engine core as a multiprocessing subprocess. `del LLM()` plus `gc.collect()` plus `torch.cuda.empty_cache()` did not release the engine subprocess's GPU allocations in our run. The second axis's LLM init saw 24 GB free out of 192 GB and crashed: `ValueError: Free memory on device cuda:0 (24.05/191.69 GiB) on startup is less than desired GPU memory utilization (0.85, 162.93 GiB)`. Fix: split each axis into its own python process (`--skip-kld` for axis A, `--skip-gtm` for axis B, etc.). Process exit guarantees teardown.

For axis C R-NIAH specifically, also bumped `REFRACT_VLLM_MAX_MODEL_LEN=33792` since the 32K probe needs ctx ≥ 32K and the LLM is cached by `max_model_len`.

### SGLang (3 issues + orchestrator)

1. **Broken `aiter.dtypes` in published Docker image.** The `lmsysorg/sglang:v0.5.10.post1-rocm720-mi30x` image's `aiter` package is missing its `dtypes` module. SGLang's Quark MXFP4 import chain references `aiter.dtypes.fp8` unconditionally at module load (the load happens regardless of whether Quark is being used at request time):
   ```
   File "sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py":
       from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import ...
   File "aiter/ops/triton/quant/fused_fp8_quant.py":
       fp8_dtype = aiter.dtypes.fp8
   AttributeError: module 'aiter' has no attribute 'dtypes'
   ```
   Fix: a `sitecustomize.py` that stubs `aiter.dtypes` mapping to `torch.float8_e4m3fnuz`, mounted at `/opt/sitecustom/sitecustomize.py:ro`, selected via `PYTHONPATH=/opt/sitecustom`. Also stubbed `dynamic_per_tensor_quant` and `static_per_tensor_quant` to raise on call (the Quark loader imports them but does not invoke them on the BF16 / fp8 KV paths used in this bench).

2. **AITER fp8 prefill rejects hybrid model.** With Quark loading patched, the next failure surfaces at request time: AITER's `mha_batch_prefill_fp8bf16` kernel raises `RuntimeError: invalid argument for batch_prefill` on hybrid Qwen3.6 for both `fp8_e4m3` and `fp8_e5m2`. Workaround: `--attention-backend triton` forces SGLang to bypass AITER's prefill kernel and route through Triton attention. This affects what "fp8 KV" actually runs (see §5).

3. **KV dtype is fixed at server launch.** SGLang has no per-request KV dtype switching. REFRACT's KLD axis wants to compare two configs in one run. Built a sequential orchestrator (`refract_sglang_seq.sh`):
   - Phase ref: launch BF16 server, run all probes via HTTP, dump to JSON, kill container
   - Phase cand: launch fp8 server, run same probes, dump to JSON, kill
   - Aggregate: load both JSONs, compute KLD per chunk, generate REFRACT-format scores

   For axes C (R-NIAH) and D (PLAD), `refract_sglang_cd_collect.py` imports REFRACT's needle generator (`refract.axes.rniah._build_prompt`, `_extract_needle_keyword`) and perturbation functions (`refract.axes.plad._PERTURBATION_FUNCS`) directly so the methodology is identical to native REFRACT. The aggregator uses HuggingFace's tokenizer for PLAD's edit distance to avoid shelling out to `llama-tokenize`.

### Total bring-up cost

8 nontrivial bugs across 3 engines. ~12 hours wall time end-to-end including the 70-minute flash-attn ROCm compile, several CUDA graph captures, two cycles of vLLM script revision, and the SGLang sequential orchestrator development.

---

## 7. Limitations

1. **Single model, single GPU.** Qwen3.6-35B-A3B on one MI300X. Other hybrid models (Qwen3-Next, Jamba2) and other GPUs (H100, B200, MI355X) are untested.

2. **Symmetric 8-bit KV only.** Asymmetric configs (e.g. K=int8 + V=fp8) are not tested in this bench.

3. **SGLang fp8 fidelity has an open question.** The 0.021 nats mean_kld may reflect real Triton-path fp8 quantization with finer scaling, or it may reflect partial higher-precision fallback in the Triton attention path. R-NIAH at 100% does not disambiguate. Pending an adversarial-distractor stress test.

4. **Single bench run per engine.** No multi-trial variance bound. PPL and decode tok/s have been observed in other benches to vary 2–5% across runs. The cross-engine differences here are larger than that variance, but tighter bounds would require multi-trial measurement.

5. **PPL chunk size differs across engines.** vLLM 3 × 32K, SGLang 9 × 8K (the 32K logprob query OOM'd), llama.cpp 9 × 32K sliding. PPL numbers in the BF16 table are not strictly apples-to-apples.

6. **Decode methodology differs across engines.** llama.cpp uses isolated `tg256`; vLLM and SGLang use `total - prefill` subtraction after a 32K prefill. The latter can underestimate steady-state decode.

7. **Fork vs stock asymmetry.** vLLM and llama.cpp are tested via the author's forks (BF16 attention path identical to upstream main for vLLM; llama.cpp fork has hybrid-model hardening). SGLang is tested via the published Docker image. Bring-up smoothness numbers reflect this asymmetry.

8. **No NVIDIA comparison.** AMD-only.

9. **No advanced KV compression schemes tested.** Baseline only.

---

## 8. Conclusion

Three production inference engines on the same model, same GPU, same `fp8_e4m3` dtype label produce measurably different output fidelity. The dtype label does not constrain the kernel implementation; the kernel implementation determines the actual fidelity outcome.

Measured composite REFRACT scores under each engine's native 8-bit KV: llama.cpp `q8_0` 89.39 PASS, SGLang `fp8_e4m3` (Triton attn) 86.97 PASS, vLLM `fp8_e4m3` 84.31 DEGRADED. All three engines score 100.0 on R-NIAH at 32K under their respective 8-bit KV.

For workload selection in the BF16 round: SGLang dominates 32K prefill; llama.cpp dominates decode and load time; vLLM sits in the middle on speed.

Methodology recommendations for cross-engine benches: (1) anchor scoring against each engine's own fp/bf16 reference, (2) match context length and tokenization where the engines permit, (3) run multi-axis fidelity scoring rather than relying on PPL alone, (4) do not assume engines with the same dtype label produce equivalent output.

---

## 9. Reproducibility

All scripts and orchestrators on the droplet at `/root/scripts/`:

- `cross_engine_bench.sh` — BF16 baseline (load / PPL / prefill / decode / KV size for all 3 engines)
- `refract_llamacpp_full.sh` — REFRACT --full on llama.cpp
- `refract_vllm_full.sh` / `refract_vllm_full_cd.sh` — REFRACT split-axis on vLLM
- `refract_sglang_seq.sh` / `refract_sglang_cd_seq.sh` — REFRACT two-phase orchestrator on SGLang
- `refract_sglang_collect.py` / `refract_sglang_cd_collect.py` — SGLang HTTP probe collectors (A+B and C+D)
- `refract_sglang_aggregate.py` / `refract_sglang_cd_aggregate.py` — REFRACT-format scoring from collected dumps
- `sitecustomize.py` — `aiter.dtypes` stub for SGLang container

REFRACT framework changes are now consolidated in the
[current monorepo](../../README.md):

- vLLM backend: working implementation, evict-on-key-change cache, env knobs for `MAX_NUM_SEQS`, `KLD_TOPK`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`
- SGLang backend with two-phase orchestration support
- Trajectory and PLAD axes refactored to batch ref/cand by KV config (helps memory-pressured backends)
- Runner `tokenize_to_ids` dispatches through the active backend (R-NIAH unblock for vLLM/SGLang)

---

## References

- [REFRACT framework](../../components/refract/README.md) — 4-axis KV-cache fidelity scoring
- [REFRACT QUICKSTART](../../components/refract/QUICKSTART.md)
- [REFRACT vLLM backend](../../components/refract/src/refract/backends/vllm.py)
- [REFRACT SGLang backend](../../components/refract/src/refract/backends/sglang.py)
- [REFRACT leaderboard](../../components/refract/LEADERBOARD.md)
- [Historical llama.cpp fork](../../docs/reference/historical-forks.md#llamacpp-experimental-forks) — public URL unavailable
- [Historical vLLM fork](../../docs/reference/historical-forks.md#vllm-experimental-forks) — public URL unavailable
- PPL artifacts on instruct models: [attn-rotation-and-ppl-artifact.md](attn-rotation-and-ppl-artifact.md)
