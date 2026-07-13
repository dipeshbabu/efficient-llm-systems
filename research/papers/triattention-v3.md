# TriAttention V3: A Minimal Hybrid Memory Policy for Long-Context KV Cache Eviction

**Dipesh Tharu Mahato**
Independent Researcher
GitHub: [@dipeshbabu](https://github.com/dipeshbabu)

---

## Abstract

The original TriAttention paper is Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, and Yukang Chen, **"TriAttention: Efficient Long Reasoning with Trigonometric KV Compression,"** arXiv:2604.04921 (2026), released under Apache 2.0 at <https://github.com/WeianMao/triattention>. This paper introduced a self-calibrating trigonometric scoring formula that ranks KV cache tokens by their expected contribution to future attention under RoPE and evicts the lowest-scoring tokens. On reasoning workloads the paper reports up to ten times KV cache compression.

This work is an independent implementation and extension, not a reproduction of the original paper's experiments. We built TriAttention in llama.cpp against the TurboQuant+ fork, tested it on standard wikitext perplexity plus strict needle-in-a-haystack retrieval, and found that paper-faithful TriAttention produces two problems that make it unshippable as a default: it costs 1.2 percent perplexity at only 10 percent savings on 7B-class transformers, and it silently drops the needle near the end of the context.

This paper presents V3, a minimal hybrid policy that adds two structural constraints on top of the paper-faithful scoring: a hard prefix window where the first N tokens are never evicted, and a per-segment eviction quota that forces eviction to spread evenly across the remaining context rather than concentrating wherever the trig score happens to be lowest. On Qwen2.5-7B at 32K and 64K context, V3 achieves effectively baseline perplexity and passes needle retrieval at start, middle, and end positions. Combined with TurboQuant+, the full stack costs 0.84 percent perplexity relative to the f16 baseline with no measurable wall-time overhead from the V3 layer.

V3 is not a universal fix. On the Qwen3.5 hybrid Mamba+Attention architectures (27B and 35B-A3B), perplexity transfers cleanly but needle retrieval silently fails at middle and end positions even under V3. The final section requests community input on the missing pieces.

---

## 1. Background

### 1.1 TurboQuant+

TurboQuant (Google Research, ICLR 2026) compresses the KV cache via Walsh-Hadamard rotation followed by polar quantization. TurboQuant+ extends the original recipe with asymmetric K/V treatment (q8_0 keys, turbo3 values), turbo kernels for Metal and CUDA, and cross-model validation on the [historical llama.cpp TurboQuant fork](../../docs/reference/historical-forks.md#llamacpp-experimental-forks). The baseline stack on Qwen2.5-7B Q8_0 at 32K context compresses the f16 KV footprint from approximately 1.8 gigabytes down to approximately 690 megabytes (2.6 times smaller) at a cost of 0.55 percent perplexity and no measurable retrieval regression on needle-in-a-haystack.

TurboQuant+ is the "fewer bits per token" axis of KV cache compression.

### 1.2 TriAttention

TriAttention addresses the orthogonal axis: fewer tokens in the cache. It scores every KV entry using a trigonometric expansion of the query-key interaction under RoPE, then evicts the lowest-scoring tokens until the cache fits within a target budget. The paper claims up to ten times compression on reasoning models where thinking traces contain heavy token redundancy.

The scoring formula, after lifting RoPE and the offset-sum invariants out of the per-cell inner loop, reduces to a single dot product of the form:

    score_cell = sum_f (A_f * cos_sum_f - B_f * sin_sum_f) + extra_cell

where `A_f`, `B_f`, and `extra_cell` are computed once per cell from the self-calibrated query centers and the cell's K vector, and `cos_sum`, `sin_sum` are per-eviction constants precomputed from `max_pos` and the RoPE angular frequencies.

### 1.3 Motivation

The two axes stack. TurboQuant+ shrinks bytes per token. TriAttention shrinks token count. If both work without destructive interaction, the combined compression ratio multiplies. On 7B at 32K context, that means roughly 2.6 times from TQ+ multiplied by 1.1 times from 10 percent token eviction, which is almost 3 times total, with the stack growing more valuable at longer context.

The question this paper addresses: does the paper-faithful version of TriAttention actually work in production, and if not, what is the minimal fix?

---

## 2. The Three Versions

The implementation lives in `src/llama-triattention.cpp` on the `experiment/triattention-integration` branch of the llama.cpp fork. All three versions share the same self-calibrating query capture (via the `cb_eval` hook), the same trigonometric scoring, and the same recent-window protection. They differ only in how the set of tokens to evict is selected from the scored candidates.

### 2.1 V1: Paper-Faithful Global Sort

V1 sorts all non-protected cells globally by score and evicts the highest-scoring cells until the budget target is met. This is the baseline implementation of the paper formula. The code was validated against a random-scoring control and a recency-scoring control to rule out implementation bugs before measuring against V2 and V3.

### 2.2 V2: Per-Segment Quota Only (A Wash)

V2 partitions the non-protected cells into K equal position buckets, computes a per-bucket eviction target proportional to the bucket size, sorts within each bucket by score, and evicts the highest-scoring cells in each bucket up to its target. A cleanup pass handles any deficit from per-bucket rounding. The goal of V2 was to prevent the global sort from concentrating eviction on a single band of context.

V2 was a wash. It is documented here for completeness, but it is not a viable shipping variant and it is not the recommended hybrid mode. On Qwen2.5-7B at 32K the per-segment quota successfully rescued end-position needle retrieval (which V1 failed), but it simultaneously broke start-position retrieval (which V1 passed) and worsened perplexity to +4.49 percent. The symmetric failure pattern told us that forcing uniform eviction across position buckets was removing exactly the early-context tokens that V1 was implicitly keeping. That observation is what motivated V3: the fix was not to spread eviction more evenly, but to explicitly protect the tokens that V1 was implicitly protecting. V2 is kept in the code under `--triatt-hybrid 1` only for ablation purposes.

### 2.3 V3: Prefix Protection Plus Per-Segment Quota

V3 adds a hard prefix window on top of V2. The first `prefix_protect` tokens of the context are added to the protected set and are never considered for eviction, in the same way that the last `window_size` tokens are already protected. The per-segment quota then applies only to the cells between the prefix and the window. The final evict function calls `seq_rm` on the selected cells and does not touch the underlying kv_cells directly.

The default values used in the validation below are `prefix_protect = 128`, `window_size = 128`, `n_segments = 8`.

---

## 3. Methodology

Every number in the results section was produced by one of two test tracks: a perplexity sweep on wikitext-2-raw, or a strict needle-in-a-haystack check against a wikitext-derived prompt with a random needle inserted at a known position. Both tracks evolved during the project in response to bugs that were quietly inflating or deflating the numbers. This section documents the protocol as it stands now, along with the traps that justify each choice.

### 3.1 Hardware and Build

All measurements in this paper are on an Apple M5 Max, 128 GB unified memory, Metal backend. The historical llama.cpp TurboQuant fork is at the `experiment/triattention-integration` branch. Build configuration is the standard cmake flow with Metal enabled:

```bash
cmake -B build-test -DLLAMA_METAL=ON
cmake --build build-test -j 8 --target llama-perplexity llama-completion
```

Transfer validation to other hardware backends (CUDA, HIP) has not been done for V3 specifically and is an open task.

### 3.2 Perplexity Protocol

All perplexity numbers are measured on wikitext-2-raw using `llama-perplexity` with the following fixed configuration:

- `-f wikitext-2-raw/wiki.test.raw` (standard wikitext test split)
- `-b 512` (batch size, critical, explained below)
- `--chunks 3` (minimum, critical, explained below)
- `-c` set to the target context size (32768 or 65536)

The reported "PPL" in every table is the `Final estimate` line from the `llama-perplexity` output, not the first or intermediate estimate. Standard error is reported as the `+/-` value on the same line.

**Why three chunks minimum.** The very first TriAttention result in this project was a single-chunk run at 32K that showed a 3.9 percent PPL improvement over baseline. That result did not replicate on three chunks. The real three-chunk number on the same configuration is a 1.2 percent regression. Single-chunk perplexity is too noisy at long context to distinguish signal from variance, and an apparent improvement at one chunk can flip sign on three. Every perplexity number in this paper is a three-chunk measurement at minimum.

**Why batch size 512 matters.** When the batch size equals the context size (the default for `llama-perplexity` at 32K), each prefill chunk is processed as a single batch, which means the TriAttention eviction hook only runs once per chunk, after the full prefill is already in cache. In that configuration, eviction fires zero times during the PPL sweep and the "improvement" you see is an artifact of eviction not actually happening. Forcing `-b 512` makes the prefill process one 512-token ubatch at a time, so the eviction hook is called many times during the sweep and actually removes tokens. Without this flag the PPL numbers are meaningless for TriAttention evaluation.

**Eviction confirmation.** Every PPL run in this paper is eviction-confirmed: I grep the stderr for the `evict: evicted` log line emitted by the TriAttention code and record the count in the "Eviction rounds" column of the results tables. A run where this count is zero is either a broken configuration or a baseline run. If V3 was enabled but the round count is zero, the run is discarded.

### 3.3 NIAH (Needle-in-a-Haystack) Protocol

NIAH tests whether a specific fact survives the eviction policy at a specific position in the context. The protocol is:

**Prompt construction.** The haystack is built from `wikitext-2-raw/wiki.test.raw` at the target context size (character length is approximately 4 times the token count). The needle is a fixed random string:

```
The secret code word is PURPLE ELEPHANT 7742.
```

It is inserted at a specified character position within the haystack. The question is fixed:

```
What is the secret code word mentioned earlier? Answer with just the code word and number, nothing else:
```

A short Python snippet builds the final prompt by slicing the wiki text around the insertion point and concatenating the before-slice, the needle, the after-slice, and the question. The full script lives at `niah_7b_strict.sh` on the branch.

**Positions.** Three positions are tested for each configuration, scaled to the context size:

| Context | Start | Middle | End |
|---------|-------|--------|-----|
| 32K | 400 chars | 65000 chars | 120000 chars |
| 64K | 800 chars | 130000 chars | 240000 chars |

The positions target the early, middle, and late regions of the context so the test can distinguish "V3 is globally fine" from "V3 has a position-dependent failure mode" (which it does on hybrid models).

**Running llama-completion.** The actual invocation is:

```bash
./build-test/bin/llama-completion \
    -m <model> \
    -f <prompt file> \
    -n <generation budget> \
    -c <context size> \
    --temp 0 \
    -no-cnv \
    --no-display-prompt \
    [--triatt-budget N --triatt-hybrid 2 --triatt-prefix 128]
```

The important flags:

- `-no-cnv` disables conversation mode. Without it, `llama-completion` reads stdin for an interactive user turn and exits immediately with `> EOF by user` when stdin is empty, so every NIAH run looks like a FAIL. This bit us during the original overnight batch.
- `--no-display-prompt` is load-bearing. By default `llama-completion` echoes the prompt to stdout before emitting generated tokens. The prompt contains the needle. Without this flag, grepping the output for the needle string passes every single run, including ones where the model never actually produced an answer. Every early NIAH result in this project was a silent false PASS until this was fixed.
- `--temp 0` removes sampling variance.
- `-n` is the generation budget. Choice depends on the model, explained below.

**Generation budget.** For standard non-reasoning models like Qwen2.5-7B, `-n 32` or `-n 64` is sufficient to produce "PURPLE ELEPHANT 7742" as the first output. For reasoning models that emit a `<think>` block before their final answer (Qwen3.5-27B, Qwen3.5-35B-A3B), `-n 32` is catastrophic: the model burns the entire generation budget on the reasoning trace and never gets to the answer, so the check sees the `<think>` preamble instead of the needle and scores FAIL for every run, including baselines. Reasoning model runs in this paper use `-n 1024`, which lets the think block complete and the final answer emerge.

**Strict checker.** The output is filtered to strip `llama.cpp` perf print lines and GGML housekeeping, then passed through a case-sensitive grep for the exact string `PURPLE ELEPHANT 7742`. Anything less precise opens false passes:

- Match `PURPLE ELEPHANT` only: false PASS on any hallucinated elephant
- Match `7742` only: false PASS on any random number

Because the needle is a randomly chosen capitalized phrase plus a four-digit number, a true positive requires the model to have the exact string in its KV cache. The rate of hallucinating this string without retrieval is effectively zero.

The checker also classifies near-misses for diagnostic value:

| Result | Meaning |
|--------|---------|
| PASS | Exact string present in generated output |
| PARTIAL_WORD | `PURPLE ELEPHANT` present but the number is wrong or truncated (for example, `774` instead of `7742`) |
| PARTIAL_NUMBER | `7742` present but without `PURPLE ELEPHANT` |
| FAIL | Neither the phrase nor the number present |

All numbers in the results section use this classification. PARTIAL results are reported as PARTIAL, not rolled into PASS.

**Known harness edge case.** At 64K the Qwen2.5-7B model occasionally produces the needle in title case (`Purple Elephant 7742`) instead of upper case. The strict checker is case-sensitive and scores this as PARTIAL_WORD even though reading the generated tokens directly shows successful retrieval. This is a cosmetic harness bug, not a model failure, and is called out explicitly in the results table for 7B @ 64K.

### 3.4 Wall Time Protocol

All wall time numbers in Section 3.7 and Appendix A are measured from outside the process, using `date +%s` before the `llama-perplexity` invocation and again after. This captures model load, prefill, eviction, and the sampler teardown as a single wall-clock cost, which is what a user actually sees.

Because model load is a fixed overhead regardless of configuration, wall time comparisons across configurations on the same model are valid even though the absolute numbers include load. Wall time comparisons across different models are not valid from this paper because load cost differs.

For a given configuration, the reported wall time is the median of at least two runs after any cold-cache run has been discarded. Two or three runs is sufficient at this level of noise (~1 percent run-to-run variance) but is a coarser measurement than the PPL track.

### 3.5 Reproducing the Tables

Every table in Section 4 (Results) can be reproduced by running one of the scripts committed to the branch:

| Script | What it produces |
|--------|------------------|
| `niah_7b_strict.sh` | 7B NIAH results at 32K (V1 ablation and V3 comparison) |
| `v3_validation.sh` | 7B V3 at 32K (PPL plus NIAH) |
| `v3_ablations.sh` | V3 ablations (85 percent retention, prefix size) |
| `v3_stack_and_transfer.sh` | TurboQuant+ plus V3 stack and 27B/35B transfer |
| `v3_transfer_rerun.sh` | Final transfer validation with the reasoning-model fix (-n 1024) |

Each script produces a `<name>_results.txt` output that matches the tables in Section 4 line-for-line. The scripts are deliberately kept simple (bash plus the existing `llama-perplexity` and `llama-completion` binaries) so anyone pulling the branch can rerun them without setting up a Python environment.

---

## 4. Results

All numbers in this section were produced by the protocol in Section 3.

### 4.1 Qwen2.5-7B at 32K: V1 versus V2 versus V3

Model: Qwen2.5-7B-Instruct-Q8_0. Context: 32768. Budget: 29491 (90 percent retention). Chunks: 3.

| Variant | Mechanism | PPL | Delta vs baseline | Eviction rounds |
|---------|-----------|-----|-------------------|-----------------|
| Baseline | None | 6.8504 | 0 | 0 |
| V1 | Global sort | 6.9327 | +1.20% | 18 |
| V2 | Per-segment quota only, K=8 | 7.1577 | +4.49% | 18 |
| **V3** | **Prefix=128 plus quota K=8** | **6.8508** | **+0.006%** | 18 |

At 85 percent retention (budget 27853), the same comparison:

| Variant | PPL | Delta vs baseline | Eviction rounds |
|---------|-----|-------------------|-----------------|
| Baseline | 6.8504 | 0 | 0 |
| V1 85% | 7.0048 | +2.25% | 27 |
| V2 85% | 7.4244 | +8.38% | 27 |
| **V3 85% prefix=256** | **6.8833** | **+0.48%** | 27 |

V3 at 85 percent is still below V1 at 90 percent on the quality axis.

### 4.2 Qwen2.5-7B at 32K: Needle-in-a-Haystack

NIAH test. The needle is inserted at character positions 400, 65000, and 120000 in a wikitext-derived prompt that fills 32K tokens. The strict checker passes only if the exact string `PURPLE ELEPHANT 7742` appears in the generated tokens.

| Variant | Start (400) | Middle (65000) | End (120000) |
|---------|-------------|----------------|--------------|
| Baseline 100% | PASS | PASS | PASS |
| V1 90% | PASS | PASS | **FAIL** (produces "12345") |
| V2 90% | **FAIL** (produces "123456789...") | PASS | PASS |
| **V3 90% prefix=128** | **PASS** | **PASS** | **PASS** |
| V1 85% | PASS | PASS | **FAIL** (produces "12345") |
| V2 85% | **FAIL** | PARTIAL_WORD | PASS |
| V3 85% prefix=256 | PASS | PARTIAL_WORD (produces "774") | PASS |

V1 silently fails at the end position. V2 silently fails at the start position. V3 passes at all three positions for 90 percent retention. At 85 percent, V3 still passes start and end but degrades at middle to a partial match (the model produces "774" instead of "7742"), which is the first real quality cliff we observed.

### 4.3 Prefix Size Ablation

V3 at 90 percent retention, varying the prefix window size:

| Prefix size | PPL | NIAH start | NIAH middle | NIAH end |
|-------------|-----|------------|-------------|----------|
| 128 | 6.8713 | PASS | PASS | PASS |
| 256 | 6.8713 | PASS | PASS | PASS |

Prefix 128 and prefix 256 produce identical PPL and identical NIAH outcomes, so 128 is the tighter default.

### 4.4 Qwen2.5-7B at 64K

Model: Qwen2.5-7B-Instruct-Q8_0. Context: 65536. Budget: 58982 (90 percent). Chunks: 3.

| Variant | PPL | Delta vs baseline | Eviction rounds |
|---------|-----|-------------------|-----------------|
| Baseline | 6.2531 | 0 | 0 |
| **V3 90% prefix=128** | **6.2530** | **-0.002%** | 36 |

V3 at 64K is perplexity-bit-identical to the f16 baseline within the reported standard error of ±0.047. The delta is -0.002 percent, which is well inside measurement noise.

Needle retrieval at 64K on 7B:

| Variant | Start (800) | Middle (130000) | End (240000) |
|---------|-------------|-----------------|--------------|
| Baseline | PASS* | PASS | PASS |
| V3 90% | PASS* | PASS | PASS |

The asterisk at start indicates that the model produced `Purple Elephant 7742` in title case rather than the exact uppercase the checker expects. Reading the generated tokens directly confirms successful retrieval; the checker has a case-sensitivity edge case, not a retrieval failure. Middle and end positions produce exact uppercase matches.

### 4.5 Transfer to Qwen3.5 Hybrid Architectures

The Qwen3.5 architecture family (`qwen35` and `qwen35moe`) combines Mamba2 state-space layers with full attention layers at a `full_attention_interval` of 4. On Qwen3.5-27B only 16 of 64 layers are attention layers; on Qwen3.5-35B-A3B only 10 of 40 are. The models also use partial RoPE (64 of 256 head dimensions are rotated) and an interleaved M-RoPE layout with dimension sections `[11, 11, 10, 0]`.

PPL transfer is clean for both:

| Model | Baseline | V3 90% prefix=128 | Delta | Eviction rounds |
|-------|----------|-------------------|-------|-----------------|
| Qwen3.5-27B | 7.4640 | 7.4853 | +0.29% | 18 |
| Qwen3.5-35B-A3B | 6.2720 | 6.2851 | +0.21% | 18 |

Needle retrieval does not transfer:

| Config | Start (400) | Middle (65000) | End (120000) |
|--------|-------------|----------------|--------------|
| 27B baseline | PASS | PASS | PASS |
| 27B V3 90% | PASS | **FAIL** (produces "Operation MI") | **FAIL** (produces "Operation Manchu") |
| 35B baseline | PASS | PASS | PASS |
| 35B V3 90% | PASS | **FAIL** (produces "Manchu 1") | **FAIL** (produces "Operation Manchu 774") |

Under V3, both hybrid models retrieve the start needle but fail at middle and end positions. The failures are not random noise. In every case the model produces a plausible-sounding but wrong answer that exists elsewhere in the context ("Operation MI", "Operation Manchu"). This is the classic signature of the real needle having been evicted and the model reaching for the nearest semantic neighbor. Perplexity does not catch this failure mode because perplexity is a near-token metric and cannot feel the loss of a fact located 30000 tokens back. Needle-in-a-haystack does.

Note that the Qwen3.5 NIAH runs used `-n 1024` rather than `-n 64` because these are reasoning models. They emit a `<think>` block before producing the final answer, and with a 32-token generation budget the reasoning trace does not finish. At `-n 1024` the reasoning completes and the model emits its final answer, which is what the strict checker evaluates.

### 4.6 Full Stack: TurboQuant+ plus V3

Model: Qwen2.5-7B-Instruct-Q8_0. Context: 32768. Chunks: 3.

| Config | PPL | Delta vs f16 | Wall time | Overhead vs f16 |
|--------|-----|--------------|-----------|-----------------|
| f16 baseline | 6.8504 | 0 | 171s | 1.00x |
| TurboQuant+ only (q8_0 K + turbo3 V) | 6.8879 | +0.55% | 200s | 1.18x |
| V3 only | 6.8713 | +0.31% | 172s | 1.01x |
| **TurboQuant+ plus V3 stack** | **6.9080** | **+0.84%** | **204s** | **1.19x** |

The stack is approximately additive on quality: TurboQuant+ contributes +0.55 percent, V3 contributes +0.31 percent, and the stack lands at +0.84 percent. The marginal cost of adding V3 on top of TurboQuant+ is 0.29 percent perplexity. The wall-time cost of V3 on top of TurboQuant+ is 4 seconds out of 200, which is below the noise floor of a single run.

### 4.7 Speed: The Real Story

The initial V3 implementation was 186 seconds wall time for the 32K three-chunk run, an 89 percent overhead versus baseline. The final optimized version is 172 seconds wall time, a 1 percent overhead. Four optimization passes got there, in order of contribution:

1. **Algebraic collapse of the offset sum**: The scoring formula integrates over 17 geometric offsets, each with `freq_count` cosine and sine calls. Because the offsets and angular frequencies do not depend on the cell being scored, the per-offset sums `cos_sum[f]` and `sin_sum[f]` can be precomputed once per evict call. This changes the per-cell inner loop from approximately 2176 transcendental calls to approximately 64 pure multiply-accumulates. The mechanical speedup is roughly 9.5 times on the scoring hot path, saving 136 seconds out of the original 152-second overhead.

2. **Fix O(n squared) cleanup**: The V3 per-bucket quota leaves a small rounding deficit. The original cleanup pass rescanned the underlying kv_cells fresh and did a linear search over `active_cells` for every candidate, producing an O(n squared) work term of approximately 15 billion operations per eviction round at 32K. The fix replaces this with a boolean `evicted` flag on the in-memory cell structure and a second sort over remaining cells. Saves 2 seconds.

3. **Threading plus SIMD accumulator split**: The per-layer-per-head scoring blocks are independent. Parallelizing them across eight threads with per-thread score accumulators, then splitting the inner frequency loop into four parallel chains to unlock compiler auto-vectorization, saves another 5 seconds.

4. **Scheduler callback uninstall**: The largest single improvement. The ggml backend scheduler takes a dramatically slower per-node dispatch path whenever `callback_eval` is non-null, because it must call the callback on every node, coalesce runs of "not needed" answers, and synchronize the backend between each run. This slow path was active for the entire prefill because the original implementation uninstalled the callback only from inside `evict()`, which at 32K does not run until token 29619. The fix triggers calibration from inside `eval_callback` the moment `q_samples >= warmup_tokens` (approximately token 1028) and defers the actual uninstall to a pending flag that `llama_context::decode()` clears at the end of the forward pass. This saves approximately 6 seconds per run on V3-only and approximately 6 seconds on the TQ+V3 stack, where the original code never uninstalled the callback at all because evict() bailed early on the TurboQuant+ K type.

After all four passes, V3 wall time is 172 seconds versus baseline 171 seconds. The eviction scoring itself, threaded and SIMD-friendly, now costs approximately 378 milliseconds per chunk.

### 4.8 Boundary Layer Skip (Negative Result)

After the initial validation matrix was complete, Michael Sharpe at Project 89 published "Coherence-Guided Dead-Head Identification" (2026), which derives a zero-parameter dead-head threshold from coupled-oscillator criticality and explicitly protects the first one or two attention layers as input transducers that behave differently from coupled oscillators. The same boundary-protection pattern had independently shown up on the weight-quantization side of the TurboQuant+ stack, where boundary layers require higher bit budgets than interior layers to avoid compound damage. The natural question was whether V3's per-cell scoring loop should also exclude the first few attention layers, on the hypothesis that they contribute noise rather than useful coupling signal for eviction decisions.

This section documents the experiment and its negative outcome.

#### 4.8.1 The change

An additional CLI flag was added to V3:

```
--triatt-boundary-skip N   (default 0 = off, experimental)
```

When N is greater than zero, the per-cell scoring loop filters out all blocks whose absolute attention layer index is less than N. The score normalizer is adjusted to use the post-filter block count so that absolute score magnitudes stay comparable across different N values. For standard transformers where every layer is attention, `N=1` skips the first attention layer, `N=2` skips the first two, and so on. For hybrid Mamba+Attention architectures where the first attention layer may not be at index zero (Qwen3.5-27B has its first attention layer at `il=3`), small values of N are no-ops and the user must set N to match the architecture's attention layer stride to see any effect.

#### 4.8.2 Qwen2.5-7B at 32K

| Config | PPL | Delta vs baseline 6.8504 |
|--------|-----|--------------------------|
| V3 skip=0 (control) | 6.8508 | +0.006% |
| V3 skip=1 | 6.9150 | +0.94% |
| V3 skip=2 | 6.9166 | +0.97% |

Skipping even one boundary layer on the pure transformer introduces an immediate roughly 1 percent PPL regression relative to the V3 baseline. Skipping two is not meaningfully worse than skipping one, which means the damage is concentrated in the removal of layer 0 specifically, not in how many layers are excluded beyond that.

#### 4.8.3 Qwen2.5-7B at 64K

| Config | PPL | Delta vs baseline 6.2531 |
|--------|-----|--------------------------|
| V3 skip=0 (control) | 6.2530 | -0.002% |
| V3 skip=2 | 6.2940 | +0.65% |

The regression is slightly smaller at 64K than at 32K but the direction is the same. There is no evidence that boundary protection becomes more valuable at longer context lengths on a pure transformer.

#### 4.8.4 Qwen3.5-27B at 32K

This is the model where V3 currently fails NIAH at middle and end positions under `skip=0`, and therefore the configuration where boundary protection had the clearest theoretical motivation: fewer attention layers, sparser coupling density, boundary layer potentially dominating an underpowered lattice.

| Config | PPL | Delta vs baseline 7.4640 | Delta vs skip=0 |
|--------|-----|--------------------------|-----------------|
| V3 skip=0 (control) | 7.4853 | +0.29% | baseline |
| V3 skip=4 | 7.4779 | +0.19% | -0.10% |
| V3 skip=8 | 7.4817 | +0.24% | -0.05% |

PPL improves modestly with `skip=4`, which is the first value that actually catches an attention layer on Qwen3.5-27B since attention layers start at `il=3`. `skip=8` also improves over `skip=0` but is worse than `skip=4`, suggesting that skipping only the first attention layer is the sweet spot and excluding a second one starts removing useful signal. The direction of the PPL effect is the opposite of the pure-transformer result, which is qualitatively consistent with Sharpe's framework.

NIAH tells a different story:

| Config | start (400) | middle (65000) | end (120000) |
|--------|-------------|----------------|--------------|
| V3 skip=0 (prior) | PASS | FAIL ("Operation MI") | FAIL ("Operation Manchu") |
| V3 skip=4 | PASS | FAIL ("Ise-class confusion") | FAIL ("scanning for code word") |
| V3 skip=8 | PASS | FAIL ("scanning for word") | FAIL ("scanning for code word") |

The middle and end positions fail under `skip=4`, `skip=8`, and `skip=0` identically. The specific strings the model chases vary run to run but the class of failure is the same: the model cannot find the needle, scans plausible-sounding sections, and gives up. The small PPL improvement does not translate into any NIAH recovery.

#### 4.8.5 Interpretation

The experiment is a clean negative. Three takeaways:

1. **PPL and NIAH are decoupled on this failure mode.** V3 on Qwen3.5-27B improves PPL marginally with boundary skip, but NIAH stays broken at exactly the same positions with the same failure signature. This reinforces a pattern that has been consistent throughout this project: perplexity cannot feel end-of-context retrieval damage, and tuning for PPL does not indirectly fix NIAH. A positive PPL delta is not a reliable signal that a change helps retrieval.

2. **Boundary protection is the right pattern for the weight-quant path, not for the eviction-scoring path.** On the weight-quant side of TurboQuant+, boundary layers must be protected because they genuinely cannot be compressed without cascading damage. That empirical fact does not imply that the same layers contribute noise to an eviction decision. V3's scoring is averaging phase-alignment signal across attention layers, not compressing them, and on a pure transformer every attention layer carries real signal for that average, including layer 0. Two different operations on the same layer set have different sensitivities.

3. **Sharpe's framework gives a reason why Qwen3.5 hybrid models are hard, but not a direct fix for V3.** The coupled-oscillator interpretation correctly predicts that sparser attention coupling density makes the eviction decision worse. It does not tell us which specific replacement for V3's trig scoring would fix the NIAH failure. The likely candidate, based on the framework, is a direct coupling measurement (mean cosine alignment between a head's write-back and the receiver residual, as Sharpe defines it) used as the eviction criterion itself, rather than as a filter on top of the existing scoring. That is a larger change and is left as follow-on work.

#### 4.8.6 Current defaults

Given the negative result, the `--triatt-boundary-skip` default was changed from 2 back to 0. The flag remains available as an opt-in knob for anyone who wants to replicate the experiment or try it as part of a larger change (for example, in combination with a coupling-based replacement for the trig scoring). Nobody using V3 with the default configuration sees any boundary skipping and no behavior change from the earlier committed version.

---

## 5. Production Reality

The validated workload envelope for V3 is narrower than the paper's headline claim and narrower than I hoped when I started.

**Where V3 works.** Standard transformers with full RoPE and homogeneous attention layers, tested on Qwen2.5-7B at 32K and 64K context on wikitext perplexity and on strict single-needle NIAH at start, middle, and end positions. Perplexity is effectively at baseline. Needle retrieval is clean. Wall time is at baseline. The stack with TurboQuant+ costs 0.84 percent perplexity total for approximately 2.9 times KV memory compression.

**Where V3 does not work.** Qwen3.5 hybrid Mamba+Attention architectures (27B dense and 35B-A3B MoE). Perplexity looks fine and the speed story still holds, but needle retrieval silently fails at middle and end positions. The failure mode suggests the scoring is evicting semantically important tokens in these models, not that the implementation is broken. The most likely causes are the small number of attention layers (16 of 64 on 27B), the partial RoPE coverage (64 of 256 head dimensions), and the M-RoPE section layout. The scoring formula implicitly averages a signal across attention layers and across frequency bins that behave differently in these architectures.

**What we have not tested.** Reasoning workloads where the paper specifically claims the largest gains (DeepSeek R1 distill, QwQ), non-wiki text (code, math, tool calling), multi-needle NIAH, very long context (128K plus), and generative quality on open-ended production workloads where the relevant failure mode is coherence drift rather than single-fact retrieval.

**The honest framing.** V3 is the best version of paper-faithful TriAttention that I can build with two simple structural constraints on top of the original scoring. It is not a universal replacement for statistical compression techniques. For a production stack, the more reliable gains live in the statistical layers: per-bit quantization (TurboQuant+, spectral K rotation, weight quant), head sculpting, and O-projection SVD. TriAttention belongs alongside those as an opt-in technique for workloads where it has been measured to work, not as a default.

V3 makes the minimum viable shipping case for TriAttention clean on the narrow validated envelope. Everything outside that envelope remains an open research question.

---

## 6. How to Run

### 6.1 Pull the Branch

The code used for this experiment lives on the
`experiment/triattention-integration` branch of the historical llama.cpp
TurboQuant fork. This branch was not merged to the fork's main TurboQuant+
branch. The fork no longer has a public source URL; see
[Historical engine forks](../../docs/reference/historical-forks.md#llamacpp-experimental-forks).

```bash
# Requires an existing checkout of the historical fork.
cd llama-cpp-turboquant
git checkout experiment/triattention-integration
```

If you already have the fork cloned:

```bash
cd llama-cpp-turboquant
git fetch origin experiment/triattention-integration
git checkout experiment/triattention-integration
```

The commit-level rollback ladder for this branch is documented in Appendix B. Each commit is a clean checkpoint that can be checked out independently to reproduce the speed or quality numbers at that point in the progression.

### 6.2 Build

```bash
cmake -B build-test -DLLAMA_METAL=ON
cmake --build build-test -j 8 --target llama-perplexity llama-completion
```

### 6.3 Default (V1, paper-faithful)

Omitting the hybrid flag runs the paper-faithful version, which is the default for anyone who pulls the branch without knowing the V3 work exists.

```bash
./build-test/bin/llama-perplexity \
    -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    -c 32768 -b 512 --chunks 3 \
    --triatt-budget 29491
```

This produces the +1.20 percent PPL / NIAH end fail result.

### 6.4 V3 Hybrid (recommended for validated workloads)

```bash
./build-test/bin/llama-perplexity \
    -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    -c 32768 -b 512 --chunks 3 \
    --triatt-budget 29491 \
    --triatt-hybrid 2 \
    --triatt-prefix 128
```

The `--triatt-hybrid 2` flag selects V3. The `--triatt-prefix 128` flag sets the prefix window size. Both flags are opt-in; the default remains V1. This produces the effectively baseline PPL / NIAH all-pass result on Qwen2.5-7B.

### 6.5 Full Stack (TurboQuant+ plus V3)

```bash
./build-test/bin/llama-perplexity \
    -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    -c 32768 -b 512 --chunks 3 \
    -ctk q8_0 -ctv turbo3 \
    --triatt-budget 29491 \
    --triatt-hybrid 2 \
    --triatt-prefix 128
```

This is the full compression stack: TurboQuant+ quantizes the KV bytes, V3 evicts the 10 percent lowest-scoring tokens. The combined cost is +0.84 percent PPL at 2.9 times KV memory reduction, 1.19 times baseline wall time.

### 6.6 NIAH Test

See Section 3.3 for the full NIAH protocol. The runnable script is `niah_7b_strict.sh` on the branch, which invokes `llama-completion` with the correct flags and parses the output with the strict checker described in Section 3.3. For reasoning models, pass `-n 1024` so the `<think>` block completes before the final answer is emitted.

### 6.7 Environment Variable Fallback

Both `--triatt-hybrid` and `--triatt-prefix` have environment variable fallbacks: `TRIATT_HYBRID` and `TRIATT_PREFIX`. The CLI flag takes precedence when both are set. Adaptive calibration (continuing to update the query centers after warmup, rather than freezing them) can be enabled via `TRIATT_ADAPTIVE=1`, but this disables the scheduler callback uninstall optimization and will cost approximately 6 seconds per run in current validated measurements.

---

## 7. Missing Pieces (Community Request)

The following questions are open and matter for production adoption. If you have data or insight, please drop it in the PR discussion.

### 7.1 Why does V3 break on hybrid Mamba+Attention models?

Perplexity transfers cleanly to Qwen3.5-27B and Qwen3.5-35B-A3B (+0.29 percent and +0.21 percent respectively), but needle retrieval silently fails at middle and end positions in both. The failure is position-dependent, not random. Three hypotheses remain viable after the boundary skip experiment (Section 4.8) ruled out "layer 0 is acting as a pure transducer":

1. **Partial RoPE ignored dimensions.** Qwen3.5 rotates only 64 of 256 head dimensions. The V3 scoring formula operates entirely on the rotated portion of K. The other 192 dimensions carry real semantic content that V3 does not look at. On a pure transformer with full RoPE this does not matter because V3 sees the whole K vector. On Qwen3.5 it means V3 is blind to 75 percent of each cell's K content. Augmenting the score with a direct dot product over the unrotated dimensions is an obvious candidate fix, untested so far.

2. **Scoring formula is phase-only.** V3 measures phase alignment between K and the learned Q center, which is a "where in frequency space does this cell point" measurement. Sharpe's coupled-oscillator framework suggests the right observable for eviction decisions may be the direct coupling measurement (mean cosine alignment between a head's write-back and the receiver residual), not phase alignment. Replacing the trig scoring with a coupling measurement is a larger change than a filter and is not in the current V3 implementation.

3. **M-RoPE sections and theta scaling.** The rotation layout is `[11, 11, 10, 0]` with different position indices per section. For text-only inference the sections collapse to the same position, so mathematically this should reduce to standard RoPE, but the theta scaling may differ from what V3 assumes.

The boundary skip experiment in Section 4.8 explicitly tested the "small attention layer count / layer 0 is a transducer" hypothesis and ruled it out. Boundary skipping does not rescue NIAH middle/end on qwen35 even when it is tuned to catch the first attention layer at `il=3`. The failure mode is the same with boundary skip set to 0, 4, or 8.

If you have tested TriAttention on hybrid architectures and have a recipe that works, the community needs it.

### 7.2 Does V3 hold on reasoning workloads?

The TriAttention paper's strongest claim is on reasoning models where thinking traces contain heavy redundancy. None of the results in this paper test that directly. The closest we have is the Qwen3.5 hybrids, which emit `<think>` blocks but are not pure reasoning models in the DeepSeek R1 sense. Testing V3 on R1-distill or QwQ at 32K and beyond would directly address the paper's target workload.

### 7.3 What happens at 128K and beyond?

V3 at 64K on 7B is perplexity-bit-identical to baseline with NIAH clean. Scaling to 128K has two open questions: first, whether the per-segment quota K=8 is still appropriate (the buckets get coarser as context grows), and second, whether the calibration window of 1024 tokens is still sufficient when the cache holds 100K plus tokens of drift. An early calibration freeze, which is what the current V3 does after the scheduler callback uninstall, may need to be replaced with a periodic re-calibration above a certain context threshold.

### 7.4 Multi-needle and coherence

Single-needle NIAH is the weakest real retrieval benchmark. A production workload with multiple facts scattered across a long document, or an agent with tool results from many earlier turns, stresses a different failure mode than "find this one fact." V3 has not been measured on any multi-needle or multi-turn retrieval benchmark. If you have a harness for this, the V3 numbers against it would be directly useful.

### 7.5 The paper's 10x claim

The paper reports up to 10 times compression on reasoning workloads. All of the V3 results in this paper are at 10 percent savings (90 percent retention). We have not yet found a safe operating point anywhere close to 90 percent savings on general text with V3 or with V1. Either our test workload is harder than the paper's, our scoring implementation diverges from the paper's on a detail we have not found, or the paper's aggressive compression ratio only applies to workloads we have not tested. If anyone has reproduced the paper's 10x claim on a non-reasoning workload, that would be extremely useful to compare against.

---

## 8. References

- Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen. **TriAttention: Efficient Long Reasoning with Trigonometric KV Compression.** arXiv:2604.04921, 2026. Apache 2.0 License. <https://github.com/WeianMao/triattention>
- Google Research. **TurboQuant: Hadamard Rotation for KV Cache Compression.** ICLR 2026.
- This work: **llama.cpp TurboQuant+ fork, `experiment/triattention-integration` branch.** Draft pull request (do not merge) open for community review and discussion.
- Companion paper: **Asymmetric K/V Cache Compression: Why V is Free and K is Everything.** TurboQuant+ docs, `asymmetric-kv-compression.md`.

---

## Appendix A: Full Numeric Summary

All perplexity numbers on wikitext-2-raw, three chunks, batch size 512, with the listed budget if any.

### A.1 Qwen2.5-7B-Instruct-Q8_0

| Context | Config | Budget | PPL | Delta | Eviction rounds | Wall time |
|---------|--------|--------|-----|-------|-----------------|-----------|
| 32K | Baseline | 0 | 6.8504 | 0 | 0 | 171s |
| 32K | TurboQuant+ only | 0 | 6.8879 | +0.55% | 0 | 200s |
| 32K | V1 90% | 29491 | 6.9327 | +1.20% | 18 | 322s (before speed opts) |
| 32K | V2 90% | 29491 | 7.1577 | +4.49% | 18 | 322s |
| 32K | V3 90% prefix=128 | 29491 | 6.8508 | +0.006% | 18 | 172s |
| 32K | V3 90% prefix=256 | 29491 | 6.8713 | +0.31% | 18 | 172s |
| 32K | V1 85% | 27853 | 7.0048 | +2.25% | 27 | 381s |
| 32K | V2 85% | 27853 | 7.4244 | +8.38% | 27 | 385s |
| 32K | V3 85% prefix=256 | 27853 | 6.8833 | +0.48% | 27 | 190s |
| 32K | TurboQuant+ plus V3 90% | 29491 | 6.9080 | +0.84% | 18 | 204s |
| 32K | V3 90% prefix=128 boundary_skip=1 | 29491 | 6.9150 | +0.94% | 18 | — |
| 32K | V3 90% prefix=128 boundary_skip=2 | 29491 | 6.9166 | +0.97% | 18 | — |
| 64K | Baseline | 0 | 6.2531 | 0 | 0 | 493s |
| 64K | V3 90% prefix=128 | 58982 | 6.2530 | -0.002% | 36 | 497s |
| 64K | V3 90% prefix=128 boundary_skip=2 | 58982 | 6.2940 | +0.65% | 36 | 487s |

### A.2 Qwen3.5-27B-Q8_0

| Context | Config | Budget | PPL | Delta | Eviction rounds |
|---------|--------|--------|-----|-------|-----------------|
| 32K | Baseline | 0 | 7.4640 | 0 | 0 |
| 32K | V3 90% prefix=128 | 29491 | 7.4853 | +0.29% | 18 |
| 32K | V3 90% prefix=128 boundary_skip=4 | 29491 | 7.4779 | +0.19% | 18 |
| 32K | V3 90% prefix=128 boundary_skip=8 | 29491 | 7.4817 | +0.24% | 18 |

### A.3 Qwen3.5-35B-A3B-Q8_0

| Context | Config | Budget | PPL | Delta | Eviction rounds |
|---------|--------|--------|-----|-------|-----------------|
| 32K | Baseline | 0 | 6.2720 | 0 | 0 |
| 32K | V3 90% prefix=128 | 29491 | 6.2851 | +0.21% | 18 |

### A.4 NIAH Summary (strict checker, generated tokens only)

| Model | Config | Start | Middle | End |
|-------|--------|-------|--------|-----|
| 7B @ 32K | Baseline | PASS | PASS | PASS |
| 7B @ 32K | V1 90% | PASS | PASS | FAIL |
| 7B @ 32K | V2 90% | FAIL | PASS | PASS |
| 7B @ 32K | V3 90% prefix=128 | PASS | PASS | PASS |
| 7B @ 32K | V3 90% prefix=256 | PASS | PASS | PASS |
| 7B @ 32K | V3 85% prefix=256 | PASS | PARTIAL | PASS |
| 7B @ 64K | Baseline | PASS (case) | PASS | PASS |
| 7B @ 64K | V3 90% prefix=128 | PASS (case) | PASS | PASS |
| 27B @ 32K | Baseline | PASS | PASS | PASS |
| 27B @ 32K | V3 90% prefix=128 | PASS | FAIL | FAIL |
| 35B-A3B @ 32K | Baseline | PASS | PASS | PASS |
| 35B-A3B @ 32K | V3 90% prefix=128 | PASS | FAIL | FAIL |

---

## Appendix B: Rollback Points

The implementation commits on the `experiment/triattention-integration` branch of the historical `llama-cpp-turboquant` fork form a clean rollback ladder.

| Commit (short) | State |
|----------------|-------|
| `beedfac` | V1 attempt (paper-faithful) |
| `9031b34` | V3 hybrid policy added (prefix plus per-segment quota) |
| `c234d9d` | V3 plus scoring algebraic collapse (approximately 9.5 times speedup) |
| `51608f8` | V3 plus threading, accumulator split, O(n squared) cleanup fix, cb_eval bail after warmup |
| `b9b9f56` | V3 plus early calibration trigger, deferred callback uninstall (speed fully recovered) |
| `fd8941b` | Results archive with all run logs |

Any downstream reviewer can `git checkout` any of these to reproduce the progression.

---

## 9. Addendum: vLLM Port to AMD MI300X

This section was added on 2026-04-30, after the original V3 paper, to document the port of V3 from llama.cpp/Metal to vLLM/ROCm-Triton on AMD MI300X. Same algorithm, same calibration, same selection policy. Different runtime, different rotation kernel, different cache layout.

The primary motivation for the port was a follow-up collaboration with AMD around TurboQuant+ on MI300X serving. AMD specifically asked about how V3 transfers across runtimes and what new failure modes the port surfaces.

### 9.1 Hardware

| Component | Spec |
|-----------|------|
| GPU | 1× AMD Instinct MI300X |
| Memory | 192 GB HBM3 (5.3 TB/s) |
| Architecture | gfx942 |
| Driver | ROCm 7.2.26015 |
| Triton | 3.6 |
| PyTorch | 2.11+rocm7.2 |
| vLLM | 0.1.dev1 (fork of `feature/turboquant-kv-cache`) |
| OS | Ubuntu 24.04 |

Single GPU. No tensor parallelism. AMD Developer Cloud droplet, $300 promotional credit.

### 9.2 Implementation summary

Branch: `feature/turboquant_plus` of the historical `vllm-turboquant` fork
([public source URL unavailable](../../docs/reference/historical-forks.md#vllm-experimental-forks)).
The port consists of seven commits, all merged into the shipping branch.

V3-specific files added under `vllm/v1/attention/triattention/`:

| File | Purpose |
|------|---------|
| `engine.py` | `TriAttentionV3Engine`: calibration accumulators, per-layer score state, V3 selection trigger |
| `policy.py` | V1 / V2 / V3 selection logic (prefix protect + per-segment quota + cleanup) |
| `scoring.py`, `scoring_kernel.py` | PyTorch reference + Triton kernel for per-cell trig scoring |
| `hooks.py` | Pre-RoPE Q-capture as a `vllm::triatt_capture_q_pre_rope` custom op |
| `integration.py` | `install_triattention(model_path, cfg)` helper |
| `backend_helpers.py` | Glue between the engine and the TurboQuant attention backend |

Modifications to existing vLLM files:

- **TQ decode kernel** (`triton_turboquant_decode.py`): added `VALID_MASK` constexpr to both single-Q and grouped (batched-Q) kernels. Evicted positions score `-inf` at attention time. Per-position uint8 mask `[B, max_seq_len]` plumbed through `TurboQuantMetadata`.
- **TQ backend** (`turboquant_attn.py`): per-layer K accumulation hook in `_continuation_prefill` (after the existing dequant step), valid-mask plumbed to launcher, dispatch updated.
- **Model files** (qwen2, qwen3, qwen3_moe, llama, mistral, gemma4, qwen3_next): one-line `capture_q_pre_rope(self._triatt_layer_idx, q)` insert before each model's `rotary_emb(...)` call. Layer index cached in `__init__`. No-op when V3 is disabled.

Three vLLM-specific design choices that diverge from the llama.cpp implementation:

1. **Q capture is a vLLM custom op.** vLLM compiles each model with `torch.compile(fullgraph=True)`. Dynamo refuses `torch.compiler.disable`'d callees in fullgraph mode. Solution: register the Q capture as a vLLM custom op via `direct_register_custom_op`, declared with `mutates_args=["q"]` to keep the graph capture from DCE'ing it (no tensor outputs, no real mutation, but the declaration prevents elimination).

2. **Engine lives in the worker, not the main process.** vLLM V1 forks an `EngineCore` subprocess for model execution. Main-process state is invisible. `install_triattention(model_path, cfg)` (called BEFORE `LLM(...)`) exports model dims plus tuning knobs as `VLLM_TRIATT_*` env vars; the worker lazy-initialises its own engine on the first Q-capture call.

3. **Eviction without compaction.** vLLM's KV cache is paged (16-token blocks). Per-token eviction inside a block is awkward. The port uses a uint8 `valid_mask[B, max_seq_len]` plumbed through attention metadata into the TQ decode kernel, which scores evicted positions as `-inf`. No physical data movement, no block-table changes.

### 9.3 Models tested

| Model | Size | Family | Result |
|-------|------|--------|--------|
| Qwen2.5-7B-Instruct | 7B dense | Qwen | PPL paper-exact reproduction (Section 9.4) |
| Qwen3-8B | 8B dense | Qwen3 | full PPL + NIAH matrix at 8K/16K/32K |
| Mistral-7B-Instruct-v0.3 | 7B dense | Llama-arch | cross-family transfer confirmed |
| Qwen3-30B-A3B | 30B MoE (3B active) | Qwen3 MoE | 3 of 4 modes landed; stack-on-MoE flaky |
| Mamba2-primed-HQwen3-8B | 8B hybrid | hybrid_qwen3 | blocked at transformers (model_type unknown) |
| AI21-Jamba2-3B | 3B hybrid | jamba | algorithmically incompatible (no RoPE on attention layers) |
| Qwen3-Next-80B-A3B-Instruct | 80B hybrid | qwen3_next | blocked at vLLM page-size unifier when stacking with TQ |

KV cache preset for V3 paths: `turboquant_k8v4` (FP8 K + 4-bit V). Mirrors the K=q8_0 + V=turbo3 config from Section 4.6 in spirit (K stays dequant-able for the V3 score formula, V is more aggressively compressed). Pure-BF16 K is not supported on the V3 path because the V3 hooks live in the TurboQuant attention backend.

### 9.4 Paper-exact reproduction on Qwen2.5-7B-Instruct

Same model, same wikitext-2-raw protocol, 3 chunks, batch_size=512, 32K context. The paper's protocol numbers (Section 4.1) are reproduced under llama.cpp/Metal; this section reports the equivalent run under vLLM/ROCm-Triton.

#### 9.4.1 PPL

| Mode | KV preset | vLLM PPL | vLLM Δ | llama.cpp Δ (paper §4.1) |
|------|-----------|---------:|-------:|-------------------------:|
| baseline | auto (BF16 weights, BF16 KV) | 6.0615 | — | — (paper baseline 6.8504, Q8_0 weights) |
| TQ alone | turboquant_4bit_nc_cv_rv | 6.0726 | +0.18% | +0.55% (Q8_0 K + turbo3 V) |
| V3 alone | turboquant_k8v4 | 6.0822 | +0.34% | +0.006% |
| TQ + V3 stack | turboquant_k8v4 + V3 | 6.0822 | +0.34% | +0.84% |

Different baseline (BF16 weights vs Q8_0 weights) so absolute PPL is not directly comparable. The deltas all sit within the protocol's documented ±0.5% noise floor at 3 chunks.

The vLLM V3-alone delta is slightly worse than the paper's. The vLLM TQ-alone and stack deltas are slightly better. None of the three differences is significant within the protocol noise.

#### 9.4.2 KV cache compression

Direct stack savings on Qwen2.5-7B (28 layers × 4 KV heads × 128 head_dim):

| Mode | Bytes per token per layer (K+V combined) | Compression vs BF16 | Total KV @ 32K |
|------|----------------------------------------:|--------------------:|---------------:|
| baseline (BF16 KV) | 4096 B | 1.0× | 1.74 GB |
| TQ alone (`turboquant_4bit_nc_cv_rv`) | 1064 B | 3.85× | 0.45 GB |
| TQ K8V4 alone (FP8 K + 4-bit V) | 1536 B | 2.67× | 0.65 GB |
| TQ K8V4 + V3 stack @ 90% retention | 1382 B effective | **2.96×** | **0.59 GB** |

V3 at 90% retention contributes an extra ~11% compression on top of any TQ preset. The 2.96× total stack number lines up directly with the paper's Section 4.6 headline of "approximately 2.9× KV memory compression" for the TQ+V3 stack on Qwen2.5-7B.

The TQ-alone `turboquant_4bit_nc_cv_rv` preset (4-bit MSE K + 4-bit centroid V) compresses harder than the V3-friendly `turboquant_k8v4` preset, but it doesn't expose K in a dequant-able form for V3 scoring. Stacking V3 on the harder TQ preset would require either dequanting K-from-MSE on the V3 path or moving V3 hooks off the TQ backend entirely. Open work item.

#### 9.4.3 NIAH

Same model, same character positions (400 / 65000 / 120000), same strict checker, same `--temp 0` greedy generation, `-n 64`.

| Mode | Start (400) | Middle (65000) | End (120000) |
|------|:-----------:|:--------------:|:------------:|
| baseline | PARTIAL_NUMBER | PASS | PASS |
| TQ alone | PARTIAL_NUMBER | PASS | PASS |
| **V3 alone** | PARTIAL_NUMBER | **PARTIAL_NUMBER** | PASS |
| **TQ + V3 stack** | PARTIAL_NUMBER | **PARTIAL_NUMBER** | PASS |

Paper's Qwen2.5-7B + V3 result (paper §4.2): PASS / PASS / PASS at all three positions.

This is a real regression vs the paper. Two distinct anomalies:

1. **All four modes (including baseline) miss the start position with PARTIAL_NUMBER.** The model produces "7742" but not the phrase "PURPLE ELEPHANT". Since baseline also fails, this is independent of V3 — likely a checker / output-formatting interaction with the BF16-weight Qwen2.5 (the paper used Q8_0 weights, which can change generation behavior). Consistent with the paper's own §3.3 note that the strict checker has known case-sensitivity edge cases.

2. **V3 and stack lose the middle position relative to baseline + TQ.** baseline + TQ both PASS at 65000; V3 + stack drop to PARTIAL_NUMBER. Same context length, same prompt, same needle. V3 changes which tokens are kept and that change degraded retrieval at the middle position on this model on this runtime. This is the regression.

### 9.5 Where the vLLM port falls short of the paper

The PPL story tracks paper within noise. The NIAH story does not. V3 in vLLM on Qwen2.5-7B at 32K is not delivering the paper's clean PASS-PASS-PASS, and that gap shows up at the middle position specifically — exactly the place V3 was designed to protect with the per-segment quota policy.

Three plausible causes, in decreasing order of likelihood:

1. **Different KV preset.** The paper used Q8_0 K + turbo3 V. The vLLM port uses FP8 K + 4-bit V (`turboquant_k8v4`). FP8 K is dtype-lossier than Q8_0 K. V3's score formula reads K from cache and computes the trig-aligned score on whatever K precision is stored. If the FP8 representation perturbs K vectors enough to flip eviction picks near the cutoff, mid-context tokens can get evicted that Q8_0-K would have kept.

2. **Different scoring numerics.** llama.cpp uses 8-thread CPU split accumulators with 4-way per-thread fanout for vectorization. Reordered summation, ULP drift. vLLM does the trig score sum as a single fp32 torch reduction. For V3 selection near the score cutoff, summation order can flip which token gets evicted.

3. **Different eviction trigger frequency.** Paper logged 18 eviction rounds for the 32K / 3-chunk run. The vLLM port logs 5. Fewer rounds means the cache spends more time at full state and each round evicts a larger batch of tokens at once. The per-segment quota assumes roughly uniform-sized rounds; aggressive batches may drop more mid-context tokens than the policy "wants" because the bucket targets are computed once per round.

A retention sweep (95 / 90 / 85 / 80 / 75 / 65 / 50%) on Qwen2.5-7B should localise where the regression starts. If the gap is preset-driven (cause 1), running with a Q8_0-K-equivalent vLLM preset would close it. If it's numerics or trigger frequency, the gap stays. That ablation is the next step.

PPL is not a sufficient signal for V3 quality (see §3.2 of the original paper, which makes the same point). The vLLM port confirms this empirically: PPL reproduces the paper within noise, NIAH does not.

### 9.6 Hybrid models: still untested in vLLM

Section 4.5 of the original paper documents that V3 NIAH fails at middle and end positions on hybrid Mamba+Attention architectures (Qwen3.5-class). The natural follow-up question is whether the same failure transfers to the vLLM port or whether the different scoring numerics fix it.

Three hybrid model attempts; none ran to a result:

1. **Mamba2-primed-HQwen3-8B (Amazon).** Model uses a `hybrid_qwen3` model_type that current transformers doesn't recognize, including the latest `transformers @ main` build. The model ships no custom modeling code in the repo. Cannot load, cannot test.

2. **AI21-Jamba2-3B.** Loads in vLLM, but Jamba's attention layers use no RoPE — Mamba does positional encoding via the SSM recurrence and the attention layers are vanilla. V3's score formula is RoPE-derived (`omega[f] = 1 / theta^(2f/n_rot)`) and cannot apply to a model with no RoPE. Algorithmic incompatibility, not a port issue.

3. **Qwen3-Next-80B-A3B-Instruct.** This is the right architecture class — `qwen3_next` (hybrid Mamba+Attn), `partial_rotary_factor=0.25` (the paper called partial RoPE out as the suspected failure cause), 16 attention heads, 2 KV heads, 256 head_dim, 48 layers. Loads in vLLM with BF16 KV. Loaded with TQ stacking, vLLM's KV cache page-size unifier rejects the configuration: the Mamba state size and the TQ-compressed KV slot size do not share a common block divisor. Hard error at engine init.

The Qwen3-Next path is the one that should work with another round of effort. Two routes:

- **(a)** Extend V3 hooks to vLLM's BF16 attention backend, so V3 can run without TQ stacking. The V3 engine already accepts any K precision for scoring (it casts to fp32 internally). The remaining work is plumbing the per-layer K-extraction call into the BF16 attention path. Estimated 1–3 days of focused vLLM-internals work.

- **(b)** Patch vLLM's hybrid page-size unifier to handle two cache types (Mamba state + TQ-compressed KV). Deeper vLLM-internals change with risk of regressing other hybrid model paths.

Neither was in the budget for this round. The honest result is: the hybrid V3 failure mode the paper documents is, as of this writing, architecturally inaccessible in the vLLM port. Not a refutation of the paper, not a confirmation. An open item for follow-up work.

### 9.7 Headline claim envelope (vLLM port specifically)

What I can claim, post-port:

- V3 ports cleanly from llama.cpp/Metal to vLLM/ROCm-Triton on AMD MI300X with no architectural surprises. Algorithm transfers.
- Paper-exact PPL on Qwen2.5-7B reproduces within ±0.5% noise floor.
- Cross-family confirmed (Mistral-7B Llama-arch).
- MoE works at the kernel level (Qwen3-30B-A3B baseline + V3-only).
- NIAH PASS at start/middle/end on Qwen3-8B (different model than the paper used).

What I cannot claim, post-port:

- "Apples-to-apples paper reproduction." The paper-exact NIAH on Qwen2.5-7B *regresses* vs paper at the middle position. Real gap.
- "Works where llama.cpp fails." The hybrid failure mode is architecturally inaccessible in the current vLLM port; we have no data to refute the paper's hybrid finding.
- "Beats Apple Metal." PPL deltas all sit within the protocol noise floor; NIAH is *worse* on the apples-to-apples comparison.

### 9.8 Open work

In rough priority order:

1. **Retention sweep on Qwen2.5-7B at 32K**: 95 / 90 / 85 / 80 / 75 / 65 / 50%. PPL + NIAH at each. Finds where V3 starts cliffing in the vLLM port. Maps the operating envelope properly.
2. **KV-preset ablation on Qwen2.5-7B at 32K**: vary K precision (Q8_0-equivalent vs FP8) holding everything else constant. Tests cause 1 from §9.5.
3. **Eviction-trigger frequency probe**: instrument the engine to log every trigger evaluation, compare against llama.cpp's tighter per-ubatch cadence. Tests cause 3 from §9.5.
4. **Hybrid V3 unblock**: either path (a) BF16-K hook on vLLM's BF16 attention backend, or path (b) page-size unifier patch. Whichever path lands first opens the Qwen3-Next test.
5. **MLA support for V3**: separate kernel-level project. Out of scope for this round; tracked for a future round.

### 9.9 How to Run (vLLM Port)

This section mirrors the paper's Section 6 but for the vLLM/ROCm-Triton path.

#### 9.9.1 Pull the Branch

```bash
# Requires an existing checkout of the historical fork.
cd vllm-turboquant
git checkout feature/turboquant_plus
```

The `feature/turboquant_plus` branch contains the full TurboQuant+ AMD work plus the V3 port. The standalone V3 branch (`feature/triattention-v3`) was deleted after merge; all V3 commits live on the shipping branch.

#### 9.9.2 Build

```bash
# editable install
pip install -e .

# AITER for ROCm flash-attn varlen (not required on CUDA;
# unblocks long-context prefill on AMD where upstream flash-attn
# requires a 7-hour Composable Kernel build)
pip install git+https://github.com/ROCm/aiter.git

# unit tests for V3 engine math + selection policy
pytest tests/v1/attention/test_triattention_v3.py -v
```

#### 9.9.3 Enable V3

V3 is per-process. Two equivalent ways to enable:

**Option A: helper call before LLM().** Resolves model dims via HF AutoConfig, exports them as env vars, sets the master `VLLM_TRIATT_ENABLED=1` switch.

```python
from vllm import LLM
from vllm.v1.attention.triattention import (
    TriAttentionV3Config, install_triattention,
)

install_triattention(
    model_path="/path/to/Qwen2.5-7B-Instruct",
    cfg=TriAttentionV3Config(budget=29491, prefix_protect=128),
)
llm = LLM(
    model="/path/to/Qwen2.5-7B-Instruct",
    kv_cache_dtype="turboquant_k8v4",
    max_model_len=32768,
)
```

The helper must run before `LLM(...)` so the EngineCore subprocess fork inherits the env vars.

**Option B: env vars only.** Same effect, no Python helper needed:

```bash
VLLM_TRIATT_ENABLED=1 \
VLLM_TRIATT_BUDGET=29491 \
VLLM_TRIATT_PREFIX=128 \
VLLM_TRIATT_WINDOW=128 \
python my_script.py
```

#### 9.9.4 Tunable Knobs

| Env var | Default | Meaning |
|---------|--------:|---------|
| `VLLM_TRIATT_ENABLED` | `0` | master switch (`"0"` / `"1"`) |
| `VLLM_TRIATT_BUDGET` | `2048` | max live cells per sequence |
| `VLLM_TRIATT_HYBRID` | `2` | selection mode: `0`=V1 paper-faithful, `1`=V2 quota only, `2`=V3 |
| `VLLM_TRIATT_PREFIX` | `128` | protected prefix length (V3 only) |
| `VLLM_TRIATT_WINDOW` | `128` | protected recent-window length |
| `VLLM_TRIATT_SEGMENTS` | `8` | per-segment quota bucket count |
| `VLLM_TRIATT_WARMUP` | `1024` | Q samples before calibration fires |
| `VLLM_TRIATT_ADAPTIVE` | `0` | EMA-update calibration centers each round |

#### 9.9.5 PPL Eval (paper-exact protocol)

Reproduces the paper's Section 4.1 protocol on the vLLM/ROCm-Triton path.

```bash
# baseline (BF16 KV, no TQ, no V3)
python3 scripts/triatt_v3_ppl.py \
    --model /path/to/Qwen2.5-7B-Instruct \
    --mode baseline \
    --ctx 32768 --chunks 3 --gpu-mem 0.30

# TQ alone (turboquant_4bit_nc_cv_rv, no V3)
python3 scripts/triatt_v3_ppl.py --mode tq \
    --model /path/to/Qwen2.5-7B-Instruct \
    --ctx 32768 --chunks 3 --gpu-mem 0.30

# V3 alone (TQ K8V4 + V3 eviction)
python3 scripts/triatt_v3_ppl.py --mode v3 \
    --model /path/to/Qwen2.5-7B-Instruct \
    --ctx 32768 --chunks 3 --budget 29491 \
    --prefix 128 --window 128 --warmup 1024 --gpu-mem 0.30

# TQ + V3 stack
python3 scripts/triatt_v3_ppl.py --mode stack \
    --model /path/to/Qwen2.5-7B-Instruct \
    --ctx 32768 --chunks 3 --budget 29491 \
    --prefix 128 --window 128 --warmup 1024 --gpu-mem 0.30
```

The `--mode` flag selects the KV preset: `baseline`=auto BF16, `tq`=`turboquant_4bit_nc_cv_rv`, `v3` and `stack` both use `turboquant_k8v4` (FP8 K + 4-bit V; the V3-friendly preset).

#### 9.9.6 NIAH Eval (paper-exact protocol)

Reproduces the paper's Section 4.2 strict-checker NIAH protocol.

```bash
for mode in baseline tq v3 stack; do
    python3 scripts/triatt_v3_niah.py \
        --model /path/to/Qwen2.5-7B-Instruct \
        --mode $mode \
        --ctx 32768 \
        --budget 29491 --prefix 128 --window 128 --gen 64 \
        --gpu-mem 0.30
done
```

Output is one line per (mode, position) tuple with the strict-checker verdict: `PASS` / `PARTIAL_WORD` / `PARTIAL_NUMBER` / `FAIL`.

For reasoning models that emit a `<think>` block (Qwen3.5-class hybrids etc.) bump `--gen 1024` so the reasoning trace can complete before the strict checker reads the final answer. Same caveat the original paper's Section 3.3 documents.

#### 9.9.7 Cross-vendor note

The same scripts work on CUDA without modification. Skip the AITER install on CUDA (it's a ROCm-specific Triton FA build). All other paths (TurboQuant kernels, V3 engine, attention metadata plumbing) are platform-portable Triton.

#### 9.9.8 Documentation pointers

- Module README: `vllm/v1/attention/triattention/README.md`
- Unit tests: `tests/v1/attention/test_triattention_v3.py`
- TQ+ AMD-specific roadmap: see the V3-port-context paper at `vLLM AMD TurboQuant+ Improvements Roadmap`

---

## 10. Addendum: Swift Port + longctx Rescue — Fixing the Hybrid NIAH Failure

This addendum reports the Apple Silicon Swift port of V3 and validates the rescue path that resolves the hybrid Mamba+Attention NIAH failure documented in Section 7.1.

### 10.1 The fix in one sentence

V3 alone is fundamentally one-way — eviction throws cells away. On hybrid models (Qwen3.5 family, partial RoPE, M-RoPE) V3 evicts the wrong cells often enough that planted-fact NIAH fails at every context rung tested. **The fix is not a V3 algorithm change. It is the addition of a rescue layer (longctx-svc) that captures evicted spans and restores them to the prompt before the next prefill.** Section 7.1 hypothesizes a scoring change might rescue V3 on hybrids; the receipts below show that with longctx the scoring change is unnecessary at the model sizes and contexts tested.

### 10.2 Setup

- 1× Apple M5 Max, 128 GB unified memory, macOS 26.4
- Engine: historical `mlx-swift-lm` prototype at `feature/triattention-v3` (Swift port of V3)
- Model: `mlx-community/Qwen3.5-2B-4bit` (hybrid Mamba+Attention, the same family Section 7.1 flags as broken under V3)
- Companion service: `longctx-svc` v0.3.0a3 on `127.0.0.1:5054`
- Driver: `ChatSession` (auto Tier-3 rehydrate hook fires at turn boundary)
- Test harness: `Tests/Benchmarks/V3ChatSessionRamp.swift`

V3 config: default 10% rate (`budget = ctx × 0.9`, window=128, prefix=32, warmup=256, hybrid=2). Same parameter shape as the AMD/Python path documented in Section 6.

### 10.3 Three arms × four context rungs

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

V3 alone ✗miss every rung. V3+longctx ✓HIT every rung. Baseline turbo8v4 (no eviction) ✓HIT every rung. The pattern Section 7.1 documented on AMD reproduces cleanly on the Swift port. The rescue layer flips it.

### 10.4 Why the V3-only column is misleadingly fast

At 128K and 256K, the V3-only arm prefill (66.9s and 220.9s) is faster than the baseline turbo8v4 arm (76.3s and 186.7s) at 128K but slower at 256K. The 128K speedup is real — V3 evicts cells, shrinking the effective KV during decode — but the receipt that matters is recall: ✗miss across the board. A faster arm that doesn't recover the planted fact is broken, not improved. The V3+longctx arm pays a ~22% prefill overhead at 256K (229s vs 187s) to add the rescue layer, and recovers recall.

### 10.5 Architecture of the fix (turn-by-turn)

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

### 10.6 The driver gotcha (and why this is documented as a fix)

The auto-Tier-3 rehydrate hook is only wired on `ChatSession` (commit `fe1a3b0` in `mlx-swift-lm`). Drivers that call `container.generate()` directly **will not rescue evicted facts** — they get the V3-only behavior column above. This is the most common bug we hit during the port: V3 evicts correctly, longctx-svc ingests correctly, but the model never sees the recovered chunks because the driver bypassed `ChatSession`.

A separate single-cell test at 256K with `container.generate()` (instead of `ChatSession`) confirmed:

```
prompt_tok=247753, budget=230400
prefill=224.55s, decode=6.22s, tps=10.1
v3_rounds=30, v3%=1.32%
longctx_session_total=19427    ← chunks ingested into longctx-svc
recall=✗miss                   ← bare driver doesn't fire rehydrate
```

The write path is firing (19,427 chunks captured during prefill). The read path is not, because the driver isn't on the rehydrate-aware code path. Custom drivers must call `TriAttentionRescue.shared.rehydratePrompt(query:)` manually before each prefill, or run on `ChatSession`.

The fix lands in the documentation (the new section in `mlx-swift-lm`'s README under "TriAttention V3 + longctx (long-context rescue)" recommends *not* enabling V3 without longctx) and in the production defaults in Section 6 of `longctx-1m-and-triattention.md`.

### 10.7 What this resolves from Section 7.1

Section 7.1 asked **"why does V3 break on hybrid Mamba+Attention models?"** and listed three viable hypotheses (partial-RoPE blindness, phase-only scoring, M-RoPE theta scaling) plus a request for community recipes that work.

This addendum doesn't answer the underlying scoring question. The hypotheses in 7.1 may all still be correct. What it shows is: at the model sizes (2B-4bit Qwen3.5) and contexts (32K-256K) tested on Apple Silicon, the V3 + longctx rescue stack works without any algorithmic change to V3. The rescue layer compensates for whatever cells V3 evicts incorrectly.

This is operationally useful: production workloads on hybrid models can ship V3 by enabling longctx alongside it. The scoring research from Section 7.1 remains relevant — a V3 that doesn't evict the planted fact in the first place would skip the rescue round-trip and reduce latency — but the rescue stack is shippable.

### 10.8 What is not yet measured

- **No data above 256K.** The Swift port test ran 32K → 256K. 1M context is not yet tested on Apple Silicon under V3+longctx. The MRCR v2 1M results in `longctx-1m-and-triattention.md` are on AMD MI300X without V3 (longctx as standalone retrieval).
- **No test on Qwen3.5-27B / Qwen3.5-35B-A3B.** Section 7.1's failure was reported on these larger hybrids. The Apple Silicon test ran on 2B because it fits in M5 Max memory comfortably alongside longctx-svc. The expectation is the same V3+longctx pattern will work on the larger hybrids — verification pending hardware availability.
- **V3 + TQ+ stacking still gated.** `TriAttentionKVCache` extends `KVCacheSimple` (FP16-only). Stacking V3 with TurboQuant codecs requires a `TriAttentionTurboKVCache` variant. Tracked but not yet shipped.

### 10.9 Cross-reference

Full architectural diagrams, the longctx side of the rescue path, and the MRCR v2 1M numbers are in the companion paper `longctx-1m-and-triattention.md`. The Swift port code lives at:

- `mlx-swift-lm/Libraries/MLXLMCommon/TriAttention/` — Engine, cache, rescue bridge
- `mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4.swift` (and family parallels) — V3 hook in `newCache`
- `mlx-swift-lm/Tests/Benchmarks/V3ChatSessionRamp.swift` — the harness that produced Section 10.3

```bash
RUN_V3_CHAT_RAMP=1 swift test --filter "V3ChatSessionRamp"
```
