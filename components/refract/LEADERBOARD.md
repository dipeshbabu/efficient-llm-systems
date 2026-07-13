# REFRACT Leaderboard

> **⚠️ READ THIS BEFORE USING THESE NUMBERS.**
>
> This leaderboard ranks **how faithfully a quantized KV-cache config
> preserves a model's own fp16-KV behaviour**. It is NOT a model
> quality leaderboard. Specifically:
>
> - A high REFRACT score does NOT mean the model is good. It means
>   the quantization didn't make it materially worse than its own fp16
>   baseline.
> - A low REFRACT score does NOT necessarily mean the model is bad. It
>   means the quantization is materially diverging from the fp16
>   reference; the underlying model could be excellent at fp16.
> - **REFRACT scores do not represent how a model plays in the field.**
>   Real downstream performance depends on prompt distribution, task
>   difficulty, sampling choices, the user's prompts, and hundreds of
>   factors REFRACT can't see. For real-world task evaluation use
>   HELM, lm-eval, or your own task suite. REFRACT is the tool to
>   answer: "did my quantization break this model relative to its own
>   fp16 self?"
>
> Treat this leaderboard as **"which models tolerate which KV configs"**,
> not **"which models are best"**.

---

Each row is tagged with the framework version that produced it. Within
a single REFRACT version, rows are directly comparable. Across versions,
expect ±2 composite points of methodology drift.

## Best models for `ctk=q8_0,ctv=turbo4` candidate (REFRACT v0.3.0)

> What this asks: **does this candidate KV config faithfully preserve
> this model's fp16 behaviour?**
> What this does NOT ask: which of these models is smartest, best at
> code, best at reasoning, etc.

| Rank | Model | Composite | Band | Trajectory | KLD | KLD nats | R-NIAH | PLAD | Notes |
|------|-------|-----------|------|------------|-----|----------|--------|------|-------|
| 1 | Mistral-Small-24B Q4_K_M | **90.86** | EXCELLENT | 76.65 | 99.71 | 0.0029 | 100.00 | 91.34 | Tolerates q8/turbo4 best of the matrix. |
| 2 | phi-4 Q8 | **90.25** | EXCELLENT | 77.95 | 99.55 | 0.0046 | 100.00 | 87.35 | Microsoft's phi-4. Distribution-faithful under turbo. |
| 3 | Llama-4-Scout-17B-16E Q4_K_M | **89.77** | PASS | 73.58 | 97.32 | 0.0272 | 100.00 | 93.54 | Just below EXCELLENT. PLAD highest in the matrix. |
| 4 | Qwen3.5-2B Q8 | **81.48** | PASS | 60.07 | 98.35 | 0.0167 | 100.00 | 81.47 | Trajectory drift visible but distribution + retrieval intact. |
| 5 | Gemma-4-E2B Q4_K_L | **78.51** | DEGRADED | 52.73 | 93.50 | 0.0672 | 100.00 | 88.57 | Borderline pass; audit before deploying. |
| 6 | Qwen2.5-7B Q8 | **77.98** | DEGRADED | 55.13 | 98.75 | 0.0126 | 100.00 | 76.73 | KLD high but Trajectory + PLAD struggle. |
| 7 | Gemma-4-31B Q8 | **50.78** | FAIL | 26.41 | 49.23 | 0.7086 | 100.00 | 94.45 | Distribution materially shifted (0.7 nats). Don't ship. |
| 8 | Gemma-4-26B-A4B Q8 | **29.12** | FAIL | 17.32 | 17.59 | 1.7381 | 100.00 | 78.40 | Catastrophic distribution shift (1.7 nats). The motivating model from the paper. |

### Negative control (sanity check, not a real candidate)

| Model | Candidate KV | Composite | Band | Notes |
|-------|--------------|-----------|------|-------|
| Gemma-4-26B-A4B Q8 | **turbo4/turbo4 (symmetric)** | **~11** | FAIL | Deliberately broken config from the paper. KLD = 2.13 nats, Trajectory = 3.93. If your run on this combo does NOT FAIL, your framework setup has a problem. |

## How to read the rankings

The rank order means: **which model tolerates this exact KV config
best while staying close to its own fp16 reference.** Several non-obvious
properties of these numbers:

- **Mistral-24B at #1** doesn't mean it's the strongest model. It means
  q8/turbo4 quantization barely moves it from its own fp16 distribution.
  Mistral happens to be robust to KV quantization at this bit budget.

- **Gemma-4-31B at #7 with KLD 49.23** doesn't mean Gemma is a bad
  model. It means q8/turbo4 *specifically* doesn't preserve its
  distribution. Try `ctk=q8_0,ctv=q8_0` (no turbo) on the same model
  and the score will jump dramatically.

- **Llama-4-Scout PASS at #3** doesn't mean Llama-4 is good. It means
  the q8/turbo4 quant doesn't degrade it relative to its own fp16
  baseline. Llama-4's broader reputation issues (hallucination,
  reasoning shallowness) are model-architecture problems REFRACT
  cannot see — REFRACT only measures quant fidelity.

In short: **REFRACT ranks the quant, not the model.**

## What REFRACT does measure

For each candidate KV config × model:

- **Trajectory** — does greedy decode produce the same tokens as fp16?
- **KLD@D** — does the per-token distribution match fp16's on a corpus?
- **R-NIAH** — does long-context retrieval work as well as fp16?
- **PLAD** — does the model's robustness to typos / casing / paraphrase
  match fp16's?

A high composite says all four surfaces look like fp16. It says nothing
about whether fp16 itself was any good for your workload.

## What REFRACT does NOT measure

- Real downstream task accuracy (HELM, lm-eval, MMLU territory)
- Code quality, reasoning depth, instruction-following
- Agentic capability, tool-use fidelity (T-Call axis is v0.4)
- Hallucination rate
- Safety, refusal calibration
- Production latency, throughput, memory footprint
- Anything the user actually cares about end-to-end

A model can pass REFRACT and still be bad at your task. A model can
pass REFRACT under one quant and fail at your task because the model
itself is a poor fit. **You still need real-world evaluation.**

## How to submit a result

Alpha-stage informal: open an issue or PR with:

1. **The JSON** of your run (`refract score ... --json-out report.json`).
   It embeds `framework_version`, `environment.backend`, and (for
   llama.cpp) `environment.llama_cpp_commit` so we can validate the
   row is from a known-good methodology version.
2. **The HTML report** if you want a visual share (`--html-out`).
3. **`refract selftest --backend X --model Y` output**.
4. **A repeatability run** (`refract repeatability --runs 4 ...`) if
   the score is in a band that surprises you.

We accept rows where:
  - `framework_version >= 0.3.0` (chat-template handling fixed)
  - Confidence flags are clean (no R-NIAH `low` without an explanation)

## Reproducibility info for the rows above

  - REFRACT version: v0.3.0
  - Backend: llamacpp (dipeshbabu/llama.cpp `feature/turboquant-kv-cache`)
  - Hardware: M5 Max 128 GB RAM, macOS Tahoe 26.4
  - Reference KV: ctk=f16,ctv=f16
  - R-NIAH ctx_max: 16384
  - Total matrix runtime: 7370s (2h 2min) for v0.2.0 + 7336s for v0.3.0

Sample report files: `src/refract/examples/*.json` + `*.html`.

## Versioning

REFRACT version dictates methodology. Major changes that affect the
score surface:

| Version | Methodology change | Effect |
|---------|--------------------|--------|
| v0.1.4  | Trajectory axis replacing buggy GTM | Most v0.1.4+ scores 5–15 points different from v0.1.x |
| v0.2.0  | Added R-NIAH + PLAD axes | Composite went from harmonic_mean(2) to harmonic_mean(2..4) |
| v0.2.1  | R-NIAH neutral needle + n_predict=256 | Fixed refusal artifacts |
| v0.3.0  | Chat-template handling via `--jinja -rea off` | Fixed instruct-model engagement |
| v0.3.1  | Backend abstraction + confidence guards + version stamps | No score impact |
| v0.3.2  | HTML reports + repeatability subcommand | No score impact |

Cite results as "Mistral-24B got 90.86 EXCELLENT on REFRACT v0.3.0,
candidate ctk=q8_0/ctv=turbo4, llamacpp backend, M5 Max" — the version
+ candidate are load-bearing.

## Future automation

A `refract leaderboard --in dir/ --out LEADERBOARD.md` subcommand
that walks a directory of submitted JSONs is a v0.4 target. For now
the table is curated from the matrix output JSONs.
