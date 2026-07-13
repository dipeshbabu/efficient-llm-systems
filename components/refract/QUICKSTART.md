# REFRACT QUICKSTART

> ## Runtime expectations (7B Q8 model on Apple Silicon)
>
> | Mode | Axes | Time | When |
> |---|---|---|---|
> | `selftest` | preflight only | **~1s static / ~30s with `--model`** | Before your first real run. Free. |
> | **`score` (default)** | Trajectory + KLD | **~5–7 min** | Most runs. Go/no-go on a candidate KV config. |
> | `score --full` | + R-NIAH + PLAD | ~25–30 min | Pre-ship audit. Adds long-context retrieval + brittleness. |
> | `repeatability --runs 4` | repeats default | 4× default | Sanity-check reproducibility. |
>
> **Default is quick.** Most users should never wait 30 minutes unless
> they're explicitly running `--full` for a ship-decision audit.
>
> ---

> **BETA — validate on your workload before deployment.**
>
> The framework is packaged and all axes are implemented, but inference
> engines and model files remain user-managed:
> - Install `refract-llm` 0.3.4 or newer from your package index, or use an
>   editable source checkout. The bundled prompt set removes the need for a
>   prompt path.
> - **All four backends are implemented**: llama.cpp, MLX, vLLM,
>   SGLang. vLLM and SGLang were verified on AMD MI300X / ROCm 7.2 in the
>   cross-engine bench at `../../research/papers/cross-engine-mi300x.md`.
> - **Confidence guards exist but aren't exhaustive** — you may find
>   edge cases. Please open an issue with the JSON.
> - **Score interpretation is calibrated on one matrix run** of 7
>   models. Bands (90/80/60) are provisional and may shift in v0.4.
>
> If you hit a wall, open an issue with your `selftest`
> output and the JSON of the failing run.

Goal: get from "git clone" to a real REFRACT score in under **5–7 minutes**
on the default (quick) mode.

## What REFRACT does (one paragraph)

REFRACT scores how faithful a quantized KV-cache config is to the same
model's fp16-KV reference. Score 0–100, higher is better. It's a
multi-axis composite (Trajectory + KLD + R-NIAH + PLAD), bit-exact on
Metal, fail-loud (any single broken axis tanks the composite). Replaces
"lower PPL = better" because PPL inverts sign on instruct-tuned models.

## Step 0 — install REFRACT

### PyPI (version 0.3.4 or newer)

```bash
# Apple Silicon
pip install 'refract-llm[refract-mlx]>=0.3.4'

# CUDA / ROCm (vLLM in-process)
pip install 'refract-llm[refract-vllm]>=0.3.4'

# SGLang HTTP client (you run the SGLang server separately, e.g. via Docker)
pip install 'refract-llm[refract-sglang]>=0.3.4'

# All three backends in one shot
pip install 'refract-llm[full]>=0.3.4'
```

The version floor prevents an older package-index build from being installed
silently. If 0.3.4 is not available from your index yet, use the source
installation below.

After install, the `refract` CLI is on your PATH and the v0.1 prompt set plus
example reports ship inside the wheel. Inference engines and the
version-sensitive llama.cpp trajectory extension are installed separately.

> **macOS gotcha — use Python 3.10 or newer.** Older macOS installations may
> provide `/usr/bin/python3` as 3.9, which this release no longer supports.
> Use a newer Python (for example, `brew install python@3.13`, then create
> a virtual environment with `python3.13 -m venv ...`)
> for the framework and MLX backend.

### Source install (for hacking / contributing)

```bash
git clone https://github.com/dipeshbabu/efficient-llm-systems.git
cd efficient-llm-systems/components/refract

pip install -e .                   # editable install, base
pip install -e .[refract-mlx]      # editable + MLX backend
pip install -e .[refract-vllm]     # editable + vLLM backend
pip install -e .[refract-sglang]   # editable + SGLang backend
pip install -e .[dev]              # editable + pytest + coverage + build tooling
```

Every later command (`python3 -m refract.cli ...`) assumes you installed
the component and are running from `components/refract/`.

The llamacpp backend needs compatible patched binaries on `PATH` /
`LD_LIBRARY_PATH`, or in the directory named by `LLAMA_CPP_BIN_DIR`. The source
used for the cited TurboQuant experiments is a
[historical fork whose public URL is unavailable](../../docs/reference/historical-forks.md#llamacpp-experimental-forks).

For the standard vLLM backend on CUDA / ROCm, install upstream `vllm` with
`pip install vllm`. Fork-specific TurboQuant schemes require the
[historical vLLM implementation](../../docs/reference/historical-forks.md#vllm-experimental-forks),
which does not currently have a public source URL.

For SGLang, the simplest path is the published Docker image (the bench
in `../../research/papers/cross-engine-mi300x.md` uses
`lmsysorg/sglang:v0.5.10.post1-rocm720-mi30x` for AMD MI300X — see §6
for the in-container patches that image needs).

## Prereqs

Once REFRACT is installed and you're inside `components/refract/`, you need:

  - Python 3.10+
  - One of:
    - **llama.cpp build** with `--jinja` support and the REFRACT v0.1.4
      patch in `tools/completion/completion.cpp`. (Patch emits per-token
      JSONL when `REFRACT_TRAJECTORY` env var is set.)
    - **mlx-lm** (`pip install mlx mlx-lm`). MLX backend is native
      Python; no patches needed.
    - **vllm** (`pip install vllm` or `pip install -e .[refract-vllm]`).
      Working backend as of v0.3.2. Caches one LLM at a time, evicts
      on KV-config change. Tunable via `REFRACT_VLLM_*` env knobs.
    - **SGLang server** (Docker recommended; `pip install -e
      .[refract-sglang]` for the HTTP client). Backend posts to a
      pre-launched SGLang server. KV dtype is fixed at server launch,
      so `run_kld` requires either two simultaneous servers
      (`REFRACT_SGLANG_REF_URL` + `REFRACT_SGLANG_CAND_URL`) or a
      two-phase orchestrator (example in `../../research/papers/cross-engine-mi300x.md`).
  - A model in the right format for your backend:
    - `.gguf` for llama.cpp
    - directory with `config.json + model.safetensors` for mlx
    - HF safetensors directory for vllm and sglang
  - **Corpus + haystack: automatic.** REFRACT auto-downloads
    wikitext-2-raw (~10MB) to `~/.cache/refract/` on first run and uses
    `wiki.test.raw` for KLD + `wiki.train.raw` for R-NIAH unless you
    pass paths explicitly. Pre-fetch with:
    ```
    python3 -m refract.cli fetch
    ```
    Disable network access with `--no-auto-fetch`; already-cached files and
    explicit paths still work (CI-friendly).
  - The prompts JSONL is bundled in the package. `--prompts` is optional and
    remains available when you want to pin a custom prompt set.

## Constrained VRAM? Pass extra llama.cpp flags

REFRACT defaults to `-ngl 99` (all layers on GPU) for the llama.cpp
backend. Consumer-card users running large MoE models (e.g.
Qwen3.6-35B-A3B on a 12 GB 3060) won't fit that — they need
`-ncmoe N` to offload some MoE expert layers to CPU.

Pass any extra llama.cpp flags via `REFRACT_LLAMA_EXTRA_FLAGS`:

```bash
# 12 GB consumer GPU running Qwen3.6-35B-A3B with MoE offload
export REFRACT_LLAMA_EXTRA_FLAGS="-ngl 28 -ncmoe 32"
python3 -m refract.cli score --backend llamacpp --model /path/to/model.gguf ...
```

The flags get appended to every `llama-cli`, `llama-completion`, and
`llama-perplexity` invocation **after** REFRACT's own. llama.cpp uses
last-wins for repeated flags, so `REFRACT_LLAMA_EXTRA_FLAGS="-ngl 28
-ncmoe 32"` overrides the default `-ngl 99`. Parsed with `shlex` so
quoted args work the same as on the command line.

Confirmed working scenarios:
- Consumer 12 GB GPU + 35B-A3B MoE: `-ngl 28 -ncmoe 32`
- CPU-only fallback: `-ngl 0`
- Tensor split across multiple GPUs: `-ts 1,1`

If a flag REFRACT doesn't recognize trips up its own subprocess,
open an issue with the failing command line and we'll plumb it.

## Step 2 — preflight (~30 seconds)

```bash
# llama.cpp model (.gguf)
python3 -m refract.cli selftest --backend auto --model /path/to/model.gguf

# OR an MLX model (directory with config.json + model.safetensors)
python3 -m refract.cli selftest --backend auto --model /path/to/mlx-model-dir/

# Without --model: static checks only (~1 second)
python3 -m refract.cli selftest
```

`--backend auto` infers from the path: `.gguf` → llamacpp; directory →
mlx (or vllm if `REFRACT_BACKEND=vllm`). Override with
`--backend llamacpp|mlx|vllm|sglang` or set `REFRACT_BACKEND` env var.

Verifies binaries, flags, env vars, and a tiny generation. If it bails,
fix the reported issue before going further. Don't burn a long run
finding out your setup is broken.

## Step 3 — first quick score (5–7 min on a 7B Q8)

```
python3 -m refract.cli score \
    --model /path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --json-out my-first-report.json \
    --html-out my-first-report.html
```

The bundled prompt set is used automatically, and `--corpus` is resolved
from `~/.cache/refract/` (downloaded on first run). This runs Trajectory +
KLD@D — the two cheap axes. You'll get a composite score, a band
(EXCELLENT/PASS/DEGRADED/FAIL), and a plain-English diagnosis of what the
per-axis pattern means.

## Step 4 — full audit (25–30 min on a 7B Q8)

Add `--full`. Both haystack file and corpus are auto-resolved from the cache.

```
python3 -m refract.cli score \
    --model /path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --full \
    --rniah-up-to 16384 \
    --json-out my-full-report.json \
    --html-out my-full-report.html
```

### Long-context audit knob

`--rniah-up-to N` controls how deep R-NIAH probes. Lengths are
auto-generated as a doubling step-up from 4K up to N:

| `--rniah-up-to` | Lengths tested | R-NIAH wall-time on 7B Q8 |
|---|---|---|
| `16384` (default) | 4K, 8K, 16K | ~10–15 min |
| `32768` | 4K, 8K, 16K, 32K | ~25–35 min |
| `65536` | 4K, 8K, 16K, 32K, 64K | ~60–90 min |
| `131072` | 4K … 128K | ~3+ hours |

Pick a value matching your model's actual usable context. If the model
fails at 64K under fp16, R-NIAH will report `confidence: low` for those
cells (per-cell `base_acc = 0`) — cleaner to cap below that.

Power users: `--rniah-lengths 4096,16384,65536` overrides the doubling
step-up with an explicit list.

### Generating the HTML report

Pass `--html-out path.html` to any `score` invocation. The HTML report
is a **single self-contained file** (~40 KB) you can email, paste into
Discord, or open offline:

```bash
python3 -m refract.cli score \
    --model /path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --json-out report.json \
    --html-out report.html
```

What's in it:
- Composite + per-axis stats strip at the top
- Plain-English diagnosis (colored callout)
- Per-axis breakdown with bars and bands
- R-NIAH per-cell heatmap + PLAD per-perturbation table when `--full`
- Run details (model size, hardware, KV configs)
- Reproduce command (sanitized — no personal paths)
- Embedded raw JSON in a `<details>` section
- Sun/moon toggle in the top-right for light/dark mode (follows OS by default)

What's bundled vs external:
- HTML, CSS, JS, raw JSON: all inline. Works offline.
- **No external fonts or assets.** Reports use the native system-font stack
  and render identically without network access.
- Dark mode uses `light-dark()` CSS — needs Chrome 123+ / Safari 17.5+ /
  Firefox 120+ (all 2024). Older browsers see the light theme cleanly;
  dark mode is progressive enhancement.

Sample reports live in [`examples/`](src/refract/examples/) (4 real reports from
the 2026-04-30 matrix run). Open one to preview the format before
running your own.

## Step 5 — interpret the result

Quick table:

| Composite | Band      | What it means                                  |
|-----------|-----------|------------------------------------------------|
| 90–100    | EXCELLENT | Indistinguishable from fp16. Safe to deploy.   |
| 80–90     | PASS      | Minor drift; safe to deploy in most uses.      |
| 60–80     | DEGRADED  | Visible drift; audit on your workload first.   |
| 0–60      | FAIL      | Material quality loss; treat as broken.        |

If the composite is below 90, look at the per-axis breakdown and the
**Diagnosis** block in the report. It will tell you in plain English
which surface broke (e.g., "decode distribution drift detected;
candidate generates different tokens than fp16 on short-context
prompts") and a suggested next move.

For deeper interpretation see [`INTERPRETATION.md`](INTERPRETATION.md).

## Step 6 — compare candidates side by side

```
python3 -m refract.cli compare \
    report-q8q8.json report-q8turbo4.json report-q4q4.json
```

Prints a comparison table. Useful for finding the breaking point of a
model under increasingly aggressive quants.

## Backends

| Backend  | Status   | Use for                                         |
|----------|----------|-------------------------------------------------|
| llamacpp | shipping | .gguf models, all four axes, TurboQuant configs |
| mlx      | shipping | MLX models (directory layout); Trajectory + R-NIAH + PLAD work; KLD has limitations on RotatingKVCache models |
| vllm     | shipping | HF safetensors models on CUDA / ROCm; all four axes; in-process LLM (caches one at a time, evicts on KV-config change). Verified on MI300X (Qwen3.6-35B-A3B). |
| sglang   | shipping | HF safetensors models served via a pre-launched SGLang server (HTTP). KV dtype is fixed at server launch — see `../../research/papers/cross-engine-mi300x.md` §6 for a two-phase orchestrator that handles this. |

Override default with `--backend mlx` (or `REFRACT_BACKEND=mlx`).

## Common pitfalls (also see [PITFALLS.md](PITFALLS.md))

- **Don't use the v0.1.x `gtm` axis** — it has a known
  detokenize→retokenize unit-mismatch bug. v0.3.1 default is
  `--axis-a trajectory` (the proper fix).
- **Instruct models need chat-template handling** — REFRACT v0.3.0+
  applies it automatically via `--jinja`. If you see all-zero
  retrieval (R-NIAH `base_acc = 0` everywhere), your llama.cpp build
  may be too old.
- **Thinking-mode models** — auto-detected at run start; reasoning
  disabled via `-rea off`. The detection line in the banner says
  whether your model triggered it.
- **R-NIAH with `base_acc < 0.2` averaged across cells** flags
  `confidence: low` in the JSON — the model isn't engaging the task
  and the score is noise-floor.
- **PLAD `paraphrase = NaN`** means no synonym matches in your prompts
  set. Other perturbations (typo/case/punct) still produce valid
  numbers; the cell is recorded as `skipped_perturbations` in JSON.

## Reproducibility

Run the same configuration repeatedly without restating the default inputs:

```
python3 -m refract.cli repeatability \
    --model /path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --runs 4
```

`repeatability` uses the same bundled prompts and cached/auto-downloaded
corpus as `score`. Add `--full` to include R-NIAH and PLAD; the haystack is
resolved from cached `wiki.train.raw`. Explicit paths and `--no-auto-fetch`
have the same meaning on both commands.

Reports embed:
  - `framework_version` (REFRACT version)
  - `environment.backend` (llamacpp / mlx / vllm)
  - `environment.llama_cpp_commit` (when llamacpp)
  - `environment.mlx_lm_version` (when mlx)
  - `score_direction` and `score_range` (so machine consumers can't
    accidentally invert the comparison)

When sharing scores ("I got 87 on Mistral-7B"), include the JSON. The
number alone is not reproducible without the version stamp.
