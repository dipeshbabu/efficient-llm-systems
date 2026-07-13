# REFRACT v0.3.4

> **BETA.** The Python package, CLI, prompts, reports, and four backends are
> implemented and unit-tested. Backend-specific engine builds and model files
> remain user-managed. Scores are evidence for a deployment decision, not a
> substitute for workload-specific validation.

**REF**erence-anchored **R**obust **A**cid-test for **C**ompressed **T**ransformers.

## Install

### From PyPI (version 0.3.4 or newer)

```bash
pip install "refract-llm>=0.3.4"                       # base — zero non-stdlib deps
pip install "refract-llm[refract-mlx]>=0.3.4"          # + MLX backend (Apple Silicon)
pip install "refract-llm[refract-vllm]>=0.3.4"         # + vLLM backend (CUDA / ROCm)
pip install "refract-llm[refract-sglang]>=0.3.4"       # + SGLang HTTP client backend
pip install "refract-llm[full]>=0.3.4"                 # + all three backends
```

The minimum version is intentional: 0.3.4 is the first release built from
this repository's current package metadata. If your package index does not
offer it yet, use the source installation below.

After install, the `refract` CLI is on your PATH and the prompt set plus
example reports ship in the wheel. REFRACT does not ship inference engines or
a version-sensitive llama.cpp source patch.

### From source (for hacking / contributing)

```bash
git clone https://github.com/dipeshbabu/efficient-llm-systems.git
cd efficient-llm-systems/components/refract
pip install -e .                              # editable install, base
pip install -e .[refract-mlx]                 # editable + MLX backend
pip install -e .[refract-vllm]                # editable + vLLM backend
pip install -e .[refract-sglang]              # editable + SGLang backend
pip install -e .[dev]                         # editable + pytest + coverage + build tooling
```

The base install gives you the `refract` CLI with no third-party
dependencies. Backends are extras you opt into.

## Platform support

REFRACT itself (Python framework + report renderer + CLI) is
platform-portable. The constraint is which **backend** runs on your OS:

| OS | llamacpp | mlx | vllm | sglang |
|---|---|---|---|---|
| macOS (Apple Silicon) | ✅ primary dev target | ✅ production | n/a | n/a |
| Linux (Ubuntu 24.04, x86_64, ROCm 7.2 MI300X) | ✅ verified | n/a (Apple Silicon only) | ✅ verified | ✅ verified (HTTP client; SGLang server runs separately) |
| Windows | Python/CLI tested; engine build required | n/a | backend-dependent | HTTP client tested |

vLLM and SGLang backends were both verified end-to-end on the AMD MI300X
droplet on hybrid Qwen3.6-35B-A3B during the cross-engine bench documented
in [`research/papers/cross-engine-mi300x.md`](../../research/papers/cross-engine-mi300x.md).

The llama.cpp backend needs compatible binaries (`llama-cli`,
`llama-completion`, `llama-tokenize`, `llama-perplexity`) on the loader
path — `LD_LIBRARY_PATH` / `ldconfig` on Linux, DLLs next to the `.exe`
or on `PATH` on Windows. `selftest` detects this and prints the right
remediation per OS. The trajectory capture extension is available in the
[historical llama.cpp TurboQuant fork](../../docs/reference/historical-forks.md#llamacpp-experimental-forks)
builds described by the root README; it is not embedded in this wheel. MLX is
Apple Silicon only by upstream design.

The vLLM backend uses `vllm.LLM` in-process. Each call instantiates an
LLM at the requested KV config; backend caches one LLM at a time and
evicts on key change for memory-pressured deployments. Env knobs:
`REFRACT_VLLM_GPU_MEMORY_UTILIZATION`, `REFRACT_VLLM_MAX_NUM_SEQS`,
`REFRACT_VLLM_KLD_TOPK`, `REFRACT_VLLM_MAX_MODEL_LEN`.

The SGLang backend is HTTP-based — the user runs an SGLang server
separately (typically via the published Docker image), and REFRACT
posts to it. KV dtype is fixed at SGLang server-launch time, so
`run_kld` (which compares two configs) requires either two simultaneous
servers (`REFRACT_SGLANG_REF_URL` + `REFRACT_SGLANG_CAND_URL`) or a
two-phase orchestrator that launches them sequentially. See
`../../research/papers/cross-engine-mi300x.md` §6 for a working orchestrator.

Friend-tester input on Windows is welcome — open an issue with your
`refract selftest` output.

## Where do I go?

| If you want to… | Read |
|---|---|
| Understand what REFRACT is and why it exists | This file (below) + [`research/papers/attn-rotation-and-ppl-artifact.md`](../../research/papers/attn-rotation-and-ppl-artifact.md) |
| Get to a real score in 30 minutes | [QUICKSTART.md](QUICKSTART.md) |
| Read your own report (figure out what your score means) | [INTERPRETATION.md](INTERPRETATION.md) |
| See which models score how on which KV configs | [LEADERBOARD.md](LEADERBOARD.md) |
| Avoid known setup / interpretation traps | [PITFALLS.md](PITFALLS.md) |
| See what v0.3 explicitly does NOT do | [LIMITATIONS.md](LIMITATIONS.md) |
| See what changed across versions | [CHANGELOG.md](CHANGELOG.md) |
| Compare your run to known-good reference numbers | [examples/](src/refract/examples/) (4 sample JSONs + HTMLs) |
| See the methodology evolution data | [MATRIX-RESULTS.md](MATRIX-RESULTS.md) |


A benchmaxx-resistant alternative to corpus PPL for evaluating KV-cache
quantization quality. Replaces "lower PPL = better" — a metric the paper
[`research/papers/attn-rotation-and-ppl-artifact.md`](../../research/papers/attn-rotation-and-ppl-artifact.md)
shows can invert sign on instruct-tuned models — with a 4-axis composite
that ranks configurations by *distance from the fp16-KV reference*, not
by absolute corpus likelihood.

## Why this exists

The motivation paper documents a real failure of corpus PPL: on
**gemma-4-26B-A4B-Q8 with q8/turbo4 KV**, wikitext-2 PPL says rotation
OFF "wins" by 42%, but **KLD vs the fp16-KV reference says the same
configuration is 1.7 nats away from fp16** — the largest distribution
drift on the row. The KLD codepath is bit-exact zero on Metal, so the
signal is real. PPL is reading miscalibration as improvement.

REFRACT rejects the PPL framing entirely: nothing matters except how
close the quantized model's behaviour stays to its fp16 self.

Read [`research/papers/attn-rotation-and-ppl-artifact.md`](../../research/papers/attn-rotation-and-ppl-artifact.md)
for the full motivation.

## What ships in v0.3.4

Four axes, each scored 0–100 (higher is better) against the model's own
fp16-KV reference:

| Axis | Name | What it measures | Notes |
|------|------|------------------|-------|
| A | **Trajectory** | Token-level agreement on greedy decode (decode-time IDs, no detokenize round-trip) | Symmetric length normalization penalizes unilateral early stops |
| B | **KLD@D** | Distribution-level divergence on a natural-text corpus | Bit-exact zero on Metal at ref==cand |
| C | **R-NIAH** | Long-context retrieval quality (needle-in-haystack at multiple lengths/positions) | v0.2.0+; opt-in via `--full` |
| D | **PLAD** | Robustness to small prompt perturbations (typo/case/punct/paraphrase) | v0.2.0+; opt-in via `--full` |

**Composite** = harmonic mean of the axes that ran. Any single broken
axis tanks the composite — the framework is intentionally fail-loud.

**Bands**: `[90,100]` EXCELLENT · `[80,90)` PASS · `[60,80)` DEGRADED · `[0,60)` FAIL.

**Backends**: llama.cpp, MLX, vLLM, and SGLang. Engine availability and
supported KV dtypes differ by backend; run `refract selftest` before a score.
Native vLLM/SGLang KLD is explicitly labeled as a normalized top-k estimate,
not full-vocabulary KL.

## Subcommands

```
refract score          # score a candidate KV config
refract selftest       # 30s preflight: binaries, flags, model probe
refract compare        # multi-report side-by-side
refract repeatability  # run N times, report spread (stdev/range)
refract fetch          # download wikitext-2-raw corpus to ~/.cache/refract/
```

## Reports

Every `score` run can emit two formats via `--json-out` and `--html-out`:

- **JSON** (`--json-out report.json`) — schema `refract.report.v0.3.2`,
  consumable by `refract compare` or any JSON-aware tool.
- **HTML** (`--html-out report.html`) — single **self-contained file**
  (~40 KB) with composite stats, diagnosis callout, per-axis bars,
  R-NIAH heatmap, PLAD per-perturbation table, run details (hardware +
  model + env), the sanitized repro command, and the raw JSON embedded
  in a collapsible section. Sun/moon toggle in the top-right for
  light/dark mode (follows OS by default). Pasteable in Discord/X.
  See [`examples/`](src/refract/examples/) for 4 real samples.

The HTML uses `light-dark()` CSS (Chrome 123+ / Safari 17.5+ / Firefox
120+) for dark mode and a native system-font stack. It contains no external
font, script, or stylesheet dependency.

## Quickstart

See [QUICKSTART.md](QUICKSTART.md) for full setup. Short version:

```bash
# 1. Verify your setup
python3 -m refract.cli selftest --backend auto --model path/to/model.gguf

# 2. First quick score (~5-7 min on a 7B Q8)
python3 -m refract.cli score \
    --model path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --json-out report.json \
    --html-out report.html

# 3. Full audit (~25-30 min on a 7B Q8)
python3 -m refract.cli score \
    --model path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --full \
    --rniah-ctx-max 16384 \
    --json-out report.json --html-out report.html

# 4. Verify reproducibility (4 runs, expect stdev ≤ 1.0)
python3 -m refract.cli repeatability \
    --model path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --runs 4
```

These commands use the prompt set bundled in the package. The corpus is
resolved from `~/.cache/refract/` or downloaded on first use; `--full` also
resolves the cached haystack. Explicit `--prompts`, `--corpus`, and
`--rniah-haystack` paths remain available for pinned runs. For offline/CI
execution, run `refract fetch` first or pass explicit paths, then add
`--no-auto-fetch`.

## Documentation

| File | When to read |
|------|--------------|
| [QUICKSTART.md](QUICKSTART.md) | First-time setup + first run |
| [INTERPRETATION.md](INTERPRETATION.md) | What does my score mean? Per-axis "what to do if low" |
| [LEADERBOARD.md](LEADERBOARD.md) | Cross-model rankings on which KV configs (with the strong "this is NOT a model-quality leaderboard" disclaimer) |
| [PITFALLS.md](PITFALLS.md) | Things that have actually bitten us — avoid them |
| [LIMITATIONS.md](LIMITATIONS.md) | What v0.3 explicitly does NOT do |
| [CHANGELOG.md](CHANGELOG.md) | Full history including the v0.2 / v0.3 discoveries |
| [MATRIX-RESULTS.md](MATRIX-RESULTS.md) | Reference numbers from the 7-model 2026-04-30 matrix |
| [examples/](src/refract/examples/) | Sample JSONs + HTML reports (clean / degraded / distribution-broken / catastrophic) |
| [research/papers/attn-rotation-and-ppl-artifact.md](../../research/papers/attn-rotation-and-ppl-artifact.md) | Why this framework exists at all (the motivation paper) |

## File layout

```
components/refract/
  pyproject.toml          # refract-llm distribution metadata
  src/refract/
    __init__.py           # version stamp
    cli.py                # CLI: score / selftest / compare / repeatability
    score.py              # composite + bands + diagnosis
    report.py             # text + JSON report formatter
    report_html.py        # self-contained HTML report (v0.3.2+)
    runner.py             # llama.cpp subprocess wrappers + KVConfig
    axes/                 # GTM, trajectory, KLD, R-NIAH, and PLAD axes
    backends/             # llama.cpp, MLX, vLLM, and SGLang backends
    prompts/v0.1.jsonl    # 30 CC0 prompts shipped in the wheel
    examples/             # sample reports shipped in the wheel
  tests/                  # unit and integration-contract tests
  README.md               # this file
  QUICKSTART.md           # setup + first run
  INTERPRETATION.md       # how to read a report
  PITFALLS.md             # known traps
  LIMITATIONS.md          # what v0.3 doesn't do
  CHANGELOG.md            # reverse-chronological
  MATRIX-RESULTS.md       # 2026-04-30 7-model matrix
```

## Status

  - **Implemented**: all four axes plus llama.cpp, MLX, vLLM, and SGLang
    backends. Hardware-specific end-to-end support depends on the selected
    engine and KV dtype.
  - **Open**: T-Call axis (tool-call fidelity) — v0.4 target;
    multi-prompt-set support; bundled corpus distribution.

## Contributing

This is beta software. Open issues with:
  - Your `selftest` output (so we know what you have)
  - The full JSON of any failing run (`--json-out`)
  - The HTML report if you want a visual share (`--html-out`)
  - Your model + KV config

Especially valuable feedback: surfaces where REFRACT fails silently
(low base_acc, NaN perturbations, etc.) before the confidence guards
catch them.
