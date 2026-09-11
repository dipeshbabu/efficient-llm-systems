# KV Fidelity 0.3.5 development

> **BETA.** The Python package, CLI, prompts, reports, and four backends are
> implemented and unit-tested. Backend-specific engine builds and model files
> remain user-managed. Scores are evidence for a deployment decision, not a
> substitute for workload-specific validation.

Multi-axis KV-cache fidelity evaluation anchored to each model's
full-precision reference.

## Install

### Installation status

This repository has not published KV Fidelity to PyPI. The planned distribution is
`kv-fidelity`, with import `kv_fidelity` and command `kv-fidelity`. Do not install the
legacy `refract-llm` project as a substitute: its releases were not published
from this repository and use different ownership and license metadata.

Install from source until a verified `kv-fidelity` release exists. The
[package-identity record](https://github.com/dipeshbabu/metria/blob/main/components/kv-fidelity/PACKAGE-IDENTITY.md)
explains the rename and the legacy release boundary.

After PyPI shows version 0.3.5 with provenance from this repository's
`publish-kv-fidelity.yml` workflow, install it with:

```bash
pip install "kv-fidelity==0.3.5"          # base
pip install "kv-fidelity[mlx]==0.3.5"     # Apple Silicon
pip install "kv-fidelity[sglang]==0.3.5"  # SGLang client
pip install "kv-fidelity[full]==0.3.5"    # managed backends
```

The vLLM adapter remains implemented, but the `vllm` dependency extra
is temporarily unavailable. vLLM 0.25.1 pins PyTorch 2.11.0, which is affected
by [GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47).
Do not force a PyTorch override because vLLM's compiled extensions require a
matching build. Use an existing audited environment or another backend until
vLLM publishes support for PyTorch 2.13 or newer.

After install, the `kv-fidelity` CLI is on your PATH and the prompt set plus
example reports ship in the wheel. KV Fidelity does not ship inference engines or
a version-sensitive llama.cpp source patch.

### From source (for hacking / contributing)

```bash
git clone https://github.com/dipeshbabu/metria.git
cd metria
uv sync --all-packages              # root core + components + development tools
uv run kv-fidelity --help
# Add --extra mlx or --extra sglang to uv sync for the selected backend.
```

KV Fidelity uses the stdlib-only Metria core for shared artifact infrastructure.
The current development checkout requires `metria>=0.1.1.dev0,<0.2`; the workspace
resolves that dependency locally until the corresponding root release is
published. Inference backend dependencies remain optional extras.

Default WikiText-2 inputs use an immutable upstream revision and pinned
archive/member hashes. `kv-fidelity fetch` verifies cached contents, enforces
download/extraction limits, and rejects unsafe archive members. Scoring JSON
retains source, revision, license metadata, and full input hashes under
`extras.input_artifacts`. See [artifact resolution](../../docs/guides/artifact-resolution.md).

## Platform support

KV Fidelity itself (Python framework + report renderer + CLI) is
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
`KV_FIDELITY_VLLM_GPU_MEMORY_UTILIZATION`, `KV_FIDELITY_VLLM_MAX_NUM_SEQS`,
`KV_FIDELITY_VLLM_KLD_TOPK`, `KV_FIDELITY_VLLM_MAX_MODEL_LEN`.

The SGLang backend is HTTP-based — the user runs an SGLang server
separately (typically via the published Docker image), and KV Fidelity
posts to it. KV dtype is fixed at SGLang server-launch time, so
`run_kld` (which compares two configs) requires either two simultaneous
servers (`KV_FIDELITY_SGLANG_REF_URL` + `KV_FIDELITY_SGLANG_CAND_URL`) or a
two-phase orchestrator that launches them sequentially. See
`../../research/papers/cross-engine-mi300x.md` §6 for a working orchestrator.

Friend-tester input on Windows is welcome — open an issue with your
`kv-fidelity selftest` output.

## Where do I go?

| If you want to… | Read |
|---|---|
| Understand what KV Fidelity is and why it exists | This file (below) + [`research/papers/attn-rotation-and-ppl-artifact.md`](../../research/papers/attn-rotation-and-ppl-artifact.md) |
| Get to a real score in 30 minutes | [QUICKSTART.md](QUICKSTART.md) |
| Read your own report (figure out what your score means) | [INTERPRETATION.md](INTERPRETATION.md) |
| See which models score how on which KV configs | [LEADERBOARD.md](LEADERBOARD.md) |
| Avoid known setup / interpretation traps | [PITFALLS.md](PITFALLS.md) |
| See what v0.3 explicitly does NOT do | [LIMITATIONS.md](LIMITATIONS.md) |
| See what changed across versions | [CHANGELOG.md](CHANGELOG.md) |
| Inspect historical example reports | [examples/](src/kv_fidelity/examples/) (4 sample JSONs + HTMLs) |
| Check whether reports support direct comparison | [Comparison guide](../../docs/guides/kv-fidelity-comparison.md) |
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

KV Fidelity rejects the PPL framing entirely: nothing matters except how
close the quantized model's behaviour stays to its fp16 self.

Read [`research/papers/attn-rotation-and-ppl-artifact.md`](../../research/papers/attn-rotation-and-ppl-artifact.md)
for the full motivation.

## What ships in the current source

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
supported KV dtypes differ by backend; run `kv-fidelity selftest` before a score.
Native vLLM/SGLang KLD is explicitly labeled as a normalized top-k estimate,
not full-vocabulary KL.

## Subcommands

```
kv-fidelity score          # score a candidate KV config
kv-fidelity selftest       # 30s preflight: binaries, flags, model probe
kv-fidelity compare        # shared Metria compatibility checks and score table
kv-fidelity repeatability  # run N times, report spread (stdev/range)
kv-fidelity fetch          # download wikitext-2-raw corpus to ~/.cache/kv-fidelity/
```

## Reports

Every `score` run can emit two formats via `--json-out` and `--html-out`:

- **JSON** (`--json-out report.json`) — schema `kv_fidelity.report.v0.3.3`,
  consumable by `kv-fidelity compare` or any JSON-aware tool.
- **HTML** (`--html-out report.html`) — single **self-contained file**
  (~40 KB) with composite stats, diagnosis callout, per-axis bars,
  R-NIAH heatmap, PLAD per-perturbation table, run details (hardware +
  model + env), the sanitized repro command, and the raw JSON embedded
  in a collapsible section. Sun/moon toggle in the top-right for
  light/dark mode (follows OS by default). Pasteable in Discord/X.
  See [`examples/`](src/kv_fidelity/examples/) for 4 real samples.

The HTML uses `light-dark()` CSS (Chrome 123+ / Safari 17.5+ / Firefox
120+) for dark mode and a native system-font stack. It contains no external
font, script, or stylesheet dependency.

Development reports now include named metric methods and available local input
identities under `extras.comparison_evidence`. `compare` requires at least two
valid reports and rejects missing evidence or incompatible methods by default.
Full-vocabulary KLD and top-k estimates remain different metrics. Historical
reports can be inspected with a reasoned `--allow-incompatible` override, which
retains their incompatibilities in text and JSON. See the
[comparison and migration guide](../../docs/guides/kv-fidelity-comparison.md).

## Quickstart

See [QUICKSTART.md](QUICKSTART.md) for full setup. Short version:

```bash
# 1. Verify your setup
python3 -m kv_fidelity.cli selftest --backend auto --model path/to/model.gguf

# 2. First quick score (~5-7 min on a 7B Q8)
python3 -m kv_fidelity.cli score \
    --model path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --json-out report.json \
    --html-out report.html

# 3. Full audit (~25-30 min on a 7B Q8)
python3 -m kv_fidelity.cli score \
    --model path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --full \
    --rniah-ctx-max 16384 \
    --json-out report.json --html-out report.html

# 4. Verify reproducibility (4 runs, expect stdev ≤ 1.0)
python3 -m kv_fidelity.cli repeatability \
    --model path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --runs 4
```

These commands use the prompt set bundled in the package. The corpus is
resolved from `~/.cache/kv-fidelity/` or downloaded on first use; `--full` also
resolves the cached haystack. Explicit `--prompts`, `--corpus`, and
`--rniah-haystack` paths remain available for pinned runs. For offline/CI
execution, run `kv-fidelity fetch` first or pass explicit paths, then add
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
| [examples/](src/kv_fidelity/examples/) | Sample JSONs + HTML reports (clean / degraded / distribution-broken / catastrophic) |
| [research/papers/attn-rotation-and-ppl-artifact.md](../../research/papers/attn-rotation-and-ppl-artifact.md) | Why this framework exists at all (the motivation paper) |

## File layout

```
components/kv-fidelity/
  pyproject.toml          # kv-fidelity distribution metadata
  src/kv_fidelity/
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

Especially valuable feedback: surfaces where KV Fidelity fails silently
(low base_acc, NaN perturbations, etc.) before the confidence guards
catch them.
