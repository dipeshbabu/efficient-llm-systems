# Metria

[![PyPI](https://img.shields.io/pypi/v/metria)](https://pypi.org/project/metria/)
[![CI](https://github.com/dipeshbabu/metria/actions/workflows/ci.yml/badge.svg)](https://github.com/dipeshbabu/metria/actions/workflows/ci.yml)
[![Metria core](https://github.com/dipeshbabu/metria/actions/workflows/metria-core.yml/badge.svg)](https://github.com/dipeshbabu/metria/actions/workflows/metria-core.yml)
[![Root package](https://github.com/dipeshbabu/metria/actions/workflows/root-package.yml/badge.svg)](https://github.com/dipeshbabu/metria/actions/workflows/root-package.yml)

**Verify inference changes with reproducible evidence.**

Metria is an open-source experiment and evidence layer for LLM inference
systems. It helps researchers answer a question that raw benchmark numbers do
not:

> For this model, runtime, workload, hardware, and treatment, what did we ask
> for, what actually ran, what changed, and are the resulting measurements
> valid to compare?

Metria sits **above** inference runtimes and optimization libraries. It is not a
serving engine, scheduler, kernel library, or universal quantizer. Systems such
as vLLM, llama.cpp, SGLang, MLX, TensorRT-LLM, torchao, LLM Compressor, and
custom research code remain responsible for execution; Metria provides the
study, provenance, measurement, and comparison layer around them.

> **Status:** [Metria 0.1.0 is available on PyPI](https://pypi.org/project/metria/0.1.0/).
> This first release is Alpha; public APIs remain provisional.

The first complete CLI workflow verifies a **local llama.cpp CPU thread-count
change** using a pinned model and qualified capture provider. It checks what ran,
compares sampled token trajectories, and saves a report with both run records.
See the [0.1.0 release notes and downloads](https://github.com/dipeshbabu/metria/releases/tag/metria-v0.1.0)
for the supported scope, source archive, wheel, and checksums.

## Why Metria

Inference research is easy to benchmark and surprisingly hard to compare well.
A result can look reproducible while hiding differences in model revisions,
tokenizers, runtime settings, cache formats, hardware, measurement methods, or
what the runtime actually applied.

Metria is being built around four principles:

1. **Requested is not observed.** Asking for FP8 KV cache is not proof that the
   runtime used FP8 KV cache.
2. **Comparability is study-specific.** A runtime or hardware change may be the
   variable under study rather than something that makes two runs globally
   incompatible.
3. **Measurements carry method identity.** Values with different methods or
   versions are not silently treated as the same metric.
4. **Failed and partial runs are evidence too.** Missing observation, timeout,
   unsupported configuration, or failed cleanup should not be converted into an
   apparently successful result.

## What works today

| Capability | Current state |
|---|---|
| Study design | `StudySpec`, `RunSpec`, `ComparisonPlan`, treatments, controls, blocking dimensions |
| Evidence model | Requested → resolved → observed state with immutable run evidence |
| Typed identity | `ModelRef`, `RuntimeConfig`, `WorkloadSpec`, `CapabilitySet`, `HardwareFingerprint`, `ArtifactManifest` |
| Capability inspection | Conservative model-geometry normalization, TurboQuant KV guardrails, and `metria inspect` |
| Hardware evidence | Privacy-conscious stdlib host/software fingerprinting; accelerator identity remains runtime-observed |
| Runtime lifecycle | `RuntimeAdapter` / `RuntimeSession` plus reusable runtime contract tests |
| First-party runtimes | llama.cpp and vLLM adapters |
| Execution | Failure-aware `execute_run()` and `execute_study()` Python APIs |
| Measurements | Decode-time token trajectory capture with retained prompt fingerprints |
| Pairwise analysis | KV Fidelity-compatible trajectory agreement analysis |
| Recipes | Versioned `metria.study_recipe.v1` JSON with deterministic SHA-256 digesting |
| Run records | Versioned `metria.run_record.v1` JSON with typed metrics plus full-record/evidence digests |
| CLI | Local CPU-thread `metria verify`, recipe `validate` / `digest` / `normalize`, `metria inspect`, and saved-record `metria compare` |
| Packaging | Dependency-free root `metria` wheel and source archive; focused components stay independent |

The standalone [KV Fidelity](components/kv-fidelity/README.md) package also
supports llama.cpp, MLX, vLLM, and SGLang for its focused KV-cache evaluation
workflow. Those component backends should not be confused with the smaller set
of first-party runtime adapters already exposed by the Metria core.

## Quick start

### 1. Install Metria

Use Python 3.10–3.14:

```bash
python -m pip install metria==0.1.0
metria --version
metria verify --help
```

For development, install the current source checkout:

```bash
git clone https://github.com/dipeshbabu/metria.git
cd metria
python -m pip install .

metria --version
metria --help
```

The root package has no runtime dependencies. Install the native inference
runtime and model separately; the guide below provides a pinned CPU example.

### 2. Verify a local CPU thread change

On Linux or Ubuntu WSL, follow the
[local verification guide](https://github.com/dipeshbabu/metria/blob/main/docs/guides/metria-verify.md)
to obtain the setup helpers from the source archive, build the pinned llama.cpp
capture provider, and prepare the small model and `study.json`. Git, a C++17
compiler, CMake, and curl are needed for that setup. Then run:

```bash
metria verify study.json --output verification
```

The output contains `manifest.json`, `report.md`, `reference.run.json`, and
`candidate.run.json`. The command distinguishes completed comparisons,
insufficient evidence, invalid comparisons, and execution failures. It reports
behavioral observations; it does not apply a universal quality threshold.

Use a new output directory for each run. `VERIFIED` means the comparison has
sufficient evidence and its analysis completed; it does not certify quality,
a speedup, or deployment acceptance. GPU settings, quantization, runtime upgrades,
chat templates, and repeated-trial policies are outside this first CLI scope.

The development version adds [optional acceptance policies](docs/guides/verification-policies.md)
that produce PASS/FAIL after the evidence and comparison gates succeed. These
use your criteria, with no universal safety thresholds; they are not included
in the published 0.1.0 release.

## Supporting recipe and comparison tools

### Define a study recipe

Metria recipes describe requested experiment intent as versioned data. For
example:

```json
{
  "schema": "metria.study_recipe.v1",
  "study": {
    "name": "runtime-comparison",
    "runs": [
      {
        "model": {"id": "example/model"},
        "runtime": {"name": "llamacpp"},
        "scenario": {"name": "decode"},
        "measurements": ["kv_fidelity.decode_time_trajectory"]
      },
      {
        "model": {"id": "example/model"},
        "runtime": {"name": "vllm"},
        "scenario": {"name": "decode"},
        "measurements": ["kv_fidelity.decode_time_trajectory"]
      }
    ],
    "comparison": {
      "vary": ["runtime"],
      "control": ["model", "scenario", "measurements"],
      "analyses": ["kv_fidelity.trajectory_match"]
    }
  },
  "measurement_configs": {
    "kv_fidelity.decode_time_trajectory": {
      "prompts": [
        {"id": "p1", "prompt": "The capital of France is"}
      ]
    }
  },
  "environment": {}
}
```

Validate and fingerprint it:

```bash
metria recipe validate study.json
metria recipe digest study.json
```

Normalize a validated recipe to deterministic JSON with:

```bash
metria recipe normalize study.json --output normalized-study.json
```

`normalize` reproduces the complete recipe, including prompt text or other
sensitive input. Do not treat normalized private recipes as safe-to-publish
artifacts.

Inspect data-only geometry/capability and local hardware evidence before a run:

```bash
metria inspect study.json
metria inspect study.json --json
```

Inspection is conservative: it does not infer model geometry from a model name,
and an accelerator is not claimed present merely because an environment
variable mentions it. See the
[capability inspection guide](docs/guides/metria-inspection.md).

### Persist and compare run evidence

The Python execution APIs return `RunRecord` values. Persist them with the
versioned record API:

```python
from metria import dump_run_record

dump_run_record("run-0001.json", record)
```

Then compare saved records under the recipe's explicit `ComparisonPlan`:

```bash
metria compare run-0001.json run-0002.json --recipe study.json
metria compare run-0001.json run-0002.json --recipe study.json --json
```

Use `metria verify` for the qualified local CPU-thread workflow. General study
execution remains available through the Python APIs; the CLI does not expose a
generic `metria run` command.

See the [CLI guide](docs/guides/metria-recipe-cli.md) and
[run-record guide](docs/guides/metria-run-records.md).

## Core model

Metria does not model an experiment as a fixed list of six peer objects. The
study decides what varies and what must stay fixed:

```text
Study = factors + controls + comparison plan

Run = system under test
    × scenario / workload
    × measurement protocol
    @ observed environment
```

Every run then separates:

```text
requested
    ↓
resolved
    ↓
observed
    ↓
evidence + metrics + artifacts + lifecycle events
```

- **Requested** — what the user or recipe asked for.
- **Resolved** — exact settings, revisions, artifacts, and choices selected
  before launch.
- **Observed** — what the runtime and environment report actually ran.

That distinction is central to Metria. A configuration request is experiment
intent; observed state is evidence.

See [Metria core architecture](docs/architecture/metria-core.md) for the full
provisional contract.

## Comparison semantics

A `ComparisonPlan` declares the role of experiment dimensions:

```text
vary      dimensions intentionally changed
control   dimensions that must match
block_by  dimensions used to form comparable groups
```

For example, a study may intentionally vary runtime and KV-cache treatment,
control model/workload/measurement method, and block by hardware class.

Metria therefore does **not** use one universal "same fingerprint = comparable"
rule. Missing controlled evidence is not equality, and methodologically
different metrics require an explicit analysis that defines how they may be
combined.

Saved-record comparison deliberately requires the study recipe that supplies the
comparison plan. Record/evidence digests identify serialized evidence; they do
not replace study semantics.

## Versioned run evidence

`metria.run_record.v1` stores one executed run as strict JSON while preserving:

- requested `RunSpec` using the same schema as study recipes;
- resolved and observed runtime state;
- lifecycle status;
- metric identity, raw samples, aggregation, uncertainty, and coverage;
- measurement evidence and artifact references;
- lifecycle events and execution provenance.

`run_record_digest()` covers the full record, including requested intent and
local run identity. `run_evidence_digest()` covers produced evidence while
excluding study/run IDs and requested intent. Neither digest is a universal
comparability proof.

See [run records and comparison](docs/guides/metria-run-records.md).

## Runtime adapters

The Metria core currently includes two first-party runtime adapters:

### llama.cpp

The current adapter supports local llama.cpp command-line execution, GGUF model
paths, explicit runtime settings, KV-cache treatments, binary identity hashing,
requested/resolved/observed evidence, and optional decode-time token-ID capture
with a compatible patched binary.

See the [llama.cpp runtime guide](docs/guides/metria-llamacpp-runtime.md).

### vLLM

The current adapter uses the offline `vllm.LLM` API, keeps vLLM as an optional
lazy dependency, supports native token-ID capture, and separates configured
runtime state from introspected applied engine state.

See the [vLLM runtime guide](docs/guides/metria-vllm-runtime.md).

Runtime support is intentionally narrow while the common adapter contract is
being hardened. Metria should prefer adapters over reimplementing upstream
runtimes.

## Measurement and fidelity

The first Metria measurement bridge is decode-time token trajectory capture.
Each run retains its own token-ID trajectory evidence and prompt fingerprints;
the trajectory agreement score is derived only when a valid reference/candidate
pair is compared.

This keeps two concepts separate:

```text
run-local evidence != pairwise fidelity metric
```

The pairwise trajectory analysis is compatible with the current KV Fidelity
trajectory methodology while avoiding KV Fidelity's legacy module-global
backend dispatch.

See the [trajectory measurement guide](docs/guides/metria-trajectory-measurement.md).

## Focused components

Metria remains a monorepo with focused components that keep their own package
identities and release lifecycles.

### KV Fidelity

Reference-anchored behavioral evaluation for KV-cache compression and runtime
changes.

```bash
python -m pip install "./components/kv-fidelity"
kv-fidelity --help
```

See [KV Fidelity](components/kv-fidelity/README.md) and its
[quick start](components/kv-fidelity/QUICKSTART.md).

### TurboQuant Reference

Portable NumPy/SciPy reference implementations of PolarQuant, QJL, and
TurboQuant KV-cache compression.

```bash
python -m pip install "./components/turboquant-reference"
python components/turboquant-reference/benchmarks/examples/demo.py
```

See [TurboQuant Reference](components/turboquant-reference/README.md).

## What Metria is not

Metria is not intended to become:

- another inference server;
- a replacement for vLLM, llama.cpp, SGLang, MLX, or TensorRT-LLM;
- a reimplementation of every quantization algorithm;
- a universal scalar fidelity score;
- an automatic recommendation engine before measurement uncertainty and
  failure semantics are mature;
- a single environment containing every inference runtime;
- a training-optimization framework.

## Roadmap

The current design roadmap is tracked in
[issue #50](https://github.com/dipeshbabu/metria/issues/50).

Near-term work is focused on:

1. **Observed runtime identity** — stronger served model/tokenizer/applied-config
   evidence.
2. **Verifier coverage** — extend the local CPU-thread workflow to additional
   qualified inference changes, with trustworthy performance and acceptance policies.
3. **Runtime qualification** — exercise the shared contract against first-party
   adapters and add hardware-qualified evidence lanes.
4. **Artifact provenance** — immutable model/data verification and manifest
   identity.
5. **Shared systems APIs** — move benchmark, timeout, diagnostics, and reusable
   KV Fidelity logic behind Metria protocols rather than adding more standalone
   scripts.

Later phases can add broader evaluation suites, more runtime/optimization
adapters, experiment matrices, Pareto visualization, and constrained search.
Automatic recommendation should come only after the evidence layer is mature
enough to support it.

## Repository layout

```text
src/metria/                 Shared Metria core
metria_tests/               Core contract tests
components/
  kv-fidelity/              Focused fidelity evaluator
  turboquant-reference/     Portable algorithm reference
docs/                       Current guidance and architecture
research/                   Dated studies and investigations
artifacts/                  Retained experiment evidence
tools/                      Existing diagnostics and benchmark utilities
```

Current guidance belongs in `docs/`. Dated research conclusions and negative
results belong in `research/`. Generated evidence belongs in `artifacts/`.
Historical evidence should remain historical rather than being silently
rewritten to match newer conclusions.

## Development

```bash
git clone https://github.com/dipeshbabu/metria.git
cd metria

uv sync --all-packages
uv run pre-commit install
uv run pytest
```

Core-only checks:

```bash
uv run pytest metria_tests -v
uv run mypy src/metria
uv run ruff check src/metria metria_tests
uv run ruff format --check src/metria metria_tests
```

Build the workspace distributions independently:

```bash
uv run python -m build .
uv run python -m build components/kv-fidelity
uv run python -m build components/turboquant-reference
```

Backend-specific inference dependencies remain optional. Install only the stack
needed for the runtime under test.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), and
[SUPPORT.md](SUPPORT.md).

Changes to experiment semantics, metric methodology, comparison rules,
packaging, or release policy should be discussed before those contracts are
stabilized. New runtime, evaluator, benchmark, or optimization work should plug
into the common Metria contracts rather than creating a parallel architecture.

## Citation

Use [CITATION.cff](CITATION.cff) when Metria supports your work. When relying on
a specific result under `research/`, cite that report as well so readers can
recover its model, runtime, hardware, configuration, and date.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
