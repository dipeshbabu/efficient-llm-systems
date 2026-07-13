# Efficient LLM Systems

Research, reference implementations, evaluation tools, and reproducible
evidence for making large-language-model inference more efficient without
losing behavioral fidelity.

This repository is an umbrella for work on KV-cache and weight compression,
quantization, sparse and long-context attention, inference kernels, hardware
diagnostics, cross-engine validation, and deployment-quality evaluation.
Production engine integrations live in their respective upstream projects;
this repository keeps the portable Python components, experimental tools,
current guidance, and the evidence behind the recommendations.

## Components

| Component | Purpose | Stability |
|---|---|---|
| [REFRACT](components/refract/README.md) | Reference-anchored fidelity evaluation across llama.cpp, MLX, vLLM, and SGLang | Beta; distribution name `refract-llm` |
| [TurboQuant Reference](components/turboquant-reference/README.md) | NumPy/SciPy implementation of PolarQuant, QJL, KV-cache compression, packing, and related experiments | Research reference |
| [Tools](tools/README.md) | Diagnostics, quality validation, benchmarking, and model-conversion utilities | Mixed; see each tool's requirements |
| [Research](research/README.md) | Dated papers, investigations, negative results, and archived plans | Evidence record |
| [Artifacts](artifacts/README.md) | Retained raw benchmark output, NIAH proofs, ablations, and hardware profiles | Immutable evidence where noted |

The repository name is the umbrella identity. Existing public component
contracts remain unchanged:

- PyPI distribution: `refract-llm`
- REFRACT import and command: `refract`
- TurboQuant reference import: `turboquant`

## Research areas

- KV-cache and weight compression
- Scalar, vector, and residual quantization
- Sparse, selective, and long-context attention
- Layer-aware and asymmetric K/V policies
- Decode and prefill kernel performance
- Apple Silicon, CUDA, ROCm, Vulkan, and CPU behavior
- Cross-engine reproducibility and fidelity evaluation
- Hardware diagnostics and benchmark methodology

## Start here

- [Documentation index](docs/index.md)
- [Getting started](docs/guides/getting-started.md)
- [TurboQuant configuration recommendations](docs/guides/turboquant-recommendations.md)
- [Benchmark reference](docs/reference/benchmarks.md)
- [REFRACT quick start](components/refract/QUICKSTART.md)
- [Repository contribution guide](CONTRIBUTING.md)

## Development setup

Install `uv`, then synchronize the workspace from the repository root:

```bash
git clone https://github.com/dipeshbabu/efficient-llm-systems.git
cd efficient-llm-systems

uv sync --all-packages
uv run pre-commit install
uv run pytest
```

Backend-specific REFRACT dependencies are optional. Add only the extra needed
for the backend under test:

```bash
uv sync --all-packages --extra refract-mlx
uv sync --all-packages --extra refract-vllm
uv sync --all-packages --extra refract-sglang
```

Avoid `--all-extras`: the backend stacks have different platform and hardware
requirements.

Python code follows PEP 8. Ruff enforces linting and import order and formats
the codebase using the repository's Python 3.9 target and 88 character line
length. Line-length rule `E501` is delegated to the formatter. Pre-commit runs
Ruff, mypy, and the repository's lightweight file checks before each commit.
See the [contribution guide](CONTRIBUTING.md) for check and fix commands.

## Current findings

The repository's controlled experiments support three recurring conclusions
within the tested model and hardware matrix:

1. Value-cache compression can often be substantially more aggressive than
   key-cache compression. See the
   [asymmetric K/V study](research/papers/asymmetric-kv-compression.md).
2. Key precision usually dominates quality because K controls attention
   routing. See the
   [M5 Max stress test](research/papers/m5-max-stress-test.md).
3. Boundary layers are disproportionately sensitive on several tested
   architectures. See the
   [layer-aware V study](research/papers/layer-aware-v-compression.md).

These are evidence-bounded findings, not universal guarantees. Validate every
new model, engine, context length, and hardware target. REFRACT exists to make
that comparison behavioral rather than relying on perplexity alone.

## Production ecosystem

The production implementations are maintained outside this research
monorepo:

| Project | Role |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | Upstream TurboQuant attention backend |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Upstream Hadamard KV rotation and platform kernels |
| [Historical llama.cpp TurboQuant fork](docs/reference/historical-forks.md#llamacpp-experimental-forks) | Full TurboQuant KV and weight formats across Metal, CUDA, HIP, and CPU; public fork URL currently unavailable |
| [mlx-swift-lm](https://github.com/ekryski/mlx-swift-lm) | Apple Silicon inference and TurboQuant collaboration |
| [Historical vllm-swift prototype](docs/reference/historical-forks.md#swift-and-long-context-prototypes) | Swift serving on Apple Silicon; public prototype URL currently unavailable |

Use the component and engine documentation for supported formats and current
runtime flags.

## Repository layout

```text
components/
  refract/                 Published fidelity-evaluation package
  turboquant-reference/    Portable quantization reference package
docs/
  guides/                  Current operational guidance
  reference/               Curated benchmark and compatibility references
research/
  papers/                  Dated research reports
  investigations/          Engineering experiments and validation records
  archive/                 Superseded plans and historical documentation
tools/
  diagnostics/             Hardware and runtime diagnostics
  validation/              Quality, NIAH, and regression gates
  benchmarks/              System benchmark drivers
  conversion/              Model conversion helpers
  maintenance/             Repository integrity checks
artifacts/
  benchmarks/              Retained raw benchmark output
  niah/                    Retrieval evidence
  mlx/                     MLX quality output
  ablations/               Controlled ablation logs
  profiles/                Hardware baselines
```

Current guidance belongs in `docs/`. Dated claims and negative results belong
in `research/`. Generated evidence belongs in `artifacts/`. Executable
workflows belong in `tools/` or the component that owns them.

## Verification

Run the complete Python gate:

```bash
uv run pre-commit run --all-files
uv run pytest
```

Build components independently:

```bash
uv run python -m build components/refract
uv run python -m build components/turboquant-reference
```

The root is deliberately not a publishable Python distribution. Each
component owns its dependencies, tests, package data, and release lifecycle.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
