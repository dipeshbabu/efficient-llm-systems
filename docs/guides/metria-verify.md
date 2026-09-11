# Verify a local llama.cpp CPU thread change

`metria verify` runs one reference and one candidate, checks the evidence needed
for their comparison, and writes a result you can inspect or retain in CI.
The first supported workflow changes CPU thread count while holding the local
model, binary, prompt workload, batch threads, and other settings fixed.

This workflow uses plain greedy completion, a SHA-256-pinned GGUF model, and a
qualified llama.cpp capture provider. It does not yet verify GPU settings,
quantization treatments, runtime upgrades, chat templating, or repeated-trial
policies. General study execution remains available through the Python APIs.

## Install and prepare the example

Use Linux or Ubuntu WSL for the pinned example below. You need Git, a C++17
compiler, CMake, Python 3.10-3.14, and curl. Inference stays on the local CPU.

Install the published root package in a Python virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install metria==0.1.0
```

The Python package does not bundle the native runtime or model. The source
archive contains the setup helpers. Download and extract it from PyPI:

```bash
python -m pip download --no-deps --no-binary=:all: metria==0.1.0
tar -xzf metria-0.1.0.tar.gz
cd metria-0.1.0
```

The archive and checksums are also available in the
[GitHub release](https://github.com/dipeshbabu/metria/releases/tag/metria-v0.1.0).
From the extracted directory, prepare the pinned local example:

```bash
sh tools/qualification/build_llamacpp_cpu.sh /var/tmp/metria-example-build
curl -fL "https://huggingface.co/ggml-org/models-moved/resolve/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories260K.gguf" -o stories260K.gguf
python tools/qualification/prepare_cpu_verification.py \
  --bin-dir /var/tmp/metria-example-build/build/bin \
  --model stories260K.gguf \
  --output study.json
```

Choose a new build workspace and new output filenames. The build helper pins
upstream llama.cpp and applies the small capture patch documented in
[qualification tools](../../tools/qualification/README.md). It builds a CPU-only
`llama-completion`; no inference server or GPU dependency is installed.

The preparation tool verifies the model's published SHA-256 and exercises the
provider before writing a recipe. Its probe record is saved alongside the recipe
as `study.qualification.run.json`. An unpatched provider, incorrect model digest,
or missing native readback fails qualification. The tiny 1.19 MB model is an
integration fixture, not a model-quality benchmark.

The default recipe compares one CPU thread with two, holds batch threads at one,
and uses three short prompts with 16 generated tokens each. You can choose thread
counts with `--reference-threads` and `--candidate-threads`. For another local GGUF,
supply its independently trusted digest with `--model-sha256`.

## Run verification

```bash
metria verify study.json --output verification
```

For machine-readable stdout:

```bash
metria verify study.json --output verification-json --json
```

Each output directory must be new. A completed verification contains:

```text
verification/
  manifest.json
  report.md
  reference.run.json
  candidate.run.json
```

The reference record is saved before the candidate starts. Runtime failures,
timeouts, incomplete observation, and interruption are retained as evidence.
Writes use temporary files, and the manifest is published last. A filesystem
failure leaves completed records intact and does not produce a success manifest.

The report omits workload prompt text and generated text. Run records retain
prompt IDs, fingerprints, sampled token IDs, requested configuration, and local
artifact paths. Review configuration and paths before sharing the records.
The plain-completion workflow rejects system prompts and expert runtime flags
instead of silently ignoring them.

## Read the outcome

| Verdict | Meaning |
|---|---|
| `VERIFIED` | Both runs completed, the scoped evidence checks passed, and behavioral comparison completed. |
| `PASS` / `FAIL` | Development version: a valid comparison met / did not meet an explicit user-defined acceptance policy. |
| `NOT_COMPARABLE` | An undeclared or controlled difference prevents a valid comparison. |
| `INSUFFICIENT_EVIDENCE` | Required model/provider identity, runtime readback, or token captures are absent or inconsistent with the request. |
| `EXECUTION_FAILED` | Execution, timeout, interruption, or behavioral analysis prevented completion. |

Exit status is `0` for `VERIFIED` or `PASS`, `1` for other verification outcomes, `2` for
invalid input or filesystem errors, and `130` for interruption.

`VERIFIED` is not a task-quality or deployment-acceptance verdict. Token prefix
agreement and exact sequence matches describe behavioral change on the supplied
prompts. Without a policy, the report explicitly records that no acceptance
policy was evaluated. The development version adds
[optional typed acceptance policies](verification-policies.md); published 0.1.0
does not include that feature.

Process wall-time samples include startup, model loading, prompt evaluation, and
generation. They are descriptive observations from this workload; they are not
decode-only throughput, TTFT, isolated kernel timing, or a statistically qualified
performance claim. Repeated-trial and verifier-native performance measurement
remain separate follow-up work; current policy targets cover behavioral analysis.

The observed thread count and context come from the running llama.cpp context.
Missing readback never becomes a match. For example, if llama.cpp rounds a
requested context to a different value, the verifier reports insufficient
evidence for the requested configuration rather than silently accepting it.

## Python API

```python
from metria import load_study_recipe, verify_recipe

result = verify_recipe(load_study_recipe("study.json"), "verification")
print(result.manifest["verdict"])
```

The JSON manifest uses `metria.verification.v1` and records the scoped contract,
recipe digest, run/evidence digests, hardware evidence, observed facts, comparison
issues, analysis identity, diagnostics, and process wall-time method.
