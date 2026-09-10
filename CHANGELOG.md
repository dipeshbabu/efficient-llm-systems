# Metria changelog

This changelog covers the root `metria` distribution. KV Fidelity and
TurboQuant reference packages have independent versions and release notes.

## 0.1.0

First Alpha release, prepared for maintainer publication.

### Local llama.cpp verification

- `metria verify` compares two CPU thread counts using the same pinned GGUF,
  qualified llama.cpp binary, greedy completion workload, and batch settings.
- Native readback checks what actually ran before token trajectories are
  compared. Missing or inconsistent evidence cannot produce a verified result.
- Each execution retains a manifest, readable report, and both run records.
  Failures, timeouts, interruption, and partial evidence remain inspectable.
- Source archives include the pinned CPU runtime build helper, capture patch
  and its upstream MIT license, and model/recipe preparation tools.

### Evidence APIs and supporting commands

- Immutable study, run, model, runtime, workload, and hardware identities.
- Failure-aware Python execution APIs with llama.cpp and vLLM adapters.
- Versioned JSON recipes and run records with deterministic SHA-256 digests.
- Recipe validation, normalization and digesting; capability inspection;
  saved-record comparison; and token-trajectory agreement analysis.
- A dependency-free Python wheel and source distribution for Python 3.10–3.14.
  Inference runtimes and models are installed separately.

### Scope and qualification

The end-to-end CLI is qualified on Linux/Ubuntu WSL with local CPU inference.
Core tests also run on Windows and macOS. GPU settings, quantization treatments,
runtime upgrades, chat templates, and repeated-trial policies are outside the
first verifier's supported scope. The vLLM Python adapter is available but is
not an end-to-end release-qualified CLI workflow.

`VERIFIED` means the requested comparison had sufficient evidence and its
analysis completed. It does not mean task quality, speedup, or deployment
acceptance. Process timing includes startup and model loading. The retained
tiny-model example is an integration check, not a performance or quality study.

Prompt and generated text are omitted from summary reports. Run records retain
requested configuration, which can include prompts and local paths; review them
before sharing. Public APIs remain provisional during the 0.1 series.

See the [verification guide](docs/guides/metria-verify.md) for installation,
exit codes, evidence interpretation, and setup requirements.
