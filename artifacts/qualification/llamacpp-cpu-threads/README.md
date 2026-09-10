# Local llama.cpp CPU-thread qualification

These bundles were generated on 2026-09-10 by an installed Metria wheel, using a
pinned, CPU-only llama.cpp build and the SHA-256-verified stories260K GGUF.
They exercise the real runtime, token capture, evidence checks, persistence,
comparison, and public `metria verify` command.

- [One thread to two threads](threads-1-to-2/report.md): both runs completed,
  actual thread counts matched the request, and all three prompt trajectories
  matched. The command returned `VERIFIED` with exit status `0`.
- [Rounded context rejected](context-rounding-rejected/report.md): both recipes
  requested context 128, while the runtime reported 256. The command returned
  `INSUFFICIENT_EVIDENCE` with exit status `1` and did not report a behavioral
  score. The failed verification's records were retained.

Each case includes its recipe, two run records, manifest, and report. File hashes,
source revisions, the tested wheel hash, and build/model identities are recorded
in [qualification.json](qualification.json).

The model is a small integration fixture with three prompts and 16 generated
tokens per prompt. These single-pass process wall-time observations are not a
model-quality or statistically qualified performance result. Qualification is
limited to the recorded CPU build, model, host, and verification scope. Broader
runtime and hardware qualification remains tracked by issue #12.

To reproduce the workflow with your own local paths, follow the
[local verification guide](../../../docs/guides/metria-verify.md). The retained
absolute `/var/tmp` paths describe the qualification environment; they are not
portable installation locations.
