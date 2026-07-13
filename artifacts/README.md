# Research artifacts

This directory contains retained outputs used to support or reproduce the
repository's research record.

## Layout

- `benchmarks/legacy-raw/` — immutable historical benchmark logs, including
  negative and later-invalidated runs.
- `niah/` — timestamped needle-in-a-haystack reports and machine-readable
  results.
- `mlx/` — generated MLX quality-suite reports.
- `ablations/sparse-v-threshold/` — raw threshold-ablation logs.
- `profiles/` — reusable hardware diagnostic baselines.

Artifacts are evidence, not current recommendations. Consult
[`docs/`](../docs/index.md) for maintained guidance and
[`research/`](../research/README.md) for the interpretation of an experiment.

New artifacts should include enough metadata to identify the model, engine
version, configuration, hardware, command, and date. Do not commit downloaded
models, corpora, caches, or temporary build output.
