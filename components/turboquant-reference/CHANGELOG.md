# Changelog

All notable user-visible changes to `turboquant-reference` are recorded here.
The package follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- Removed Accelerate from the `bench` extra to address
  GHSA-4j2p-28q2-5m79 without relying on an unpublished upstream fix.
  The real-model CPU validator now loads through Transformers and uses explicit
  PyTorch placement. Models must fit host memory; automatic device dispatch and
  offload are not part of these single-device examples.

### Added

- NumPy/SciPy reference implementations of PolarQuant, QJL, TurboQuant, and
  KV-cache compression.
- Packing helpers, hardware-profile replay utilities, tests, and runnable
  experiment examples.

Before a release, maintainers replace `Unreleased` with a heading in the form
`[VERSION] - YYYY-MM-DD` and add a new empty `Unreleased` section above it.
