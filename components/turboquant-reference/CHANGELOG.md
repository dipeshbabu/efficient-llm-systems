# Changelog

All notable user-visible changes to `turboquant-reference` are recorded here.
The package follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Performance

- Scalar and batch rotations share a vectorized Walsh-Hadamard kernel.
  Scalar transforms no longer run Python loops over individual elements,
  and batch transforms reuse scratch space across stages. Inputs remain
  unchanged and supported vector outputs retain the existing normalization.
  Non-vector input to the scalar transform now raises a clear error.

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
