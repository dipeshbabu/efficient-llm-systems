# Changelog

All notable user-visible changes to `turboquant-reference` are recorded here.
The package follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Public quantizer, rotation, packing, and memory APIs now validate numeric
  types, finite values, dimensions, bit widths, indices, signs, norms, and
  compressed metadata before computation. Invalid types raise `TypeError` and
  invalid values/shapes raise `ValueError`; validation remains active with
  `python -O`. QJL's orthogonality invariant uses an explicit exception.
- Norm calculation handles extreme representable magnitudes without silently
  producing zero or infinity. Packed float32 norms reject unrepresentable values.
- Single-row outlier batches preserve their batch axis, and outlier memory
  estimates count only norms belonging to active channel groups. Memory reports
  require positive counts and use integral byte accounting.

### Performance

- Eight-bit packing makes one owned conversion instead of two copies; empty
  packing/unpacking avoids work on unused data. Scalar structured rotations use
  the shared in-place kernel without an extra transform copy.

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
