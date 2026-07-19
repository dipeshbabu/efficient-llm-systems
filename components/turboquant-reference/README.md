# TurboQuant Reference

`turboquant-reference` is the NumPy/SciPy reference implementation of
TurboQuant KV-cache compression in the Efficient LLM Systems repository. The
distribution name is `turboquant-reference`; the stable Python import remains
`turboquant`.

This component is intended for algorithm inspection, reproducibility, and
experimentation. Production inference integrations live in their respective
engine projects.

## Install

After a tagged release has been published, install it from your package index:

```bash
python -m pip install "turboquant-reference>=0.1,<0.2"
```

Install the optional real-model benchmark stack with:

```bash
python -m pip install "turboquant-reference[bench]>=0.1,<0.2"
```

Before the first package-index release, or when validating a checkout, use a
non-editable source install from the repository root:

```bash
python -m pip install "./components/turboquant-reference"
```

For development, install the component in editable mode:

```bash
python -m pip install -e "./components/turboquant-reference[dev]"
```

Use the editable benchmark extra when changing real-model experiments:

```bash
python -m pip install -e "./components/turboquant-reference[bench]"
```

## Verify

```bash
python -m pytest components/turboquant-reference/tests -q
python components/turboquant-reference/benchmarks/examples/demo.py
```

## Public API

The package exports `PolarQuant`, `QJL`, `TurboQuant`, `TurboQuantMSE`,
`CompressedVector`, and `KVCacheCompressor`. `QJL` and the full
`TurboQuant` pipeline are retained for paper-oriented reproduction;
`TurboQuantMSE` provides the direct reconstruction-oriented path used by most
of the component experiments.

## Layout

```text
src/turboquant/       Python reference package
tests/                Component unit tests
benchmarks/examples/  Small runnable demonstrations
benchmarks/runners/   Server and llama.cpp benchmark drivers
benchmarks/validation Real-model and dimension-scale validation
benchmarks/experiments Exploratory compression experiments
benchmarks/archive/   Historical comparisons and result snapshots
```

Run benchmark files from an environment where this component has been
installed. They intentionally import `turboquant` from the installed package
instead of modifying `sys.path`.

## Stability and compatibility

The package is an alpha research reference and follows semantic versioning.
During the `0.x` series, incompatible public-API changes may ship in a minor
release; patch releases preserve the documented API. Deprecations receive a
changelog entry and a migration path when practical.

Released wheels support Python 3.10 through 3.13 on operating systems where
NumPy and SciPy satisfy the declared dependencies. The implementation is
portable Python, but that does not imply support for any particular production
inference engine or accelerator kernel.

See the [changelog](CHANGELOG.md) for user-visible changes. Maintainers follow
the repository's [protected release procedure](../../docs/guides/releasing.md)
and tag releases as `turboquant-reference-v<VERSION>`.

## License

Apache License 2.0. See `LICENSE` and `NOTICE` in this component directory.
