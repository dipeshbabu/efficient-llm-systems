# TurboQuant Reference

`turboquant-reference` is the NumPy/SciPy reference implementation of
TurboQuant KV-cache compression in the Efficient LLM Systems repository. The
distribution name is `turboquant-reference`; the stable Python import remains
`turboquant`.

This component is intended for algorithm inspection, reproducibility, and
experimentation. Production inference integrations live in their respective
engine projects.

## Install

From the repository root:

```bash
python -m pip install -e "./components/turboquant-reference[dev]"
```

Install the optional real-model benchmark stack with:

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

## License

Apache License 2.0. See `LICENSE` and `NOTICE` in this component directory.
