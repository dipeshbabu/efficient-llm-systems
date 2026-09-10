# Documentation

This directory contains current, maintained guidance for Metria. Dated
experiments and historical conclusions live under [`research/`](../research/README.md);
generated evidence lives under [`artifacts/`](../artifacts/README.md).

## Architecture

- [Metria core architecture](architecture/metria-core.md)

## Guides

- [Verify a local llama.cpp CPU thread change](guides/metria-verify.md)
- [Metria llama.cpp runtime adapter](guides/metria-llamacpp-runtime.md)
- [Metria vLLM runtime adapter](guides/metria-vllm-runtime.md)
- [Metria trajectory measurement bridge](guides/metria-trajectory-measurement.md)
- [Metria pairwise analyses](guides/metria-pairwise-analysis.md)
- [TurboQuant configuration recommendations](guides/turboquant-recommendations.md)
- [MLX port](guides/mlx-port.md)
- [Windows and AMD RDNA 4 setup](guides/windows-rdna4-setup.md)

## Reference

- [Benchmarks](reference/benchmarks.md)
- [Hardware comparison matrix](reference/hardware-comparison-matrix.md)
- [Test-suite definition](reference/test-suite-definition.md)
- [Weight-compression results](reference/weight-compression-results.md)

## Historical engine setup

- [Archived TurboQuant+ fork setup](../research/archive/turboquant-engine-setup.md)

This archived guide describes an experimental fork whose public source is
unavailable. For the published Metria CLI, use the local verification guide above.

## Components

- [KV Fidelity](../components/kv-fidelity/README.md)
- [TurboQuant reference](../components/turboquant-reference/README.md)

## Maintainers

- [Project governance](../GOVERNANCE.md)
- [Maintainers and component ownership](../MAINTAINERS.md)
- [Repository settings baseline](maintainers/repository-settings.md)
- [Publishing the root Metria package](guides/releasing-metria.md)
- [Independent component release procedures](guides/releasing.md)

When documents disagree, prefer current guidance backed by the newer
controlled experiment. Preserve older results as dated evidence rather than
rewriting them to match the latest recommendation.
