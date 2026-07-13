# Historical engine forks

Some dated experiments in this repository used implementation forks and
prototype repositories that are not currently published under the project
account. Their repository, branch, commit, pull-request, and issue identifiers
are retained as provenance, but they are intentionally not linked to a URL
that does not exist.

Do not substitute an upstream checkout when reproducing a fork-specific
result. Upstream links below identify lineage only; they do not imply that the
experimental TurboQuant changes are present upstream.

## llama.cpp experimental forks

`llama-cpp-turboquant` and the experimental `llama.cpp` fork were based on
[upstream llama.cpp](https://github.com/ggml-org/llama.cpp). The historical
forks, branches, pull requests, issues, and commits cited in this repository
do not currently have a public source URL.

## MLX experimental fork

The TurboQuant MLX work was based on
[upstream MLX](https://github.com/ml-explore/mlx). The historical
`feature/turboquant-plus` fork and its cited commits do not currently have a
public source URL.

## vLLM experimental forks

The TurboQuant vLLM work was based on
[upstream vLLM](https://github.com/vllm-project/vllm). The historical `vllm`
and `vllm-turboquant` forks and their cited branches do not currently have a
public source URL.

## Swift and long-context prototypes

The historical `vllm-swift`, `mlx-swift-lm`, and `longctx` prototype
repositories do not currently have public source URLs. Their names and branch
identifiers remain in research reports solely to identify the measured code.

## Retired TurboQuant+ repository

The former `turboquant_plus` repository was consolidated into this monorepo.
Current material is available in the [documentation index](../index.md), the
[REFRACT component](../../components/refract/README.md), and the
[research archive](../../research/README.md).

## Archived identifiers

References such as `PR #45`, issues `#32`, `#47`, `#87`, `#88`, and `#89`, or
fork-specific commit hashes are historical identifiers. They remain useful for
interpreting dated reports, but no live endpoint is asserted for them here.

## Other unavailable citations

The third-party `signalnine/llama-cpp-turboquant-cuda` fork and its cited
`PR #24` do not currently have a public endpoint. The identifier is retained
only to preserve the provenance of the benchmark that cited it.
