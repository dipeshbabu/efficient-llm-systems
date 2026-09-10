# Metria vLLM runtime adapter

Metria's vLLM adapter provides an optional, in-process offline inference runtime
for the shared `RuntimeAdapter` / `RuntimeSession` contracts.

The adapter is intentionally instance-scoped. It does not reuse KV Fidelity's
module-global vLLM model cache: each `VLLMSession` owns the `vllm.LLM` instance
created for one resolved run configuration.

## Optional dependency

vLLM is not a root Metria dependency and is imported only when a vLLM session
is launched. Keep vLLM in a runtime-specific environment rather than forcing it
into environments used for llama.cpp, MLX, or other engines.

`probe()` reports the run unsupported when the optional `vllm` distribution is
not importable. A requested `runtime.version` is also fail-closed: if it differs
from the installed vLLM distribution version, the run fails preflight rather
than silently using another runtime build.

## Run specification

A run may pin both runtime and tokenizer identity:

```python
from metria import RunSpec, TreatmentSpec, TreatmentType

spec = RunSpec(
    model={
        "id": "org/model",
        "revision": "model-commit",
        "tokenizer_id": "org/tokenizer",
        "tokenizer_revision": "tokenizer-commit",
    },
    runtime={
        "name": "vllm",
        "version": "installed-version",
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 32,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": False,
    },
    scenario={"context": 4096, "max_tokens": 128},
    measurements=("kv_fidelity.decode_time_trajectory",),
    treatments=(
        TreatmentSpec(
            name="vllm.kv_cache",
            kind=TreatmentType.RUNTIME_FEATURE,
            config={"dtype": "auto"},
        ),
    ),
)
```

When `tokenizer_id` is explicit, Metria passes it to vLLM instead of assuming
the model tokenizer. A tokenizer revision defaults to the model revision only
when the tokenizer itself is implicit. An explicitly different tokenizer with
no tokenizer revision does not inherit an unrelated model revision.

The adapter deliberately owns a small runtime surface. Unknown runtime or
generation fields fail rather than being silently ignored.

## KV-cache treatment

The vLLM adapter uses a vLLM-native KV-cache treatment:

```text
vllm.kv_cache
  dtype = auto | fp8 | fp8_e4m3 | fp8_e5m2
```

Metria does not translate llama.cpp-specific `q8_0`, `q4_0`, or asymmetric K/V
formats into vLLM formats and call them equivalent. A cross-runtime study must
state the actual treatment each runtime received.

## Generation and capture

The initial semantic generation surface supports:

- `max_tokens`
- `seed`
- `temperature`
- `chat_template`
- `system`

The session batches all requests into one `LLM.generate()` call and can return
native output token IDs through:

```python
CaptureRequest(kind="token_ids")
```

`TokenTrajectoryProtocol.requirements()` declares this capture before runtime
launch. Once the ordinary vLLM probe is supported, its
`token_ids_capture=native_output_token_ids` evidence satisfies that requirement,
and `execute_run()` retains the conclusion under
`provenance.preflight.captures` before resolving or constructing `vllm.LLM`.

An unrecognized capture kind is never inferred to be available. It remains
`unknown` during preflight and therefore blocks verification before launch.
Capture options are currently reserved and rejected rather than silently
ignored.

Raw prompt and system text are not retained in invocation evidence. Metria
stores SHA-256 fingerprints for the original prompt, rendered prompt, and system
message along with non-sensitive generation settings and output token counts.

## Identity verification before measurement

Immediately after `vllm.LLM` is constructed and before a measurement receives
the session, Metria inspects the live engine and tokenizer. The normalized
result is retained under:

```text
observed.identity
  schema: metria.runtime_identity.v1
  status: verified | partial | unknown | mismatch
  model: ...
  tokenizer: ...
  runtime: ...
  chat_template: ...
  applied: ...
```

The adapter checks independently observable model identifier/revision,
tokenizer identifier/revision, loaded runtime version, and selected applied
configuration fields. A concrete mismatch aborts launch before any measurement
prompt is executed. Missing metadata is retained as partial or unknown rather
than copied from the request.

The active tokenizer chat template is represented by SHA-256 when inspectable;
raw template text is not stored in identity evidence. The loaded module version
and installed distribution metadata are both retained. If both are available
and disagree, identity verification fails.

Applied configuration remains intentionally conservative. Metria checks fields
that the engine exposes and that have stable resolved counterparts, including
maximum model length, KV-cache dtype, GPU memory utilization, prefix caching,
and tensor-parallel size. A detected mismatch fails before measurement. Even
when those fields agree, the applied component remains `partial` because vLLM
does not expose authoritative readback for every constructor/runtime choice.

## Requested, resolved, and observed state

The adapter keeps configured and applied state separate.

`resolve()` records:

- model identifier/path and requested revisions;
- resolved tokenizer identifier/revision;
- installed and requested vLLM versions;
- exact runtime settings passed to `LLM`;
- native KV-cache dtype;
- resolved maximum model length.

After launch, the session retains both the existing introspection view and the
normalized identity envelope. If a field cannot be recovered from the live
engine, Metria does not copy the configured value into observed identity and
pretend it was verified.

## Cleanup

The adapter releases its session-owned vLLM and tokenizer references at close.
It calls a public `shutdown()` only if the installed runtime exposes one. When
identity verification fails during launch, Metria also performs best-effort
cleanup before propagating the failure so a rejected candidate does not leave a
known engine instance behind.

The cleanup record distinguishes explicit shutdown from ordinary Python
reference release; Metria does not claim that reference release proves complete
device allocator teardown.

## Cross-runtime trajectory studies

With this adapter and a capture-qualified llama.cpp adapter, the same trajectory
measurement can be executed independently on two different runtimes:

```text
RunSpec -> capture preflight -> identity verification -> vLLM -> measurement
RunSpec -> capture preflight -> llama.cpp -> measurement -> observed identity
```

Runtime execution differs, while the measurement semantics and pairwise
comparison method stay fixed.

## Current limitations

The initial vLLM adapter does not yet provide:

- online/server-mode load generation;
- log-probability or KLD capture;
- concurrency/throughput benchmarking;
- runtime-level retries or worker restart policy;
- automatic treatment equivalence across engines;
- a universal dependency environment.

Those should be added as explicit protocols/capabilities rather than folded into
the basic offline trajectory path.
