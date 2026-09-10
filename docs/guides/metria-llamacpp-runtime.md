# Metria llama.cpp runtime adapter

The first Metria runtime adapter drives local llama.cpp command-line binaries
without using KV Fidelity's module-global backend dispatch.

It is intentionally narrow. The current adapter proves the runtime lifecycle and
evidence model before Metria adds more engines.

## What it supports

- local `llama-cli` text generation;
- local GGUF model paths;
- explicit `n_gpu_layers`, flash-attention, and expert `extra_args`;
- one runtime treatment: `llamacpp.kv_cache`;
- requested → resolved → observed runtime evidence;
- content hashing for llama.cpp binaries;
- normalized `observed.identity` with explicit authority states;
- exact redacted command records for invocations;
- optional decode-time `token_ids` capture through a qualified patched
  `llama-completion` binary;
- timeout and non-zero-exit failure reporting;
- instance-local sessions with no Metria global backend selector.

It does **not** yet provide server mode, batching/concurrency load generation,
per-token log probabilities, perplexity/KLD measurement, remote model download,
or a stable public CLI.

## Run specification

```python
from metria import RunSpec, TreatmentSpec, TreatmentType
from metria.protocols import InferenceRequest
from metria.runtimes import LlamaCppAdapter

spec = RunSpec(
    model={"path": "/models/model.gguf", "id": "org/model"},
    runtime={
        "name": "llamacpp",
        "bin_dir": "/opt/llama.cpp/build/bin",
        "n_gpu_layers": 99,
        "flash_attention": True,
    },
    scenario={"context": 8192, "max_tokens": 128},
    measurements=("text",),
    treatments=(
        TreatmentSpec(
            name="llamacpp.kv_cache",
            kind=TreatmentType.RUNTIME_FEATURE,
            config={"key_dtype": "q8_0", "value_dtype": "q4_0"},
        ),
    ),
)

adapter = LlamaCppAdapter()
environment = {"hardware_class": "local-gpu"}

support = adapter.probe(spec, environment)
if support.status != "supported":
    raise RuntimeError(support.reasons)

resolved = adapter.resolve(spec, environment)
session = adapter.launch(resolved, environment)
try:
    batch = session.infer((InferenceRequest(prompt="Explain KV caches briefly."),))
    observed = adapter.observe(session)
finally:
    session.close()
```

## KV-cache treatment

The adapter currently accepts one `runtime_feature` treatment named either
`llamacpp.kv_cache` or `kv_cache`.

Supported fields are:

| Field | Meaning |
|---|---|
| `key_dtype` | llama.cpp `-ctk` value; default `f16` |
| `value_dtype` | llama.cpp `-ctv` value; default `f16` |
| `attention_rotation_k` | patched `LLAMA_ATTN_ROT_K_OVERRIDE` value |
| `attention_rotation_v` | patched `LLAMA_ATTN_ROT_V_OVERRIDE` value |
| `attention_rotation_disable` | patched `LLAMA_ATTN_ROT_DISABLE` value |

Unknown treatments and unknown fields fail explicitly. Metria does not silently
ignore a treatment it cannot prove it applied.

## Token-ID capture

`CaptureRequest(kind="token_ids")` selects `llama-completion` and uses the
existing `KV_FIDELITY_TRAJECTORY` patch ABI. Merely finding that binary is
recorded as `binary_present_unverified` and is **not sufficient** for Metria's
verification path.

Before `execute_run()` resolves or launches llama.cpp, the trajectory
measurement declares its `token_ids` requirement. Capture preflight then has
three conservative outcomes:

- no `llama-completion` provider: `unsupported`;
- provider present but not qualified: `unknown`;
- provider present and its content SHA-256 exactly matches
  `environment["llama_cpp_token_ids_capture_sha256"]`: `supported`.

A malformed qualification digest or a digest mismatch is `unsupported`. Any
status other than `supported` becomes a `PREFLIGHT_FAILED` run, so an
unqualified trajectory experiment cannot consume model/GPU work and fail only
after launch.

For example, a controlled qualification runner can pass:

```python
environment = {
    "hardware_class": "h100",
    "llama_cpp_token_ids_capture_sha256": "<64-hex-qualified-binary-digest>",
}
```

The selected provider is hashed again at preflight. The expected and observed
digests are retained in `provenance.preflight.captures`.

This only qualifies the capture provider's binary identity. It does not claim
that every model, llama.cpp revision, or hardware target has been validated.
The pinned real-engine/hardware evidence remains tracked by the runtime
qualification work in #12.

## Observed identity and authority

`observe()` now exposes the same `metria.runtime_identity.v1` envelope used by
other first-party runtimes:

```text
observed.identity
  status: partial
  runtime:
    status: verified
    cli_sha256: ...
    completion_sha256: ...
  model:
    status: partial
  tokenizer:
    status: unknown
  chat_template:
    status: unknown
  applied:
    status: unknown | partial
```

The runtime executable identity is based on the content hashes retained during
resolution. Metria does not infer a runtime version string from a filename or
path.

Supply `model.sha256` from a trusted artifact manifest to bind a local GGUF file
to expected content. Resolution streams and verifies that digest, and launch
checks the content again. A mismatch rejects the run. Pinned sessions also
reject size or modification-time changes before each invocation.

A matching digest is retained in observed model identity with status `verified`
and source `verified_local_file_sha256`. Without a pin, file metadata remains
`partial`. Requested model IDs and revisions are never copied into observation.
Keep pinned model files immutable throughout execution; this is local content
verification, not a guarantee against concurrent filesystem tampering. Download
resolution, split model artifacts, and broader provenance remain tracked by #16.

Likewise, the current llama.cpp CLI path does not independently expose the
embedded tokenizer or active chat-template identity, so both stay `unknown`.
After an invocation, the applied identity component becomes `partial` because
Metria can prove the exact command it issued but cannot claim that command-line
intent is authoritative readback of llama.cpp internal state.

This asymmetry is intentional. `partial` and `unknown` are useful evidence; they
are safer than presenting requested values as facts.

## Evidence and privacy

The resolved runtime record includes the content hash of each llama.cpp binary
and file metadata for the model. Model content is hashed when `model.sha256` is
provided. Broader model-artifact resolution remains tracked by #16.

For each invocation, Metria records the actual managed command flags and managed
environment overrides. Prompt and system-message contents are replaced with
`<redacted>` in command evidence and represented by SHA-256 fingerprints.

`extra_args` is available for expert llama.cpp options, but cannot override
flags that Metria manages directly. This prevents the recorded resolved state
from disagreeing with the effective command line.
