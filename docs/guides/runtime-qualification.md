# Metria runtime adapter qualification

Metria separates **semantic adapter conformance** from **real-engine/hardware
qualification**. Passing unit or mocked integration tests does not prove that a
runtime works on every upstream version, model architecture, kernel, or device.

## Qualification levels

| Level | What it proves | Where it runs |
|---|---|---|
| Shared semantic contract | Adapter obeys the Metria probe → capture-requirement negotiation → resolve → launch → infer → observe → reset → close lifecycle, keeps evidence privacy boundaries, resolves deterministically, and fails after close | PR-required Metria core tests |
| First-party identity contract | First-party adapters expose structurally valid `metria.runtime_identity.v1` evidence and do not promote unavailable facts to verified | PR-required Metria core tests |
| Engine-specific contract tests | Runtime-specific flags, token capture, launch arguments, redaction, cleanup, identity checks, and configured-vs-observed semantics behave as implemented | PR-required Metria core tests |
| External-boundary adapter conformance | The real Metria adapter class satisfies the shared semantic contract while only the external engine process/module is replaced by a deterministic fake | PR-required Metria core tests |
| Real-engine qualification | A pinned upstream runtime actually launches and produces expected observed identity/capture semantics | Manual or scheduled environment with the runtime installed |
| Hardware-qualified evidence | A real runtime/model executes on identified accelerator hardware and retains the model/runtime/hardware evidence needed to reproduce the qualification | Manual or scheduled hardware runner |

The first four levels are normal pull-request gates. The last two require an
explicit environment and should not be inferred from mocked tests.

## Measurement evidence negotiation

Before a runtime is resolved or launched, `execute_run()` evaluates the selected
measurement's `requirements()` and negotiates every requested capture against
runtime support evidence. The conclusion is retained under
`provenance.preflight.captures` together with the exact semantic capture list.

Only `supported` capture requirements proceed to launch. `unsupported`,
`unknown`, and `experimental` conclusions fail closed as `PREFLIGHT_FAILED`.
This is intentionally stricter than discovering a missing capture during the
measurement itself.

Adapters can expose a dedicated `probe_captures()` method while that capability
surface evolves. Existing first-party adapters also expose conservative
`<kind>_capture` markers in ordinary probe evidence. Metria translates only
known markers; an absent or unfamiliar marker is `unknown`, never implicitly
supported.

Capture options are currently reserved. A nonempty `CaptureRequest.options`
fails validation instead of being silently ignored by a runtime.

## Observed identity contract

First-party runtime observations expose `observed.identity` with schema
`metria.runtime_identity.v1`. Model, tokenizer, runtime, chat-template, and
applied-configuration components each carry their own authority status:

- `verified`: independently observed evidence matches the resolved expectation;
- `partial`: useful observed evidence exists but does not prove the complete identity;
- `unknown`: the runtime does not expose enough evidence to make the claim;
- `mismatch`: independently observed evidence contradicts the resolved expectation.

The aggregate status is conservative and cannot be stronger than its component
states. A concrete mismatch in a runtime that supports pre-measurement identity
inspection must stop the candidate before measurement rather than being treated
as a normal result.

Raw chat templates are not retained; an inspectable template is content hashed.
Future endpoint evidence is restricted to non-secret identity fields. Credential
material and authenticated request headers are never valid identity evidence.

## Current first-party adapters

### llama.cpp

PR-required coverage includes:

- shared runtime semantic contract exercised through `LlamaCppAdapter`;
- first-party `observed.identity` structure and authority states;
- binary/model existence and executable content hashes during resolution;
- command construction and managed-flag rules;
- timeout/non-zero exit behavior;
- prompt/system redaction in retained invocation evidence;
- KV-cache runtime-feature handling;
- fail-closed token-ID capture negotiation before launch;
- reset/close semantics.

Finding `llama-completion` is not itself a support claim. The existing probe
reports `binary_present_unverified`. For `token_ids` verification, that remains
`unknown` unless the run environment supplies
`llama_cpp_token_ids_capture_sha256` and the actual completion binary matches
that digest exactly. A missing provider or digest mismatch is `unsupported`.

The resolved llama.cpp executable content hash is authoritative binary identity.
The local GGUF model is still only partial identity because the current adapter
does not hash the potentially large model file under #15. Requested model IDs,
revisions, and digests remain claims rather than observed facts. The embedded
tokenizer and chat template remain unknown. Recorded command invocations provide
partial applied evidence but do not prove runtime-internal readback.

Immutable model content identity belongs to #16. Pinned real-engine and hardware
qualification remains tracked by #12.

### vLLM

PR-required coverage includes:

- shared runtime semantic contract exercised through `VLLMAdapter`;
- first-party `observed.identity` structure and authority states;
- fail-closed requested-vs-installed runtime version checking;
- explicit tokenizer ID/revision routing;
- model, model-revision, tokenizer, tokenizer-revision, and loaded-runtime identity checks before measurement;
- active chat-template hashing when inspectable;
- selected applied-configuration checks against live engine introspection;
- lazy optional dependency behavior;
- constructor/runtime configuration;
- native output-token-ID capture and pre-launch capture negotiation;
- prompt/system redaction;
- reset/close semantics.

When the ordinary vLLM probe is supported, its
`token_ids_capture=native_output_token_ids` evidence satisfies the trajectory
capture requirement without launching the engine merely to discover support.

A concrete observed model/tokenizer/runtime/applied-config mismatch aborts the
launch before a measurement can execute. Missing tokenizer revision, missing
chat-template metadata, or incomplete applied introspection remains partial or
unknown instead of being filled from requested configuration.

Real-engine qualification still needs a pinned vLLM environment, immutable model
identity, and accelerator evidence. The default CI suite does not make that
claim.

## Evidence required for a real qualification

A real-engine qualification should retain at least:

```text
Metria version / commit
runtime name + upstream version/commit
model identifier + immutable revision/digest
tokenizer identifier + immutable revision/digest
chat-template digest or explicit unknown status
requested runtime configuration
resolved configuration
observed.identity authority envelope
observed/applied runtime evidence
hardware fingerprint + accelerator identity
driver/runtime software versions
measurement/capture method + version
capture requirement + preflight support conclusion
run status and lifecycle events
```

Where an engine cannot expose authoritative applied state, the qualification
must say `partial`/`unknown` rather than copying requested values into the
observed record.

## CI policy

- Mocked/fixture conformance remains required for every PR because it is
  deterministic and cross-platform.
- Real-engine/hardware qualification is **not** a required GitHub-hosted PR gate
  until Metria has a controlled runner, immutable model artifacts, and pinned
  upstream runtime inputs.
- When a scheduled/manual qualification lane is added, its artifacts should be
  versioned Metria run records/manifests rather than ad-hoc console logs.
- A failed or stale real-engine qualification should downgrade published support
  claims; it must not be hidden by passing mocked tests.

## Future runtimes

MLX, SGLang, TensorRT-LLM, or other engines should not be advertised as
first-party Metria runtime support merely because a focused component has a
backend for them. A new Metria runtime should first implement the common adapter
protocol, provide the identity evidence it can actually observe, and pass the
same semantic/conformance levels described here.
