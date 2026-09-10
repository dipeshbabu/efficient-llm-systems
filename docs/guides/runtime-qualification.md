# Metria runtime adapter qualification

Metria separates **semantic adapter conformance** from **real-engine/hardware
qualification**. Passing unit or mocked integration tests does not prove that a
runtime works on every upstream version, model architecture, kernel, or device.

## Qualification levels

| Level | What it proves | Where it runs |
|---|---|---|
| Shared semantic contract | Adapter obeys the Metria probe → capture-requirement negotiation → resolve → launch → infer → observe → reset → close lifecycle, keeps evidence privacy boundaries, resolves deterministically, and fails after close | PR-required Metria core tests |
| Engine-specific contract tests | Runtime-specific flags, token capture, launch arguments, redaction, cleanup, and configured-vs-observed semantics behave as implemented | PR-required Metria core tests |
| External-boundary adapter conformance | The real Metria adapter class satisfies the shared semantic contract while only the external engine process/module is replaced by a deterministic fake | PR-required Metria core tests |
| Real-engine qualification | A pinned upstream runtime actually launches and produces expected observed identity/capture semantics | Manual or scheduled environment with the runtime installed |
| Hardware-qualified evidence | A real runtime/model executes on identified accelerator hardware and retains the model/runtime/hardware evidence needed to reproduce the qualification | Manual or scheduled hardware runner |

The first three levels are normal pull-request gates. The last two require an
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

## Current first-party adapters

### llama.cpp

PR-required coverage includes:

- shared runtime semantic contract exercised through `LlamaCppAdapter`;
- binary/model existence and content hashes during resolution;
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

This SHA qualification bridge proves only the identity of the instrumented
binary selected for the run. It does not replace the pinned real-engine and
hardware qualification tracked in #12.

### vLLM

PR-required coverage includes:

- shared runtime semantic contract exercised through `VLLMAdapter`;
- lazy optional dependency behavior;
- constructor/runtime configuration;
- native output-token-ID capture and pre-launch capture negotiation;
- prompt/system redaction;
- configured-vs-introspected applied state;
- reset/close semantics.

When the ordinary vLLM probe is supported, its
`token_ids_capture=native_output_token_ids` evidence satisfies the trajectory
capture requirement without launching the engine merely to discover support.

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
requested runtime configuration
resolved configuration
observed/applied runtime evidence
hardware fingerprint + accelerator identity
driver/runtime software versions
measurement/capture method + version
capture requirement + preflight support conclusion
run status and lifecycle events
```

Where an engine cannot expose authoritative applied state, the qualification
must say `unknown`/`unverified` rather than copying requested values into the
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
protocol and pass the same semantic/conformance levels described here.
