# Metria core architecture

Metria is a neutral evidence and experiment layer for LLM inference systems.
Its job is not to reimplement serving engines, quantizers, kernels, or schedulers.
Instead, Metria defines how an inference-systems study is specified, executed,
recorded, and compared.

The initial design deliberately keeps the public model small:

```text
StudySpec
  ├── runs: RunSpec[]
  └── comparison: ComparisonPlan

RunSpec
  ├── model
  ├── runtime
  ├── treatments
  ├── scenario
  ├── measurements
  ├── trial policy
  └── environment selector

MeasurementResult
  ├── metrics
  ├── evidence
  └── artifacts

RunRecord
  ├── requested
  ├── resolved
  ├── observed
  ├── status
  ├── metrics
  ├── evidence
  ├── events
  ├── artifacts
  └── provenance

StudyExecutionResult
  ├── records: RunRecord[]
  └── comparisons: StudyPairComparison[]
```

## Study semantics

The study, rather than a hard-coded fingerprint, decides what may vary.
A comparison plan separates:

- `vary`: dimensions intentionally changed by the study;
- `control`: dimensions that must match for a valid comparison;
- `block_by`: dimensions used to form comparable groups.

For example, a study may compare two runtimes and two KV-cache treatments while
controlling the model, scenario, and trial policy and blocking by hardware
class. Runtime therefore is not globally "compatible" or "incompatible"; its
role is study-specific.

## Requested, resolved, observed

Metria keeps three states separate:

1. **Requested**: what the recipe asked for.
2. **Resolved**: exact revisions, runtime settings, artifacts, and choices
   selected before execution.
3. **Observed**: what the runtime and environment report actually ran.

This distinction is required for trustworthy systems evidence. A request for
FP8 KV cache, for example, is not evidence that the launched engine used FP8.
Runtime adapters must provide independent applied-configuration evidence when
the runtime exposes it and must retain unknown state when it does not.

## Observed identity authority

First-party runtimes normalize identity evidence under `observed.identity` using
`metria.runtime_identity.v1`. The envelope separates five authority-bearing
components:

- model;
- tokenizer;
- runtime/build;
- chat template;
- applied configuration.

Each component declares `verified`, `partial`, `unknown`, or `mismatch`. The
overall identity status is derived conservatively from those component states;
an adapter cannot label the whole identity verified while retaining an unknown
component.

Observed identity contains facts learned independently from the launched runtime
or resolved local executable. Requested model IDs, revisions, digests, or runtime
features are not copied into this section and presented as observation. A
concrete vLLM model, tokenizer, revision, runtime-version, or introspected applied
configuration mismatch aborts launch before measurement. Missing upstream
metadata stays partial or unknown.

Chat templates are represented by digest when inspectable rather than by raw
template text. Future endpoint identity is constrained to non-secret fields;
credentials, authorization headers, API keys, and raw authenticated URLs do not
belong in durable identity evidence.

llama.cpp currently has asymmetric authority: Metria can content-identify the
resolved executable, but the GGUF model, embedded tokenizer, chat template, and
runtime-applied internal settings are not all independently observable. Those
components therefore remain partial or unknown until immutable artifact and
real-engine qualification work provides stronger evidence.

Comparison uses the semantic identity facts and component authority states.
Diagnostic source labels and explanatory reason strings are not comparison
dimensions. A comparison plan's `model` role governs observed model, tokenizer,
and chat-template identity; its `runtime` role governs runtime/build, endpoint,
and applied-runtime identity. A difference in overall identity authority remains
fail-closed unless explicitly accounted for.

## Treatments

`TreatmentSpec` is intentionally broader than "optimization". The initial
taxonomy is:

- model transformation;
- runtime feature;
- execution policy;
- instrumentation.

The common study model treats these as experimental treatments while leaving
their different lifecycle mechanics to specialized adapters.

## Metrics

A metric identity includes its name, unit, optimization direction, method, and
method version. Raw values are directly comparable only when those identities
match and the study comparison plan permits the run comparison.

This prevents methodologically different measurements, such as full-vocabulary
KL divergence and top-k KL estimates, from being silently treated as the same
metric. Cross-method studies can still compare matched-baseline effect sizes,
but that analysis must be explicit.

## Measurement results and evidence

`MeasurementProtocol` returns a `MeasurementResult`, not only a metric mapping.
The result separates compact numerical summaries from the observations needed
to reproduce or derive them:

- `metrics` contains typed `MetricSummary` values;
- `evidence` contains immutable method-specific observations;
- `artifacts` points to large external payloads when embedding them would make a
  run record impractical.

When a measurement completes, its run-local evidence is retained under
`RunRecord.evidence`. `RunRecord.provenance` is reserved for how the run was
executed and resolved rather than being used as a catch-all for method output.

This distinction is important for fidelity methods. A decode-time token
trajectory, for example, is evidence produced by one run. Its agreement score
is not a property of that run by itself; the score exists only after comparing a
reference trajectory with a candidate trajectory. Metria therefore retains the
run-local trajectories first and derives the pairwise score at comparison time.

Pairwise derived metrics must verify that the underlying evidence refers to the
same experimental object. The trajectory bridge checks prompt identifiers,
prompt fingerprints, capture schema, and method version before computing a
score. This avoids silently comparing two different prompt sets that happen to
share a benchmark label.

## Runtime and measurement boundaries

The provisional protocols separate runtime lifecycle from measurement
methodology:

- `RuntimeAdapter` is explicitly named, probes support, resolves configuration,
  launches a session, and records observed runtime evidence.
- `CaptureSupportProbe` is an optional runtime contract for adapters that can
  negotiate measurement capture requirements directly before launch.
- `RuntimeSession` performs inference and owns reset/cleanup behavior.
- `MeasurementProtocol` is explicitly named/versioned, declares evidence
  requirements, and returns a `MeasurementResult` containing metrics and
  retained evidence.

When an adapter does not implement `CaptureSupportProbe`, Metria may consume
recognized conservative capture-support markers from its ordinary runtime probe.
Missing or unfamiliar capture evidence is `unknown`, never implicitly supported.
Only an explicit `supported` capture conclusion permits launch.

These protocols are intentionally provisional until exercised by at least two
materially different runtimes.

## Run execution lifecycle

`execute_run()` is the first orchestration boundary. It intentionally executes
one runtime and one measurement protocol at a time:

```text
RunSpec
  -> measurement requirements
  -> shared capability inspection
  -> runtime probe
  -> capture support negotiation
  -> resolve
  -> launch + identity verification
  -> measure
  -> observe
  -> close
  -> RunRecord
```

Measurement requirements are validated before runtime work begins. The exact
required capture set and its support conclusion are retained under
`RunRecord.provenance.preflight.captures`. Unsupported, unknown, or experimental
required captures fail before resolve or launch rather than consuming runtime
resources and discovering the incompatibility during measurement.

A runtime that can independently inspect identity immediately after construction
may reject a mismatch during launch, before the measurement receives a session.
Identity that can only be learned later remains observed evidence and is not
silently promoted to verified.

Failed runs are still evidence. The executor uses the lifecycle status to avoid
turning missing data into apparent success:

- unsupported requests, invalid measurement requirements, capture negotiation
  failures, and other failures before launch become `PREFLIGHT_FAILED`;
- launch or measurement failures become `FAILED`;
- a measurement-level `TimeoutError` becomes `TIMED_OUT`;
- completed metrics with missing observed runtime evidence or failed cleanup
  become `PARTIAL`;
- only completed measurement, observation, and cleanup yield `COMPLETED`.

The executor attempts to observe a launched runtime even after measurement
failure, and it always attempts to close a successfully launched session.
Lifecycle exception text is not embedded verbatim because third-party runtimes
may include prompts or other sensitive input in errors. The record retains the
exception type and a SHA-256 fingerprint of the message instead.

## Study execution lifecycle

`execute_study()` routes every `RunSpec` through registered runtime adapters and
measurement protocols, then evaluates the study's pairwise comparison semantics:

```text
StudySpec
  -> validate all registry routes
  -> execute run-0000
  -> execute run-0001
  -> ...
  -> compare every run pair with ComparisonPlan
  -> StudyExecutionResult
```

Registry mistakes are configuration errors, not experimental outcomes. The
study executor therefore validates every requested runtime and measurement
before the first run starts. Once execution begins, however, runtime or
measurement failures are preserved as `RunRecord` values and do not prevent
later runs from executing.

Direct pairwise comparison is lifecycle-aware. `compare_runs()` requires both
records to be `COMPLETED`; failed, timed-out, or partial records remain useful
evidence but are not analysis-ready by default. Controlled and blocking
requirements are still checked and reported alongside the lifecycle issue so a
study can diagnose more than one incompatibility at once.

The first study executor deliberately supports exactly one measurement per run
and one shared environment mapping. Multi-measurement scheduling, heterogeneous
host placement, retries, persistence, parallel execution, and CLI recipes remain
separate later concerns rather than hidden behavior in the initial contract.

## What is not in the first core

The first Metria core does not provide:

- automated configuration recommendation;
- active search;
- a universal optimization plugin;
- a universal fidelity scalar;
- production serving;
- production TurboQuant integration;
- a root all-runtime dependency bundle.

The first stable milestone is narrower:

> Metria can reproducibly run, record, and validly compare a defined
> inference-systems study across at least two runtimes.

KV Fidelity and TurboQuant Reference remain independent focused components while
Metria's shared study and evidence model matures.
