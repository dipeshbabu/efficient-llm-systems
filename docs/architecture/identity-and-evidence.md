# Typed identity and evidence primitives

Metria's experiment model remains **Study → Run → Evidence**.

The typed primitives in `metria.identity` and `metria.identity_evidence` do not
introduce a second experiment object graph. They provide validated constructors
and reusable evidence objects at boundaries that were previously represented
only by unstructured mappings.

## Relationship to the study core

```text
StudySpec
  └── RunSpec
      ├── model      ← ModelRef or an equivalent mapping
      ├── runtime    ← RuntimeConfig or an equivalent mapping
      ├── scenario   ← WorkloadSpec or an equivalent mapping
      ├── treatments
      └── measurements

RunRecord
  ├── requested
  ├── resolved
  ├── observed
  │   ├── identity   ← RuntimeIdentityEvidence mapping
  │   └── hardware   ← may include HardwareFingerprint-shaped evidence
  ├── metrics
  ├── evidence
  └── artifacts      ← may include ArtifactManifest
```

`ModelRef`, `RuntimeConfig`, and `WorkloadSpec` implement Python's `Mapping`
interface. `RunSpec` therefore deep-freezes them through the same path used for
ordinary dictionaries. The retained `RunSpec` representation remains a mapping,
so existing runtime adapters do not need a parallel typed code path.

This also preserves `metria.study_recipe.v1`: a typed constructor and the
semantically equivalent dictionary produce the same recipe data and canonical
digest. The same RunSpec serializer/parser is reused by
`metria.run_record.v1`, so saved evidence does not introduce a second requested
configuration schema.

## Requested identity is not observed identity

A `ModelRef` records requested model and tokenizer identity. Fields such as
`revision` or `geometry` are not proof that a launched runtime actually used
those values.

The normal evidence sequence remains:

```text
requested → resolved → observed
```

Runtime adapters are responsible for resolving requested identity and collecting
whatever authoritative post-launch evidence their runtime can expose. Missing
observed evidence remains unknown rather than being inferred from the request.

## RuntimeIdentityEvidence

`RuntimeIdentityEvidence` is the stable mapping-compatible shape for normalized
identity under `RunRecord.observed["identity"]`. Its schema identifier is
`metria.runtime_identity.v1`.

The envelope carries model, tokenizer, runtime/build, chat-template, and applied
configuration components. Each non-empty component declares one authority
state:

- `verified`;
- `partial`;
- `unknown`;
- `mismatch`.

The top-level status is computed conservatively from the component states. A
caller cannot construct a `verified` aggregate while one retained component is
`unknown`, `partial`, or `mismatch`.

The object deep-freezes nested evidence and remains JSON/run-record compatible.
Diagnostic source labels and explanatory reason strings are retained for humans
but are not comparison dimensions. Semantic identity facts and their authority
states remain comparison relevant.

An inspectable chat template is represented by digest rather than raw text.
Optional endpoint identity accepts only a small allowlist of non-secret fields.
Authorization headers, tokens, API keys, and arbitrary authenticated URLs are
rejected at the identity boundary.

### First-party authority today

vLLM can expose enough live engine/tokenizer metadata to check model,
model-revision, tokenizer, tokenizer-revision, loaded runtime version, template
digest, and selected applied configuration fields before measurement. Concrete
mismatches abort the candidate before prompt execution. Missing metadata remains
partial or unknown.

llama.cpp verifies explicitly pinned local GGUF bytes and executable content.
The qualified capture provider also supplies native configuration readback.
The embedded tokenizer/template are not independently inspected, and requested
IDs are not copied into observed identity. The shared
[artifact resolver](../guides/artifact-resolution.md) provides bounded,
digest-verified downloads and extraction for external model/data inputs.

## ModelRef and geometry inspection

`ModelRef` requires at least a model identifier or a local path. It can retain:

- model id and revision;
- local path;
- tokenizer id and revision;
- model geometry supplied by an inspector or caller;
- additional metadata.

`ModelGeometry` now normalizes explicit geometry evidence and derives `head_dim`
only when `hidden_size / num_attention_heads` is exact and internally
consistent. Metria does not infer architecture facts from a model-family name.
Contradictory or missing evidence remains `unknown`.

The first enforced capability consumer is the documented TurboQuant KV-cache
head-dimension guardrail. Known unsupported/unknown active configurations fail
before runtime probing; explicit experimental overrides are retained as
requested study intent rather than being hidden command-line bypasses.

See [capability inspection](../guides/metria-inspection.md).

## RuntimeConfig and WorkloadSpec

`RuntimeConfig` requires a non-empty runtime name and optionally a version. Its
`config` fields are flattened into the existing runtime mapping, which preserves
adapter compatibility:

```python
RuntimeConfig(
    name="vllm",
    config={"dtype": "bfloat16", "kv_cache_dtype": "fp8"},
)
```

normalizes to the same requested mapping as:

```python
{
    "name": "vllm",
    "dtype": "bfloat16",
    "kv_cache_dtype": "fp8",
}
```

`WorkloadSpec` follows the same pattern for the `RunSpec.scenario` mapping.
Reserved identity keys cannot be redefined through extension configuration.

## Capability states

Capability discovery uses four conservative states:

- `supported` — the implemented rule has sufficient evidence to support the
  requested capability;
- `experimental` — the capability is intentionally available but outside the
  normal validated support envelope;
- `unsupported` — available evidence proves the request is not supported;
- `unknown` — evidence is missing, contradictory, or insufficient.

`unknown` is distinct from `unsupported`. An adapter should not turn missing
metadata into a confident incompatibility claim, and it should not turn a user
request into proof of support.

`SupportReport` uses the same `SupportLevel` vocabulary so runtime preflight and
`metria inspect` converge on one meaning. `IdentityStatus` is separate because
runtime identity also needs `partial` and `mismatch` states after launch.

## HardwareFingerprint

`HardwareFingerprint` is structured **observed evidence**, not a placement
request. It can retain platform, host, accelerator, software, and additional
metadata while remaining deeply immutable.

`capture_hardware_fingerprint()` provides a stdlib-only baseline containing
platform/software identity, CPU count when exposed, and a domain-separated
hostname correlation digest rather than the raw host name. That digest supports
record correlation; it is not a secrecy guarantee for guessable host names.
Accelerator identity remains runtime/adapter-observed until an authoritative
shared probe is implemented.

## ArtifactManifest

`ArtifactManifest` provides a shared identity/provenance shape for models,
datasets, reports, generated outputs, patches, containers, or other retained
artifacts. It can represent:

- artifact name and kind;
- URI and/or local path;
- immutable revision;
- SHA-256 digest;
- byte size;
- upstream source/license information;
- additional provenance metadata.

The type validates SHA-256 syntax and byte sizes but does not itself claim that
a hash was verified. The shared artifact resolver checks content before returning
resolved manifests, and KV Fidelity uses it for its default corpus cache.

## Versioned run evidence

`metria.run_record.v1` is the durable JSON boundary for one `RunRecord`. It
preserves requested, resolved, and observed state together with lifecycle status,
typed metric identity and raw samples, measurement evidence, artifacts, events,
and provenance.

Two digests serve different identity needs:

- `run_record_digest()` covers the complete versioned record, including local
  run identity and requested intent;
- `run_evidence_digest()` covers produced evidence while excluding `study_name`,
  `run_id`, and requested intent.

Neither digest implies that two records are valid to compare. Study-specific
`ComparisonPlan` semantics and metric method/version identity remain
authoritative. The `metria compare` CLI therefore requires an explicit study
recipe instead of treating a digest match as a comparison rule.

`RuntimeIdentityEvidence` is stored as an ordinary immutable mapping inside the
observed record, so it round-trips through the existing run-record schema and
participates in record/evidence digests without a second serializer.

See [run records and comparison](../guides/metria-run-records.md).

## What this layer still does not solve

Remaining follow-on work includes:

- authoritative accelerator inventory beyond runtime-observed evidence;
- exact runtime/model qualification on pinned real engines (#12);

New runtime, verification, measurement, and provenance work should reuse these
primitives and schemas instead of introducing incompatible identity paths.
