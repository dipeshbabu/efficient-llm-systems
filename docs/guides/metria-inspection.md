# Metria capability inspection

Metria can inspect a versioned study recipe before execution and report the
capability evidence that can be determined without launching an inference
runtime.

```bash
metria inspect study.json
metria inspect study.json --json
```

The inspection is intentionally conservative. It does not infer model geometry
from a model-family name and does not treat a requested runtime feature as proof
that the feature will be applied.

## Geometry evidence

A recipe can provide authoritative or externally resolved model geometry under
`model.geometry`:

```json
{
  "model": {
    "id": "example/model",
    "revision": "abc123",
    "geometry": {
      "hidden_size": 4096,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "num_hidden_layers": 32,
      "max_position_embeddings": 8192
    }
  }
}
```

Metria normalizes common metadata field names and derives `head_dim` only when
`hidden_size / num_attention_heads` is exact and internally consistent. If an
explicit `head_dim` contradicts that arithmetic, the geometry conclusion is
`unknown` rather than silently choosing one value.

The current geometry object can retain:

- hidden size;
- attention-head count;
- KV-head count;
- head dimension;
- layer count;
- context limit;
- an explicit attention-layout label.

The evidence source remains recorded as model metadata. A successful geometry
inspection means the supplied fields are internally consistent; it is not a
claim that Metria independently queried the upstream model repository.

## TurboQuant KV-cache guardrail

This domain rule is supplied by `metria.integrations.turboquant`. The generic
core invokes registered checks and applies preflight/override policy; see the
[capability integration contract](capability-integrations.md) for adding an
application check without changing core support allowlists.

The first enforced geometry rule covers the documented TurboQuant KV-cache
head-dimension boundary.

For a KV treatment whose key or value dtype starts with `turbo`, ignoring case:

| Geometry evidence | Capability result |
|---|---|
| `head_dim <= 64` | `unsupported` |
| `head_dim` 128 or 256 | `supported` |
| another consistent `head_dim > 64` | `experimental` |
| missing or contradictory `head_dim` | `unknown` |

Non-Turbo KV configurations do not require this rule and remain supported by
the geometry guardrail.

`execute_run()` evaluates the shared guardrail before invoking the runtime
adapter. `unsupported` and `unknown` active TurboQuant conclusions fail
preflight. An `experimental` conclusion also fails unless the recipe records an
explicit capability override.

## Explicit experimental override

Overrides are part of requested study intent, not a command-line bypass. Record
them in the run's trial policy:

```json
{
  "trial_policy": {
    "capability_overrides": [
      "turboquant.kv_cache.geometry"
    ]
  }
}
```

For the known `head_dim=64` boundary, an explicit override changes the
capability conclusion from `unsupported` to `experimental` and allows preflight
to continue. The override remains present in the retained capability evidence.

Missing or contradictory geometry remains `unknown` even when an override name
is present; Metria does not turn absent evidence into a supported claim.

Override names are validated for every run, even when no matching treatment is
active. Unrecognized names are retained as `metria.capability_overrides`
evidence and block preflight so a typo cannot silently disable a guardrail.

## Hardware fingerprint

`metria inspect` also captures a portable stdlib-only hardware/software
fingerprint. The initial fingerprint includes:

- operating-system and machine architecture;
- processor string when the platform exposes it;
- CPU count when the platform exposes it;
- a domain-separated SHA-256 correlation identifier derived from the host name;
- Python version and implementation.

The hostname digest supports correlation between run records without storing the
raw host name. It is **not** a confidentiality guarantee: an attacker who can
guess likely host names may still recover a matching name by enumeration.

Accelerator identity is deliberately left to runtime/adapter evidence until an
authoritative accelerator probe is implemented. Environment variables such as
`CUDA_VISIBLE_DEVICES` are not treated as proof that a specific GPU is present.

## Evidence boundary

`capability.status = supported` means the implemented rule is satisfied by the
available evidence. It is not deployment certification, a behavioral-quality
guarantee, or a statement that every upstream runtime/kernel combination is
safe.

Runtime launch still has its own `RuntimeAdapter.probe()` and
requested → resolved → observed evidence. Capability inspection adds an earlier,
shared fail-closed layer; it does not replace runtime observation.
