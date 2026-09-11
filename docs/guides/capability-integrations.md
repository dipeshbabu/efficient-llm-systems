# Capability integration boundary

Metria core owns geometry normalization, capability evidence, and preflight
decisions. Integrations own treatment names and domain-specific support rules.
The bundled TurboQuant KV rule lives in
`metria.integrations.turboquant`; its support boundary and override behavior are
unchanged. Core does not infer model-family properties from names.

Application code registers additional checks explicitly. There is no entry-point
discovery, recipe-directed import, or plugin marketplace. A check receives the
immutable run specification, one normalized geometry inspection shared by all
checks, and a boolean saying whether its experimental override was requested and
supported. It returns `CapabilityCheckResult`, or `None` when it does not apply.

For example, this application rule checks a requested context against a supplied
model limit; it does not independently query the runtime:

```python
from metria import (
    Capability, CapabilityCheck, CapabilityCheckRegistry, CapabilityCheckResult,
    RunSpec, SupportLevel, inspect_run_capabilities,
)

def context_budget(spec, inspection, experimental_override):
    requested = spec.scenario.get("context")
    limit = inspection.geometry.context_length if inspection.geometry else None
    known = (
        isinstance(requested, int) and not isinstance(requested, bool)
        and requested > 0 and limit is not None
    )
    status = (
        SupportLevel.UNKNOWN if not known else
        SupportLevel.SUPPORTED if requested <= limit else
        SupportLevel.UNSUPPORTED
    )
    return CapabilityCheckResult(Capability(
        "example.context_budget", status,
        evidence={"requested_context": requested, "supplied_limit": limit},
    ))

checks = CapabilityCheckRegistry().with_check(CapabilityCheck(
    "example.context_budget", context_budget, version="1",
))
run = RunSpec(
    model={"geometry": {"context_length": 4096}},
    runtime={"name": "example"}, scenario={"context": 2048}, measurements=(),
)
result = inspect_run_capabilities(run, capability_checks=checks)
assert not result.blocking
```

Pass the same optional `capability_checks` registry to `execute_run`,
`execute_study`, or the Python `verify_recipe` API. The executor evaluates checks
before probing or launching a runtime. Failures in check implementation or
contract validation become preflight failures; study/verifier registry errors
are rejected before execution or output creation.

The registry is immutable. Additional checks extend mandatory built-ins, and
duplicate/reserved names are rejected. Applications cannot replace or remove a
built-in through this API. Adding an application rule requires no core treatment
allowlist change. Bundled composition lives in the integrations package.

## Overrides and provenance

A required `unknown` or `unsupported` conclusion always blocks. A required
`experimental` conclusion needs both an explicit name in
`trial_policy.capability_overrides` and a check registered with
`supports_experimental_override=True`. A domain rule may return `experimental`
for an explicitly overridden case; core never upgrades its `unknown` or
`unsupported` result. Unknown override names remain blocking errors.

Checks with `required=False` contribute informational conclusions. Core records
check versions, applicability, and requested/supported/applied override state
alongside the capabilities, without overwriting integration-owned evidence.
Execution retains this under `record.provenance["capabilities"]["checks"]`.
The existing TurboQuant `experimental_override` evidence also remains intact.

The root import `metria.evaluate_turboquant_kv_capability` remains available for
0.1 API compatibility. New integration-specific imports should use
`metria.integrations.turboquant.evaluate_turboquant_kv_capability`; the evaluator
is no longer defined in the generic `metria.capabilities` module.
