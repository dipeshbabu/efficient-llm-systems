"""Generic capability inspection and fail-closed preflight policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._freeze import freeze_mapping
from .capabilities import inspect_model_geometry
from .capability_checks import CapabilityCheckRegistry, CapabilityCheckResult
from .identity import Capability, CapabilitySet, SupportLevel
from .models import RunSpec


@dataclass(frozen=True)
class PreflightCapabilityResult:
    """Capability conclusions plus any rules that must block execution."""

    capabilities: CapabilitySet
    blocking: tuple[Capability, ...] = ()
    checks: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocking", tuple(self.blocking))
        object.__setattr__(self, "checks", freeze_mapping(self.checks))


def _override_names(spec: RunSpec) -> frozenset[str]:
    raw = spec.trial_policy.get("capability_overrides", ())
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise TypeError("trial_policy.capability_overrides must be a sequence of names")
    names = tuple(raw)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("capability override names must be non-empty strings")
    return frozenset(names)


def resolve_capability_checks(
    additional: CapabilityCheckRegistry | None = None,
) -> CapabilityCheckRegistry:
    """Combine explicit application checks with mandatory built-in checks."""
    from .integrations import builtin_capability_checks

    if additional is not None and not isinstance(additional, CapabilityCheckRegistry):
        raise TypeError("capability_checks must be a CapabilityCheckRegistry")
    builtins = builtin_capability_checks()
    return CapabilityCheckRegistry(
        (*builtins.checks, *(additional.checks if additional else ()))
    )


def _unknown_override_capability(
    overrides: frozenset[str], recognized: frozenset[str]
) -> Capability | None:
    unknown = tuple(sorted(overrides - recognized))
    if not unknown:
        return None
    return Capability(
        name="metria.capability_overrides",
        status=SupportLevel.UNKNOWN,
        reasons=("unrecognized capability override names: " + ", ".join(unknown),),
        evidence={
            "requested": tuple(sorted(overrides)),
            "recognized": tuple(sorted(overrides & recognized)),
            "unrecognized": unknown,
        },
    )


def inspect_run_capabilities(
    spec: RunSpec, *, capability_checks: CapabilityCheckRegistry | None = None
) -> PreflightCapabilityResult:
    """Invoke domain checks and apply generic preflight/override semantics.

    Required UNKNOWN/UNSUPPORTED results always block. Required EXPERIMENTAL
    results need an explicit, supported override. Evaluator failures propagate;
    the executor retains them as preflight failures before loading a runtime.
    """
    registry = resolve_capability_checks(capability_checks)
    geometry = inspect_model_geometry(spec.model)
    capabilities: list[Capability] = [geometry.capability]
    blocking: list[Capability] = []
    check_evidence: dict[str, Any] = {}
    overrides = _override_names(spec)
    recognized = frozenset(
        check.name for check in registry.checks if check.supports_experimental_override
    )
    unknown_override = _unknown_override_capability(overrides, recognized)
    if unknown_override is not None:
        capabilities.append(unknown_override)
        blocking.append(unknown_override)

    for check in registry.checks:
        override = check.name in overrides and check.supports_experimental_override
        descriptor: dict[str, Any] = {
            "version": check.version,
            "override": {
                "requested": check.name in overrides,
                "supported": check.supports_experimental_override,
                "applied": False,
            },
        }
        result = check.evaluate(spec, geometry, override)
        if result is None:
            check_evidence[check.name] = {
                **descriptor,
                "applicable": False,
                "required": False,
            }
            continue
        if not isinstance(result, CapabilityCheckResult):
            raise TypeError(
                f"capability check {check.name!r} returned an invalid result"
            )
        if result.capability.name != check.name:
            raise ValueError(
                f"capability check {check.name!r} returned a different capability name"
            )
        capability = result.capability
        descriptor["override"]["applied"] = (
            override
            and result.required
            and capability.status is SupportLevel.EXPERIMENTAL
        )
        check_evidence[check.name] = {
            **descriptor,
            "applicable": True,
            "required": result.required,
        }
        capabilities.append(capability)
        if result.required and (
            capability.status in {SupportLevel.UNSUPPORTED, SupportLevel.UNKNOWN}
            or (capability.status is SupportLevel.EXPERIMENTAL and not override)
        ):
            blocking.append(capability)

    return PreflightCapabilityResult(
        capabilities=CapabilitySet(tuple(capabilities)),
        blocking=tuple(blocking),
        checks=check_evidence,
    )


def capability_inspection_to_mapping(
    result: PreflightCapabilityResult,
) -> Mapping[str, Any]:
    """Return a JSON-friendly capability inspection representation."""
    return {
        "capabilities": result.capabilities.to_mapping(),
        "blocking": tuple(capability.name for capability in result.blocking),
        "allowed": not result.blocking,
        "checks": result.checks,
    }
