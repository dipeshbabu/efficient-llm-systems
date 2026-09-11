"""TurboQuant KV support knowledge for the explicit built-in integration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..capabilities import GeometryInspection, inspect_model_geometry
from ..capability_checks import CapabilityCheck, CapabilityCheckResult
from ..identity import Capability, SupportLevel
from ..models import RunSpec, TreatmentSpec, TreatmentType

_VALIDATED_TURBO_HEAD_DIMS = frozenset({128, 256})
_KV_TREATMENT_NAMES = frozenset(
    {"kv_cache", "llamacpp.kv_cache", "turboquant.kv_cache"}
)


def _uses_turboquant(kv_config: Mapping[str, Any]) -> bool:
    for key in ("key_dtype", "value_dtype"):
        value = kv_config.get(key)
        if isinstance(value, str) and value.lower().startswith("turbo"):
            return True
    return False


def _evaluate_geometry(
    inspection: GeometryInspection | None,
    kv_config: Mapping[str, Any],
    *,
    experimental_override: bool = False,
) -> Capability:
    """Evaluate the documented TurboQuant KV-cache head-dimension guardrail."""

    if not _uses_turboquant(kv_config):
        return Capability(
            name="turboquant.kv_cache.geometry",
            status=SupportLevel.SUPPORTED,
            reasons=("requested KV cache does not use a TurboQuant dtype",),
            evidence={"active": False},
        )

    if inspection is None:
        raise ValueError("active TurboQuant checks require geometry inspection")
    if inspection.geometry is None or inspection.geometry.head_dim is None:
        reasons = inspection.capability.reasons
        if inspection.geometry is not None:
            reasons = ("head_dim is unavailable from consistent geometry metadata",)
        return Capability(
            name="turboquant.kv_cache.geometry",
            status=SupportLevel.UNKNOWN,
            reasons=reasons,
            evidence={
                "active": True,
                "experimental_override": experimental_override,
                "geometry": (
                    inspection.geometry.to_mapping()
                    if inspection.geometry is not None
                    else None
                ),
                "geometry_status": SupportLevel(inspection.capability.status).value,
            },
        )

    head_dim = inspection.geometry.head_dim
    evidence = {
        "active": True,
        "head_dim": head_dim,
        "validated_head_dims": tuple(sorted(_VALIDATED_TURBO_HEAD_DIMS)),
        "experimental_override": experimental_override,
        "geometry": inspection.geometry.to_mapping(),
    }
    if head_dim <= 64:
        if experimental_override:
            return Capability(
                name="turboquant.kv_cache.geometry",
                status=SupportLevel.EXPERIMENTAL,
                reasons=(
                    "head_dim <= 64 is at/below the documented TurboQuant boundary; "
                    "explicit experimental override requested",
                ),
                evidence=evidence,
            )
        return Capability(
            name="turboquant.kv_cache.geometry",
            status=SupportLevel.UNSUPPORTED,
            reasons=(
                "head_dim <= 64 is at/below the documented TurboQuant boundary; "
                "use a non-Turbo KV dtype or opt into an explicit experimental override",
            ),
            evidence=evidence,
        )
    if head_dim in _VALIDATED_TURBO_HEAD_DIMS:
        return Capability(
            name="turboquant.kv_cache.geometry",
            status=SupportLevel.SUPPORTED,
            reasons=(f"head_dim={head_dim} is in the documented validated set",),
            evidence=evidence,
        )
    return Capability(
        name="turboquant.kv_cache.geometry",
        status=SupportLevel.EXPERIMENTAL,
        reasons=(
            f"head_dim={head_dim} is internally consistent but outside the documented "
            "validated set {128, 256}",
        ),
        evidence=evidence,
    )


def _kv_treatment(spec: RunSpec) -> TreatmentSpec | None:
    matches = [
        treatment
        for treatment in spec.treatments
        if treatment.name in _KV_TREATMENT_NAMES
        and treatment.kind is TreatmentType.RUNTIME_FEATURE
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            "only one KV-cache runtime treatment may be capability-checked"
        )
    return matches[0]


def evaluate_turboquant_kv_capability(
    model: Mapping[str, Any],
    kv_config: Mapping[str, Any],
    *,
    experimental_override: bool = False,
) -> Capability:
    """Evaluate the integration's existing head-dimension support boundary."""
    inspection = inspect_model_geometry(model) if _uses_turboquant(kv_config) else None
    return _evaluate_geometry(
        inspection, kv_config, experimental_override=experimental_override
    )


def _check_run(
    spec: RunSpec, geometry: GeometryInspection, experimental_override: bool
) -> CapabilityCheckResult | None:
    treatment = _kv_treatment(spec)
    if treatment is None:
        return None
    capability = _evaluate_geometry(
        geometry, treatment.config, experimental_override=experimental_override
    )
    return CapabilityCheckResult(
        capability, required=bool(capability.evidence.get("active"))
    )


TURBOQUANT_KV_CHECK = CapabilityCheck(
    name="turboquant.kv_cache.geometry",
    evaluate=_check_run,
    supports_experimental_override=True,
)
