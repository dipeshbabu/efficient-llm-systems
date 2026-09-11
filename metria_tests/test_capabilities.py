from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from metria import (
    Capability,
    CapabilityCheck,
    CapabilityCheckRegistry,
    CapabilityCheckResult,
    ComparisonPlan,
    RunSpec,
    RunStatus,
    StudySpec,
    TreatmentSpec,
    TreatmentType,
    execute_run,
    execute_study,
)
from metria.capabilities import (
    ModelGeometry,
    inspect_model_geometry,
)
from metria.identity import SupportLevel
from metria.inspection import inspect_run_capabilities
from metria.integrations.turboquant import evaluate_turboquant_kv_capability
from metria.protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    MeasurementResult,
    SupportReport,
)


def _turbo_spec(
    *,
    head_dim: int | None,
    override: bool = False,
) -> RunSpec:
    geometry = {} if head_dim is None else {"head_dim": head_dim}
    trial_policy: dict[str, Any] = {}
    if override:
        trial_policy["capability_overrides"] = ("turboquant.kv_cache.geometry",)
    return RunSpec(
        model={"id": "example/model", "geometry": geometry},
        runtime={"name": "llamacpp"},
        scenario={"max_tokens": 4},
        measurements=("test.measurement",),
        treatments=(
            TreatmentSpec(
                name="llamacpp.kv_cache",
                kind=TreatmentType.RUNTIME_FEATURE,
                config={"key_dtype": "q8_0", "value_dtype": "turbo3"},
            ),
        ),
        trial_policy=trial_policy,
    )


def _plain_spec(*, trial_policy: Mapping[str, Any]) -> RunSpec:
    return RunSpec(
        model={"id": "example/model"},
        runtime={"name": "llamacpp"},
        scenario={"max_tokens": 4},
        measurements=("test.measurement",),
        trial_policy=trial_policy,
    )


def test_model_geometry_derives_head_dim_from_consistent_metadata() -> None:
    inspection = inspect_model_geometry(
        {
            "geometry": {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "max_position_embeddings": 8192,
            }
        }
    )

    assert inspection.capability.status is SupportLevel.SUPPORTED
    assert inspection.geometry == ModelGeometry(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=32,
        context_length=8192,
        evidence=inspection.geometry.evidence if inspection.geometry else {},
    )
    assert inspection.geometry is not None
    assert inspection.geometry.head_dim == 128


def test_model_geometry_conflict_fails_to_unknown() -> None:
    inspection = inspect_model_geometry(
        {
            "geometry": {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "head_dim": 64,
            }
        }
    )

    assert inspection.geometry is None
    assert inspection.capability.status is SupportLevel.UNKNOWN
    assert "contradicts" in inspection.capability.reasons[0]


def test_turboquant_validated_head_dims_are_supported() -> None:
    for head_dim in (128, 256):
        capability = evaluate_turboquant_kv_capability(
            {"geometry": {"head_dim": head_dim}},
            {"key_dtype": "q8_0", "value_dtype": "turbo3"},
        )
        assert capability.status is SupportLevel.SUPPORTED
        assert capability.evidence["head_dim"] == head_dim


def test_turboquant_head_dim_64_fails_closed_without_override() -> None:
    result = inspect_run_capabilities(_turbo_spec(head_dim=64))
    capability = result.capabilities.get("turboquant.kv_cache.geometry")

    assert capability is not None
    assert capability.status is SupportLevel.UNSUPPORTED
    assert result.blocking == (capability,)


def test_turboquant_head_dim_64_explicit_override_is_experimental_and_allowed() -> None:
    result = inspect_run_capabilities(_turbo_spec(head_dim=64, override=True))
    capability = result.capabilities.get("turboquant.kv_cache.geometry")

    assert capability is not None
    assert capability.status is SupportLevel.EXPERIMENTAL
    assert capability.evidence["experimental_override"] is True
    assert result.blocking == ()


def test_unvalidated_consistent_head_dim_requires_explicit_override() -> None:
    blocked = inspect_run_capabilities(_turbo_spec(head_dim=96))
    allowed = inspect_run_capabilities(_turbo_spec(head_dim=96, override=True))
    capability = blocked.capabilities.get("turboquant.kv_cache.geometry")

    assert capability is not None
    assert capability.status is SupportLevel.EXPERIMENTAL
    assert blocked.blocking
    assert allowed.blocking == ()


def test_missing_turboquant_geometry_remains_unknown_even_with_override() -> None:
    result = inspect_run_capabilities(_turbo_spec(head_dim=None, override=True))
    capability = result.capabilities.get("turboquant.kv_cache.geometry")

    assert capability is not None
    assert capability.status is SupportLevel.UNKNOWN
    assert result.blocking == (capability,)


def test_non_turbo_kv_configuration_does_not_require_geometry() -> None:
    capability = evaluate_turboquant_kv_capability(
        {},
        {"key_dtype": "q8_0", "value_dtype": "q8_0"},
    )

    assert capability.status is SupportLevel.SUPPORTED
    assert capability.evidence["active"] is False


def test_override_shape_is_validated_without_matching_treatment() -> None:
    spec = _plain_spec(
        trial_policy={"capability_overrides": "turboquant.kv_cache.geometry"}
    )

    with pytest.raises(TypeError, match="must be a sequence"):
        inspect_run_capabilities(spec)


def test_unrecognized_override_is_retained_as_blocking_evidence() -> None:
    spec = _plain_spec(trial_policy={"capability_overrides": ("turboquant.kv_cache",)})

    result = inspect_run_capabilities(spec)
    capability = result.capabilities.get("metria.capability_overrides")

    assert capability is not None
    assert capability.status is SupportLevel.UNKNOWN
    assert capability.evidence["unrecognized"] == ("turboquant.kv_cache",)
    assert result.blocking == (capability,)


class _CountingSession:
    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        del capture
        return InferenceBatch(outputs=tuple("ok" for _ in requests))

    def reset(self, scope: str = "measurement") -> None:
        del scope

    def close(self) -> None:
        pass


class _CountingAdapter:
    name = "llamacpp"

    def __init__(self) -> None:
        self.probes = 0

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        del spec, environment
        self.probes += 1
        return SupportReport(status="supported")

    def resolve(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del spec, environment
        return {"runtime": {"name": self.name}}

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> _CountingSession:
        del resolved, environment
        return _CountingSession()

    def observe(self, session: _CountingSession) -> Mapping[str, Any]:
        del session
        return {"runtime": self.name}


class _Measurement:
    name = "test.measurement"
    version = "1"

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]:
        del config
        return ()

    def execute(
        self,
        session: _CountingSession,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        del session, scenario, config
        return MeasurementResult()


def test_execute_run_blocks_known_unsupported_capability_before_adapter_probe() -> None:
    adapter = _CountingAdapter()

    record = execute_run(
        study_name="guardrail-study",
        run_id="run-0",
        spec=_turbo_spec(head_dim=64),
        adapter=adapter,
        measurement=_Measurement(),
        measurement_config={},
        environment={},
    )

    assert record.status is RunStatus.PREFLIGHT_FAILED
    assert adapter.probes == 0
    assert record.events[-1]["kind"] == "capability_blocked"
    assert record.provenance["capabilities"]["allowed"] is False


def test_execute_run_retains_experimental_override_in_provenance() -> None:
    adapter = _CountingAdapter()

    record = execute_run(
        study_name="guardrail-study",
        run_id="run-0",
        spec=_turbo_spec(head_dim=64, override=True),
        adapter=adapter,
        measurement=_Measurement(),
        measurement_config={},
        environment={},
    )

    assert record.status is RunStatus.COMPLETED
    assert adapter.probes == 1
    capability = record.provenance["capabilities"]["capabilities"][
        "turboquant.kv_cache.geometry"
    ]
    assert capability["status"] == "experimental"
    assert capability["evidence"]["experimental_override"] is True


def _application_checks(status, *, required=True, supports_override=False):
    def evaluate(spec, geometry, override):
        return CapabilityCheckResult(
            Capability(
                "example.layout",
                status,
                reasons=("synthetic application check",),
                evidence={"check": "domain-owned field", "received_override": override},
            ),
            required=required,
        )

    return CapabilityCheckRegistry(
        (
            CapabilityCheck(
                "example.layout",
                evaluate,
                version="test-1",
                supports_experimental_override=supports_override,
            ),
        )
    )


@pytest.mark.parametrize(
    "status,blocked",
    [
        (SupportLevel.SUPPORTED, False),
        (SupportLevel.EXPERIMENTAL, True),
        (SupportLevel.UNSUPPORTED, True),
        (SupportLevel.UNKNOWN, True),
    ],
)
def test_application_checks_use_generic_fail_closed_policy(status, blocked):
    result = inspect_run_capabilities(
        _plain_spec(trial_policy={}), capability_checks=_application_checks(status)
    )
    assert bool(result.blocking) is blocked
    assert result.capabilities.get("example.layout").status is status


def test_generic_override_preserves_domain_evidence_and_records_its_effect():
    spec = _plain_spec(trial_policy={"capability_overrides": ["example.layout"]})
    result = inspect_run_capabilities(
        spec,
        capability_checks=_application_checks(
            SupportLevel.EXPERIMENTAL, supports_override=True
        ),
    )
    assert not result.blocking
    assert (
        result.capabilities.get("example.layout").evidence["check"]
        == "domain-owned field"
    )
    assert result.checks["example.layout"]["override"] == {
        "requested": True,
        "supported": True,
        "applied": True,
    }
    assert result.checks["example.layout"]["version"] == "test-1"


@pytest.mark.parametrize("status", [SupportLevel.UNKNOWN, SupportLevel.UNSUPPORTED])
def test_generic_override_cannot_upgrade_unknown_or_unsupported(status):
    spec = _plain_spec(trial_policy={"capability_overrides": ["example.layout"]})
    result = inspect_run_capabilities(
        spec, capability_checks=_application_checks(status, supports_override=True)
    )
    assert result.blocking
    assert result.capabilities.get("example.layout").status is status
    assert not result.checks["example.layout"]["override"]["applied"]


def test_unavailable_override_is_not_silently_authorized():
    spec = _plain_spec(trial_policy={"capability_overrides": ["example.layout"]})
    result = inspect_run_capabilities(
        spec, capability_checks=_application_checks(SupportLevel.SUPPORTED)
    )
    assert result.blocking[0].name == "metria.capability_overrides"
    assert (
        result.capabilities.get("example.layout").evidence["received_override"] is False
    )


def test_inactive_and_non_required_checks_do_not_block():
    registry = _application_checks(SupportLevel.UNSUPPORTED, required=False).with_check(
        CapabilityCheck("example.inactive", lambda spec, geometry, override: None)
    )
    result = inspect_run_capabilities(
        _plain_spec(trial_policy={}), capability_checks=registry
    )
    assert not result.blocking
    assert result.capabilities.get("example.inactive") is None
    assert result.checks["example.inactive"]["applicable"] is False


def test_application_registry_cannot_remove_or_replace_builtin_checks():
    spec = _turbo_spec(head_dim=64)
    assert inspect_run_capabilities(
        spec, capability_checks=CapabilityCheckRegistry()
    ).blocking
    replacement = CapabilityCheckRegistry(
        (CapabilityCheck("turboquant.kv_cache.geometry", lambda *args: None),)
    )
    with pytest.raises(ValueError, match="duplicate"):
        inspect_run_capabilities(spec, capability_checks=replacement)


def test_registry_detaches_its_entries_and_rejects_duplicate_or_reserved_names():
    check = CapabilityCheck("example.one", lambda *args: None)
    entries = [check]
    registry = CapabilityCheckRegistry(entries)
    entries.clear()
    assert registry.checks == (check,)
    with pytest.raises(ValueError, match="duplicate"):
        registry.with_check(check)
    with pytest.raises(ValueError, match="reserved"):
        registry.with_check(CapabilityCheck("model.geometry", lambda *args: None))


@pytest.mark.parametrize(
    "bad_result",
    [object(), CapabilityCheckResult(Capability("wrong.name", SupportLevel.SUPPORTED))],
)
def test_bad_check_results_fail_before_runtime_probe(bad_result):
    registry = CapabilityCheckRegistry(
        (CapabilityCheck("example.bad", lambda *args: bad_result),)
    )
    adapter = _CountingAdapter()
    record = execute_run(
        study_name="checks",
        run_id="one",
        spec=_plain_spec(trial_policy={}),
        adapter=adapter,
        measurement=_Measurement(),
        measurement_config={},
        environment={},
        capability_checks=registry,
    )
    assert record.status is RunStatus.PREFLIGHT_FAILED
    assert adapter.probes == 0
    assert record.events[-1]["stage"] == "capability_inspection"


def test_custom_checks_are_applied_to_every_study_run():
    adapter = _CountingAdapter()
    study = StudySpec(
        "application-checks",
        (_plain_spec(trial_policy={}), _plain_spec(trial_policy={})),
        ComparisonPlan(),
    )
    result = execute_study(
        study,
        adapters={"llamacpp": adapter},
        measurements={"test.measurement": _Measurement()},
        measurement_configs={},
        environment={},
        capability_checks=_application_checks(SupportLevel.UNSUPPORTED),
    )
    assert all(record.status is RunStatus.PREFLIGHT_FAILED for record in result.records)
    assert adapter.probes == 0


@pytest.mark.parametrize(
    "head_dim,without,with_override",
    [
        (None, SupportLevel.UNKNOWN, SupportLevel.UNKNOWN),
        (0, SupportLevel.UNKNOWN, SupportLevel.UNKNOWN),
        (32, SupportLevel.UNSUPPORTED, SupportLevel.EXPERIMENTAL),
        (64, SupportLevel.UNSUPPORTED, SupportLevel.EXPERIMENTAL),
        (96, SupportLevel.EXPERIMENTAL, SupportLevel.EXPERIMENTAL),
        (128, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED),
        (256, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED),
        (512, SupportLevel.EXPERIMENTAL, SupportLevel.EXPERIMENTAL),
    ],
)
@pytest.mark.parametrize("override", [False, True])
def test_turboquant_integration_preserves_the_support_boundary(
    head_dim, without, with_override, override
):
    spec = _turbo_spec(head_dim=head_dim, override=override)
    result = inspect_run_capabilities(spec)
    capability = result.capabilities.get("turboquant.kv_cache.geometry")
    assert capability.status is (with_override if override else without)
    assert capability == evaluate_turboquant_kv_capability(
        spec.model, spec.treatments[0].config, experimental_override=override
    )


def test_generic_policy_modules_do_not_own_domain_names_or_rules():
    import inspect

    import metria
    import metria.capabilities as geometry
    import metria.capability_checks as contracts
    import metria.inspection as policy

    for module in (geometry, contracts, policy):
        assert "turboquant" not in inspect.getsource(module).lower()
    assert metria.evaluate_turboquant_kv_capability is evaluate_turboquant_kv_capability
    assert (
        evaluate_turboquant_kv_capability.__module__ == "metria.integrations.turboquant"
    )
