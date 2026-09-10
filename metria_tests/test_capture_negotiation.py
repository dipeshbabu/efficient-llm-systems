from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from metria import MeasurementResult, RunSpec, RunStatus, execute_run
from metria.protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    SupportReport,
)


class _Session:
    def __init__(self) -> None:
        self.close_calls = 0

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
        self.close_calls += 1


class _Adapter:
    name = "capture-fake"

    def __init__(
        self,
        *,
        capture_status: str = "supported",
        capture_error: Exception | None = None,
    ) -> None:
        self.capture_status = capture_status
        self.capture_error = capture_error
        self.calls: list[str] = []
        self.capture_requests: tuple[CaptureRequest, ...] = ()
        self.session = _Session()

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        del spec, environment
        self.calls.append("probe")
        return SupportReport(status="supported", evidence={"runtime": self.name})

    def probe_captures(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
        capture: Sequence[CaptureRequest],
    ) -> SupportReport:
        del spec, environment
        self.calls.append("probe_captures")
        if self.capture_error is not None:
            raise self.capture_error
        self.capture_requests = tuple(capture)
        reasons = ()
        if self.capture_status != "supported":
            reasons = (f"capture status is {self.capture_status}",)
        return SupportReport(
            status=self.capture_status,
            reasons=reasons,
            evidence={
                "mechanism": "fake_capture_probe",
                "kinds": tuple(request.kind for request in capture),
            },
        )

    def resolve(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del environment
        self.calls.append("resolve")
        return {"runtime": {"name": self.name}, "model": spec.model}

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> _Session:
        del resolved, environment
        self.calls.append("launch")
        return self.session

    def observe(self, session: _Session) -> Mapping[str, Any]:
        assert session is self.session
        self.calls.append("observe")
        return {"runtime": self.name}


class _LegacyAdapter:
    """Adapter predating dedicated capture negotiation."""

    name = "legacy-no-capture"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.session = _Session()

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        del spec, environment
        self.calls.append("probe")
        return SupportReport(status="supported", evidence={"runtime": self.name})

    def resolve(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del environment
        self.calls.append("resolve")
        return {"runtime": {"name": self.name}, "model": spec.model}

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> _Session:
        del resolved, environment
        self.calls.append("launch")
        return self.session

    def observe(self, session: _Session) -> Mapping[str, Any]:
        assert session is self.session
        self.calls.append("observe")
        return {"runtime": self.name}


class _Measurement:
    name = "test.capture.measurement"
    version = "1"

    def __init__(
        self,
        requirements: Any = (),
        *,
        requirement_error: Exception | None = None,
    ) -> None:
        self._requirements = requirements
        self._requirement_error = requirement_error
        self.execute_calls = 0

    def requirements(self, config: Mapping[str, Any]) -> Any:
        del config
        if self._requirement_error is not None:
            raise self._requirement_error
        return self._requirements

    def execute(
        self,
        session: _Session,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        del session, scenario, config
        self.execute_calls += 1
        return MeasurementResult()


def _spec(measurement: _Measurement, runtime: str = "capture-fake") -> RunSpec:
    return RunSpec(
        model={"id": "example/model"},
        runtime={"name": runtime},
        scenario={"name": "decode"},
        measurements=(measurement.name,),
    )


def _execute(adapter: Any, measurement: _Measurement):
    return execute_run(
        study_name="capture-negotiation",
        run_id="run-0",
        spec=_spec(measurement, runtime=adapter.name),
        adapter=adapter,
        measurement=measurement,
        measurement_config={},
        environment={"hardware_class": "test"},
    )


def test_supported_capture_is_negotiated_and_retained_before_launch() -> None:
    measurement = _Measurement((CaptureRequest(kind="token_ids"),))
    adapter = _Adapter()

    record = _execute(adapter, measurement)

    assert record.status is RunStatus.COMPLETED
    assert adapter.calls == ["probe", "probe_captures", "resolve", "launch", "observe"]
    assert adapter.capture_requests == (CaptureRequest(kind="token_ids"),)
    captures = record.provenance["preflight"]["captures"]
    assert captures["status"] == "supported"
    assert captures["required"] == ({"kind": "token_ids", "options": {}},)
    assert captures["evidence"]["mechanism"] == "fake_capture_probe"
    assert measurement.execute_calls == 1


@pytest.mark.parametrize("status", ["unsupported", "unknown", "experimental"])
def test_non_supported_capture_state_fails_before_resolve_or_launch(
    status: str,
) -> None:
    measurement = _Measurement((CaptureRequest(kind="token_ids"),))
    adapter = _Adapter(capture_status=status)

    record = _execute(adapter, measurement)

    assert record.status is RunStatus.PREFLIGHT_FAILED
    assert adapter.calls == ["probe", "probe_captures"]
    assert adapter.session.close_calls == 0
    assert measurement.execute_calls == 0
    assert record.provenance["preflight"]["captures"]["status"] == status
    assert record.events[-1]["kind"] == "capture_requirements_blocked"
    assert not any(event["stage"] == "launch" for event in record.events)


def test_multiple_unique_capture_requirements_are_negotiated_atomically() -> None:
    requirements = (
        CaptureRequest(kind="token_ids"),
        CaptureRequest(kind="attention_trace"),
    )
    measurement = _Measurement(requirements)
    adapter = _Adapter()

    record = _execute(adapter, measurement)

    assert record.status is RunStatus.COMPLETED
    assert adapter.capture_requests == requirements
    assert record.provenance["preflight"]["captures"]["required"] == (
        {"kind": "token_ids", "options": {}},
        {"kind": "attention_trace", "options": {}},
    )


def test_duplicate_capture_requirements_fail_before_runtime_probe() -> None:
    measurement = _Measurement(
        (
            CaptureRequest(kind="token_ids"),
            CaptureRequest(kind="token_ids"),
        )
    )
    adapter = _Adapter()

    record = _execute(adapter, measurement)

    assert record.status is RunStatus.PREFLIGHT_FAILED
    assert adapter.calls == []
    assert measurement.execute_calls == 0
    event = record.events[-1]
    assert event["stage"] == "measurement_requirements"
    assert event["error_type"] == "ValueError"


@pytest.mark.parametrize(
    "requirements",
    ["token_ids", (object(),)],
)
def test_malformed_measurement_requirements_fail_before_runtime_probe(
    requirements: Any,
) -> None:
    measurement = _Measurement(requirements)
    adapter = _Adapter()

    record = _execute(adapter, measurement)

    assert record.status is RunStatus.PREFLIGHT_FAILED
    assert adapter.calls == []
    assert record.events[-1]["stage"] == "measurement_requirements"


def test_requirement_validation_failure_is_privacy_conscious() -> None:
    measurement = _Measurement(
        requirement_error=ValueError("private prompt must never be retained")
    )
    adapter = _Adapter()

    record = _execute(adapter, measurement)

    assert record.status is RunStatus.PREFLIGHT_FAILED
    assert adapter.calls == []
    event = record.events[-1]
    assert event["stage"] == "measurement_requirements"
    assert event["error_type"] == "ValueError"
    assert len(event["message_sha256"]) == 64
    assert "private prompt" not in repr(record)


def test_capture_probe_failure_is_privacy_conscious_and_prelaunch() -> None:
    measurement = _Measurement((CaptureRequest(kind="token_ids"),))
    adapter = _Adapter(capture_error=RuntimeError("private capture detail"))

    record = _execute(adapter, measurement)

    assert record.status is RunStatus.PREFLIGHT_FAILED
    assert adapter.calls == ["probe", "probe_captures"]
    assert measurement.execute_calls == 0
    assert adapter.session.close_calls == 0
    assert record.events[-1]["stage"] == "capture_probe"
    assert "private capture detail" not in repr(record)


def test_no_capture_measurement_keeps_legacy_adapter_source_compatible() -> None:
    measurement = _Measurement()
    adapter = _LegacyAdapter()

    record = _execute(adapter, measurement)

    assert record.status is RunStatus.COMPLETED
    assert adapter.calls == ["probe", "resolve", "launch", "observe"]
    captures = record.provenance["preflight"]["captures"]
    assert captures["status"] == "supported"
    assert captures["required"] == ()
    assert captures["evidence"]["mechanism"] == "no_capture_required"


def test_capture_request_rejects_empty_kind_and_reserved_options() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        CaptureRequest(kind="   ")
    with pytest.raises(ValueError, match="options are reserved"):
        CaptureRequest(kind="token_ids", options={"top_k": 4})
