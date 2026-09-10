from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from metria import (
    MeasurementResult,
    MetricDefinition,
    MetricDirection,
    MetricSummary,
    RunSpec,
    RunStatus,
    execute_run,
)
from metria.measurements import TokenTrajectoryProtocol
from metria.protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    SupportReport,
)


class FakeSession:
    def __init__(
        self,
        *,
        token_batches: Sequence[Sequence[int]] = (),
        close_error: bool = False,
    ) -> None:
        self.token_batches = tuple(tuple(tokens) for tokens in token_batches)
        self.close_error = close_error
        self.close_calls = 0
        self.infer_calls = 0

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        self.infer_calls += 1
        if capture and tuple(item.kind for item in capture) != ("token_ids",):
            raise ValueError("unexpected capture")
        return InferenceBatch(
            outputs=tuple("unused" for _ in requests),
            captures={"token_ids": self.token_batches} if capture else {},
        )

    def reset(self, scope: str = "measurement") -> None:
        del scope

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error:
            raise RuntimeError("close secret should not be retained")


class FakeAdapter:
    def __init__(
        self,
        session: FakeSession,
        *,
        support_status: str = "supported",
        fail_stage: str | None = None,
    ) -> None:
        self.session = session
        self.support_status = support_status
        self.fail_stage = fail_stage
        self.calls: list[str] = []

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        del spec, environment
        self.calls.append("probe")
        if self.fail_stage == "probe":
            raise RuntimeError("probe secret")
        return SupportReport(
            status=self.support_status,
            reasons=("not available",) if self.support_status != "supported" else (),
            evidence={
                "capability": "fake",
                "token_ids_capture": "native_output_token_ids",
            },
        )

    def resolve(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del environment
        self.calls.append("resolve")
        if self.fail_stage == "resolve":
            raise RuntimeError("resolve secret")
        return {
            "runtime": {"name": "fake", "version": "1"},
            "model": spec.model,
            "scenario": spec.scenario,
        }

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> FakeSession:
        del resolved, environment
        self.calls.append("launch")
        if self.fail_stage == "launch":
            raise RuntimeError("launch secret")
        return self.session

    def observe(self, session: Any) -> Mapping[str, Any]:
        assert session is self.session
        self.calls.append("observe")
        if self.fail_stage == "observe":
            raise RuntimeError("observe secret")
        return {"runtime": "fake", "applied": True}


class FakeMeasurement:
    name = "test.measurement"
    version = "1"

    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls = 0

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]:
        del config
        return ()

    def execute(
        self,
        session: Any,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        del session, scenario, config
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        metric = MetricSummary(
            definition=MetricDefinition(
                name="latency",
                unit="seconds",
                direction=MetricDirection.LOWER_IS_BETTER,
                method=self.name,
                version=self.version,
            ),
            value=1.5,
        )
        return MeasurementResult(
            metrics={"latency": metric},
            evidence={"samples": (1.0, 2.0)},
            artifacts=({"kind": "raw-log", "uri": "artifact://fake"},),
        )


def _spec(*, measurement: str = "test.measurement") -> RunSpec:
    return RunSpec(
        model={"id": "example/model"},
        runtime={"name": "fake"},
        scenario={"name": "decode"},
        measurements=(measurement,),
    )


def _execute(
    adapter: FakeAdapter,
    measurement: Any,
    *,
    spec: RunSpec | None = None,
):
    return execute_run(
        study_name="executor-study",
        run_id="run-1",
        spec=spec or _spec(measurement=measurement.name),
        adapter=adapter,
        measurement=measurement,
        measurement_config={},
        environment={"hardware_class": "test-gpu"},
    )


def test_execute_run_builds_complete_record_and_closes_session() -> None:
    session = FakeSession()
    adapter = FakeAdapter(session)
    measurement = FakeMeasurement()

    record = _execute(adapter, measurement)

    assert record.status is RunStatus.COMPLETED
    assert record.resolved["runtime"]["name"] == "fake"
    assert record.observed["applied"] is True
    assert record.metrics["latency"].value == 1.5
    assert record.evidence["measurements"][measurement.name]["samples"] == (1.0, 2.0)
    assert record.artifacts[0]["kind"] == "raw-log"
    assert record.provenance["preflight"]["status"] == "supported"
    assert record.provenance["measurement"]["name"] == measurement.name
    assert session.close_calls == 1
    assert adapter.calls == ["probe", "resolve", "launch", "observe"]


def test_measurement_not_declared_fails_before_adapter_probe() -> None:
    session = FakeSession()
    adapter = FakeAdapter(session)
    measurement = FakeMeasurement()

    record = _execute(adapter, measurement, spec=_spec(measurement="other.measurement"))

    assert record.status is RunStatus.PREFLIGHT_FAILED
    assert adapter.calls == []
    assert session.close_calls == 0
    assert record.events[0]["kind"] == "measurement_not_requested"


def test_unsupported_preflight_returns_record_without_launching() -> None:
    session = FakeSession()
    adapter = FakeAdapter(session, support_status="unsupported")

    record = _execute(adapter, FakeMeasurement())

    assert record.status is RunStatus.PREFLIGHT_FAILED
    assert adapter.calls == ["probe"]
    assert record.provenance["preflight"]["reasons"] == ("not available",)
    assert session.close_calls == 0


@pytest.mark.parametrize("stage", ["probe", "resolve"])
def test_prelaunch_exceptions_become_preflight_failed(stage: str) -> None:
    session = FakeSession()
    adapter = FakeAdapter(session, fail_stage=stage)

    record = _execute(adapter, FakeMeasurement())

    assert record.status is RunStatus.PREFLIGHT_FAILED
    assert record.events[-1]["stage"] == stage
    assert "secret" not in repr(record.events)
    assert len(record.events[-1]["message_sha256"]) == 64
    assert session.close_calls == 0


def test_launch_failure_preserves_resolved_state() -> None:
    session = FakeSession()
    adapter = FakeAdapter(session, fail_stage="launch")

    record = _execute(adapter, FakeMeasurement())

    assert record.status is RunStatus.FAILED
    assert record.resolved["runtime"]["name"] == "fake"
    assert record.observed == {}
    assert session.close_calls == 0


def test_measurement_failure_still_observes_and_closes_runtime() -> None:
    session = FakeSession()
    adapter = FakeAdapter(session)
    measurement = FakeMeasurement(failure=ValueError("private prompt in failure"))

    record = _execute(adapter, measurement)

    assert record.status is RunStatus.FAILED
    assert record.observed["applied"] is True
    assert session.close_calls == 1
    assert "private prompt" not in repr(record.events)
    assert any(event["stage"] == "measurement" for event in record.events)


def test_timeout_failure_uses_timed_out_status() -> None:
    session = FakeSession()
    adapter = FakeAdapter(session)
    measurement = FakeMeasurement(failure=TimeoutError("private timeout detail"))

    record = _execute(adapter, measurement)

    assert record.status is RunStatus.TIMED_OUT
    assert session.close_calls == 1
    assert record.observed["applied"] is True


def test_observation_failure_after_metrics_marks_record_partial() -> None:
    session = FakeSession()
    adapter = FakeAdapter(session, fail_stage="observe")

    record = _execute(adapter, FakeMeasurement())

    assert record.status is RunStatus.PARTIAL
    assert record.metrics["latency"].value == 1.5
    assert record.evidence["measurements"]["test.measurement"]["samples"] == (1.0, 2.0)
    assert record.observed == {}
    assert session.close_calls == 1


def test_cleanup_failure_after_metrics_marks_record_partial() -> None:
    session = FakeSession(close_error=True)
    adapter = FakeAdapter(session)

    record = _execute(adapter, FakeMeasurement())

    assert record.status is RunStatus.PARTIAL
    assert record.metrics["latency"].value == 1.5
    assert record.observed["applied"] is True
    assert session.close_calls == 1
    close_event = next(event for event in record.events if event["stage"] == "close")
    assert close_event["kind"] == "error"
    assert "secret" not in repr(close_event)


def test_executor_integrates_trajectory_protocol_into_run_record() -> None:
    session = FakeSession(token_batches=((1, 2, 3), (4, 5)))
    adapter = FakeAdapter(session)
    measurement = TokenTrajectoryProtocol()
    spec = RunSpec(
        model={"id": "example/model"},
        runtime={"name": "fake"},
        scenario={"name": "decode"},
        measurements=(measurement.name,),
    )

    record = execute_run(
        study_name="trajectory-study",
        run_id="trajectory-run",
        spec=spec,
        adapter=adapter,
        measurement=measurement,
        measurement_config={
            "prompts": (
                {"id": "p1", "prompt": "private one"},
                {"id": "p2", "prompt": "private two"},
            )
        },
        environment={"hardware_class": "test-gpu"},
    )

    assert record.status is RunStatus.COMPLETED
    assert record.metrics["trajectory_mean_steps"].value == 2.5
    trajectory_evidence = record.evidence["measurements"][measurement.name]
    assert trajectory_evidence["prompts"][0]["token_ids"] == (1, 2, 3)
    assert "private one" not in repr(trajectory_evidence)
    assert record.provenance["preflight"]["captures"]["status"] == "supported"
    assert session.infer_calls == 1
    assert session.close_calls == 1
