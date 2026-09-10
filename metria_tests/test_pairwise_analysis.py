from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from metria import (
    ComparisonPlan,
    MeasurementResult,
    MetricDefinition,
    MetricDirection,
    MetricSummary,
    PairwiseAnalysisStatus,
    RunRecord,
    RunSpec,
    StudySpec,
    execute_study,
)
from metria.measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from metria.protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    SupportReport,
)


class TokenSession:
    def __init__(self, token_batches: Sequence[Sequence[int]]) -> None:
        self.token_batches = tuple(tuple(tokens) for tokens in token_batches)
        self.closed = False

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        assert tuple(item.kind for item in capture) == ("token_ids",)
        assert len(requests) == len(self.token_batches)
        return InferenceBatch(
            outputs=tuple("unused" for _ in requests),
            captures={"token_ids": self.token_batches},
        )

    def reset(self, scope: str = "measurement") -> None:
        del scope

    def close(self) -> None:
        self.closed = True


class TokenAdapter:
    def __init__(self, name: str, token_batches: Sequence[Sequence[int]]) -> None:
        self.name = name
        self.token_batches = tuple(tuple(tokens) for tokens in token_batches)
        self.calls: list[str] = []

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        del spec, environment
        self.calls.append("probe")
        return SupportReport(
            status="supported",
            evidence={"token_ids_capture": "native_output_token_ids"},
        )

    def resolve(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del environment
        self.calls.append("resolve")
        return {"runtime": {"name": self.name}, "scenario": spec.scenario}

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> TokenSession:
        del resolved, environment
        self.calls.append("launch")
        return TokenSession(self.token_batches)

    def observe(self, session: Any) -> Mapping[str, Any]:
        assert isinstance(session, TokenSession)
        self.calls.append("observe")
        return {"runtime": self.name, "hardware_class": "gpu-a"}


class ScalarSession:
    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        del requests, capture
        return InferenceBatch(outputs=())

    def reset(self, scope: str = "measurement") -> None:
        del scope

    def close(self) -> None:
        return None


class ScalarAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[str] = []

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        del spec, environment
        self.calls.append("probe")
        return SupportReport(status="supported")

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
    ) -> ScalarSession:
        del resolved, environment
        self.calls.append("launch")
        return ScalarSession()

    def observe(self, session: Any) -> Mapping[str, Any]:
        assert isinstance(session, ScalarSession)
        self.calls.append("observe")
        return {"runtime": self.name, "hardware_class": "gpu-a"}


class ScalarMeasurement:
    name = "test.scalar"
    version = "1"

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]:
        del config
        return ()

    def execute(
        self,
        session: Any,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        del session, scenario
        value = float(config.get("value", 1.0))
        return MeasurementResult(
            metrics={
                "value": MetricSummary(
                    definition=MetricDefinition(
                        name="value",
                        unit="units",
                        direction=MetricDirection.DESCRIPTIVE,
                        method=self.name,
                        version=self.version,
                    ),
                    value=value,
                )
            },
            evidence={"value": value},
        )


class DeltaAnalysis:
    name = "test.delta"
    version = "1"

    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[tuple[str, str]] = []

    def analyze(self, left: RunRecord, right: RunRecord) -> MeasurementResult:
        self.calls.append((left.run_id, right.run_id))
        if self.failure is not None:
            raise self.failure
        delta = right.metrics["value"].value - left.metrics["value"].value
        return MeasurementResult(
            metrics={
                "delta": MetricSummary(
                    definition=MetricDefinition(
                        name="delta",
                        unit="units",
                        direction=MetricDirection.DESCRIPTIVE,
                        method=self.name,
                        version=self.version,
                    ),
                    value=delta,
                )
            },
            evidence={"left": left.run_id, "right": right.run_id},
        )


def _scalar_run(runtime: str, *, model: str = "example/model") -> RunSpec:
    return RunSpec(
        model={"id": model},
        runtime={"name": runtime},
        scenario={"name": "decode"},
        measurements=(ScalarMeasurement.name,),
    )


def _scalar_study(*runs: RunSpec, analyses: tuple[str, ...]) -> StudySpec:
    return StudySpec(
        name="pairwise-analysis-study",
        runs=tuple(runs),
        comparison=ComparisonPlan(
            vary=frozenset({"runtime"}),
            control=frozenset({"model", "scenario", "measurements"}),
            block_by=frozenset({"observed.hardware_class"}),
            analyses=analyses,
        ),
    )


def test_comparison_plan_rejects_duplicate_analysis_names() -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        ComparisonPlan(analyses=("test.delta", "test.delta"))


def test_study_runs_declared_pairwise_analysis_for_compatible_pair() -> None:
    measurement = ScalarMeasurement()
    analysis = DeltaAnalysis()

    result = execute_study(
        _scalar_study(
            _scalar_run("a"),
            _scalar_run("b"),
            analyses=(analysis.name,),
        ),
        adapters={"a": ScalarAdapter("a"), "b": ScalarAdapter("b")},
        measurements={measurement.name: measurement},
        measurement_configs={measurement.name: {"value": 3.0}},
        environment={"hardware_class": "gpu-a"},
        analyses={analysis.name: analysis},
    )

    pair = result.comparisons[0]
    assert pair.report.compatible
    assert len(pair.analyses) == 1
    outcome = pair.analyses[0]
    assert outcome.status is PairwiseAnalysisStatus.COMPLETED
    assert outcome.result is not None
    assert outcome.result.metrics["delta"].value == 0.0
    assert analysis.calls == [("run-0000", "run-0001")]


def test_incompatible_pair_skips_analysis_without_calling_plugin() -> None:
    measurement = ScalarMeasurement()
    analysis = DeltaAnalysis()

    result = execute_study(
        _scalar_study(
            _scalar_run("a", model="model/a"),
            _scalar_run("b", model="model/b"),
            analyses=(analysis.name,),
        ),
        adapters={"a": ScalarAdapter("a"), "b": ScalarAdapter("b")},
        measurements={measurement.name: measurement},
        measurement_configs={},
        environment={"hardware_class": "gpu-a"},
        analyses={analysis.name: analysis},
    )

    pair = result.comparisons[0]
    assert not pair.report.compatible
    outcome = pair.analyses[0]
    assert outcome.status is PairwiseAnalysisStatus.SKIPPED
    assert outcome.reason == "pair is not directly comparable under the study plan"
    assert outcome.result is None
    assert analysis.calls == []


def test_analysis_failure_is_retained_without_raw_error_text() -> None:
    measurement = ScalarMeasurement()
    analysis = DeltaAnalysis(failure=RuntimeError("private analysis detail"))

    result = execute_study(
        _scalar_study(
            _scalar_run("a"),
            _scalar_run("b"),
            analyses=(analysis.name,),
        ),
        adapters={"a": ScalarAdapter("a"), "b": ScalarAdapter("b")},
        measurements={measurement.name: measurement},
        measurement_configs={},
        environment={"hardware_class": "gpu-a"},
        analyses={analysis.name: analysis},
    )

    outcome = result.comparisons[0].analyses[0]
    assert outcome.status is PairwiseAnalysisStatus.FAILED
    assert outcome.error_type == "RuntimeError"
    assert outcome.message_sha256 is not None
    assert len(outcome.message_sha256) == 64
    assert "private analysis detail" not in repr(outcome)


def test_missing_analysis_registration_fails_before_any_run_starts() -> None:
    measurement = ScalarMeasurement()
    adapter = ScalarAdapter("a")

    with pytest.raises(ValueError, match="unregistered pairwise analysis 'test.delta'"):
        execute_study(
            _scalar_study(_scalar_run("a"), analyses=("test.delta",)),
            adapters={"a": adapter},
            measurements={measurement.name: measurement},
            measurement_configs={},
            environment={},
        )

    assert adapter.calls == []


def test_analysis_registry_key_must_match_plugin_name() -> None:
    measurement = ScalarMeasurement()

    with pytest.raises(ValueError, match="does not match analysis.name"):
        execute_study(
            _scalar_study(_scalar_run("a"), analyses=("alias",)),
            adapters={"a": ScalarAdapter("a")},
            measurements={measurement.name: measurement},
            measurement_configs={},
            environment={},
            analyses={"alias": DeltaAnalysis()},
        )


def _trajectory_run(runtime: str) -> RunSpec:
    return RunSpec(
        model={"id": "example/model"},
        runtime={"name": runtime},
        scenario={"name": "decode"},
        measurements=(TokenTrajectoryProtocol.name,),
    )


def test_builtin_trajectory_analysis_is_derived_automatically() -> None:
    measurement = TokenTrajectoryProtocol()
    analysis = TrajectoryAgreementAnalysis()
    study = StudySpec(
        name="trajectory-study",
        runs=(_trajectory_run("reference"), _trajectory_run("candidate")),
        comparison=ComparisonPlan(
            vary=frozenset({"runtime"}),
            control=frozenset({"model", "scenario", "measurements"}),
            block_by=frozenset({"observed.hardware_class"}),
            analyses=(analysis.name,),
        ),
    )

    result = execute_study(
        study,
        adapters={
            "reference": TokenAdapter("reference", ((1, 2, 3),)),
            "candidate": TokenAdapter("candidate", ((1, 2),)),
        },
        measurements={measurement.name: measurement},
        measurement_configs={
            measurement.name: {"prompts": ({"id": "p1", "prompt": "same prompt"},)}
        },
        environment={"hardware_class": "gpu-a"},
        analyses={analysis.name: analysis},
    )

    outcome = result.comparisons[0].analyses[0]
    assert outcome.status is PairwiseAnalysisStatus.COMPLETED
    assert outcome.result is not None
    assert outcome.result.metrics["trajectory_agreement_score"].value == pytest.approx(
        100.0 * 2 / 3
    )
    assert outcome.result.evidence["mean_reference_steps"] == 3.0
    assert outcome.result.evidence["mean_candidate_steps"] == 2.0
