"""Study-level orchestration across runtimes, measurements, and analyses."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any

from .capability_checks import CapabilityCheckRegistry
from .comparison import compare_runs
from .execution import execute_run
from .inspection import resolve_capability_checks
from .models import CompatibilityReport, RunRecord, RunSpec, StudySpec
from .protocols import (
    MeasurementProtocol,
    MeasurementResult,
    PairwiseAnalysis,
    RuntimeAdapter,
)


class PairwiseAnalysisStatus(str, Enum):
    """Lifecycle status of one requested analysis for one run pair."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class StudyPairAnalysis:
    """Outcome of one named pairwise analysis."""

    name: str
    version: str
    status: PairwiseAnalysisStatus
    result: MeasurementResult | None = None
    reason: str | None = None
    error_type: str | None = None
    message_sha256: str | None = None

    def __post_init__(self) -> None:
        """Reject contradictory outcome payloads."""

        if self.status is PairwiseAnalysisStatus.COMPLETED and self.result is None:
            raise ValueError("completed pairwise analysis must contain a result")
        if (
            self.status is not PairwiseAnalysisStatus.COMPLETED
            and self.result is not None
        ):
            raise ValueError(
                "non-completed pairwise analysis must not contain a result"
            )


@dataclass(frozen=True)
class StudyPairComparison:
    """Compatibility plus derived analyses for one deterministic run pair."""

    left_run_id: str
    right_run_id: str
    report: CompatibilityReport
    analyses: tuple[StudyPairAnalysis, ...] = ()

    def __post_init__(self) -> None:
        """Normalize pairwise analysis outcomes into an immutable tuple."""

        object.__setattr__(self, "analyses", tuple(self.analyses))


@dataclass(frozen=True)
class StudyExecutionResult:
    """All run records and pairwise study results."""

    study: StudySpec
    records: tuple[RunRecord, ...]
    comparisons: tuple[StudyPairComparison, ...]

    def __post_init__(self) -> None:
        """Normalize result collections into immutable tuples."""

        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "comparisons", tuple(self.comparisons))


def _message_fingerprint(exc: Exception) -> str:
    """Fingerprint analysis errors without retaining possibly sensitive text."""

    return hashlib.sha256(str(exc).encode("utf-8")).hexdigest()


def _runtime_name(spec: RunSpec, *, index: int) -> str:
    """Return the explicit runtime registry name for one requested run."""

    value = spec.runtime.get("name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"study run[{index}] runtime.name must be a non-empty string")
    return value


def _measurement_name(spec: RunSpec, *, index: int) -> str:
    """Return the one measurement supported by the initial study executor."""

    if len(spec.measurements) != 1:
        raise ValueError(
            f"study run[{index}] must request exactly one measurement; "
            "multi-measurement scheduling is not implemented yet"
        )
    value = spec.measurements[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"study run[{index}] measurement name must be a non-empty string"
        )
    return value


def _validate_registries(
    study: StudySpec,
    adapters: Mapping[str, RuntimeAdapter],
    measurements: Mapping[str, MeasurementProtocol],
    analyses: Mapping[str, PairwiseAnalysis],
) -> tuple[tuple[str, str], ...]:
    """Validate every run and pairwise-analysis route before experiment work."""

    routes: list[tuple[str, str]] = []
    for key, adapter in adapters.items():
        if not isinstance(key, str) or not key:
            raise ValueError("runtime registry keys must be non-empty strings")
        if adapter.name != key:
            raise ValueError(
                f"runtime registry key {key!r} does not match adapter.name "
                f"{adapter.name!r}"
            )
    for key, measurement in measurements.items():
        if not isinstance(key, str) or not key:
            raise ValueError("measurement registry keys must be non-empty strings")
        if measurement.name != key:
            raise ValueError(
                f"measurement registry key {key!r} does not match measurement.name "
                f"{measurement.name!r}"
            )
    for key, analysis in analyses.items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                "pairwise analysis registry keys must be non-empty strings"
            )
        if analysis.name != key:
            raise ValueError(
                f"pairwise analysis registry key {key!r} does not match analysis.name "
                f"{analysis.name!r}"
            )
        if not isinstance(analysis.version, str) or not analysis.version.strip():
            raise ValueError(f"pairwise analysis {key!r} must define a version")

    for analysis_name in study.comparison.analyses:
        if analysis_name not in analyses:
            raise ValueError(
                f"comparison plan requires unregistered pairwise analysis "
                f"{analysis_name!r}"
            )

    for index, spec in enumerate(study.runs):
        runtime_name = _runtime_name(spec, index=index)
        measurement_name = _measurement_name(spec, index=index)
        if runtime_name not in adapters:
            raise ValueError(
                f"study run[{index}] requires unregistered runtime {runtime_name!r}"
            )
        if measurement_name not in measurements:
            raise ValueError(
                f"study run[{index}] requires unregistered measurement "
                f"{measurement_name!r}"
            )
        routes.append((runtime_name, measurement_name))
    return tuple(routes)


def _analyze_pair(
    left: RunRecord,
    right: RunRecord,
    report: CompatibilityReport,
    analysis_names: tuple[str, ...],
    analyses: Mapping[str, PairwiseAnalysis],
) -> tuple[StudyPairAnalysis, ...]:
    """Run declared analyzers only when the pair is directly comparable."""

    outcomes: list[StudyPairAnalysis] = []
    for name in analysis_names:
        analysis = analyses[name]
        if not report.compatible:
            outcomes.append(
                StudyPairAnalysis(
                    name=analysis.name,
                    version=analysis.version,
                    status=PairwiseAnalysisStatus.SKIPPED,
                    reason="pair is not directly comparable under the study plan",
                )
            )
            continue
        try:
            result = analysis.analyze(left, right)
        except Exception as exc:
            outcomes.append(
                StudyPairAnalysis(
                    name=analysis.name,
                    version=analysis.version,
                    status=PairwiseAnalysisStatus.FAILED,
                    error_type=type(exc).__name__,
                    message_sha256=_message_fingerprint(exc),
                )
            )
        else:
            outcomes.append(
                StudyPairAnalysis(
                    name=analysis.name,
                    version=analysis.version,
                    status=PairwiseAnalysisStatus.COMPLETED,
                    result=result,
                )
            )
    return tuple(outcomes)


def execute_study(
    study: StudySpec,
    *,
    adapters: Mapping[str, RuntimeAdapter],
    measurements: Mapping[str, MeasurementProtocol],
    measurement_configs: Mapping[str, Mapping[str, Any]],
    environment: Mapping[str, Any],
    analyses: Mapping[str, PairwiseAnalysis] | None = None,
    record_sink: Callable[[RunRecord], None] | None = None,
    capability_checks: CapabilityCheckRegistry | None = None,
) -> StudyExecutionResult:
    """Execute every run, validate pair compatibility, and derive analyses.

    Registry/configuration mistakes are validated before the first run starts so
    a programming error does not create a half-executed study. Once execution
    begins, experimental failures are preserved as ``RunRecord`` values and do
    not abort later runs.

    Pairwise analyzers are declared by ``study.comparison.analyses`` and run only
    after ``compare_runs`` reports the pair directly compatible. Analyzer
    failures are retained as hashed failure outcomes rather than aborting the
    study. The left/right order is deterministic study order; directional
    analyzers should interpret the earlier run deliberately (for example as a
    reference baseline).

    The initial study executor uses one shared environment mapping for all runs
    and supports exactly one measurement per ``RunSpec``. These constraints are
    explicit so future environment placement and multi-measurement scheduling
    can be added without silently changing current study semantics.

    ``record_sink`` receives each immutable record as soon as that run finishes,
    before the next run starts. Persistence failures propagate to the caller;
    already-written evidence can therefore survive a later interrupted run.
    """

    resolve_capability_checks(capability_checks)
    analysis_registry: Mapping[str, PairwiseAnalysis] = analyses or {}
    routes = _validate_registries(
        study,
        adapters,
        measurements,
        analysis_registry,
    )
    unknown_configs = sorted(set(measurement_configs) - set(measurements))
    if unknown_configs:
        raise ValueError(
            "measurement_configs contains unregistered measurements: "
            + ", ".join(unknown_configs)
        )

    records: list[RunRecord] = []
    for index, (spec, route) in enumerate(zip(study.runs, routes, strict=True)):
        runtime_name, measurement_name = route
        record = execute_run(
            study_name=study.name,
            run_id=f"run-{index:04d}",
            spec=spec,
            adapter=adapters[runtime_name],
            measurement=measurements[measurement_name],
            measurement_config=measurement_configs.get(measurement_name, {}),
            environment=environment,
            capability_checks=capability_checks,
        )
        records.append(record)
        if record_sink is not None:
            record_sink(record)

    pairwise: list[StudyPairComparison] = []
    for left, right in combinations(records, 2):
        report = compare_runs(left, right, study.comparison)
        pairwise.append(
            StudyPairComparison(
                left_run_id=left.run_id,
                right_run_id=right.run_id,
                report=report,
                analyses=_analyze_pair(
                    left,
                    right,
                    report,
                    study.comparison.analyses,
                    analysis_registry,
                ),
            )
        )

    return StudyExecutionResult(
        study=study,
        records=tuple(records),
        comparisons=tuple(pairwise),
    )
