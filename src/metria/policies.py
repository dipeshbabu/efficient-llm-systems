"""Versioned, typed acceptance criteria for verified inference comparisons."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from ._freeze import freeze_mapping
from .models import MetricDefinition, MetricDirection, MetricSummary
from .study_execution import PairwiseAnalysisStatus, StudyPairAnalysis

POLICY_SCHEMA = "metria.verification_policy.v1"
POLICY_RESULT_SCHEMA = "metria.policy_evaluation.v1"
_ANALYSIS = "kv_fidelity.trajectory_match"
_METHOD_VERSION = "0.3.4"


class PolicyDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclass(frozen=True)
class _Target:
    name: str
    kind: str
    unit: str
    metric: MetricDefinition | None = None
    aggregation: str | None = None
    divisor: float = 1.0
    fact: str | None = None

    def to_data(self) -> dict[str, Any]:
        source: dict[str, Any] = {
            "analysis": {"name": _ANALYSIS, "version": _METHOD_VERSION}
        }
        if self.metric is not None:
            source["metric"] = _metric_identity(self.metric)
            source["aggregation"] = self.aggregation
            source["normalization_divisor"] = self.divisor
        if self.fact is not None:
            source["fact"] = self.fact
        return {
            "name": self.name,
            "version": _METHOD_VERSION,
            "kind": self.kind,
            "unit": self.unit,
            "direction": self.metric.direction.value if self.metric else "descriptive",
            "source": source,
        }


_TARGETS = MappingProxyType(
    {
        target.name: target
        for target in (
            _Target(
                "behavior.trajectory_agreement",
                "number",
                "fraction",
                MetricDefinition(
                    "trajectory_agreement_score",
                    "score_0_100",
                    MetricDirection.HIGHER_IS_BETTER,
                    _ANALYSIS,
                    _METHOD_VERSION,
                ),
                "weighted_by_max_trajectory_length",
                100.0,
            ),
            _Target(
                "behavior.full_match_rate",
                "number",
                "fraction",
                MetricDefinition(
                    "trajectory_full_match_rate",
                    "fraction",
                    MetricDirection.HIGHER_IS_BETTER,
                    _ANALYSIS,
                    _METHOD_VERSION,
                ),
                "mean",
            ),
            _Target(
                "behavior.all_trajectories_match",
                "boolean",
                "boolean",
                fact="all_trajectories_match",
            ),
            _Target("analysis.status", "status", "status", fact="analysis_status"),
        )
    }
)


def _target(name: str, version: str) -> _Target:
    if not isinstance(name, str) or not isinstance(version, str):
        raise TypeError("policy target and version must be strings")
    if name not in _TARGETS or version != _METHOD_VERSION:
        raise ValueError(f"unknown policy target or version: {name!r}@{version!r}")
    return _TARGETS[name]


def _threshold(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if not 0 <= number <= 1:
        raise ValueError(f"{name} must be within [0, 1] for a fraction target")
    return number


@dataclass(frozen=True)
class PolicyCriterion:
    """One named/versioned target with numeric bounds or an exact expectation."""

    target: str
    version: str
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    equals: bool | str | None = None

    def __post_init__(self) -> None:
        target = _target(self.target, self.version)
        if self.unit is not None and self.unit != target.unit:
            raise ValueError(
                f"policy target {self.target!r} requires unit {target.unit!r}"
            )
        object.__setattr__(self, "unit", target.unit)
        if target.kind == "number":
            if self.equals is not None or (
                self.minimum is None and self.maximum is None
            ):
                raise ValueError(
                    "numeric criteria require min/max bounds and do not support equals"
                )
            for field in ("minimum", "maximum"):
                value = getattr(self, field)
                if value is not None:
                    object.__setattr__(self, field, _threshold(value, field))
            if (
                self.minimum is not None
                and self.maximum is not None
                and self.minimum > self.maximum
            ):
                raise ValueError("contradictory policy bounds: min exceeds max")
        else:
            if self.minimum is not None or self.maximum is not None:
                raise ValueError("boolean/status criteria require equals, not min/max")
            if target.kind == "boolean" and not isinstance(self.equals, bool):
                raise TypeError("boolean policy equals must be a boolean")
            if target.kind == "status":
                if not isinstance(self.equals, str):
                    raise TypeError("status policy equals must be a status string")
                if self.equals not in {
                    status.value for status in PairwiseAnalysisStatus
                }:
                    raise ValueError(
                        "analysis status equals must be completed, failed, or skipped"
                    )

    def to_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "target": self.target,
            "version": self.version,
            "unit": self.unit,
        }
        for name, value in (
            ("min", self.minimum),
            ("max", self.maximum),
            ("equals", self.equals),
        ):
            if value is not None:
                data[name] = value
        return data


@dataclass(frozen=True)
class VerificationPolicy:
    """A conjunction of explicit criteria; no default acceptance thresholds."""

    criteria: tuple[PolicyCriterion, ...]
    schema: str = POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != POLICY_SCHEMA:
            raise ValueError(f"unsupported policy schema: {self.schema!r}")
        criteria = tuple(self.criteria)
        if not criteria:
            raise ValueError("a verification policy requires at least one criterion")
        seen = set()
        for criterion in criteria:
            if not isinstance(criterion, PolicyCriterion):
                raise TypeError("policy criteria must be PolicyCriterion values")
            if criterion.target in seen:
                raise ValueError(
                    f"duplicate policy criterion target: {criterion.target}"
                )
            seen.add(criterion.target)
        object.__setattr__(self, "criteria", criteria)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "criteria": [criterion.to_data() for criterion in self.criteria],
        }

    @property
    def required_analyses(self) -> frozenset[str]:
        return frozenset({_ANALYSIS})


def _fields(
    value: Any, name: str, required: set[str], optional: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed object")
    if missing := required - set(value):
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if unknown := set(value) - required - optional:
        raise ValueError(
            f"{name} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    return value


def policy_from_data(value: Any) -> VerificationPolicy:
    """Validate the policy schema, typed targets, units, and thresholds."""
    data = _fields(value, "policy", {"schema", "criteria"}, set())
    raw = data["criteria"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("policy.criteria must be an array")
    criteria = []
    for index, item in enumerate(raw):
        entry = _fields(
            item,
            f"policy.criteria[{index}]",
            {"target", "version"},
            {"unit", "min", "max", "equals"},
        )
        if any(
            entry[key] is None
            for key in ("unit", "min", "max", "equals")
            if key in entry
        ):
            raise TypeError("policy criterion fields cannot be null")
        criteria.append(
            PolicyCriterion(
                entry["target"],
                entry["version"],
                entry.get("unit"),
                entry.get("min"),
                entry.get("max"),
                entry.get("equals"),
            )
        )
    return VerificationPolicy(tuple(criteria), schema=data["schema"])


def _metric_identity(definition: MetricDefinition) -> dict[str, str]:
    name, unit, direction, method, version = definition.identity
    return {
        "name": name,
        "unit": unit,
        "direction": direction,
        "method": method,
        "version": version,
    }


def _finite_observation(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _observe(
    target: _Target, analyses: Sequence[StudyPairAnalysis]
) -> tuple[Any, dict[str, Any], str | None]:
    matches = [analysis for analysis in analyses if analysis.name == _ANALYSIS]
    if len(matches) != 1:
        return None, {}, "required analysis is missing or ambiguous"
    analysis = matches[0]
    identity: dict[str, Any] = {
        "analysis": {"name": analysis.name, "version": analysis.version}
    }
    if analysis.version != _METHOD_VERSION:
        return None, identity, "analysis version does not match the policy target"
    if (
        analysis.status is not PairwiseAnalysisStatus.COMPLETED
        or analysis.result is None
    ):
        return None, identity, "required analysis is incomplete"
    if target.kind == "status":
        return analysis.status.value, identity, None
    result = analysis.result
    if (
        result.evidence.get("schema"),
        result.evidence.get("method"),
        result.evidence.get("method_version"),
    ) != ("metria.trajectory_comparison.v1", _ANALYSIS, _METHOD_VERSION):
        return (
            None,
            identity,
            "analysis methodology evidence is missing or incompatible",
        )
    if target.kind == "boolean":
        if target.fact is None:
            raise RuntimeError("boolean policy targets require a named fact")
        value = result.evidence.get(target.fact)
        if not isinstance(value, bool):
            return None, identity, "required boolean analysis evidence is missing"
        identity["fact"] = target.fact
        return value, identity, None
    definition = target.metric
    if definition is None:
        raise RuntimeError("numeric policy targets require a metric identity")
    metric = result.metrics.get(definition.name)
    if not isinstance(metric, MetricSummary):
        return None, identity, "required policy metric is missing"
    try:
        identity["metric"] = _metric_identity(metric.definition)
    except (AttributeError, TypeError, ValueError):
        return None, identity, "policy metric identity is invalid"
    coverage = _finite_observation(metric.coverage)
    raw = _finite_observation(metric.value)
    identity.update(
        {"aggregation": metric.aggregation, "coverage": coverage, "raw_value": raw}
    )
    if (
        metric.definition.identity != definition.identity
        or metric.aggregation != target.aggregation
    ):
        return (
            None,
            identity,
            "metric unit, direction, method, version, or aggregation differs",
        )
    if coverage != 1.0:
        return None, identity, "policy metric coverage is incomplete or unknown"
    if raw is None:
        return (
            None,
            identity,
            "policy metric value is missing, non-finite, or not numeric",
        )
    number = raw / target.divisor
    if not math.isfinite(number) or not 0 <= number <= 1:
        return None, identity, "policy metric value is non-finite or outside its domain"
    return number, identity, None


@dataclass(frozen=True)
class PolicyEvaluation:
    status: PolicyDecision
    criteria: tuple[Mapping[str, Any], ...]
    gate: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "criteria", tuple(freeze_mapping(row) for row in self.criteria)
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": POLICY_RESULT_SCHEMA,
            "policy_schema": POLICY_SCHEMA,
            "status": self.status.value,
            "gate": self.gate,
            "criteria": [_plain(row) for row in self.criteria],
        }


def evaluate_policy(
    policy: VerificationPolicy,
    analyses: Sequence[StudyPairAnalysis],
    *,
    verification_status: str,
) -> PolicyEvaluation:
    """Evaluate only after the verifier established complete, comparable evidence."""
    rows = []
    for criterion in policy.criteria:
        target = _target(criterion.target, criterion.version)
        value: Any = None
        source: dict[str, Any] = {}
        reason: str | None = f"verification gate: {verification_status}"
        status = PolicyDecision.NOT_EVALUATED
        if verification_status == "VERIFIED":
            value, source, reason = _observe(target, analyses)
            if reason is not None:
                status = PolicyDecision.INSUFFICIENT_EVIDENCE
            else:
                passed = (
                    value == criterion.equals
                    if target.kind != "number"
                    else (
                        (criterion.minimum is None or value >= criterion.minimum)
                        and (criterion.maximum is None or value <= criterion.maximum)
                    )
                )
                status = PolicyDecision.PASS if passed else PolicyDecision.FAIL
        rows.append(
            {
                "criterion": criterion.to_data(),
                "target_identity": target.to_data(),
                "observed_value": value,
                "observed_source": source,
                "result": status.value,
                "reason": reason,
            }
        )
    if verification_status != "VERIFIED":
        decision = PolicyDecision.NOT_EVALUATED
    elif any(
        row["result"] == PolicyDecision.INSUFFICIENT_EVIDENCE.value for row in rows
    ):
        decision = PolicyDecision.INSUFFICIENT_EVIDENCE
    elif all(row["result"] == PolicyDecision.PASS.value for row in rows):
        decision = PolicyDecision.PASS
    else:
        decision = PolicyDecision.FAIL
    return PolicyEvaluation(decision, tuple(rows), verification_status)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def render_policy_evaluation(data: Mapping[str, Any]) -> list[str]:
    """Render the same criteria and outcomes retained in the JSON report."""
    import json

    lines = [f"  Policy: {data['status']}", "", "Policy checks:"]
    for row in data["criteria"]:
        criterion = row["criterion"]
        constraints = []
        for key, symbol in (("min", ">="), ("max", "<="), ("equals", "==")):
            if key in criterion:
                constraints.append(f"{symbol} {json.dumps(criterion[key])}")
        value = (
            "unavailable"
            if row["observed_value"] is None
            else json.dumps(row["observed_value"])
        )
        lines.append(
            f"  {criterion['target']}@{criterion['version']} {' and '.join(constraints)}: {row['result']} (observed {value} {criterion['unit']})"
        )
        if row["reason"]:
            lines.append(f"    {row['reason']}")
    lines.append(
        "  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment."
    )
    return lines
