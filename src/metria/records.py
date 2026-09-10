"""Versioned JSON serialization for durable Metria run evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import (
    MetricDefinition,
    MetricDirection,
    MetricSample,
    MetricSummary,
    RunRecord,
    RunStatus,
)
from .recipes import (
    _json_value,
    _keys,
    _mapping,
    _unique_json_object,
    run_spec_from_data,
    run_spec_to_data,
)

RUN_RECORD_SCHEMA = "metria.run_record.v1"


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return result


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array")
    return value


def _metric_definition_from_data(value: Any, *, name: str) -> MetricDefinition:
    mapping = _mapping(value, name=f"{name}.definition")
    _keys(
        mapping,
        name=f"{name}.definition",
        required=frozenset({"name", "unit", "direction", "method", "version"}),
    )
    direction = _string(mapping["direction"], name=f"{name}.definition.direction")
    try:
        metric_direction = MetricDirection(direction)
    except ValueError as exc:
        supported = ", ".join(item.value for item in MetricDirection)
        raise ValueError(
            f"{name}.definition.direction must be one of: {supported}"
        ) from exc
    return MetricDefinition(
        name=_string(mapping["name"], name=f"{name}.definition.name"),
        unit=_string(mapping["unit"], name=f"{name}.definition.unit"),
        direction=metric_direction,
        method=_string(mapping["method"], name=f"{name}.definition.method"),
        version=_string(mapping["version"], name=f"{name}.definition.version"),
    )


def _metric_sample_from_data(value: Any, *, name: str) -> MetricSample:
    mapping = _mapping(value, name=name)
    _keys(
        mapping,
        name=name,
        required=frozenset({"value"}),
        optional=frozenset({"metadata"}),
    )
    return MetricSample(
        value=_number(mapping["value"], name=f"{name}.value"),
        metadata=_mapping(mapping.get("metadata", {}), name=f"{name}.metadata"),
    )


def _metric_summary_from_data(value: Any, *, key: str) -> MetricSummary:
    name = f"record.metrics.{key}"
    mapping = _mapping(value, name=name)
    _keys(
        mapping,
        name=name,
        required=frozenset({"definition", "value", "samples", "aggregation"}),
        optional=frozenset({"uncertainty", "coverage"}),
    )
    definition = _metric_definition_from_data(mapping["definition"], name=name)
    if definition.name != key:
        raise ValueError(
            f"{name}.definition.name is {definition.name!r}; expected metric key {key!r}"
        )
    raw_samples = _sequence(mapping["samples"], name=f"{name}.samples")
    uncertainty_raw = _mapping(
        mapping.get("uncertainty", {}),
        name=f"{name}.uncertainty",
    )
    uncertainty = {
        field: _number(item, name=f"{name}.uncertainty.{field}")
        for field, item in uncertainty_raw.items()
    }
    coverage_raw = mapping.get("coverage")
    coverage = (
        None if coverage_raw is None else _number(coverage_raw, name=f"{name}.coverage")
    )
    return MetricSummary(
        definition=definition,
        value=_number(mapping["value"], name=f"{name}.value"),
        samples=tuple(
            _metric_sample_from_data(item, name=f"{name}.samples[{index}]")
            for index, item in enumerate(raw_samples)
        ),
        aggregation=_string(mapping["aggregation"], name=f"{name}.aggregation"),
        uncertainty=uncertainty,
        coverage=coverage,
    )


def _metric_summary_to_data(summary: MetricSummary, *, key: str) -> dict[str, Any]:
    name = f"record.metrics.{key}"
    definition_name = _string(
        summary.definition.name,
        name=f"{name}.definition.name",
    )
    if definition_name != key:
        raise ValueError(
            f"record metric key {key!r} does not match definition name "
            f"{definition_name!r}"
        )
    try:
        direction = MetricDirection(summary.definition.direction).value
    except ValueError as exc:
        raise ValueError(f"{name}.definition.direction is invalid") from exc
    uncertainty = {
        field: _number(item, name=f"{name}.uncertainty.{field}")
        for field, item in summary.uncertainty.items()
    }
    return {
        "definition": {
            "name": definition_name,
            "unit": _string(summary.definition.unit, name=f"{name}.definition.unit"),
            "direction": direction,
            "method": _string(
                summary.definition.method,
                name=f"{name}.definition.method",
            ),
            "version": _string(
                summary.definition.version,
                name=f"{name}.definition.version",
            ),
        },
        "value": _number(summary.value, name=f"{name}.value"),
        "samples": [
            {
                "value": _number(sample.value, name=f"{name}.samples[{index}].value"),
                "metadata": _json_value(
                    sample.metadata,
                    path=f"{name}.samples[{index}].metadata",
                ),
            }
            for index, sample in enumerate(summary.samples)
        ],
        "aggregation": _string(summary.aggregation, name=f"{name}.aggregation"),
        "uncertainty": uncertainty,
        "coverage": (
            None
            if summary.coverage is None
            else _number(summary.coverage, name=f"{name}.coverage")
        ),
    }


def run_record_to_data(record: RunRecord) -> dict[str, Any]:
    """Return the strict ``metria.run_record.v1`` JSON-compatible shape."""

    metrics = {
        key: _metric_summary_to_data(summary, key=key)
        for key, summary in record.metrics.items()
    }
    try:
        status = RunStatus(record.status).value
    except ValueError as exc:
        raise ValueError("record.status is invalid") from exc
    return {
        "schema": RUN_RECORD_SCHEMA,
        "record": {
            "study_name": _string(record.study_name, name="record.study_name"),
            "run_id": _string(record.run_id, name="record.run_id"),
            "requested": run_spec_to_data(record.requested, path="record.requested"),
            "resolved": _json_value(record.resolved, path="record.resolved"),
            "observed": _json_value(record.observed, path="record.observed"),
            "status": status,
            "metrics": metrics,
            "evidence": _json_value(record.evidence, path="record.evidence"),
            "events": _json_value(record.events, path="record.events"),
            "artifacts": _json_value(record.artifacts, path="record.artifacts"),
            "provenance": _json_value(record.provenance, path="record.provenance"),
        },
    }


def run_record_from_data(value: Any) -> RunRecord:
    """Validate and parse one strict ``metria.run_record.v1`` object."""

    normalized = _json_value(value, path="run_record")
    envelope = _mapping(normalized, name="run_record")
    _keys(
        envelope,
        name="run_record",
        required=frozenset({"schema", "record"}),
    )
    schema = envelope["schema"]
    if schema != RUN_RECORD_SCHEMA:
        raise ValueError(
            f"unsupported run record schema {schema!r}; expected {RUN_RECORD_SCHEMA!r}"
        )
    record = _mapping(envelope["record"], name="record")
    _keys(
        record,
        name="record",
        required=frozenset(
            {
                "study_name",
                "run_id",
                "requested",
                "resolved",
                "observed",
                "status",
                "metrics",
                "evidence",
                "events",
                "artifacts",
                "provenance",
            }
        ),
    )
    status_raw = _string(record["status"], name="record.status")
    try:
        status = RunStatus(status_raw)
    except ValueError as exc:
        supported = ", ".join(item.value for item in RunStatus)
        raise ValueError(f"record.status must be one of: {supported}") from exc

    metrics_raw = _mapping(record["metrics"], name="record.metrics")
    metrics = {
        key: _metric_summary_from_data(item, key=key)
        for key, item in metrics_raw.items()
    }
    events_raw = _sequence(record["events"], name="record.events")
    artifacts_raw = _sequence(record["artifacts"], name="record.artifacts")
    return RunRecord(
        study_name=_string(record["study_name"], name="record.study_name"),
        run_id=_string(record["run_id"], name="record.run_id"),
        requested=run_spec_from_data(record["requested"]),
        resolved=_mapping(record["resolved"], name="record.resolved"),
        observed=_mapping(record["observed"], name="record.observed"),
        status=status,
        metrics=metrics,
        evidence=_mapping(record["evidence"], name="record.evidence"),
        events=tuple(
            _mapping(item, name=f"record.events[{index}]")
            for index, item in enumerate(events_raw)
        ),
        artifacts=tuple(
            _mapping(item, name=f"record.artifacts[{index}]")
            for index, item in enumerate(artifacts_raw)
        ),
        provenance=_mapping(record["provenance"], name="record.provenance"),
    )


def run_record_to_json(record: RunRecord, *, indent: int | None = 2) -> str:
    """Serialize a run record deterministically with sorted object keys."""

    return json.dumps(
        run_record_to_data(record),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
    )


def _digest_data(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_record_digest(record: RunRecord) -> str:
    """Digest the complete versioned record, including requested intent and IDs."""

    return _digest_data(run_record_to_data(record))


def run_evidence_digest(record: RunRecord) -> str:
    """Digest produced evidence without study/run identifiers or requested intent.

    This digest supports evidence identity and deduplication. It is not a
    universal comparability proof; comparison remains governed by
    ``ComparisonPlan`` and metric-method identity.
    """

    data = run_record_to_data(record)["record"]
    evidence = {
        key: data[key]
        for key in (
            "resolved",
            "observed",
            "status",
            "metrics",
            "evidence",
            "events",
            "artifacts",
            "provenance",
        )
    }
    return _digest_data(evidence)


def load_run_record(path: str | Path) -> RunRecord:
    """Load and validate one UTF-8 JSON run record."""

    record_path = Path(path)
    try:
        raw = json.loads(
            record_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON run record {record_path}: {exc.msg}") from exc
    return run_record_from_data(raw)


def dump_run_record(path: str | Path, record: RunRecord) -> None:
    """Write one deterministic UTF-8 run record with a trailing newline."""

    Path(path).write_text(run_record_to_json(record) + "\n", encoding="utf-8")
