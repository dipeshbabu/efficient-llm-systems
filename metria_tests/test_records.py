from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from metria import (
    MetricDefinition,
    MetricDirection,
    MetricSample,
    MetricSummary,
    RunRecord,
    RunSpec,
    RunStatus,
)
from metria.records import (
    RUN_RECORD_SCHEMA,
    dump_run_record,
    load_run_record,
    run_evidence_digest,
    run_record_digest,
    run_record_from_data,
    run_record_to_data,
    run_record_to_json,
)


def _record(*, run_id: str = "run-0001", requested_runtime: str = "vllm") -> RunRecord:
    definition = MetricDefinition(
        name="latency_ms",
        unit="ms",
        direction=MetricDirection.LOWER_IS_BETTER,
        method="wall_clock",
        version="2",
    )
    return RunRecord(
        study_name="serialization-study",
        run_id=run_id,
        requested=RunSpec(
            model={"id": "example/model", "revision": "abc123"},
            runtime={"name": requested_runtime, "dtype": "float16"},
            scenario={"name": "decode", "max_tokens": 16},
            measurements=("latency",),
            trial_policy={"repetitions": 2},
        ),
        resolved={
            "runtime": {"name": "vllm", "version": "1.2.3"},
            "model": {"id": "example/model", "revision": "abc123"},
        },
        observed={
            "runtime": "vllm",
            "model_revision": "abc123",
            "hardware": {"class": "test-gpu"},
        },
        status=RunStatus.COMPLETED,
        metrics={
            "latency_ms": MetricSummary(
                definition=definition,
                value=11.0,
                samples=(
                    MetricSample(value=10.0, metadata={"trial": 1}),
                    MetricSample(value=12.0, metadata={"trial": 2}),
                ),
                aggregation="mean",
                uncertainty={"stdev": 1.0},
                coverage=1.0,
            )
        },
        evidence={"measurement": {"clock": "monotonic"}},
        events=({"stage": "measurement", "kind": "completed"},),
        artifacts=(
            {
                "kind": "log",
                "uri": "artifact://run-0001/log.txt",
                "sha256": "a" * 64,
            },
        ),
        provenance={
            "recipe_digest": "b" * 64,
            "executor": {"name": "metria.execute_run", "version": "1"},
        },
    )


def test_run_record_v1_round_trip_preserves_typed_evidence() -> None:
    record = _record()
    data = run_record_to_data(record)
    restored = run_record_from_data(data)

    assert data["schema"] == RUN_RECORD_SCHEMA
    assert restored == record
    assert restored.status is RunStatus.COMPLETED
    summary = restored.metrics["latency_ms"]
    assert summary.definition.direction is MetricDirection.LOWER_IS_BETTER
    assert summary.definition.method == "wall_clock"
    assert summary.samples[1].metadata["trial"] == 2


def test_run_record_json_and_digest_are_deterministic() -> None:
    left = _record()
    right = replace(
        left,
        resolved={
            "model": {"revision": "abc123", "id": "example/model"},
            "runtime": {"version": "1.2.3", "name": "vllm"},
        },
        provenance={
            "executor": {"version": "1", "name": "metria.execute_run"},
            "recipe_digest": "b" * 64,
        },
    )

    assert run_record_to_json(left) == run_record_to_json(right)
    assert run_record_digest(left) == run_record_digest(right)
    assert run_evidence_digest(left) == run_evidence_digest(right)


def test_evidence_digest_excludes_run_identity_and_requested_intent() -> None:
    left = _record(run_id="run-a", requested_runtime="vllm")
    right = _record(run_id="run-b", requested_runtime="llamacpp")

    assert run_record_digest(left) != run_record_digest(right)
    assert run_evidence_digest(left) == run_evidence_digest(right)


def test_dump_and_load_run_record_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    record = _record()

    dump_run_record(path, record)

    assert path.read_text(encoding="utf-8").endswith("\n")
    assert load_run_record(path) == record


def test_run_record_parser_rejects_unknown_schema_and_fields() -> None:
    data = run_record_to_data(_record())
    data["schema"] = "metria.run_record.v999"
    with pytest.raises(ValueError, match="unsupported run record schema"):
        run_record_from_data(data)

    data = run_record_to_data(_record())
    data["record"]["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        run_record_from_data(data)


def test_run_record_parser_rejects_nonfinite_values() -> None:
    data = run_record_to_data(_record())
    data["record"]["metrics"]["latency_ms"]["value"] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        run_record_from_data(data)


def test_metric_mapping_key_must_match_definition_name() -> None:
    data = run_record_to_data(_record())
    metric = data["record"]["metrics"].pop("latency_ms")
    data["record"]["metrics"]["renamed"] = metric

    with pytest.raises(ValueError, match="expected metric key"):
        run_record_from_data(data)


@pytest.mark.parametrize(
    ("original", "duplicate"),
    [
        ('"status":', '"status": "failed", "status":'),
        ('"revision":', '"revision": "different", "revision":'),
        ('"value":', '"value": -100.0, "value":'),
    ],
)
def test_record_loader_rejects_duplicate_object_keys(
    tmp_path: Path, original: str, duplicate: str
) -> None:
    path = tmp_path / "ambiguous.json"
    text = run_record_to_json(_record()).replace(original, duplicate, 1)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_run_record(path)
