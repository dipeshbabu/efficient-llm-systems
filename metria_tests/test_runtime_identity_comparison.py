from __future__ import annotations

from metria import ComparisonPlan, RunRecord, RunSpec, RunStatus, compare_runs


def _identity(
    *,
    tokenizer: str = "example/tokenizer",
    runtime_version: str = "1",
    source_suffix: str = "a",
    reason: str = "diagnostic a",
) -> dict[str, object]:
    return {
        "schema": "metria.runtime_identity.v1",
        "status": "partial",
        "model": {
            "status": "verified",
            "identifier": "example/model",
            "revision": "model-rev",
            "source": f"model-source-{source_suffix}",
        },
        "tokenizer": {
            "status": "verified",
            "identifier": tokenizer,
            "revision": "tokenizer-rev",
            "source": f"tokenizer-source-{source_suffix}",
        },
        "runtime": {
            "status": "verified",
            "name": "vllm",
            "version": runtime_version,
            "source": f"runtime-source-{source_suffix}",
        },
        "chat_template": {
            "status": "verified",
            "sha256": "a" * 64,
            "source": f"template-source-{source_suffix}",
        },
        "applied": {
            "status": "partial",
            "fields": {"cache.cache_dtype": "fp8"},
            "source": f"applied-source-{source_suffix}",
        },
        "endpoint": {},
        "reasons": (reason,),
    }


def _record(
    *,
    run_id: str,
    requested_runtime_version: str = "1",
    identity: dict[str, object],
) -> RunRecord:
    spec = RunSpec(
        model={
            "id": "example/model",
            "revision": "model-rev",
            "tokenizer_id": "example/tokenizer",
            "tokenizer_revision": "tokenizer-rev",
        },
        runtime={"name": "vllm", "version": requested_runtime_version},
        scenario={"name": "decode"},
        measurements=("identity",),
    )
    return RunRecord(
        study_name="identity-comparison",
        run_id=run_id,
        requested=spec,
        resolved={
            "model": dict(spec.model),
            "runtime": {
                "name": "vllm",
                "version": requested_runtime_version,
            },
        },
        observed={
            "model": dict(spec.model),
            "runtime": {
                "name": "vllm",
                "version": requested_runtime_version,
            },
            "identity": identity,
        },
        status=RunStatus.COMPLETED,
    )


def test_identity_diagnostic_sources_and_reasons_do_not_break_comparability() -> None:
    left = _record(run_id="left", identity=_identity(source_suffix="left"))
    right = _record(
        run_id="right",
        identity=_identity(source_suffix="right", reason="different diagnostic text"),
    )

    report = compare_runs(
        left,
        right,
        ComparisonPlan(
            control=frozenset({"model", "runtime"}),
        ),
    )

    assert report.compatible
    assert report.issues == ()


def test_tokenizer_identity_mismatch_is_governed_by_model_control() -> None:
    left = _record(run_id="left", identity=_identity())
    right = _record(
        run_id="right",
        identity=_identity(tokenizer="wrong/tokenizer"),
    )

    report = compare_runs(
        left,
        right,
        ComparisonPlan(control=frozenset({"model", "runtime"})),
    )

    assert not report.compatible
    issue = next(
        issue
        for issue in report.issues
        if issue.dimension == "observed.identity.tokenizer.identifier"
    )
    assert issue.reason == "controlled dimension differs"


def test_runtime_identity_difference_is_allowed_when_runtime_is_intentional() -> None:
    left = _record(
        run_id="left",
        requested_runtime_version="1",
        identity=_identity(runtime_version="1"),
    )
    right = _record(
        run_id="right",
        requested_runtime_version="2",
        identity=_identity(runtime_version="2", source_suffix="right"),
    )

    report = compare_runs(
        left,
        right,
        ComparisonPlan(
            vary=frozenset({"runtime"}),
            control=frozenset({"model"}),
        ),
    )

    assert report.compatible


def test_identity_authority_difference_still_fails_closed() -> None:
    left_identity = _identity()
    right_identity = _identity()
    right_identity["status"] = "unknown"
    right_identity["model"] = {"status": "unknown", "source": "missing"}
    right_identity["tokenizer"] = {"status": "unknown", "source": "missing"}
    right_identity["runtime"] = {"status": "unknown", "source": "missing"}
    right_identity["chat_template"] = {"status": "unknown", "source": "missing"}
    right_identity["applied"] = {"status": "unknown", "source": "missing"}

    report = compare_runs(
        _record(run_id="left", identity=left_identity),
        _record(run_id="right", identity=right_identity),
        ComparisonPlan(control=frozenset({"model", "runtime"})),
    )

    assert not report.compatible
    assert any(
        issue.dimension == "observed.identity.status"
        and issue.reason == "undeclared comparison-relevant difference"
        for issue in report.issues
    )
