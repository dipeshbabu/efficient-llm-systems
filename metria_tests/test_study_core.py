from __future__ import annotations

from dataclasses import replace

import pytest

from metria import (
    ComparisonPlan,
    MetricDefinition,
    MetricDirection,
    MetricSample,
    MetricSummary,
    RunRecord,
    RunSpec,
    RunStatus,
    StudySpec,
    TreatmentSpec,
    TreatmentType,
    compare_runs,
)
from metria.protocols import InferenceBatch, InferenceRequest, SupportReport


def _run_spec(*, runtime: str = "llamacpp", treatment: str = "baseline") -> RunSpec:
    return RunSpec(
        model={"id": "example/model", "revision": "abc123"},
        runtime={"name": runtime, "version": "1"},
        scenario={"name": "decode", "context": 4096},
        measurements=("decode_tps", "distribution_drift"),
        treatments=(
            TreatmentSpec(
                name=treatment,
                kind=TreatmentType.RUNTIME_FEATURE,
                config={"kv_dtype": treatment},
            ),
        ),
        trial_policy={"warmup": 1, "repetitions": 3},
        environment_selector={"hardware_class": "gpu-a"},
    )


def _metric(*, method: str = "wall_clock") -> MetricSummary:
    definition = MetricDefinition(
        name="decode_tps",
        unit="tokens/s",
        direction=MetricDirection.HIGHER_IS_BETTER,
        method=method,
    )
    return MetricSummary(
        definition=definition,
        value=42.0,
        samples=(MetricSample(40.0), MetricSample(42.0), MetricSample(44.0)),
        aggregation="mean",
        uncertainty={"stdev": 2.0},
    )


def _record(
    *,
    runtime: str = "llamacpp",
    treatment: str = "baseline",
    method: str = "wall_clock",
    hardware_class: str = "gpu-a",
) -> RunRecord:
    return RunRecord(
        study_name="runtime-study",
        run_id=f"{runtime}-{treatment}",
        requested=_run_spec(runtime=runtime, treatment=treatment),
        resolved={"runtime": runtime, "kv_dtype": treatment},
        observed={"hardware_class": hardware_class, "runtime": runtime},
        status=RunStatus.COMPLETED,
        metrics={"decode_tps": _metric(method=method)},
    )


@pytest.mark.parametrize("role", ["undeclared", "vary", "control", "waiver"])
def test_dotted_path_collisions_cannot_hide_comparison_differences(role: str) -> None:
    left = replace(
        _record(),
        observed={"settings": {"cache": {"dtype": "fp16"}, "cache.dtype": "same"}},
    )
    right = replace(
        left,
        observed={"settings": {"cache": {"dtype": "fp8"}, "cache.dtype": "same"}},
    )
    dimension = "observed.settings.cache.dtype"
    plans = {
        "undeclared": ComparisonPlan(),
        "vary": ComparisonPlan(vary=frozenset({dimension})),
        "control": ComparisonPlan(control=frozenset({dimension})),
        "waiver": ComparisonPlan(waivers={dimension: "intentional difference"}),
    }

    for first, second in ((left, right), (right, left)):
        report = compare_runs(first, second, plans[role])

        assert not report.compatible
        assert any("ambiguous" in issue.reason for issue in report.issues)


@pytest.mark.parametrize(
    "fields",
    [
        {"cache": {"dtype": "fp16"}, "cache.dtype": "fp8"},
        {"cache": {}, "cache.scale": 1.0},
        {"cache": {}, "cache-extra": 1, "cache.scale": 1.0},
        {"cache.dtype": "fp16", "cache.dtype.format": "fp8"},
    ],
)
def test_ambiguous_top_level_evidence_is_not_comparable_even_to_itself(fields) -> None:
    record = replace(_record(), observed=fields)

    report = compare_runs(record, record, ComparisonPlan())

    assert not report.compatible
    assert any("ambiguous" in issue.reason for issue in report.issues)


def test_distinct_dotted_evidence_keys_remain_comparable() -> None:
    record = replace(
        _record(),
        observed={"cache.dtype": "fp16", "cache.scale": 1.0},
    )

    assert compare_runs(record, record, ComparisonPlan()).compatible


def _verification_record(
    *,
    run_id: str = "reference",
    kv_dtype: str = "fp16",
    applied_kv_dtype: str | None = None,
    tokenizer_revision: str = "tok-a",
    resolved_runtime_version: str = "1",
    scenario_temperature: float = 0.0,
    invocation_temperature: float | None = None,
    zone: str = "a",
    method: str = "trajectory",
) -> RunRecord:
    applied = applied_kv_dtype if applied_kv_dtype is not None else kv_dtype
    invocation_temp = (
        scenario_temperature
        if invocation_temperature is None
        else invocation_temperature
    )
    requested = RunSpec(
        model={
            "id": "example/model",
            "revision": "model-a",
            "tokenizer_revision": tokenizer_revision,
        },
        runtime={"name": "vllm", "version": "1"},
        scenario={
            "name": "decode",
            "max_tokens": 16,
            "temperature": scenario_temperature,
        },
        measurements=("trajectory",),
        treatments=(
            TreatmentSpec(
                name="vllm.kv_cache",
                kind=TreatmentType.RUNTIME_FEATURE,
                config={"dtype": kv_dtype},
            ),
        ),
        trial_policy={"warmup": 1, "repetitions": 2},
        environment_selector={"hardware_class": "gpu-a"},
    )
    return RunRecord(
        study_name="verification-study",
        run_id=run_id,
        requested=requested,
        resolved={
            "runtime": {
                "name": "vllm",
                "version": resolved_runtime_version,
                "settings": {"dtype": "bfloat16"},
            },
            "model": {
                "id": "example/model",
                "revision": "model-a",
                "tokenizer_revision": tokenizer_revision,
            },
            "scenario": requested.scenario,
            "kv_cache": {"dtype": kv_dtype},
            "support": {
                "runtime_probe": "excluded from comparison identity",
            },
        },
        observed={
            "runtime": {
                "name": "vllm",
                "version": resolved_runtime_version,
            },
            "model": {
                "id": "example/model",
                "revision": "model-a",
                "tokenizer_revision": tokenizer_revision,
            },
            "configured": {
                "runtime": {"dtype": "bfloat16"},
                "kv_cache": {"dtype": kv_dtype},
            },
            "applied": {
                "status": "introspected",
                "fields": {"cache.cache_dtype": applied},
            },
            "hardware_class": "gpu-a",
            "environment": {"zone": zone},
            "invocations": (
                {
                    "prompt_sha256": "prompt-a",
                    "rendered_prompt_sha256": "rendered-a",
                    "system_sha256": None,
                    "generation": {
                        "max_tokens": 16,
                        "temperature": invocation_temp,
                        "seed": 42,
                    },
                    "output_tokens": 12,
                },
            ),
            "reset_count": 1,
            "closed": True,
            "cleanup": {"mode": "public_shutdown"},
        },
        status=RunStatus.COMPLETED,
        metrics={"decode_tps": _metric(method=method)},
    )


def test_comparison_plan_rejects_overlapping_dimensions() -> None:
    with pytest.raises(ValueError, match="cannot overlap"):
        ComparisonPlan(vary=frozenset({"runtime"}), control=frozenset({"runtime"}))


def test_comparison_plan_rejects_nested_role_overlap() -> None:
    with pytest.raises(ValueError, match="cannot overlap"):
        ComparisonPlan(
            control=frozenset({"model"}),
            vary=frozenset({"model.revision"}),
        )


def test_comparison_plan_requires_non_empty_waiver_reasons() -> None:
    with pytest.raises(ValueError, match="waiver reasons"):
        ComparisonPlan(waivers={"observed.environment.zone": "   "})


def test_comparison_plan_requires_waiver_mapping() -> None:
    with pytest.raises(TypeError, match="waivers must be a mapping"):
        ComparisonPlan(waivers=("observed.environment.zone",))  # type: ignore[arg-type]


def test_study_requires_a_run() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        StudySpec(
            name="empty",
            runs=(),
            comparison=ComparisonPlan(),
        )


def test_runtime_can_be_an_intentional_variable() -> None:
    plan = ComparisonPlan(
        vary=frozenset({"runtime", "treatments", "resolved.kv_dtype"}),
        control=frozenset({"model", "scenario", "trial_policy"}),
        block_by=frozenset({"observed.hardware_class"}),
    )

    report = compare_runs(
        _record(runtime="llamacpp", treatment="fp16"),
        _record(runtime="vllm", treatment="fp8"),
        plan,
    )

    assert report.compatible
    assert report.comparable_metrics == ("decode_tps",)


def test_controlled_dimension_mismatch_is_reported() -> None:
    left = _record()
    right = RunRecord(
        study_name=left.study_name,
        run_id="different-model",
        requested=RunSpec(
            model={"id": "other/model", "revision": "def456"},
            runtime=left.requested.runtime,
            scenario=left.requested.scenario,
            measurements=left.requested.measurements,
            treatments=left.requested.treatments,
            trial_policy=left.requested.trial_policy,
            environment_selector=left.requested.environment_selector,
        ),
        resolved=left.resolved,
        observed=left.observed,
        status=RunStatus.COMPLETED,
        metrics=left.metrics,
    )
    plan = ComparisonPlan(
        vary=frozenset({"treatments"}),
        control=frozenset({"model", "scenario"}),
    )

    report = compare_runs(left, right, plan)

    assert not report.compatible
    assert report.issues[0].dimension == "model"
    assert report.issues[0].reason == "controlled dimension differs"


def test_blocking_dimension_must_match_for_pairwise_comparison() -> None:
    plan = ComparisonPlan(
        vary=frozenset({"runtime"}),
        block_by=frozenset({"observed.hardware_class"}),
    )

    report = compare_runs(
        _record(runtime="llamacpp", hardware_class="gpu-a"),
        _record(runtime="vllm", hardware_class="gpu-b"),
        plan,
    )

    assert not report.compatible
    assert report.issues[0].reason == "runs belong to different comparison blocks"


def test_missing_nested_dimension_cannot_prove_comparability() -> None:
    plan = ComparisonPlan(
        vary=frozenset({"runtime"}),
        block_by=frozenset({"observed.hardware_class"}),
    )
    left = replace(_record(runtime="llamacpp"), observed={"runtime": "llamacpp"})
    right = replace(_record(runtime="vllm"), observed={"runtime": "vllm"})

    report = compare_runs(left, right, plan)

    assert not report.compatible
    assert report.issues[0].left == "<missing>"
    assert report.issues[0].right == "<missing>"
    assert report.issues[0].reason == "required block dimension is missing"


def test_explicit_none_is_distinct_from_a_missing_dimension() -> None:
    plan = ComparisonPlan(
        vary=frozenset({"runtime"}),
        block_by=frozenset({"observed.hardware_class"}),
    )
    left = replace(
        _record(runtime="llamacpp"),
        observed={"runtime": "llamacpp", "hardware_class": None},
    )
    right = replace(
        _record(runtime="vllm"),
        observed={"runtime": "vllm", "hardware_class": None},
    )

    report = compare_runs(left, right, plan)

    assert report.compatible


def test_undeclared_model_difference_fails_closed() -> None:
    left = _verification_record()
    right = replace(
        _verification_record(run_id="candidate"),
        requested=replace(
            _verification_record().requested,
            model={
                "id": "other/model",
                "revision": "model-b",
                "tokenizer_revision": "tok-a",
            },
        ),
    )

    report = compare_runs(left, right, ComparisonPlan())

    dimensions = {issue.dimension for issue in report.issues}
    assert not report.compatible
    assert "model.id" in dimensions
    assert "model.revision" in dimensions


def test_undeclared_tokenizer_difference_fails_closed() -> None:
    report = compare_runs(
        _verification_record(tokenizer_revision="tok-a"),
        _verification_record(
            run_id="candidate",
            tokenizer_revision="tok-b",
        ),
        ComparisonPlan(),
    )

    dimensions = {issue.dimension for issue in report.issues}
    assert not report.compatible
    assert "model.tokenizer_revision" in dimensions
    assert "resolved.model.tokenizer_revision" in dimensions
    assert "observed.model.tokenizer_revision" in dimensions


def test_undeclared_resolved_runtime_build_difference_fails_closed() -> None:
    left = _verification_record()
    right = replace(
        _verification_record(run_id="candidate"),
        resolved={
            **dict(_verification_record().resolved),
            "runtime": {
                "name": "vllm",
                "version": "2",
                "settings": {"dtype": "bfloat16"},
            },
        },
        observed={
            **dict(_verification_record().observed),
            "runtime": {"name": "vllm", "version": "2"},
        },
    )

    report = compare_runs(left, right, ComparisonPlan())

    dimensions = {issue.dimension for issue in report.issues}
    assert not report.compatible
    assert "resolved.runtime.version" in dimensions
    assert "observed.runtime.version" in dimensions


def test_generation_identity_difference_fails_closed() -> None:
    report = compare_runs(
        _verification_record(),
        _verification_record(
            run_id="candidate",
            scenario_temperature=0.7,
            invocation_temperature=0.7,
        ),
        ComparisonPlan(),
    )

    dimensions = {issue.dimension for issue in report.issues}
    assert not report.compatible
    assert "scenario.temperature" in dimensions
    assert "resolved.scenario.temperature" in dimensions
    assert "observed.invocations.0.generation.temperature" in dimensions


def test_invocation_generation_path_can_be_intentionally_varied() -> None:
    report = compare_runs(
        _verification_record(),
        _verification_record(
            run_id="candidate",
            scenario_temperature=0.7,
            invocation_temperature=0.7,
        ),
        ComparisonPlan(
            vary=frozenset(
                {
                    "scenario.temperature",
                    "resolved.scenario.temperature",
                    "observed.invocations.0.generation.temperature",
                }
            )
        ),
    )

    assert report.compatible


def test_missing_applied_evidence_fails_even_for_intentional_variation() -> None:
    left = _verification_record()
    left_observed = dict(left.observed)
    left_observed.pop("applied")
    left = replace(left, observed=left_observed)

    report = compare_runs(
        left,
        _verification_record(run_id="candidate", kv_dtype="fp8"),
        ComparisonPlan(vary=frozenset({"observed.applied.fields.cache.cache_dtype"})),
    )

    applied_issue = next(
        issue
        for issue in report.issues
        if issue.dimension == "observed.applied.fields.cache.cache_dtype"
    )
    assert not report.compatible
    assert applied_issue.left == "<missing>"
    assert applied_issue.reason == "required vary dimension is missing"


def test_nested_controlled_path_reports_precise_mismatch() -> None:
    left = _verification_record()
    right_resolved = dict(left.resolved)
    right_model = dict(left.resolved["model"])
    right_model["revision"] = "model-b"
    right_resolved["model"] = right_model
    right = replace(
        _verification_record(run_id="candidate"),
        resolved=right_resolved,
    )

    report = compare_runs(
        left,
        right,
        ComparisonPlan(control=frozenset({"resolved.model.revision"})),
    )

    issue = next(
        issue for issue in report.issues if issue.dimension == "resolved.model.revision"
    )
    assert not report.compatible
    assert issue.reason == "controlled dimension differs"


def test_intentional_treatment_change_can_cover_applied_nested_evidence() -> None:
    plan = ComparisonPlan(
        vary=frozenset(
            {
                "treatments",
                "resolved.kv_cache",
                "observed.configured.kv_cache",
                "observed.applied.fields.cache.cache_dtype",
            }
        ),
        control=frozenset(
            {
                "model",
                "runtime",
                "scenario",
                "measurements",
                "trial_policy",
                "environment_selector",
            }
        ),
        block_by=frozenset({"observed.hardware_class"}),
    )

    report = compare_runs(
        _verification_record(kv_dtype="fp16"),
        _verification_record(
            run_id="candidate",
            kv_dtype="fp8",
        ),
        plan,
    )

    assert report.compatible
    assert report.issues == ()


def test_explicit_waiver_retains_reason_without_hiding_the_difference() -> None:
    report = compare_runs(
        _verification_record(zone="a"),
        _verification_record(run_id="candidate", zone="b"),
        ComparisonPlan(
            waivers={
                "observed.environment.zone": "cross-zone qualification",
            }
        ),
    )

    assert report.compatible
    assert len(report.waived_differences) == 1
    waiver = report.waived_differences[0]
    assert waiver.dimension == "observed.environment.zone"
    assert waiver.left == "a"
    assert waiver.right == "b"
    assert waiver.reason == "cross-zone qualification"


def test_waiver_path_must_resolve_on_both_runs() -> None:
    report = compare_runs(
        _verification_record(),
        _verification_record(run_id="candidate"),
        ComparisonPlan(
            waivers={
                "observed.applied.fields.nonexistent": "temporary exception",
            }
        ),
    )

    issue = next(
        item
        for item in report.issues
        if item.dimension == "observed.applied.fields.nonexistent"
    )
    assert not report.compatible
    assert issue.left == "<missing>"
    assert issue.right == "<missing>"
    assert issue.reason == "required waiver dimension is missing"


def test_waiver_does_not_make_metric_method_mismatch_compatible() -> None:
    report = compare_runs(
        _verification_record(zone="a", method="wall_clock"),
        _verification_record(
            run_id="candidate",
            zone="b",
            method="estimated",
        ),
        ComparisonPlan(
            waivers={
                "observed.environment.zone": "cross-zone qualification",
            }
        ),
    )

    assert not report.compatible
    assert report.waived_differences
    assert "decode_tps" in report.incompatible_metrics


def test_comparison_reports_all_undeclared_differences_in_one_pass() -> None:
    left = _verification_record()
    right = _verification_record(
        run_id="candidate",
        tokenizer_revision="tok-b",
        resolved_runtime_version="2",
        scenario_temperature=0.7,
        invocation_temperature=0.7,
    )

    report = compare_runs(left, right, ComparisonPlan())

    dimensions = {issue.dimension for issue in report.issues}
    assert {
        "model.tokenizer_revision",
        "resolved.runtime.version",
        "scenario.temperature",
        "observed.invocations.0.generation.temperature",
    }.issubset(dimensions)


def test_runtime_probe_support_evidence_is_not_a_comparison_dimension() -> None:
    left = _verification_record()
    right_resolved = dict(left.resolved)
    right_resolved["support"] = {"runtime_probe": "different diagnostics"}
    right = replace(
        _verification_record(run_id="candidate"),
        resolved=right_resolved,
    )

    report = compare_runs(left, right, ComparisonPlan())

    assert report.compatible


def test_output_dependent_invocation_bookkeeping_does_not_invalidate_pair() -> None:
    left = _verification_record()
    right_observed = dict(left.observed)
    right_invocation = dict(left.observed["invocations"][0])
    right_invocation["output_tokens"] = 99
    right_invocation["duration_seconds"] = 2.5
    right_observed["invocations"] = (right_invocation,)
    right_observed["reset_count"] = 7
    right_observed["closed"] = False
    right_observed["cleanup"] = {"mode": "different"}
    right = replace(
        _verification_record(run_id="candidate"),
        observed=right_observed,
    )

    report = compare_runs(left, right, ComparisonPlan())

    assert report.compatible


def test_metric_method_identity_prevents_invalid_raw_comparison() -> None:
    plan = ComparisonPlan(vary=frozenset({"runtime"}))

    report = compare_runs(
        _record(runtime="llamacpp", method="full_vocabulary_kld"),
        _record(runtime="vllm", method="top_k_kld"),
        plan,
    )

    assert not report.compatible
    assert report.comparable_metrics == ()
    assert "decode_tps" in report.incompatible_metrics


def test_requested_resolved_and_observed_state_are_distinct() -> None:
    record = _record(treatment="fp8")

    assert record.requested.treatments[0].config["kv_dtype"] == "fp8"
    assert record.resolved["kv_dtype"] == "fp8"
    assert record.observed["runtime"] == "llamacpp"


def test_evidence_models_detach_and_deeply_freeze_nested_values() -> None:
    config = {"kv": {"flags": ["fp8", "scaled"]}}
    treatment = TreatmentSpec(
        name="fp8",
        kind=TreatmentType.RUNTIME_FEATURE,
        config=config,
    )
    config["kv"]["flags"].append("mutated")

    assert treatment.config["kv"]["flags"] == ("fp8", "scaled")
    with pytest.raises(TypeError):
        treatment.config["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        treatment.config["kv"]["new"] = "value"  # type: ignore[index]


def test_run_record_evidence_is_detached_from_source_mappings() -> None:
    resolved = {"runtime": {"flags": ["--ctx-size", "4096"]}}
    observed = {"hardware": {"gpus": ["gpu-a"]}}
    record = RunRecord(
        study_name="immutability",
        run_id="one",
        requested=_run_spec(),
        resolved=resolved,
        observed=observed,
        status=RunStatus.COMPLETED,
        events=({"stage": "launch", "details": {"attempts": [1]}},),
        artifacts=({"path": "run.json", "hashes": ["abc"]},),
        provenance={"commits": ["deadbeef"]},
    )
    resolved["runtime"]["flags"].append("--mutated")
    observed["hardware"]["gpus"].append("gpu-b")

    assert record.resolved["runtime"]["flags"] == ("--ctx-size", "4096")
    assert record.observed["hardware"]["gpus"] == ("gpu-a",)
    assert record.events[0]["details"]["attempts"] == (1,)
    assert record.artifacts[0]["hashes"] == ("abc",)
    assert record.provenance["commits"] == ("deadbeef",)


def test_runtime_payloads_are_deeply_immutable() -> None:
    generation = {"stop": ["</s>"]}
    request = InferenceRequest(prompt="hello", generation=generation)
    batch = InferenceBatch(
        outputs=["world"],
        captures={"token_ids": [1, 2, 3]},
        metadata={"runtime": {"version": ["1"]}},
    )
    support = SupportReport(
        status="supported",
        evidence={"features": ["logprobs"]},
    )
    generation["stop"].append("mutated")

    assert request.generation["stop"] == ("</s>",)
    assert batch.outputs == ("world",)
    assert batch.captures["token_ids"] == (1, 2, 3)
    assert batch.metadata["runtime"]["version"] == ("1",)
    assert support.evidence["features"] == ("logprobs",)


def test_unsupported_live_objects_are_rejected_from_evidence() -> None:
    with pytest.raises(TypeError, match="unsupported evidence value type"):
        TreatmentSpec(
            name="bad",
            kind=TreatmentType.INSTRUMENTATION,
            config={"live_object": object()},
        )
