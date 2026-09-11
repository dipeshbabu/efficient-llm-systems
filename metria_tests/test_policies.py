from __future__ import annotations

import json
from dataclasses import replace

import pytest

from metria import (
    MetricDirection,
    PolicyCriterion,
    PolicyDecision,
    VerificationPolicy,
    policy_from_data,
)
from metria.measurements.trajectory import compare_trajectory_results
from metria.policies import POLICY_SCHEMA, evaluate_policy
from metria.protocols import MeasurementResult
from metria.study_execution import PairwiseAnalysisStatus, StudyPairAnalysis


def _analysis(candidate=(1, 2, 4)):
    def capture(tokens):
        return MeasurementResult(
            evidence={
                "schema": "metria.trajectory_capture.v1",
                "method": "kv_fidelity.decode_time_trajectory",
                "method_version": "0.3.4",
                "n_prompts": 1,
                "prompts": [
                    {
                        "id": "one",
                        "prompt_sha256": "a" * 64,
                        "token_ids": tokens,
                        "token_count": len(tokens),
                    }
                ],
            }
        )

    result = compare_trajectory_results(capture((1, 2, 3)), capture(candidate))
    return StudyPairAnalysis(
        "kv_fidelity.trajectory_match",
        "0.3.4",
        PairwiseAnalysisStatus.COMPLETED,
        result,
    )


def _policy(**bounds):
    return VerificationPolicy(
        (PolicyCriterion("behavior.trajectory_agreement", "0.3.4", **bounds),)
    )


def test_numeric_policy_has_deterministic_pass_and_fail_with_explicit_units():
    analysis = _analysis()
    passed = evaluate_policy(
        _policy(minimum=0.6, maximum=0.7), (analysis,), verification_status="VERIFIED"
    )
    failed = evaluate_policy(
        _policy(minimum=0.9), (analysis,), verification_status="VERIFIED"
    )
    assert passed.status is PolicyDecision.PASS
    assert failed.status is PolicyDecision.FAIL
    row = passed.to_data()["criteria"][0]
    assert row["observed_value"] == pytest.approx(2 / 3)
    assert row["target_identity"]["unit"] == "fraction"
    assert row["observed_source"]["metric"]["unit"] == "score_0_100"
    assert row["observed_source"]["metric"]["direction"] == "higher_is_better"
    assert passed == evaluate_policy(
        _policy(minimum=0.6, maximum=0.7), (analysis,), verification_status="VERIFIED"
    )
    json.dumps(passed.to_data(), allow_nan=False)


@pytest.mark.parametrize("candidate,expected", [((1, 2, 3), True), ((1, 2, 4), False)])
def test_exact_boolean_and_status_checks(candidate, expected):
    policy = VerificationPolicy(
        (
            PolicyCriterion(
                "behavior.all_trajectories_match", "0.3.4", equals=expected
            ),
            PolicyCriterion("analysis.status", "0.3.4", equals="completed"),
        )
    )
    result = evaluate_policy(
        policy, (_analysis(candidate),), verification_status="VERIFIED"
    )
    assert result.status is PolicyDecision.PASS
    assert result.criteria[0]["observed_value"] is expected
    assert result.criteria[1]["observed_value"] == "completed"


@pytest.mark.parametrize(
    "gate", ["NOT_COMPARABLE", "INSUFFICIENT_EVIDENCE", "EXECUTION_FAILED", "unknown"]
)
def test_failed_verifier_gates_never_become_policy_pass(gate):
    result = evaluate_policy(
        _policy(minimum=0), (_analysis(),), verification_status=gate
    )
    assert result.status is PolicyDecision.NOT_EVALUATED
    assert all(
        row["observed_value"] is None and row["result"] == "NOT_EVALUATED"
        for row in result.criteria
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("unit", "tokens"),
        ("method", "different.method"),
        ("version", "9"),
        ("direction", MetricDirection.LOWER_IS_BETTER),
    ],
)
def test_wrong_metric_identity_is_insufficient_not_pass(field, value):
    analysis = _analysis()
    metrics = dict(analysis.result.metrics)
    original = metrics["trajectory_agreement_score"]
    metrics["trajectory_agreement_score"] = replace(
        original, definition=replace(original.definition, **{field: value})
    )
    changed = replace(analysis, result=replace(analysis.result, metrics=metrics))
    result = evaluate_policy(
        _policy(minimum=0), (changed,), verification_status="VERIFIED"
    )
    assert result.status is PolicyDecision.INSUFFICIENT_EVIDENCE
    assert "differs" in result.criteria[0]["reason"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_metric",
        "missing_fact",
        "coverage",
        "aggregation",
        "version",
        "metadata",
        "value",
        "nan",
    ],
)
def test_missing_or_incomplete_policy_evidence_fails_closed(change):
    analysis = _analysis()
    metrics = dict(analysis.result.metrics)
    evidence = dict(analysis.result.evidence)
    policy = _policy(minimum=0)
    if change == "missing_metric":
        del metrics["trajectory_agreement_score"]
    elif change == "missing_fact":
        del evidence["all_trajectories_match"]
        policy = VerificationPolicy(
            (PolicyCriterion("behavior.all_trajectories_match", "0.3.4", equals=False),)
        )
    elif change == "metadata":
        evidence.clear()
    elif change == "version":
        analysis = replace(analysis, version="9")
    else:
        field, value = {
            "coverage": ("coverage", None),
            "aggregation": ("aggregation", "median"),
            "value": ("value", None),
            "nan": ("value", float("nan")),
        }[change]
        metrics["trajectory_agreement_score"] = replace(
            metrics["trajectory_agreement_score"], **{field: value}
        )
    analysis = replace(
        analysis, result=replace(analysis.result, metrics=metrics, evidence=evidence)
    )
    result = evaluate_policy(policy, (analysis,), verification_status="VERIFIED")
    assert result.status is PolicyDecision.INSUFFICIENT_EVIDENCE
    assert result.criteria[0]["observed_value"] is None
    json.dumps(result.to_data(), allow_nan=False)


def test_unknown_or_incomplete_analysis_is_not_policy_evidence():
    policy = _policy(minimum=0)
    for analyses in (
        (),
        (_analysis(), _analysis()),
        (
            StudyPairAnalysis(
                "kv_fidelity.trajectory_match", "0.3.4", PairwiseAnalysisStatus.FAILED
            ),
        ),
    ):
        assert (
            evaluate_policy(policy, analyses, verification_status="VERIFIED").status
            is PolicyDecision.INSUFFICIENT_EVIDENCE
        )


@pytest.mark.parametrize("value", [True, "0.9", float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_numeric_thresholds_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        _policy(minimum=value)


def test_duplicates_contradictions_unknown_targets_and_wrong_units_are_rejected():
    criterion = PolicyCriterion("behavior.trajectory_agreement", "0.3.4", minimum=0.9)
    with pytest.raises(ValueError, match="duplicate"):
        VerificationPolicy((criterion, criterion))
    with pytest.raises(ValueError, match="contradictory"):
        _policy(minimum=0.9, maximum=0.8)
    with pytest.raises(ValueError, match="unknown policy target"):
        PolicyCriterion("analyses.0.metrics.arbitrary", "0.3.4", minimum=0)
    with pytest.raises(ValueError, match="unknown policy target"):
        PolicyCriterion("behavior.trajectory_agreement", "next", minimum=0)
    with pytest.raises(ValueError, match="unit"):
        PolicyCriterion(
            "behavior.trajectory_agreement", "0.3.4", unit="percent", minimum=0.9
        )


@pytest.mark.parametrize(
    "data",
    [
        {"schema": POLICY_SCHEMA, "criteria": []},
        {
            "schema": "unsupported",
            "criteria": [
                {
                    "target": "behavior.trajectory_agreement",
                    "version": "0.3.4",
                    "min": 0,
                }
            ],
        },
        {"schema": POLICY_SCHEMA, "criteria": "arbitrary"},
        {
            "schema": POLICY_SCHEMA,
            "criteria": [
                {
                    "target": "behavior.trajectory_agreement",
                    "version": "0.3.4",
                    "min": None,
                }
            ],
        },
        {
            "schema": POLICY_SCHEMA,
            "criteria": [
                {
                    "target": "behavior.trajectory_agreement",
                    "version": "0.3.4",
                    "expression": "arbitrary",
                }
            ],
        },
    ],
)
def test_policy_schema_is_small_and_strict(data):
    with pytest.raises((TypeError, ValueError)):
        policy_from_data(data)


def test_policy_round_trip_detaches_input_and_preserves_zero_false_and_units():
    data = {
        "schema": POLICY_SCHEMA,
        "criteria": [
            {"target": "behavior.trajectory_agreement", "version": "0.3.4", "min": 0},
            {
                "target": "behavior.all_trajectories_match",
                "version": "0.3.4",
                "equals": False,
            },
        ],
    }
    policy = policy_from_data(data)
    data["criteria"][0]["min"] = 1
    assert policy.criteria[0].minimum == 0
    assert policy.to_data()["criteria"][1]["equals"] is False
    assert policy_from_data(policy.to_data()) == policy
