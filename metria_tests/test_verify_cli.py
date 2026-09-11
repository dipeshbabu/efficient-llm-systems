from __future__ import annotations

import hashlib
import io
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from metria import (
    Capability,
    CapabilityCheck,
    CapabilityCheckRegistry,
    CapabilityCheckResult,
    ComparisonPlan,
    HardwareFingerprint,
    PolicyCriterion,
    RunSpec,
    RunStatus,
    StudyRecipe,
    StudySpec,
    SupportLevel,
    VerificationPolicy,
    dump_study_recipe,
    verification,
)
from metria.cli import main
from metria.measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from metria.records import load_run_record, run_record_digest
from metria.runtimes import llamacpp


@pytest.fixture
def local_case(tmp_path, monkeypatch):
    binary = tmp_path / (
        "llama-completion.exe" if llamacpp.os.name == "nt" else "llama-completion"
    )
    binary.write_bytes(b"qualified fixture provider")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"immutable model fixture")
    runs = tuple(
        RunSpec(
            model={
                "path": str(model),
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            },
            runtime={
                "name": "llamacpp",
                "bin_dir": str(tmp_path),
                "threads": threads,
                "threads_batch": 1,
                "n_gpu_layers": 0,
                "flash_attention": False,
            },
            scenario={
                "context": 256,
                "max_tokens": 3,
                "temperature": 0.0,
                "seed": 42,
                "chat_template": False,
            },
            measurements=(TokenTrajectoryProtocol.name,),
        )
        for threads in (1, 2)
    )
    recipe = StudyRecipe(
        study=StudySpec(
            name="thread-change",
            runs=runs,
            comparison=ComparisonPlan(
                vary=frozenset(
                    {
                        "runtime.threads",
                        "resolved.runtime.threads",
                        "observed.runtime.threads",
                        "observed.identity.applied.fields.threads",
                    }
                ),
                control=frozenset({"model", "scenario", "measurements"}),
                analyses=(TrajectoryAgreementAnalysis.name,),
            ),
        ),
        measurement_configs={
            TokenTrajectoryProtocol.name: {
                "prompts": [
                    {"id": "p1", "prompt": "private prompt that must not appear"}
                ]
            }
        },
        environment={
            "llama_cpp_token_ids_capture_sha256": hashlib.sha256(
                binary.read_bytes()
            ).hexdigest()
        },
    )
    case = {
        "recipe": recipe,
        "path": tmp_path / "study.json",
        "output": tmp_path / "verification",
        "mode": "normal",
        "calls": [],
        "binary": binary,
    }
    monkeypatch.setattr(
        verification,
        "capture_hardware_fingerprint",
        lambda: HardwareFingerprint(platform={"system": "fixture"}),
    )
    monkeypatch.setattr(llamacpp, "time", SimpleNamespace(monotonic=lambda: 1.0))

    def run(argv, **kwargs):
        threads = int(argv[argv.index("-t") + 1])
        case["calls"].append(threads)
        if threads == 2:
            assert (case["output"] / "reference.run.json").is_file()
        if case["mode"] == "timeout" and threads == 2:
            raise subprocess.TimeoutExpired(argv, 1)
        if case["mode"] == "interrupt" and threads == 2:
            raise KeyboardInterrupt("private interrupt text")
        if case["mode"] == "failure" and threads == 2:
            return subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="private failure text"
            )
        trajectory = Path(kwargs["env"]["KV_FIDELITY_TRAJECTORY"])
        tokens = [] if case["mode"] == "empty" else [11, 12, 13 if threads == 1 else 14]
        trajectory.write_text(
            "".join(
                json.dumps({"step": index, "token_id": token}) + "\n"
                for index, token in enumerate(tokens)
            ),
            encoding="utf-8",
        )
        if case["mode"] != "legacy":
            facts = {
                "schema": "metria.llamacpp_capture.v1",
                "threads": threads,
                "threads_batch": 1,
                "context": 256,
                "vocab_size": 512,
                "chat_template_applied": False,
            }
            if case["mode"] == "ignored_threads":
                facts["threads"] = 1
            if case["mode"] == "different_vocab" and threads == 2:
                facts["vocab_size"] = 1024
            Path(str(trajectory) + ".runtime.json").write_text(
                json.dumps(facts), encoding="utf-8"
            )
        return subprocess.CompletedProcess(
            argv, 0, stdout="private generated text", stderr=""
        )

    monkeypatch.setattr(llamacpp.subprocess, "run", run)
    return case


def _invoke(case, *, json_output=True):
    dump_study_recipe(case["path"], case["recipe"])
    stdout, stderr = io.StringIO(), io.StringIO()
    args = ["verify", str(case["path"]), "--output", str(case["output"])]
    if json_output:
        args.append("--json")
    status = main(args, stdout=stdout, stderr=stderr)
    return status, stdout.getvalue(), stderr.getvalue()


def test_python_verifier_applies_additional_checks_before_runtime_execution(local_case):
    checks = CapabilityCheckRegistry(
        (
            CapabilityCheck(
                "example.deployment",
                lambda spec, geometry, override: CapabilityCheckResult(
                    Capability(
                        "example.deployment",
                        SupportLevel.UNSUPPORTED,
                        reasons=("synthetic deployment gate",),
                    )
                ),
            ),
        )
    )
    verification.verify_recipe(
        local_case["recipe"], local_case["output"], capability_checks=checks
    )
    assert local_case["calls"] == []
    for role in ("reference", "candidate"):
        record = load_run_record(local_case["output"] / f"{role}.run.json")
        assert record.status is RunStatus.PREFLIGHT_FAILED
        assert record.provenance["capabilities"]["checks"]["example.deployment"][
            "required"
        ]


def test_invalid_check_registry_fails_before_creating_verifier_output(local_case):
    checks = CapabilityCheckRegistry(
        (CapabilityCheck("turboquant.kv_cache.geometry", lambda *args: None),)
    )
    with pytest.raises(ValueError, match="duplicate"):
        verification.verify_recipe(
            local_case["recipe"], local_case["output"], capability_checks=checks
        )
    assert local_case["calls"] == []
    assert not local_case["output"].exists()


def test_verify_saves_incremental_records_manifest_and_readable_report(local_case):
    status, output, errors = _invoke(local_case)
    assert status == 0 and errors == ""
    payload = json.loads(output)
    assert payload["schema"] == "metria.verification.v1"
    assert payload["verdict"] == "VERIFIED"
    assert payload["acceptance_policy_evaluated"] is False
    assert payload["change"] == {"reference_threads": 1, "candidate_threads": 2}
    assert payload["analyses"][0]["metrics"]["trajectory_agreement_score"][
        "value"
    ] == pytest.approx(200 / 3)
    assert local_case["calls"] == [1, 2]
    for role in ("reference", "candidate"):
        record = load_run_record(local_case["output"] / f"{role}.run.json")
        assert record.status is RunStatus.COMPLETED
        assert run_record_digest(record) == payload["records"][role]["record_digest"]
        assert (
            record.provenance["verification"]["recipe_digest"]
            == payload["recipe_digest"]
        )
    stored = json.loads(
        (local_case["output"] / "manifest.json").read_text(encoding="utf-8")
    )
    assert stored == payload
    report = (local_case["output"] / "report.md").read_text(encoding="utf-8")
    assert "CPU threads: 1 -> 2" in report
    assert "first divergence at token 2" in report
    for path in local_case["output"].iterdir():
        contents = path.read_text(encoding="utf-8")
        assert "private prompt that must not appear" not in contents
        assert "private generated text" not in contents


def test_verify_human_output_explains_scope_and_artifacts(local_case):
    status, output, errors = _invoke(local_case, json_output=False)
    assert status == 0 and not errors
    assert "VERIFIED" in output and "Evidence:" in output
    assert "No task-quality or performance acceptance policy was evaluated" in output


@pytest.mark.parametrize(
    "minimum,verdict,exit_code", [(0.6, "PASS", 0), (0.9, "FAIL", 1)]
)
def test_verify_applies_policy_after_valid_comparison(
    local_case, minimum, verdict, exit_code
):
    policy = VerificationPolicy(
        (
            PolicyCriterion("behavior.trajectory_agreement", "0.3.4", minimum=minimum),
            PolicyCriterion("behavior.all_trajectories_match", "0.3.4", equals=False),
            PolicyCriterion("analysis.status", "0.3.4", equals="completed"),
        )
    )
    local_case["recipe"] = replace(local_case["recipe"], policy=policy)
    code, output, errors = _invoke(local_case)
    assert code == exit_code and not errors
    data = json.loads(output)
    assert data["verdict"] == verdict
    assert data["acceptance_policy_evaluated"] is True
    assert data["policy"]["status"] == verdict
    assert len(data["policy"]["criteria"]) == 3
    assert data["policy"]["criteria"][0]["observed_value"] == pytest.approx(2 / 3)
    report = (local_case["output"] / "report.md").read_text()
    assert "Policy checks:" in report and verdict in report
    assert "universal deployment-safety" in report
    assert (
        "No task-quality or performance acceptance policy was evaluated" not in report
    )


@pytest.mark.parametrize(
    "mode",
    ["legacy", "empty", "ignored_threads", "different_vocab", "failure", "timeout"],
)
def test_policy_cannot_override_failed_verification_gates(local_case, mode):
    local_case["mode"] = mode
    local_case["recipe"] = replace(
        local_case["recipe"],
        policy=VerificationPolicy(
            (PolicyCriterion("behavior.trajectory_agreement", "0.3.4", minimum=0),)
        ),
    )
    code, output, errors = _invoke(local_case)
    assert code == 1 and not errors
    data = json.loads(output)
    assert data["verdict"] not in {"PASS", "FAIL"}
    assert data["policy"]["status"] == "NOT_EVALUATED"
    assert data["acceptance_policy_evaluated"] is False


@pytest.mark.parametrize("change", ["missing", "unit", "method", "invalid_value"])
def test_missing_or_incompatible_policy_metric_is_reported_as_insufficient(
    local_case, monkeypatch, change
):
    local_case["recipe"] = replace(
        local_case["recipe"],
        policy=VerificationPolicy(
            (PolicyCriterion("behavior.trajectory_agreement", "0.3.4", minimum=0),)
        ),
    )
    original = TrajectoryAgreementAnalysis.analyze

    def analyze(self, left, right):
        result = original(self, left, right)
        metrics = dict(result.metrics)
        metric = metrics["trajectory_agreement_score"]
        if change == "missing":
            del metrics["trajectory_agreement_score"]
        elif change == "invalid_value":
            metrics["trajectory_agreement_score"] = replace(metric, value=None)
        else:
            metrics["trajectory_agreement_score"] = replace(
                metric,
                definition=replace(metric.definition, **{change: "incompatible"}),
            )
        return replace(result, metrics=metrics)

    monkeypatch.setattr(TrajectoryAgreementAnalysis, "analyze", analyze)
    code, output, errors = _invoke(local_case)
    assert code == 1 and not errors
    data = json.loads(output)
    assert data["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert data["policy"]["status"] in {"INSUFFICIENT_EVIDENCE", "NOT_EVALUATED"}
    assert data["policy"]["criteria"][0]["observed_value"] is None


def test_unknown_policy_target_is_rejected_before_execution(local_case):
    from metria import study_recipe_to_data

    data = study_recipe_to_data(local_case["recipe"])
    data["policy"] = {
        "schema": "metria.verification_policy.v1",
        "criteria": [{"target": "arbitrary.json.path", "version": "0.3.4", "min": 0}],
    }
    local_case["path"].write_text(json.dumps(data))
    stdout, stderr = io.StringIO(), io.StringIO()
    code = main(
        ["verify", str(local_case["path"]), "--output", str(local_case["output"])],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 2
    assert "unknown policy target" in stderr.getvalue()
    assert not local_case["calls"] and not local_case["output"].exists()


@pytest.mark.parametrize(
    ("mode", "verdict", "status"),
    [
        ("legacy", "INSUFFICIENT_EVIDENCE", 1),
        ("empty", "INSUFFICIENT_EVIDENCE", 1),
        ("ignored_threads", "INSUFFICIENT_EVIDENCE", 1),
        ("different_vocab", "NOT_COMPARABLE", 1),
        ("timeout", "EXECUTION_FAILED", 1),
        ("failure", "EXECUTION_FAILED", 1),
        ("interrupt", "EXECUTION_FAILED", 130),
    ],
)
def test_verifier_distinguishes_incomplete_invalid_and_failed_runs(
    local_case, mode, verdict, status
):
    local_case["mode"] = mode
    actual, output, errors = _invoke(local_case)
    assert actual == status and errors == "", (actual, errors, output)
    payload = json.loads(output)
    assert payload["verdict"] == verdict
    assert "private" not in output
    assert (local_case["output"] / "reference.run.json").is_file()
    assert (local_case["output"] / "candidate.run.json").is_file()
    if mode == "timeout":
        assert payload["records"]["candidate"]["status"] == "timed_out"
    if mode == "interrupt":
        assert payload["records"]["candidate"]["status"] == "interrupted"


def test_insufficient_evidence_does_not_invoke_behavioral_analyzer(
    local_case, monkeypatch
):
    local_case["mode"] = "legacy"
    calls = []
    monkeypatch.setattr(
        TrajectoryAgreementAnalysis, "analyze", lambda *args: calls.append(True)
    )
    status, _, _ = _invoke(local_case)
    assert status == 1 and calls == []


@pytest.mark.parametrize("count", [1, 3])
def test_invalid_run_count_is_rejected_before_output_or_execution(local_case, count):
    recipe = local_case["recipe"]
    runs = (
        recipe.study.runs[:1]
        if count == 1
        else (*recipe.study.runs, recipe.study.runs[0])
    )
    local_case["recipe"] = replace(recipe, study=replace(recipe.study, runs=runs))
    status, output, errors = _invoke(local_case)
    assert status == 2 and output == "" and "exactly two" in errors
    assert not local_case["output"].exists() and not local_case["calls"]


@pytest.mark.parametrize("missing", ["adapters", "measurements", "analyses"])
def test_missing_routes_fail_before_any_execution(local_case, monkeypatch, missing):
    registry = verification._builtin_registries()
    monkeypatch.setattr(
        verification, "_builtin_registries", lambda: replace(registry, **{missing: {}})
    )
    status, _, errors = _invoke(local_case)
    assert status == 2 and "registered" in errors
    assert not local_case["calls"] and not local_case["output"].exists()


def test_unqualified_provider_has_failure_records_and_no_execution(local_case):
    local_case["binary"].unlink()
    status, output, _ = _invoke(local_case)
    assert status == 1 and not local_case["calls"]
    assert json.loads(output)["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_existing_output_is_never_overwritten(local_case):
    local_case["output"].mkdir()
    marker = local_case["output"] / "keep.txt"
    marker.write_text("keep this", encoding="utf-8")
    status, _, _ = _invoke(local_case)
    assert status == 2 and not local_case["calls"]
    assert marker.read_text(encoding="utf-8") == "keep this"
    assert list(local_case["output"].iterdir()) == [marker]


def test_write_failure_keeps_completed_record_without_success_manifest(
    local_case, monkeypatch
):
    original = Path.replace

    def fail_candidate(path, target):
        if Path(target).name == "candidate.run.json":
            raise OSError("simulated disk failure")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_candidate)
    status, _, errors = _invoke(local_case)
    assert status == 2 and "simulated disk failure" in errors
    assert (
        load_run_record(local_case["output"] / "reference.run.json").status
        is RunStatus.COMPLETED
    )
    assert not (local_case["output"] / "manifest.json").exists()
    assert not list(local_case["output"].glob("*.tmp"))


def test_fixed_evidence_produces_deterministic_manifests(local_case):
    status, first, _ = _invoke(local_case)
    assert status == 0
    local_case["output"] = local_case["output"].with_name("verification-second")
    status, second, _ = _invoke(local_case)
    assert status == 0
    assert json.loads(first) == json.loads(second)


def test_observation_failure_cannot_be_verified(local_case, monkeypatch):
    def fail(*args):
        raise RuntimeError("private observation failure")

    monkeypatch.setattr(llamacpp.LlamaCppAdapter, "observe", fail)
    status, output, errors = _invoke(local_case)
    assert status == 1 and not errors
    payload = json.loads(output)
    assert payload["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert payload["records"]["reference"]["status"] == "partial"
    assert "private" not in output


def test_unimplemented_trial_policy_is_not_silently_ignored(local_case):
    recipe = local_case["recipe"]
    runs = tuple(
        replace(run, trial_policy={"repetitions": 5}) for run in recipe.study.runs
    )
    local_case["recipe"] = replace(recipe, study=replace(recipe.study, runs=runs))
    status, _, errors = _invoke(local_case)
    assert status == 2 and "trial policies" in errors
    assert not local_case["calls"]


def test_failed_analysis_is_retained_without_zero_regression_claim(
    local_case, monkeypatch
):
    def fail(*args):
        raise RuntimeError("private analysis error")

    monkeypatch.setattr(TrajectoryAgreementAnalysis, "analyze", fail)
    status, output, errors = _invoke(local_case)
    assert status == 1 and errors == ""
    payload = json.loads(output)
    assert payload["verdict"] == "EXECUTION_FAILED"
    assert payload["analyses"][0]["error_type"] == "RuntimeError"
    report = (local_case["output"] / "report.md").read_text(encoding="utf-8")
    assert "Divergent prompts: 0" not in report
    assert "private analysis error" not in output + report


def test_matching_mismatch_statuses_do_not_make_identity_verified(
    local_case, monkeypatch
):
    observe = llamacpp.LlamaCppAdapter.observe

    def mismatch(adapter, session):
        observed = dict(observe(adapter, session))
        identity = dict(observed["identity"])
        identity["status"] = "mismatch"
        identity["applied"] = {**identity["applied"], "status": "mismatch"}
        observed["identity"] = identity
        return observed

    monkeypatch.setattr(llamacpp.LlamaCppAdapter, "observe", mismatch)
    status, output, _ = _invoke(local_case)
    assert status == 1
    payload = json.loads(output)
    assert payload["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "inconsistent or mismatched" in " ".join(
        payload["records"]["reference"]["evidence_gaps"]
    )
