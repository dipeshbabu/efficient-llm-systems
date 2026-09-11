"""Scientific compatibility is checked by the shared Metria engine."""

from __future__ import annotations

import copy
import json

import pytest

import kv_fidelity.comparison as comparison
from kv_fidelity.cli import main
from kv_fidelity.comparison import compare_reports, report_to_run_record
from kv_fidelity.provenance import begin_report_capture

from ._fixtures import make_comparable_report


def _set(data, path, value):
    keys = path.split(".")
    for key in keys[:-1]:
        data = data[key]
    data[keys[-1]] = value


def test_compatible_reports_use_shared_engine_and_allow_candidate_change(monkeypatch):
    left = make_comparable_report()
    right = copy.deepcopy(left)
    right["candidate"] = "ctk=q4_0,ctv=q4_0"
    right["axes"]["gtm"]["score"] = 70.0
    calls = []
    shared = comparison.compare_runs

    def compare(a, b, plan):
        calls.append(plan)
        return shared(a, b, plan)

    monkeypatch.setattr(comparison, "compare_runs", compare)
    result = compare_reports([left, right])
    assert len(calls) == 1
    assert result["compatible"], result["pairs"]
    assert result["pairs"][0]["method_compatible_metrics"] == [
        "composite",
        "gtm",
        "kld",
    ]


@pytest.mark.parametrize(
    ("path", "value", "dimension"),
    [
        (
            "extras.comparison_evidence.resolved.model.sha256",
            "f" * 64,
            "resolved.model.sha256",
        ),
        (
            "extras.comparison_evidence.resolved.tokenizer.sha256",
            "f" * 64,
            "resolved.tokenizer.sha256",
        ),
        (
            "extras.comparison_evidence.resolved.inputs.prompts.sha256",
            "f" * 64,
            "resolved.inputs.prompts.sha256",
        ),
        (
            "extras.comparison_evidence.resolved.inputs.corpus.sha256",
            "f" * 64,
            "resolved.inputs.corpus.sha256",
        ),
        (
            "extras.comparison_evidence.requested.trial_policy.seed",
            7,
            "trial_policy.seed",
        ),
        (
            "extras.comparison_evidence.requested.scenario.generation.ctx",
            1024,
            "scenario.generation.ctx",
        ),
        (
            "extras.comparison_evidence.observed.hardware.host.cpu_count",
            8,
            "observed.hardware.host",
        ),
        ("environment.llama_cpp_commit", "other-build", "observed.backend_metadata"),
        ("environment.backend", "vllm", "observed.backend_metadata"),
        ("framework_version", "0.3.1", "scenario.suite_version"),
        ("schema", "kv_fidelity.report.future", "scenario.schema"),
        ("reference", "ctk=q8_0,ctv=q8_0", "runtime.reference_kv"),
        (
            "axes.gtm.method",
            "trajectory_prefix_agreement",
            "scenario.metric_methods.gtm",
        ),
    ],
)
def test_control_block_and_method_mismatches(path, value, dimension):
    left = make_comparable_report()
    right = copy.deepcopy(left)
    _set(right, path, value)
    result = compare_reports([left, right])
    assert not result["compatible"]
    assert dimension in {issue["dimension"] for issue in result["pairs"][0]["issues"]}


@pytest.mark.parametrize(
    "path",
    [
        "extras.comparison_evidence.resolved.model.sha256",
        "extras.comparison_evidence.resolved.tokenizer.sha256",
        "extras.comparison_evidence.resolved.runtime.artifacts",
        "extras.comparison_evidence.resolved.inputs.prompts.sha256",
        "extras.comparison_evidence.resolved.inputs.corpus.sha256",
        "extras.comparison_evidence.requested.trial_policy.seed",
        "extras.comparison_evidence.requested.scenario.generation.ctx",
        "extras.comparison_evidence.requested.runtime.settings.n_gpu_layers",
        "extras.comparison_evidence.observed.hardware",
        "axes.kld.method",
        "axes.kld.metadata",
        "composite",
    ],
)
def test_missing_or_null_evidence_on_both_sides_is_not_equality(path):
    report = make_comparable_report()
    _set(report, path, None)
    result = compare_reports([report, copy.deepcopy(report)])
    assert not result["compatible"], path
    assert any("missing" in issue["reason"] for issue in result["pairs"][0]["issues"])


def test_full_vocabulary_and_topk_methods_stay_incompatible_under_override():
    left = make_comparable_report()
    right = copy.deepcopy(left)
    right["axes"]["kld"]["metadata"] = {
        "kld_estimator": "normalized_top_k_with_other_bucket",
        "full_vocabulary": False,
        "topk": 64,
    }
    result = compare_reports(
        [left, right], override_reason="Inspect historical methods separately"
    )
    assert not result["compatible"]
    assert result["override"]["enabled"]
    assert set(result["pairs"][0]["incompatible_metrics"]) == {"kld", "composite"}
    assert result["pairs"][0]["method_compatible_metrics"] == ["gtm"]


def test_every_pair_and_every_mismatch_is_reported():
    reports = [make_comparable_report() for _ in range(3)]
    reports[1]["extras"]["comparison_evidence"]["requested"]["trial_policy"]["seed"] = 3
    reports[1]["schema"] = "other"
    reports[2]["environment"]["llama_cpp_commit"] = "other"
    result = compare_reports(reports)
    assert len(result["pairs"]) == 3
    dimensions = {
        issue["dimension"] for pair in result["pairs"] for issue in pair["issues"]
    }
    assert {
        "trial_policy.seed",
        "scenario.schema",
        "observed.backend_metadata",
    } <= dimensions


def test_gpu_placement_does_not_invent_device_evidence():
    report = make_comparable_report()
    report["extras"]["comparison_evidence"]["requested"]["runtime"]["settings"][
        "n_gpu_layers"
    ] = 99
    result = compare_reports([report, copy.deepcopy(report)])
    assert not result["compatible"]
    assert "observed.hardware.accelerators" in {
        issue["dimension"] for issue in result["pairs"][0]["issues"]
    }


def test_adapter_is_deterministic_detached_and_preserves_raw_evidence():
    report = make_comparable_report()
    first = report_to_run_record(report)
    assert first == report_to_run_record(copy.deepcopy(report))
    report["axes"]["kld"]["metadata"]["full_vocabulary"] = False
    assert (
        first.evidence["kv_fidelity_report"]["axes"]["kld"]["metadata"][
            "full_vocabulary"
        ]
        is True
    )
    assert "model" not in first.observed  # a selected model is not native readback


@pytest.mark.parametrize("count", [0, 1])
def test_requires_at_least_two_reports(count):
    with pytest.raises(ValueError, match="at least two"):
        compare_reports([make_comparable_report()] * count)


def test_cli_rejects_legacy_reports_but_retains_reasoned_override(tmp_path, capsys):
    left, right, output = (
        tmp_path / name for name in ("a.json", "b.json", "comparison.json")
    )
    legacy = {"schema": "kv_fidelity.report.v0.3.3", "composite": 90, "axes": {}}
    for path in (left, right):
        path.write_text(json.dumps(legacy))
    assert main(["compare", str(left), str(right), "--json-out", str(output)]) == 2
    assert "NOT_COMPARABLE" in capsys.readouterr().out
    assert not json.loads(output.read_text())["compatible"]
    assert (
        main(
            [
                "compare",
                str(left),
                str(right),
                "--allow-incompatible",
                "Historical inspection",
                "--json-out",
                str(output),
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "NOT_COMPARABLE" in text and "Historical inspection" in text
    result = json.loads(output.read_text())
    assert not result["compatible"]
    assert result["override"] == {"enabled": True, "reason": "Historical inspection"}
    assert result["pairs"][0]["issues"]


@pytest.mark.parametrize(
    "bad_json", ['{"x":1,"x":2}', '{"x":NaN}', "[]", '{"x":1e999}', "invalid"]
)
def test_cli_refuses_malformed_reports_even_with_override(tmp_path, bad_json):
    left, right = tmp_path / "bad.json", tmp_path / "good.json"
    left.write_text(bad_json)
    right.write_text(json.dumps(make_comparable_report()))
    assert (
        main(["compare", str(left), str(right), "--allow-incompatible", "Inspect"]) == 2
    )


def test_local_capture_hashes_entire_inputs_and_detects_changes(tmp_path, monkeypatch):
    import kv_fidelity.runner as runner

    monkeypatch.setattr(runner, "DEFAULT_BIN_DIR", tmp_path)
    for name in ("model.gguf", "prompts.jsonl", "llama-completion", "llama-perplexity"):
        (tmp_path / name).write_bytes(b"test artifact")
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"a" * (1024 * 1024) + b"tail")
    settings = {
        "model": str(tmp_path / "model.gguf"),
        "prompts": tmp_path / "prompts.jsonl",
        "corpus": corpus,
        "axis_a": "trajectory",
        "ctx": 512,
        "n_predict": 32,
        "chunks": 1,
        "seed": 42,
        "n_gpu_layers": 0,
    }
    first = begin_report_capture(settings, backend="llamacpp")
    corpus.write_bytes(b"a" * (1024 * 1024) + b"changed tail")
    second = begin_report_capture(settings, backend="llamacpp")
    assert (
        first.data["resolved"]["inputs"]["corpus"]["sha256"]
        != second.data["resolved"]["inputs"]["corpus"]["sha256"]
    )
    assert first.finish()["changed_inputs"] == ["corpus"]
    assert second.finish()["changed_inputs"] == []


def test_remote_name_and_version_are_not_immutable_identity():
    capture = begin_report_capture(
        {"model": "organization/model", "n_gpu_layers": 0}, backend="vllm"
    ).finish()
    assert "model" not in capture["resolved"]
    assert "tokenizer" not in capture["resolved"]
    assert "runtime" not in capture["resolved"]


def test_missing_required_runtime_binary_fails_even_when_both_reports_omit_it():
    report = make_comparable_report()
    del report["extras"]["comparison_evidence"]["resolved"]["runtime"]["artifacts"][
        "llama-tokenize"
    ]
    assert not compare_reports([report, copy.deepcopy(report)])["compatible"]


def test_contradictory_backend_identity_fails_even_when_reports_match():
    report = make_comparable_report()
    report["environment"]["backend"] = "vllm"
    result = compare_reports([report, copy.deepcopy(report)])
    assert not result["compatible"]
    assert (
        "requested and reported backend identities differ"
        in result["input_issues"][0]["issues"]
    )


def test_same_version_different_scoring_build_is_not_method_compatible():
    left = make_comparable_report()
    right = copy.deepcopy(left)
    right["extras"]["comparison_evidence"]["resolved"]["suite"]["sha256"] = "a" * 64
    result = compare_reports([left, right], override_reason="Inspect code changes")
    assert not result["compatible"]
    assert set(result["pairs"][0]["incompatible_metrics"]) == {
        "gtm",
        "kld",
        "composite",
    }


def test_trailing_runtime_arguments_are_retained_as_unverified_digests(monkeypatch):
    monkeypatch.setenv("KV_FIDELITY_LLAMA_EXTRA_FLAGS", "-m private-model.gguf")
    capture = begin_report_capture(
        {"model": "missing.gguf", "n_gpu_layers": 0}, backend="llamacpp"
    ).finish()
    assert capture["requested"]["runtime"]["unverified_argument_overrides"]
    assert "private-model.gguf" not in json.dumps(capture)
    report = make_comparable_report()
    report["extras"]["comparison_evidence"] = capture
    result = compare_reports([report, copy.deepcopy(report)])
    assert not result["compatible"]
    assert any("EXTRA_FLAGS" in issue for issue in result["input_issues"][0]["issues"])


def test_candidate_extra_flags_cannot_change_controlled_inputs():
    report = make_comparable_report()
    report["candidate"] += ",model=other-model.gguf"
    result = compare_reports([report, copy.deepcopy(report)])
    assert not result["compatible"]
    assert any(
        "unverified extra command arguments" in issue
        for issue in result["input_issues"][0]["issues"]
    )


def test_cli_method_override_never_relabels_incompatible_metrics(tmp_path, capsys):
    report = make_comparable_report()
    left, right, output = (
        tmp_path / name for name in ("a.json", "b.json", "comparison.json")
    )
    left.write_text(json.dumps(report))
    report["axes"]["kld"]["metadata"] = {
        "kld_estimator": "normalized_top_k_with_other_bucket",
        "full_vocabulary": False,
        "topk": 64,
    }
    right.write_text(json.dumps(report))
    assert main(["compare", str(left), str(right), "--json-out", str(output)]) == 2
    assert "metric kld" in capsys.readouterr().out
    assert (
        main(
            [
                "compare",
                str(left),
                str(right),
                "--json-out",
                str(output),
                "--allow-incompatible",
                "Inspect estimates separately",
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert not result["compatible"]
    assert set(result["pairs"][0]["incompatible_metrics"]) == {"kld", "composite"}


def test_score_cli_retains_inputs_and_methods_for_comparison(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import kv_fidelity.backends as backends
    import kv_fidelity.cli as cli
    import kv_fidelity.runner as runner

    from ._fixtures import make_gtm, make_kld

    monkeypatch.setattr(runner, "DEFAULT_BIN_DIR", tmp_path)
    monkeypatch.setattr(runner, "_ACTIVE_BACKEND", None)
    backend = SimpleNamespace(
        name="llamacpp",
        detect_thinking_mode=lambda **kw: (False, []),
        model_metadata=lambda **kw: {
            "backend": "llamacpp",
            "llama_cpp_commit": "test-build",
        },
    )
    monkeypatch.setattr(backends, "get_backend", lambda name: backend)
    monkeypatch.setattr(cli, "run_gtm", lambda **kw: make_gtm())
    kld = make_kld()
    kld.metadata = {"kld_estimator": "llama_perplexity", "full_vocabulary": True}
    monkeypatch.setattr(cli, "run_kld", lambda **kw: kld)
    for name in (
        "model.gguf",
        "llama-cli",
        "llama-tokenize",
        "llama-perplexity",
        "corpus.txt",
    ):
        (tmp_path / name).write_bytes(b"synthetic test artifact")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"one","prompt":"test"}\n')
    reports = []
    for index, kv in enumerate(("q8_0", "q4_0")):
        output = tmp_path / f"report-{index}.json"
        assert (
            main(
                [
                    "score",
                    "--backend",
                    "llamacpp",
                    "--model",
                    str(tmp_path / "model.gguf"),
                    "--candidate",
                    f"ctk={kv},ctv={kv}",
                    "--axis-a",
                    "gtm",
                    "--prompts",
                    str(prompts),
                    "--corpus",
                    str(tmp_path / "corpus.txt"),
                    "-ngl",
                    "0",
                    "--json-out",
                    str(output),
                ]
            )
            == 0
        )
        reports.append(output)
    data = json.loads(reports[0].read_text())
    assert data["axes"]["gtm"]["method"] == "retokenized_greedy_token_match"
    assert data["extras"]["comparison_evidence"]["resolved"]["model"]["sha256"]
    assert main(["compare", *map(str, reports)]) == 0


def test_empty_override_and_input_overwrite_are_rejected(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(make_comparable_report()))
    original = report.read_bytes()
    assert main(["compare", str(report), str(report), "--allow-incompatible", " "]) == 2
    assert main(["compare", str(report), str(report), "--json-out", str(report)]) == 2
    assert report.read_bytes() == original


def test_comparison_json_write_error_is_an_error_exit(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(make_comparable_report()))
    assert main(["compare", str(report), str(report), "--json-out", str(tmp_path)]) == 2
    assert "could not write" in capsys.readouterr().out
