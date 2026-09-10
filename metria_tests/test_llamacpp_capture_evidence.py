from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from metria import RunSpec, RunStatus, execute_run
from metria.measurements import TokenTrajectoryProtocol
from metria.runtimes import llamacpp
from metria.runtimes.llamacpp_capture import read_runtime_capture, read_token_trajectory


def _capture():
    return {
        "schema": "metria.llamacpp_capture.v1",
        "context": 128,
        "threads": 2,
        "threads_batch": 1,
        "vocab_size": 512,
        "chat_template_applied": False,
    }


def test_completion_only_provider_retains_native_facts_and_cleans_sidecars(
    tmp_path, monkeypatch
):
    binary = tmp_path / (
        "llama-completion.exe" if llamacpp.os.name == "nt" else "llama-completion"
    )
    binary.write_bytes(b"qualified provider")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixed model")
    spec = RunSpec(
        model={
            "path": str(model),
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        },
        runtime={
            "name": "llamacpp",
            "bin_dir": str(tmp_path),
            "n_gpu_layers": 0,
            "threads": 2,
            "threads_batch": 1,
        },
        scenario={"context": 128, "max_tokens": 2, "chat_template": False},
        measurements=(TokenTrajectoryProtocol.name,),
    )
    paths = []

    def run(argv, **kwargs):
        assert argv[argv.index("-t") + 1] == "2"
        assert argv[argv.index("-tb") + 1] == "1"
        tokens = Path(kwargs["env"]["KV_FIDELITY_TRAJECTORY"])
        runtime = Path(str(tokens) + ".runtime.json")
        paths.extend((tokens, runtime))
        tokens.write_text(
            '{"step":0,"token_id":11}\n{"step":1,"token_id":12}\n', encoding="utf-8"
        )
        runtime.write_text(json.dumps(_capture()), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="result", stderr="")

    monkeypatch.setattr(llamacpp.subprocess, "run", run)
    record = execute_run(
        study_name="native-capture",
        run_id="reference",
        spec=spec,
        adapter=llamacpp.LlamaCppAdapter(),
        measurement=TokenTrajectoryProtocol(),
        measurement_config={"prompts": [{"id": "p1", "prompt": "private prompt"}]},
        environment={
            "llama_cpp_token_ids_capture_sha256": hashlib.sha256(
                binary.read_bytes()
            ).hexdigest()
        },
    )
    assert record.status is RunStatus.COMPLETED
    identity = record.observed["identity"]
    assert identity["applied"]["source"] == "qualified_runtime_capture"
    assert identity["applied"]["fields"]["threads"] == 2
    assert identity["chat_template"]["mode"] == "disabled"
    assert record.evidence["measurements"][TokenTrajectoryProtocol.name]["prompts"][0][
        "token_ids"
    ] == (11, 12)
    assert "private prompt" not in repr(record)
    assert all(not path.exists() for path in paths)


@pytest.mark.parametrize(
    "row",
    [
        {"token_id": True},
        {"token_id": 1.5},
        {"token_id": -1},
        {"token_id": "1"},
        {"token_id": 1, "step": 2},
        {"token_id": 1, "step": False},
        {"token_id": 1, "prompt": "private"},
    ],
)
def test_malformed_token_capture_is_rejected(tmp_path, row):
    path = tmp_path / "capture.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        read_token_trajectory(path)


def test_duplicate_capture_steps_are_rejected(tmp_path):
    path = tmp_path / "capture.jsonl"
    path.write_text(
        '{"step":0,"token_id":1}\n{"step":0,"token_id":2}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="contiguous"):
        read_token_trajectory(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threads", True),
        ("context", 0),
        ("schema", "other"),
        ("chat_template_applied", 1),
    ],
)
def test_malformed_runtime_capture_is_rejected(tmp_path, field, value):
    path = tmp_path / "runtime.json"
    payload = _capture()
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        read_runtime_capture(path)


def test_legacy_missing_runtime_capture_remains_unknown(tmp_path):
    assert read_runtime_capture(tmp_path / "absent.json") is None


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_invalid_threads_fail_before_launch(tmp_path, threads):
    spec = RunSpec(
        model={"path": str(tmp_path / "model.gguf")},
        runtime={"name": "llamacpp", "threads": threads},
        scenario={},
        measurements=("text",),
    )
    report = llamacpp.LlamaCppAdapter().probe(spec, {})
    assert report.status == "unsupported"
    assert any("runtime.threads" in reason for reason in report.reasons)
