from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

import metria.runtimes.llamacpp as llamacpp_module
from metria import RunSpec, RunStatus, TreatmentSpec, TreatmentType, execute_run
from metria.protocols import CaptureRequest, InferenceRequest, MeasurementResult
from metria.runtimes.llamacpp import LlamaCppAdapter, LlamaCppSession


def _files(tmp_path: Path, *, completion: bool = True) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cli = bin_dir / (
        "llama-cli.exe" if llamacpp_module.os.name == "nt" else "llama-cli"
    )
    cli.write_bytes(b"fake llama cli")
    if completion:
        trajectory = bin_dir / (
            "llama-completion.exe"
            if llamacpp_module.os.name == "nt"
            else "llama-completion"
        )
        trajectory.write_bytes(b"fake llama completion")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake model")
    return bin_dir, model


def _spec(
    bin_dir: Path,
    model: Path,
    *,
    key_dtype: str = "q8_0",
    value_dtype: str = "q4_0",
    extra_args: tuple[str, ...] = (),
) -> RunSpec:
    return RunSpec(
        model={"path": str(model), "id": "example/model", "revision": "abc123"},
        runtime={
            "name": "llamacpp",
            "bin_dir": str(bin_dir),
            "n_gpu_layers": 42,
            "flash_attention": True,
            "extra_args": extra_args,
        },
        scenario={"context": 4096, "max_tokens": 32},
        measurements=("text",),
        treatments=(
            TreatmentSpec(
                name="llamacpp.kv_cache",
                kind=TreatmentType.RUNTIME_FEATURE,
                config={
                    "key_dtype": key_dtype,
                    "value_dtype": value_dtype,
                    "attention_rotation_v": 0,
                },
            ),
        ),
    )


def _session(tmp_path: Path) -> tuple[LlamaCppAdapter, LlamaCppSession]:
    bin_dir, model = _files(tmp_path)
    adapter = LlamaCppAdapter()
    resolved = adapter.resolve(_spec(bin_dir, model), {"hardware_class": "test"})
    session = adapter.launch(resolved, {"hardware_class": "test"})
    assert isinstance(session, LlamaCppSession)
    return adapter, session


def test_probe_reports_binary_and_capture_availability(tmp_path: Path) -> None:
    bin_dir, model = _files(tmp_path)
    adapter = LlamaCppAdapter()

    report = adapter.probe(_spec(bin_dir, model), {})

    assert report.status == "supported"
    assert report.evidence["llama_cli"].endswith("llama-cli") or report.evidence[
        "llama_cli"
    ].endswith("llama-cli.exe")
    assert report.evidence["token_ids_capture"] == "binary_present_unverified"


def test_probe_is_fail_loud_for_missing_model_or_cli(tmp_path: Path) -> None:
    bin_dir = tmp_path / "empty-bin"
    bin_dir.mkdir()
    missing_model = tmp_path / "missing.gguf"
    adapter = LlamaCppAdapter()

    report = adapter.probe(_spec(bin_dir, missing_model), {})

    assert report.status == "unsupported"
    assert any("model file not found" in reason for reason in report.reasons)
    assert any(
        "llama-cli or llama-completion not found" in reason for reason in report.reasons
    )


def test_resolve_records_binary_hash_and_requested_model_identity(
    tmp_path: Path,
) -> None:
    bin_dir, model = _files(tmp_path)
    adapter = LlamaCppAdapter()

    resolved = adapter.resolve(_spec(bin_dir, model), {"hardware_class": "gpu-a"})

    assert resolved["runtime"]["cli"]["sha256"]
    assert resolved["runtime"]["n_gpu_layers"] == 42
    assert resolved["model"]["path"] == str(model.resolve())
    assert resolved["model"]["requested_id"] == "example/model"
    assert resolved["kv_cache"]["key_dtype"] == "q8_0"
    assert resolved["kv_cache"]["value_dtype"] == "q4_0"


def test_infer_uses_exact_managed_flags_and_redacts_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, session = _session(tmp_path)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs["env"]))
        return subprocess.CompletedProcess(argv, 0, stdout="| hello world\n", stderr="")

    monkeypatch.setattr(llamacpp_module.subprocess, "run", fake_run)
    request = InferenceRequest(
        prompt="private prompt",
        generation={"temperature": 0.25, "system": "private system"},
    )

    batch = session.infer((request,))
    observed = adapter.observe(session)

    assert batch.outputs == ("hello world",)
    argv, env = calls[0]
    assert argv[argv.index("-ctk") + 1] == "q8_0"
    assert argv[argv.index("-ctv") + 1] == "q4_0"
    assert argv[argv.index("-ngl") + 1] == "42"
    assert argv[argv.index("--temp") + 1] == "0.25"
    assert env["LLAMA_ATTN_ROT_V_OVERRIDE"] == "0"
    recorded_argv = observed["invocations"][0]["argv"]
    assert "private prompt" not in recorded_argv
    assert "private system" not in recorded_argv
    assert recorded_argv[recorded_argv.index("-p") + 1] == "<redacted>"
    assert recorded_argv[recorded_argv.index("-sys") + 1] == "<redacted>"
    assert observed["invocations"][0]["prompt_sha256"]


def test_token_capture_uses_existing_trajectory_patch_abi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, session = _session(tmp_path)

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        trajectory_path = Path(kwargs["env"]["KV_FIDELITY_TRAJECTORY"])
        trajectory_path.write_text(
            '{"step":0,"token_id":11}\n{"step":1,"token_id":22}\n',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="| captured\n", stderr="")

    monkeypatch.setattr(llamacpp_module.subprocess, "run", fake_run)

    batch = session.infer(
        (InferenceRequest(prompt="capture me"),),
        capture=(CaptureRequest(kind="token_ids"),),
    )
    observed = adapter.observe(session)

    assert batch.outputs == ("captured",)
    assert batch.captures["token_ids"] == ((11, 22),)
    assert observed["invocations"][0]["capture"] == "token_ids"
    assert observed["invocations"][0]["token_capture_observed"] is True


def test_token_capture_fails_if_completion_binary_is_unavailable(
    tmp_path: Path,
) -> None:
    bin_dir, model = _files(tmp_path, completion=False)
    adapter = LlamaCppAdapter()
    session = adapter.launch(adapter.resolve(_spec(bin_dir, model), {}), {})

    with pytest.raises(RuntimeError, match="token_ids capture requires"):
        session.infer(
            (InferenceRequest(prompt="capture me"),),
            capture=(CaptureRequest(kind="token_ids"),),
        )


def test_unknown_capture_and_generation_keys_fail_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, session = _session(tmp_path)

    with pytest.raises(ValueError, match="unsupported llama.cpp captures"):
        session.infer(
            (InferenceRequest(prompt="x"),),
            capture=(CaptureRequest(kind="logprobs"),),
        )

    monkeypatch.setattr(
        llamacpp_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="ok", stderr=""
        ),
    )
    with pytest.raises(ValueError, match="unsupported generation keys"):
        session.infer((InferenceRequest(prompt="x", generation={"mystery": 1}),))


def test_extra_args_cannot_override_managed_runtime_flags(tmp_path: Path) -> None:
    bin_dir, model = _files(tmp_path)
    adapter = LlamaCppAdapter()

    report = adapter.probe(_spec(bin_dir, model, extra_args=("-ctk", "q2_k")), {})

    assert report.status == "unsupported"
    assert "cannot override Metria-managed flags" in report.reasons[0]


def test_timeout_is_reported_without_prompt_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, session = _session(tmp_path)

    def fake_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1.0)

    monkeypatch.setattr(llamacpp_module.subprocess, "run", fake_timeout)

    with pytest.raises(TimeoutError, match="timed out") as exc_info:
        session.infer(
            (InferenceRequest(prompt="do not leak me", generation={"timeout": 1.0}),)
        )

    assert "do not leak me" not in str(exc_info.value)


def test_session_close_prevents_future_inference(tmp_path: Path) -> None:
    _, session = _session(tmp_path)

    session.close()

    with pytest.raises(RuntimeError, match="session is closed"):
        session.infer((InferenceRequest(prompt="x"),))


def test_execute_run_preserves_llamacpp_timeout_status(
    tmp_path: Path, monkeypatch
) -> None:
    bin_dir, model = _files(tmp_path)

    class TextMeasurement:
        name = "text"
        version = "1"

        def requirements(self, config):
            return ()

        def execute(self, session, scenario, config):
            session.infer((InferenceRequest(prompt="private timeout prompt"),))
            return MeasurementResult()

    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(llamacpp_module.subprocess, "run", timeout)
    record = execute_run(
        study_name="timeout",
        run_id="reference",
        spec=_spec(bin_dir, model),
        adapter=LlamaCppAdapter(),
        measurement=TextMeasurement(),
        measurement_config={},
        environment={},
    )

    assert record.status is RunStatus.TIMED_OUT
    assert "private timeout prompt" not in repr(record)
