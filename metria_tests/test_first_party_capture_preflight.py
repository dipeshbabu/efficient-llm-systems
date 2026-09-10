from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import metria.runtimes.vllm as vllm_module
from metria import RunSpec, RunStatus, execute_run
from metria.capture_support import probe_capture_support
from metria.measurements import TokenTrajectoryProtocol
from metria.protocols import CaptureRequest
from metria.runtimes.llamacpp import LlamaCppAdapter
from metria.runtimes.vllm import VLLMAdapter


def _llama_files(
    tmp_path: Path, *, completion: bool = True
) -> tuple[Path, Path, Path | None]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cli = bin_dir / "llama-cli"
    cli.write_bytes(b"fake llama cli")
    completion_path: Path | None = None
    if completion:
        completion_path = bin_dir / "llama-completion"
        completion_path.write_bytes(b"qualified fake completion")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake model")
    return bin_dir, model, completion_path


def _llama_spec(bin_dir: Path, model: Path, measurement: str = "text") -> RunSpec:
    return RunSpec(
        model={"path": str(model)},
        runtime={"name": "llamacpp", "bin_dir": str(bin_dir), "n_gpu_layers": 0},
        scenario={"context": 512, "max_tokens": 8},
        measurements=(measurement,),
    )


def test_vllm_native_token_ids_satisfy_capture_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vllm_module, "_vllm_available", lambda: True)
    monkeypatch.setattr(vllm_module, "_vllm_version", lambda: "0.test")
    adapter = VLLMAdapter()
    spec = RunSpec(
        model={"id": "example/model", "revision": "abc123"},
        runtime={"name": "vllm", "max_model_len": 1024},
        scenario={"max_tokens": 8},
        measurements=(TokenTrajectoryProtocol.name,),
    )
    runtime_support = adapter.probe(spec, {})

    capture_support = probe_capture_support(
        adapter=adapter,
        runtime_support=runtime_support,
        spec=spec,
        environment={},
        capture=(CaptureRequest(kind="token_ids"),),
    )

    assert capture_support.status.value == "supported"
    token_ids = capture_support.evidence["captures"]["token_ids"]
    assert token_ids["status"] == "supported"
    assert token_ids["probe_marker"] == "native_output_token_ids"


def test_llama_present_but_unqualified_token_capture_is_unknown(tmp_path: Path) -> None:
    bin_dir, model, _ = _llama_files(tmp_path)
    adapter = LlamaCppAdapter()
    spec = _llama_spec(bin_dir, model, TokenTrajectoryProtocol.name)
    runtime_support = adapter.probe(spec, {})

    capture_support = probe_capture_support(
        adapter=adapter,
        runtime_support=runtime_support,
        spec=spec,
        environment={},
        capture=(CaptureRequest(kind="token_ids"),),
    )

    assert runtime_support.evidence["token_ids_capture"] == "binary_present_unverified"
    assert capture_support.status.value == "unknown"
    assert "not qualified" in capture_support.reasons[0]


def test_llama_missing_token_capture_provider_is_unsupported(tmp_path: Path) -> None:
    bin_dir, model, _ = _llama_files(tmp_path, completion=False)
    adapter = LlamaCppAdapter()
    spec = _llama_spec(bin_dir, model, TokenTrajectoryProtocol.name)
    runtime_support = adapter.probe(spec, {})

    capture_support = probe_capture_support(
        adapter=adapter,
        runtime_support=runtime_support,
        spec=spec,
        environment={},
        capture=(CaptureRequest(kind="token_ids"),),
    )

    assert runtime_support.status.value == "supported"
    assert runtime_support.evidence["token_ids_capture"] == "unavailable"
    assert capture_support.status.value == "unsupported"


def test_llama_capture_qualification_requires_exact_binary_sha256(
    tmp_path: Path,
) -> None:
    bin_dir, model, completion = _llama_files(tmp_path)
    assert completion is not None
    adapter = LlamaCppAdapter()
    spec = _llama_spec(bin_dir, model, TokenTrajectoryProtocol.name)
    runtime_support = adapter.probe(spec, {})
    actual_sha = hashlib.sha256(completion.read_bytes()).hexdigest()

    supported = probe_capture_support(
        adapter=adapter,
        runtime_support=runtime_support,
        spec=spec,
        environment={"llama_cpp_token_ids_capture_sha256": actual_sha},
        capture=(CaptureRequest(kind="token_ids"),),
    )
    mismatched = probe_capture_support(
        adapter=adapter,
        runtime_support=runtime_support,
        spec=spec,
        environment={"llama_cpp_token_ids_capture_sha256": "0" * 64},
        capture=(CaptureRequest(kind="token_ids"),),
    )
    malformed = probe_capture_support(
        adapter=adapter,
        runtime_support=runtime_support,
        spec=spec,
        environment={"llama_cpp_token_ids_capture_sha256": "not-a-digest"},
        capture=(CaptureRequest(kind="token_ids"),),
    )

    assert supported.status.value == "supported"
    qualification = supported.evidence["captures"]["token_ids"]
    assert qualification["qualification"] == "sha256"
    assert qualification["observed_sha256"] == actual_sha
    assert mismatched.status.value == "unsupported"
    assert malformed.status.value == "unsupported"


def test_unverified_llama_trajectory_fails_before_resolve_or_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir, model, _ = _llama_files(tmp_path)
    adapter = LlamaCppAdapter()
    measurement = TokenTrajectoryProtocol()
    spec = _llama_spec(bin_dir, model, measurement.name)

    def must_not_resolve(*args: object, **kwargs: object) -> object:
        raise AssertionError("capture preflight must stop before resolve")

    monkeypatch.setattr(adapter, "resolve", must_not_resolve)
    record = execute_run(
        study_name="llama-capture-preflight",
        run_id="run-0",
        spec=spec,
        adapter=adapter,
        measurement=measurement,
        measurement_config={"prompts": ({"id": "p1", "prompt": "private"},)},
        environment={"hardware_class": "test"},
    )

    assert record.status is RunStatus.PREFLIGHT_FAILED
    assert record.provenance["preflight"]["captures"]["status"] == "unknown"
    assert record.events[-1]["kind"] == "capture_requirements_blocked"
    assert not any(event["stage"] == "launch" for event in record.events)
