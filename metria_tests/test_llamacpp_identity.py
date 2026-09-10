from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

import metria.runtimes.llamacpp as llamacpp_module
from metria import RunSpec
from metria.protocols import InferenceRequest
from metria.runtimes.llamacpp import LlamaCppAdapter


def _files(tmp_path: Path) -> tuple[Path, Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    suffix = ".exe" if llamacpp_module.os.name == "nt" else ""
    cli = bin_dir / f"llama-cli{suffix}"
    cli.write_bytes(b"identity-test-cli")
    completion = bin_dir / f"llama-completion{suffix}"
    completion.write_bytes(b"identity-test-completion")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model-content-not-yet-verified-by-issue-15")
    return bin_dir, cli, model


def _spec(bin_dir: Path, model: Path) -> RunSpec:
    return RunSpec(
        model={
            "path": str(model),
            "id": "claimed/model",
            "revision": "claimed-revision",
            "sha256": "0" * 64,
        },
        runtime={"name": "llamacpp", "bin_dir": str(bin_dir), "n_gpu_layers": 0},
        scenario={"context": 128, "max_tokens": 4},
        measurements=("text",),
    )


def test_llamacpp_identity_is_explicit_about_authority_before_inference(
    tmp_path: Path,
) -> None:
    bin_dir, cli, model = _files(tmp_path)
    adapter = LlamaCppAdapter()
    resolved = adapter.resolve(_spec(bin_dir, model), {})
    session = adapter.launch(resolved, {})

    identity = adapter.observe(session)["identity"]

    assert identity["schema"] == "metria.runtime_identity.v1"
    assert identity["status"] == "partial"
    assert identity["runtime"]["status"] == "verified"
    assert identity["runtime"]["cli_sha256"] == hashlib.sha256(
        cli.read_bytes()
    ).hexdigest()
    assert identity["model"]["status"] == "partial"
    assert identity["model"]["path"] == str(model.resolve())
    assert "claimed/model" not in repr(identity["model"])
    assert "claimed-revision" not in repr(identity["model"])
    assert identity["tokenizer"]["status"] == "unknown"
    assert identity["chat_template"]["status"] == "unknown"
    assert identity["applied"]["status"] == "unknown"
    session.close()


def test_llamacpp_identity_records_invocation_as_partial_applied_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir, _, model = _files(tmp_path)
    adapter = LlamaCppAdapter()
    session = adapter.launch(adapter.resolve(_spec(bin_dir, model), {}), {})

    monkeypatch.setattr(
        llamacpp_module.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv,
            0,
            stdout="| ok\n",
            stderr="",
        ),
    )
    session.infer((InferenceRequest(prompt="private prompt"),))
    identity = adapter.observe(session)["identity"]

    assert identity["status"] == "partial"
    assert identity["applied"]["status"] == "partial"
    assert identity["applied"]["invocation_count"] == 1
    assert "private prompt" not in repr(identity)
    session.close()


def test_llamacpp_identity_does_not_promote_requested_model_digest(
    tmp_path: Path,
) -> None:
    bin_dir, _, model = _files(tmp_path)
    adapter = LlamaCppAdapter()

    resolved = adapter.resolve(_spec(bin_dir, model), {})
    session = adapter.launch(resolved, {})
    try:
        identity = adapter.observe(session)["identity"]
        assert resolved["model"]["requested_sha256"] == "0" * 64
        assert "sha256" not in identity["model"]
        assert identity["model"]["status"] == "partial"
    finally:
        session.close()
