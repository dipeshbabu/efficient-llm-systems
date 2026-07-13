"""Tests for refract.runner: KVConfig, corpus identity, subprocess wrappers
(mocked), backend dispatch."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from refract.runner import (
    CORPUS_HASH_BYTES,
    KVConfig,
    _bin,
    _first_float,
    _PPL_RE,
    assert_corpus_matches,
    corpus_identity,
    get_active_backend,
    read_corpus_sidecar,
    run_completion,
    run_completion_trajectory,
    run_perplexity_kld,
    run_perplexity_kld_base,
    set_active_backend,
    tokenize_to_ids,
    write_corpus_sidecar,
)


# --- KVConfig ------------------------------------------------------------


def test_kvconfig_defaults():
    cfg = KVConfig()
    assert cfg.ctk == "f16"
    assert cfg.ctv == "f16"
    assert cfg.attn_rot_k is None


def test_kvconfig_parse_strips_whitespace():
    cfg = KVConfig.parse(" ctk = q8_0 , ctv = q4_0 ")
    assert cfg.ctk == "q8_0"
    assert cfg.ctv == "q4_0"


def test_kvconfig_parse_skips_empty_fragments():
    cfg = KVConfig.parse("ctk=q8_0,,ctv=q4_0")
    assert cfg.ctk == "q8_0"
    assert cfg.ctv == "q4_0"


def test_kvconfig_parse_bad_fragment_raises():
    with pytest.raises(ValueError, match="bad KV spec"):
        KVConfig.parse("ctk=q8_0,oops_no_equals")


def test_kvconfig_attn_rot_disable_passes_through():
    cfg = KVConfig.parse("ctk=f16,ctv=f16,attn_rot_disable=1")
    assert cfg.attn_rot_disable == 1
    assert cfg.env() == {"LLAMA_ATTN_ROT_DISABLE": "1"}


def test_kvconfig_label_includes_all_fields():
    cfg = KVConfig.parse(
        "ctk=q8_0,ctv=turbo4,attn_rot_k=1,attn_rot_v=0,attn_rot_disable=0,extra=42"
    )
    label = cfg.label()
    for fragment in (
        "ctk=q8_0", "ctv=turbo4", "attn_rot_k=1", "attn_rot_v=0",
        "attn_rot_disable=0", "extra=42",
    ):
        assert fragment in label


def test_kvconfig_cli_args_extras_use_double_dash():
    cfg = KVConfig.parse("ctk=f16,ctv=f16,custom=hello")
    args = cfg.cli_args()
    assert "--custom" in args
    assert "hello" in args


def test_kvconfig_env_only_emits_set_fields():
    cfg = KVConfig.parse("ctk=f16,ctv=f16")
    assert cfg.env() == {}


# --- _bin ----------------------------------------------------------------


def test_bin_missing_raises_filenotfound(monkeypatch, tmp_path):
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        _bin("nonexistent-binary")


def test_bin_returns_path_when_exists(monkeypatch, tmp_path):
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", tmp_path)
    (tmp_path / "llama-cli").write_text("#!/bin/sh\n")
    p = _bin("llama-cli")
    assert p == tmp_path / "llama-cli"


# --- corpus identity -----------------------------------------------------


def test_corpus_identity_returns_path_size_sha(tmp_path):
    p = tmp_path / "c.txt"
    p.write_bytes(b"hello world" * 100)
    info = corpus_identity(p)
    assert info["path"] == str(p)
    assert info["size_bytes"] == p.stat().st_size
    expected_sha = hashlib.sha256(p.read_bytes()[:CORPUS_HASH_BYTES]).hexdigest()
    assert info["sha256_head"] == expected_sha


def test_write_and_read_corpus_sidecar(tmp_path):
    base = tmp_path / "base.bin"
    base.write_bytes(b"x")
    corpus = tmp_path / "c.txt"
    corpus.write_bytes(b"hello")
    sidecar = write_corpus_sidecar(base, corpus)
    assert sidecar.exists()
    info = read_corpus_sidecar(base)
    assert info["sha256_head"] == corpus_identity(corpus)["sha256_head"]


def test_read_corpus_sidecar_missing_returns_none(tmp_path):
    assert read_corpus_sidecar(tmp_path / "nope") is None


def test_read_corpus_sidecar_invalid_json_returns_none(tmp_path):
    base = tmp_path / "base.bin"
    Path(str(base) + ".corpus.json").write_text("not json")
    assert read_corpus_sidecar(base) is None


def test_assert_corpus_matches_no_sidecar_is_no_op(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_bytes(b"x")
    # Should not raise
    assert_corpus_matches(tmp_path / "no_base", corpus)


def test_assert_corpus_matches_mismatch_raises(tmp_path):
    base = tmp_path / "base.bin"
    base.write_bytes(b"x")
    a = tmp_path / "a.txt"
    a.write_bytes(b"text A")
    b = tmp_path / "b.txt"
    b.write_bytes(b"text B (different)")
    write_corpus_sidecar(base, a)
    with pytest.raises(RuntimeError, match="corpus identity mismatch"):
        assert_corpus_matches(base, b)


def test_assert_corpus_matches_same_corpus_passes(tmp_path):
    base = tmp_path / "base.bin"
    base.write_bytes(b"x")
    corpus = tmp_path / "c.txt"
    corpus.write_bytes(b"hello world")
    write_corpus_sidecar(base, corpus)
    # Same corpus → no raise
    assert_corpus_matches(base, corpus)


# --- _first_float regex --------------------------------------------------


def test_first_float_extracts():
    assert _first_float(_PPL_RE, "blah Final estimate: PPL = 8.42 blah") == 8.42


def test_first_float_no_match_returns_none():
    assert _first_float(_PPL_RE, "no match here") is None


# --- backend dispatch ----------------------------------------------------


def test_active_backend_setter_getter():
    set_active_backend(None)
    assert get_active_backend() is None

    class _Stub:
        name = "test"
    stub = _Stub()
    set_active_backend(stub)
    assert get_active_backend() is stub
    set_active_backend(None)


def test_run_completion_dispatches_to_active_backend(tmp_path):
    """When active backend is set + non-llamacpp, run_completion delegates."""
    captured = {}

    class _FakeBackend:
        name = "mlx"
        def run_completion(self, **kw):
            captured.update(kw)
            from refract.backends.base import CompletionResult
            return CompletionResult(text="dispatched", n_tokens=0,
                                    metadata={"via": "fake"})

    set_active_backend(_FakeBackend())
    try:
        text, meta = run_completion(
            model=tmp_path / "m.gguf", prompt="hi",
            kv=KVConfig(), n_predict=4, ctx=128, n_gpu_layers=99,
        )
        assert text == "dispatched"
        assert meta["via"] == "fake"
    finally:
        set_active_backend(None)


def test_run_completion_trajectory_dispatches_to_active_backend(tmp_path):
    captured = {}

    class _FakeBackend:
        name = "mlx"
        def run_completion_trajectory(self, **kw):
            captured.update(kw)
            from refract.backends.base import TrajectoryResult
            return TrajectoryResult(token_ids=[7, 8, 9], metadata={})

    set_active_backend(_FakeBackend())
    try:
        token_ids, meta = run_completion_trajectory(
            model=tmp_path / "m.gguf", prompt="hi", kv=KVConfig(),
        )
        assert token_ids == [7, 8, 9]
    finally:
        set_active_backend(None)


# --- llama-cli wrapper (mocked subprocess) -------------------------------


def _stub_proc(stdout="", stderr="", returncode=0):
    p = mock.MagicMock(spec=subprocess.CompletedProcess)
    p.stdout = stdout
    p.stderr = stderr
    p.returncode = returncode
    return p


def test_run_completion_strips_noise_and_returns_text(tmp_path, monkeypatch):
    set_active_backend(None)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-cli").write_text("#!/bin/sh\n")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: _stub_proc(
            stdout="\u2580\u2581\u2582\nLoading model... \n| The capital of France is Paris.\n",
            stderr="", returncode=0,
        ),
    )
    text, meta = run_completion(
        model=tmp_path / "m.gguf", prompt="x",
        kv=KVConfig(), n_predict=8, ctx=64,
    )
    assert "Paris" in text
    assert "Loading model" not in text
    assert meta["returncode"] == 0


def test_run_completion_nonzero_returncode_raises(tmp_path, monkeypatch):
    set_active_backend(None)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-cli").write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: _stub_proc(stdout="", stderr="boom", returncode=2),
    )
    with pytest.raises(RuntimeError, match="exited 2"):
        run_completion(
            model=tmp_path / "m.gguf", prompt="x", kv=KVConfig(),
        )


def test_run_completion_trajectory_reads_jsonl_file(tmp_path, monkeypatch):
    set_active_backend(None)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-completion").write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)

    def fake_run(*a, **kw):
        # Find REFRACT_TRAJECTORY in env, write out a JSONL file there.
        traj_path = kw["env"]["REFRACT_TRAJECTORY"]
        with open(traj_path, "w") as f:
            f.write('{"step":0,"token_id":11}\n')
            f.write('{"step":1,"token_id":22}\n')
            f.write('\n')  # blank line — should be skipped
        return _stub_proc(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    token_ids, meta = run_completion_trajectory(
        model=tmp_path / "m.gguf", prompt="x", kv=KVConfig(),
    )
    assert token_ids == [11, 22]
    assert meta["n_tokens"] == 2


def test_run_completion_trajectory_missing_file_returns_empty(tmp_path, monkeypatch):
    set_active_backend(None)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-completion").write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **kw: _stub_proc(returncode=0))
    token_ids, meta = run_completion_trajectory(
        model=tmp_path / "m.gguf", prompt="x", kv=KVConfig(),
    )
    assert token_ids == []


def test_tokenize_to_ids_empty_text_returns_empty():
    set_active_backend(None)
    assert tokenize_to_ids(Path("anything"), "") == []


def test_tokenize_to_ids_parses_bracketed_output(tmp_path, monkeypatch):
    set_active_backend(None)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-tokenize").write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **kw: _stub_proc(stdout="[1, 2, 3]\n",
                                                    stderr="", returncode=0))
    assert tokenize_to_ids(tmp_path / "m.gguf", "hello") == [1, 2, 3]


def test_tokenize_to_ids_returncode_nonzero_raises(tmp_path, monkeypatch):
    set_active_backend(None)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-tokenize").write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **kw: _stub_proc(stderr="x", returncode=1))
    with pytest.raises(RuntimeError, match="exited 1"):
        tokenize_to_ids(tmp_path / "m.gguf", "x")


def test_tokenize_to_ids_empty_brackets_returns_empty(tmp_path, monkeypatch):
    set_active_backend(None)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-tokenize").write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **kw: _stub_proc(stdout="[]\n", returncode=0))
    assert tokenize_to_ids(tmp_path / "m.gguf", "x") == []


def test_tokenize_to_ids_unparseable_output_returns_empty(tmp_path, monkeypatch):
    set_active_backend(None)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-tokenize").write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **kw: _stub_proc(stdout="garbage", returncode=0))
    # No leading '[' → returns []
    assert tokenize_to_ids(tmp_path / "m.gguf", "x") == []


def test_run_perplexity_kld_base_nonzero_raises(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-perplexity").write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **kw: _stub_proc(returncode=3, stderr="bad"))
    with pytest.raises(RuntimeError, match="exited 3"):
        run_perplexity_kld_base(
            model=tmp_path / "m.gguf", corpus=tmp_path / "c.txt",
            kv=KVConfig(), base_path=tmp_path / "base.bin",
        )


def test_run_perplexity_kld_parses_metrics(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-perplexity").write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: _stub_proc(
            stdout=("Final estimate: PPL = 7.42\n"
                    "Mean KLD: 0.012345\n"),
            stderr="RMS \u0394p: 1.23 %\nSame top-p: 99.0 %\n",
            returncode=0,
        ),
    )
    out = run_perplexity_kld(
        model=tmp_path / "m.gguf", corpus=tmp_path / "c.txt",
        kv=KVConfig(), base_path=tmp_path / "base.bin",
    )
    assert out["ppl"] == 7.42
    assert out["mean_kld"] == 0.012345
    assert out["rms_dp_pct"] == 1.23
    assert out["same_topp_pct"] == 99.0


def test_run_perplexity_kld_no_kld_in_output_raises(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-perplexity").write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: _stub_proc(stdout="just some output\n",
                                     stderr="", returncode=0),
    )
    with pytest.raises(RuntimeError, match="Could not parse Mean KLD"):
        run_perplexity_kld(
            model=tmp_path / "m.gguf", corpus=tmp_path / "c.txt",
            kv=KVConfig(), base_path=tmp_path / "base.bin",
        )
