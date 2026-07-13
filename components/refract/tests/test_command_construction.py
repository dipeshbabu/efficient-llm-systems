"""v0.1.3 regression tests for the llama.cpp command lines we construct.

These pin the v0.1.1 + v0.1.2 fixes:

  - run_completion must pass --single-turn but NOT --no-conversation
    (this fork rejects --no-conversation and prints help; v0.1 captured
    the help text as the "completion").
  - run_perplexity_kld_base passes --kl-divergence-base.
  - run_perplexity_kld passes BOTH --kl-divergence and --kl-divergence-base.
  - All three pass errors='replace' to subprocess.run (Mistral-Small-24B
    crashed in v0.1.1 on a strict utf-8 decode of stderr).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from refract import runner
from refract.runner import (
    KVConfig,
    run_completion,
    run_perplexity_kld,
    run_perplexity_kld_base,
)


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Patch subprocess.run + _bin so the wrappers run without llama.cpp.

    Returns a list captured[0] = SimpleNamespace(cmd, kwargs) per call.
    """
    calls: list = []

    def fake_run(cmd, **kwargs):
        calls.append(SimpleNamespace(cmd=cmd, kwargs=kwargs))
        # Return enough fields for the wrappers to parse without raising.
        # KLD parser needs a Mean KLD line; perplexity-base just needs rc=0.
        stdout = "Mean    KLD:   0.000000\nFinal estimate: PPL = 6.0\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # _bin checks file existence — bypass with a temp dummy file.
    fake_bin = tmp_path / "fake-bin"
    fake_bin.touch()
    monkeypatch.setattr(runner, "_bin", lambda name: fake_bin)
    return calls


def test_run_completion_has_single_turn_no_no_conversation(captured, tmp_path):
    model = tmp_path / "m.gguf"
    model.touch()
    run_completion(model=model, prompt="hi", kv=KVConfig(), n_predict=4)
    assert len(captured) == 1
    cmd = captured[0].cmd
    assert "--single-turn" in cmd
    # Regression: --no-conversation must NOT be in the args. This fork
    # rejects it with "please use llama-completion instead" and prints help,
    # which v0.1 silently captured as the model's completion.
    assert "--no-conversation" not in cmd


def test_run_completion_passes_errors_replace(captured, tmp_path):
    model = tmp_path / "m.gguf"
    model.touch()
    run_completion(model=model, prompt="hi", kv=KVConfig(), n_predict=4)
    # Regression: Mistral-Small-24B crashed in v0.1.1 on UnicodeDecodeError
    # because subprocess.run(text=True) strict-decodes stderr. errors='replace'
    # keeps the stream usable when llama-cli emits non-utf-8 bytes.
    assert captured[0].kwargs.get("errors") == "replace"
    assert captured[0].kwargs.get("text") is True


def test_run_perplexity_kld_base_has_base_flag(captured, tmp_path):
    model = tmp_path / "m.gguf"
    model.touch()
    corpus = tmp_path / "wiki.raw"
    corpus.write_text("hello world")
    base = tmp_path / "base.bin"
    run_perplexity_kld_base(model=model, corpus=corpus, kv=KVConfig(),
                            base_path=base, chunks=4)
    cmd = captured[0].cmd
    assert "--kl-divergence-base" in cmd
    assert str(base) in cmd
    assert captured[0].kwargs.get("errors") == "replace"


def test_run_perplexity_kld_has_both_flags(captured, tmp_path):
    model = tmp_path / "m.gguf"
    model.touch()
    corpus = tmp_path / "wiki.raw"
    corpus.write_text("hello world")
    base = tmp_path / "base.bin"
    run_perplexity_kld(model=model, corpus=corpus, kv=KVConfig(),
                       base_path=base, chunks=4)
    cmd = captured[0].cmd
    assert "--kl-divergence" in cmd
    assert "--kl-divergence-base" in cmd
    assert captured[0].kwargs.get("errors") == "replace"


# --- REFRACT_LLAMA_EXTRA_FLAGS escape hatch (v0.3.2.1) --------------------


def test_extra_flags_unset_returns_empty(monkeypatch):
    """Empty / unset env -> no flags appended."""
    monkeypatch.delenv("REFRACT_LLAMA_EXTRA_FLAGS", raising=False)
    assert runner._llama_extra_flags() == []
    monkeypatch.setenv("REFRACT_LLAMA_EXTRA_FLAGS", "")
    assert runner._llama_extra_flags() == []
    monkeypatch.setenv("REFRACT_LLAMA_EXTRA_FLAGS", "   ")
    assert runner._llama_extra_flags() == []


def test_extra_flags_shlex_split(monkeypatch):
    """Multi-word value splits via shlex; quoted args stay intact."""
    monkeypatch.setenv("REFRACT_LLAMA_EXTRA_FLAGS", "-ngl 28 -ncmoe 32")
    assert runner._llama_extra_flags() == ["-ngl", "28", "-ncmoe", "32"]
    monkeypatch.setenv(
        "REFRACT_LLAMA_EXTRA_FLAGS", '-ts "1,1" --override-tensor "blk\\.\\d+\\.attn=CPU"'
    )
    assert runner._llama_extra_flags() == [
        "-ts", "1,1",
        "--override-tensor", "blk\\.\\d+\\.attn=CPU",
    ]


def test_run_completion_appends_extra_flags(captured, tmp_path, monkeypatch):
    """User's -ncmoe 32 reaches the llama-cli argv (3060/MoE OOM fix)."""
    monkeypatch.setenv("REFRACT_LLAMA_EXTRA_FLAGS", "-ngl 28 -ncmoe 32")
    model = tmp_path / "m.gguf"
    model.touch()
    run_completion(model=model, prompt="hi", kv=KVConfig(), n_predict=4)
    cmd = captured[0].cmd
    assert "-ncmoe" in cmd
    assert "32" in cmd
    # last-wins on -ngl: REFRACT puts 99 first, user's 28 must come AFTER
    ngl_indices = [i for i, a in enumerate(cmd) if a == "-ngl"]
    assert len(ngl_indices) == 2, f"expected 2 -ngl flags, got {ngl_indices}"
    assert cmd[ngl_indices[0] + 1] == "99"   # REFRACT default first
    assert cmd[ngl_indices[1] + 1] == "28"   # user override last


def test_run_perplexity_kld_base_appends_extra_flags(captured, tmp_path, monkeypatch):
    monkeypatch.setenv("REFRACT_LLAMA_EXTRA_FLAGS", "-ncmoe 32")
    model = tmp_path / "m.gguf"
    model.touch()
    corpus = tmp_path / "wiki.raw"
    corpus.write_text("hello world")
    base = tmp_path / "base.bin"
    run_perplexity_kld_base(
        model=model, corpus=corpus, kv=KVConfig(),
        base_path=base, chunks=4,
    )
    assert "-ncmoe" in captured[0].cmd


def test_run_perplexity_kld_appends_extra_flags(captured, tmp_path, monkeypatch):
    monkeypatch.setenv("REFRACT_LLAMA_EXTRA_FLAGS", "-ncmoe 32")
    model = tmp_path / "m.gguf"
    model.touch()
    corpus = tmp_path / "wiki.raw"
    corpus.write_text("hello world")
    base = tmp_path / "base.bin"
    run_perplexity_kld(
        model=model, corpus=corpus, kv=KVConfig(),
        base_path=base, chunks=4,
    )
    assert "-ncmoe" in captured[0].cmd


def test_run_completion_trajectory_appends_extra_flags(captured, tmp_path, monkeypatch):
    """Trajectory axis (run_completion_trajectory) honors the extra-flags
    knob the same as run_completion. This is the trajectory hot path that
    the 3060/MoE user would actually hit during axis A."""
    from refract.runner import run_completion_trajectory
    monkeypatch.setenv("REFRACT_LLAMA_EXTRA_FLAGS", "-ngl 28 -ncmoe 32")
    model = tmp_path / "m.gguf"
    model.touch()
    run_completion_trajectory(
        model=model, prompt="hi", kv=KVConfig(), n_predict=4,
    )
    cmd = captured[0].cmd
    assert "-ncmoe" in cmd
    assert "32" in cmd
