"""v0.1.3 regression tests for runner.tokenize_to_ids parsing.

llama-tokenize prints "[1, 2, 3]" to stdout; tokenize_to_ids parses that
into a list[int]. Lock the parser against:

  - normal output
  - empty list output
  - empty input (must short-circuit; no subprocess)
  - garbage / non-list output (must not crash silently)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from refract import runner
from refract.runner import tokenize_to_ids


@pytest.fixture
def fake_subprocess(monkeypatch, tmp_path):
    """Patch subprocess.run + _bin so tokenize_to_ids runs without llama.cpp.

    Returns a callable: set_response(stdout) / set_response_rc(stdout, rc).
    """
    state = {"stdout": "[]\n", "rc": 0, "calls": 0}

    def fake_run(cmd, **kwargs):
        state["calls"] += 1
        return SimpleNamespace(returncode=state["rc"],
                               stdout=state["stdout"], stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.touch()
    monkeypatch.setattr(runner, "_bin", lambda name: fake_bin)
    return state


def test_parses_simple_id_list(fake_subprocess, tmp_path):
    fake_subprocess["stdout"] = "[1, 2, 3, 4]\n"
    model = tmp_path / "m.gguf"
    model.touch()
    assert tokenize_to_ids(model, "hello world") == [1, 2, 3, 4]


def test_parses_empty_list(fake_subprocess, tmp_path):
    fake_subprocess["stdout"] = "[]\n"
    model = tmp_path / "m.gguf"
    model.touch()
    assert tokenize_to_ids(model, "x") == []


def test_empty_input_skips_subprocess(fake_subprocess, tmp_path):
    """Calling with empty text must NOT spawn llama-tokenize."""
    model = tmp_path / "m.gguf"
    model.touch()
    fake_subprocess["calls"] = 0
    assert tokenize_to_ids(model, "") == []
    assert fake_subprocess["calls"] == 0


def test_non_list_output_returns_empty(fake_subprocess, tmp_path):
    """If stdout doesn't start with '[' (e.g. tokenizer printed an error),
    return [] rather than crashing inside the splitter."""
    fake_subprocess["stdout"] = "error: failed to load vocab\n"
    model = tmp_path / "m.gguf"
    model.touch()
    assert tokenize_to_ids(model, "x") == []


def test_nonzero_returncode_raises(fake_subprocess, tmp_path):
    """If llama-tokenize exits non-zero, we must raise (and the GTM axis
    will then re-raise per the v0.1.3 fail-loud policy) rather than
    returning a possibly-broken token list."""
    fake_subprocess["rc"] = 1
    fake_subprocess["stdout"] = ""
    model = tmp_path / "m.gguf"
    model.touch()
    with pytest.raises(RuntimeError, match="llama-tokenize exited 1"):
        tokenize_to_ids(model, "x")
