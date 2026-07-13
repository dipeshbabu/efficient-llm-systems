"""Extra CLI tests to push coverage on selftest model probe + score floor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import pytest

import refract.cli as cli
from refract.backends.base import CompletionResult
from .test_cli_runs import (
    _FakeBackend,
    _gtm_result,
    _kld_result,
    _make_score_args,
    _patch_backends,
    _traj_result,
)


def test_run_score_measure_floor_passes(tmp_path, monkeypatch):
    """Floor=100 (perfect ref-vs-ref) → score continues normally."""
    _patch_backends(monkeypatch)
    monkeypatch.setattr(cli, "run_gtm", lambda **kw: _gtm_result(100.0))
    monkeypatch.setattr(cli, "run_kld",
                        lambda **kw: _kld_result(100.0, mean_kld=0.0))
    args = _make_score_args(tmp_path, axis_a="gtm", measure_floor=True)
    rc = cli._run_score(args)
    assert rc == 0


def test_run_score_measure_floor_uses_selected_trajectory(
    tmp_path, monkeypatch
):
    _patch_backends(monkeypatch)
    calls = []

    def fake_trajectory(**kw):
        calls.append((kw["reference_kv"], kw["candidate_kv"]))
        return _traj_result(100.0)

    monkeypatch.setattr(cli, "run_trajectory", fake_trajectory)
    monkeypatch.setattr(
        cli, "run_gtm",
        lambda **kw: pytest.fail("trajectory floor must not call legacy GTM"),
    )
    monkeypatch.setattr(
        cli, "run_kld",
        lambda **kw: _kld_result(100.0, mean_kld=0.0),
    )
    args = _make_score_args(tmp_path, measure_floor=True)

    assert cli._run_score(args) == 0
    assert len(calls) == 2  # floor plus candidate score
    assert calls[0][0] == calls[0][1]


def test_run_score_measure_floor_trajectory_identity_failure_aborts(
    tmp_path, monkeypatch, capsys
):
    _patch_backends(monkeypatch)

    def broken_trajectory(**kw):
        from refract.axes.trajectory import TrajectoryResult
        return TrajectoryResult(
            score=100.0, full_match_rate=0.0, median_first_divergence=50,
            mean_prefix_agreement_length=50.0,
            mean_cand_length=50.0, mean_ref_length=100.0,
            n_prompts=1, n_tokens_each=128, per_prompt=[], notes=[],
        )

    monkeypatch.setattr(cli, "run_trajectory", broken_trajectory)
    monkeypatch.setattr(
        cli, "run_kld",
        lambda **kw: _kld_result(100.0, mean_kld=0.0),
    )
    args = _make_score_args(tmp_path, measure_floor=True)

    assert cli._run_score(args) == 2
    output = capsys.readouterr().out
    assert "trajectory ref-vs-ref" in output
    assert "token-identical" in output


def test_run_score_measure_floor_honors_skip_kld(
    tmp_path, monkeypatch
):
    _patch_backends(monkeypatch)
    monkeypatch.setattr(cli, "run_trajectory", lambda **kw: _traj_result(100.0))
    monkeypatch.setattr(
        cli, "run_kld",
        lambda **kw: pytest.fail("--skip-kld must skip KLD floor and score"),
    )
    args = _make_score_args(
        tmp_path, measure_floor=True, skip_kld=True, corpus=None,
    )
    assert cli._run_score(args) == 0


def test_run_score_measure_floor_honors_skip_axis_a(
    tmp_path, monkeypatch
):
    _patch_backends(monkeypatch)
    calls = {"kld": 0}

    def fake_kld(**kw):
        calls["kld"] += 1
        return _kld_result(100.0, mean_kld=0.0)

    monkeypatch.setattr(cli, "run_kld", fake_kld)
    monkeypatch.setattr(
        cli, "run_gtm",
        lambda **kw: pytest.fail("--skip-gtm must skip legacy GTM"),
    )
    monkeypatch.setattr(
        cli, "run_trajectory",
        lambda **kw: pytest.fail("--skip-gtm must skip Trajectory"),
    )
    args = _make_score_args(tmp_path, measure_floor=True, skip_gtm=True)

    assert cli._run_score(args) == 0
    assert calls["kld"] == 2  # floor plus candidate score


def test_run_score_measure_floor_rejects_both_default_axes_skipped(
    tmp_path, capsys
):
    args = _make_score_args(
        tmp_path, measure_floor=True, skip_gtm=True, skip_kld=True,
    )
    assert cli._run_score(args) == 2
    assert "requires at least one active" in capsys.readouterr().out


def test_run_score_floor_token_identity_failure_aborts(
    tmp_path, monkeypatch, capsys
):
    """A strict-prefix ref-vs-ref result must fail even with score 100."""
    _patch_backends(monkeypatch)

    def _broken_gtm(**kw):
        from refract.axes.gtm import GTMResult
        return GTMResult(
            score=100.0, full_match_rate=0.0, median_first_divergence=50,
            # This ratio was 1.0 under the old candidate-only check even
            # though the reference trajectory continued for 50 more tokens.
            mean_prefix_agreement_length=50.0,
            mean_cand_length=50.0, mean_ref_length=100.0,
            n_prompts=1, n_tokens_each=128, per_prompt=[], notes=[],
        )

    monkeypatch.setattr(cli, "run_gtm", _broken_gtm)
    monkeypatch.setattr(cli, "run_kld",
                        lambda **kw: _kld_result(100.0, mean_kld=0.0))
    args = _make_score_args(tmp_path, axis_a="gtm", measure_floor=True)
    rc = cli._run_score(args)
    assert rc == 2
    assert "token-identical" in capsys.readouterr().out


def test_run_selftest_model_probe_succeeds(tmp_path, monkeypatch, capsys):
    """Selftest with --model present + working backend → probe success path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("llama-cli", "llama-completion",
                 "llama-tokenize", "llama-perplexity"):
        (bin_dir / tool).write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    # llama-completion --help check passes
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: mock.MagicMock(stdout="--jinja flag\n"),
    )

    class _GoodBackend(_FakeBackend):
        def run_completion(self, **kw):
            return CompletionResult(text="4", n_tokens=1, metadata={})

    import refract.backends as bk_mod
    monkeypatch.setattr(bk_mod, "auto_backend",
                        lambda model: _GoodBackend())

    model_path = tmp_path / "m.gguf"
    model_path.write_text("")
    args = argparse.Namespace(backend="auto", model=model_path)
    rc = cli._run_selftest(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "generation works" in out
    assert "no thinking-mode markers" in out


def test_run_selftest_model_probe_generation_fails(tmp_path, monkeypatch, capsys):
    """Selftest --model where generation raises → recorded as failure."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("llama-cli", "llama-completion",
                 "llama-tokenize", "llama-perplexity"):
        (bin_dir / tool).write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: mock.MagicMock(stdout="--jinja\n"),
    )

    class _BoomBackend(_FakeBackend):
        def run_completion(self, **kw):
            raise RuntimeError("model exploded")

    import refract.backends as bk_mod
    monkeypatch.setattr(bk_mod, "auto_backend", lambda m: _BoomBackend())

    model_path = tmp_path / "m.gguf"
    model_path.write_text("")
    args = argparse.Namespace(backend="auto", model=model_path)
    rc = cli._run_selftest(args)
    assert rc == 2
    assert "generation failed" in capsys.readouterr().out


def test_run_selftest_jinja_missing_records_failure(tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("llama-cli", "llama-completion",
                 "llama-tokenize", "llama-perplexity"):
        (bin_dir / tool).write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    # No --jinja in help
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: mock.MagicMock(stdout="some other help text"),
    )
    args = argparse.Namespace(backend="llamacpp", model=None)
    rc = cli._run_selftest(args)
    assert rc == 2
    out = capsys.readouterr().out
    assert "--jinja missing" in out


def test_run_selftest_help_probe_exception_is_warning(tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("llama-cli", "llama-completion",
                 "llama-tokenize", "llama-perplexity"):
        (bin_dir / tool).write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)

    def boom(*a, **kw):
        raise RuntimeError("subprocess died")

    monkeypatch.setattr("subprocess.run", boom)
    args = argparse.Namespace(backend="llamacpp", model=None)
    rc = cli._run_selftest(args)
    # Warnings only → rc 0
    assert rc == 0
    out = capsys.readouterr().out
    assert "warning" in out.lower()


def test_run_selftest_mlx_backend_handles_missing_mlx(monkeypatch, capsys):
    import sys
    # Hide mlx imports so _require_mlx raises BackendCapabilityError.
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    args = argparse.Namespace(backend="mlx", model=None)
    rc = cli._run_selftest(args)
    assert rc == 2
    out = capsys.readouterr().out
    assert "mlx not importable" in out


def test_run_repeatability_warning_when_json_unparseable(tmp_path, monkeypatch, capsys):
    _patch_backends(monkeypatch)

    def fake_run_score(args):
        # Write invalid JSON
        args.json_out.write_text("not valid json")
        return 0

    monkeypatch.setattr(cli, "_run_score", fake_run_score)
    args = argparse.Namespace(
        model=tmp_path / "m.gguf",
        reference="ctk=f16,ctv=f16", candidate="ctk=q8_0,ctv=q8_0",
        prompts=tmp_path / "p.jsonl", corpus=tmp_path / "c.txt",
        runs=2, n_predict=8, ctx=512, chunks=32, n_gpu_layers=99, seed=42,
        axis_a="trajectory", full=False,
        rniah_haystack=None, rniah_ctx_max=None,
        backend="auto", out_dir=tmp_path / "out",
    )
    rc = cli._run_repeatability(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "could not parse" in out


def test_run_selftest_detects_linux_shared_library_error(tmp_path, monkeypatch, capsys):
    """Bug: a binary that can't find libllama.so emits empty --help and a
    'cannot open shared object' stderr. Old selftest read empty stdout and
    blamed '--jinja missing'. New selftest detects the real launch failure
    AND emits a Linux-specific remediation hint."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("llama-cli", "llama-completion",
                 "llama-tokenize", "llama-perplexity"):
        (bin_dir / tool).write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: mock.MagicMock(
            stdout="",
            stderr=("/opt/llama.cpp/bin/llama-completion: error while "
                    "loading shared libraries: libllama.so.0: cannot open "
                    "shared object file: No such file or directory"),
            returncode=127,
        ),
    )
    args = argparse.Namespace(backend="llamacpp", model=None)
    rc = cli._run_selftest(args)
    assert rc == 2
    out = capsys.readouterr().out
    assert "can't launch" in out
    assert "LD_LIBRARY_PATH" in out or "ldconfig" in out
    # Must NOT mistakenly blame --jinja
    assert "--jinja missing" not in out


def test_run_selftest_detects_windows_dll_error(tmp_path, monkeypatch, capsys):
    """Same root cause on Windows: DLLs not on PATH → empty --help, code
    0xc0000135. Selftest gives the right remediation per OS."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("llama-cli", "llama-completion",
                 "llama-tokenize", "llama-perplexity"):
        (bin_dir / tool).write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: mock.MagicMock(
            stdout="",
            stderr=("The application failed to start because llama.dll was "
                    "not found. Error 0xc0000135."),
            returncode=3221225781,  # 0xc0000135
        ),
    )
    args = argparse.Namespace(backend="llamacpp", model=None)
    rc = cli._run_selftest(args)
    assert rc == 2
    out = capsys.readouterr().out
    assert "can't launch" in out
    assert "PATH" in out  # Windows hint mentions adding bin dir to PATH


def test_run_selftest_launch_succeeds_then_jinja_check_runs(tmp_path, monkeypatch, capsys):
    """When binary launches OK + --jinja in help → success path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("llama-cli", "llama-completion",
                 "llama-tokenize", "llama-perplexity"):
        (bin_dir / tool).write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: mock.MagicMock(
            stdout="usage: ...\n--jinja apply chat template\n",
            stderr="", returncode=0,
        ),
    )
    args = argparse.Namespace(backend="llamacpp", model=None)
    rc = cli._run_selftest(args)
    out = capsys.readouterr().out
    assert "--jinja chat template flag supported" in out
    # No misleading launch-failure message
    assert "can't launch" not in out


def test_run_repeatability_noisy_label(tmp_path, monkeypatch, capsys):
    """Composite stdev between 1.0 and 3.0 → NOISY warning, not unstable."""
    _patch_backends(monkeypatch)
    seq = iter([90.0, 92.0, 90.0, 92.0])  # stdev ~1.15

    def fake_run_score(args):
        rep = {"composite": next(seq),
               "axes": {"gtm": {"score": 90.0}, "kld": {"score": 90.0}}}
        args.json_out.write_text(json.dumps(rep))
        return 0

    monkeypatch.setattr(cli, "_run_score", fake_run_score)
    args = argparse.Namespace(
        model=tmp_path / "m.gguf",
        reference="ctk=f16,ctv=f16", candidate="ctk=q8_0,ctv=q8_0",
        prompts=tmp_path / "p.jsonl", corpus=tmp_path / "c.txt",
        runs=4, n_predict=8, ctx=512, chunks=32, n_gpu_layers=99, seed=42,
        axis_a="trajectory", full=False,
        rniah_haystack=None, rniah_ctx_max=None,
        backend="auto", out_dir=tmp_path / "out",
    )
    rc = cli._run_repeatability(args)
    assert rc == 0
    assert "NOISY" in capsys.readouterr().out
