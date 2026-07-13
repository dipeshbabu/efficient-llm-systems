"""Higher-level CLI tests: _run_score, _run_selftest, _run_repeatability.

All heavy paths (subprocess, runner, axes) are mocked so the dispatcher
logic gets exercised without real models.
"""

from __future__ import annotations

import argparse
import json
from unittest import mock

import pytest

import refract.cli as cli
from refract.axes.gtm import GTMResult
from refract.axes.kld import KLDResult as AxisKLDResult
from refract.axes.plad import PLADResult
from refract.axes.rniah import RNIAHCell, RNIAHResult
from refract.axes.trajectory import TrajectoryResult

# --- helpers --------------------------------------------------------------


class _FakeBackend:
    name = "llamacpp"

    def detect_thinking_mode(self, *, model, timeout=30.0):
        return False, []

    def model_metadata(self, *, model):
        return {"backend": self.name, "model": str(model)}


def _gtm_result(score=95.0):
    perfect = score == 100.0
    return GTMResult(
        score=score,
        full_match_rate=1.0 if perfect else 0.9,
        median_first_divergence=None if perfect else 10,
        mean_prefix_agreement_length=score,
        mean_cand_length=100,
        mean_ref_length=100,
        n_prompts=1,
        n_tokens_each=128,
        per_prompt=[],
        notes=[],
    )


def _traj_result(score=95.0):
    perfect = score == 100.0
    return TrajectoryResult(
        score=score,
        full_match_rate=1.0 if perfect else 0.9,
        median_first_divergence=None if perfect else 10,
        mean_prefix_agreement_length=score,
        mean_cand_length=100,
        mean_ref_length=100,
        n_prompts=1,
        n_tokens_each=128,
        per_prompt=[],
        notes=[],
    )


def _kld_result(score=99.0, mean_kld=0.01):
    return AxisKLDResult(
        score=score,
        mean_kld=mean_kld,
        ppl=8.5,
        rms_dp_pct=1.0,
        same_topp_pct=99.0,
        base_path="",
        chunks=32,
        ctx=512,
        is_self_reference=False,
        corpus={
            "path": "x",
            "size_bytes": 1,
            "sha256_head": "a" * 64,
            "sha256_head_bytes": 1,
        },
    )


def _patch_backends(monkeypatch):
    # Both _run_score and _run_repeatability do `from .backends import
    # auto_backend, get_backend` at call time, so patch on the backends module.
    import refract.backends as bk_mod

    monkeypatch.setattr(bk_mod, "auto_backend", lambda model: _FakeBackend())
    monkeypatch.setattr(bk_mod, "get_backend", lambda name: _FakeBackend())


# --- _run_score ----------------------------------------------------------


def _make_score_args(tmp_path, **overrides):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "p1", "prompt": "x"}))
    corpus = tmp_path / "c.txt"
    corpus.write_text("text")
    base = dict(
        model=tmp_path / "m.gguf",
        reference="ctk=f16,ctv=f16",
        candidate="ctk=q8_0,ctv=q8_0",
        prompts=prompts,
        corpus=corpus,
        chunks=32,
        ctx=512,
        n_gpu_layers=99,
        n_predict=8,
        seed=42,
        measure_floor=False,
        skip_gtm=False,
        skip_kld=False,
        axis_a="trajectory",
        full=False,
        axis_rniah=False,
        axis_plad=False,
        rniah_haystack=None,
        rniah_ctx_max=None,
        rniah_lengths=None,
        rniah_positions=None,
        rniah_trials=1,
        rniah_up_to=16384,
        json_out=None,
        html_out=None,
        no_progress=True,
        backend="auto",
        no_auto_fetch=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_run_score_default_two_axis(tmp_path, monkeypatch, capsys):
    _patch_backends(monkeypatch)
    monkeypatch.setattr(cli, "run_trajectory", lambda **kw: _traj_result(95.0))
    monkeypatch.setattr(cli, "run_kld", lambda **kw: _kld_result(99.0))
    args = _make_score_args(tmp_path)
    rc = cli._run_score(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "REFRACT score" in out


def test_run_score_axis_a_gtm_branch(tmp_path, monkeypatch):
    _patch_backends(monkeypatch)
    monkeypatch.setattr(cli, "run_gtm", lambda **kw: _gtm_result(85.0))
    monkeypatch.setattr(cli, "run_kld", lambda **kw: _kld_result(95.0))
    args = _make_score_args(tmp_path, axis_a="gtm")
    rc = cli._run_score(args)
    assert rc == 0


def test_run_score_full_flag_enables_rniah_and_plad(tmp_path, monkeypatch):
    _patch_backends(monkeypatch)
    monkeypatch.setattr(cli, "run_trajectory", lambda **kw: _traj_result(95.0))
    monkeypatch.setattr(cli, "run_kld", lambda **kw: _kld_result(99.0))
    captured = {"rniah_called": False, "plad_called": False}

    def _rniah(**kw):
        captured["rniah_called"] = True
        return RNIAHResult(
            score=95.0,
            n_cells=1,
            cells=[
                RNIAHCell(
                    length=4096,
                    position=0.5,
                    n_trials=1,
                    base_acc=1.0,
                    cand_acc=1.0,
                    degradation=0.0,
                )
            ],
            skipped_cells=[],
            needle="X",
            password_keyword="X",
        )

    def _plad(**kw):
        captured["plad_called"] = True
        return PLADResult(
            score=88.0,
            per_perturbation_score={"typo": 88.0},
            per_prompt=[],
            n_prompts=1,
            n_perturbations=1,
            notes=[],
        )

    monkeypatch.setattr(cli, "run_rniah", _rniah)
    monkeypatch.setattr(cli, "run_plad", _plad)
    haystack = tmp_path / "haystack.raw"
    haystack.write_text("haystack content")
    args = _make_score_args(tmp_path, full=True, rniah_haystack=haystack)
    rc = cli._run_score(args)
    assert rc == 0
    assert captured["rniah_called"]
    assert captured["plad_called"]


def test_run_score_excludes_low_confidence_rniah_from_composite(tmp_path, monkeypatch):
    _patch_backends(monkeypatch)
    monkeypatch.setattr(cli, "run_trajectory", lambda **kw: _traj_result(90.0))
    monkeypatch.setattr(cli, "run_kld", lambda **kw: _kld_result(90.0))
    monkeypatch.setattr(
        cli,
        "run_rniah",
        lambda **kw: RNIAHResult(
            score=100.0,
            n_cells=1,
            cells=[
                RNIAHCell(
                    length=4096,
                    position=0.5,
                    n_trials=1,
                    base_acc=0.0,
                    cand_acc=0.0,
                    degradation=0.0,
                )
            ],
            skipped_cells=[],
            needle="X",
            password_keyword="X",
        ),
    )
    haystack = tmp_path / "haystack.raw"
    haystack.write_text("haystack", encoding="utf-8")
    report_path = tmp_path / "report.json"
    args = _make_score_args(
        tmp_path,
        axis_rniah=True,
        rniah_haystack=haystack,
        json_out=report_path,
    )
    assert cli._run_score(args) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["composite"] == pytest.approx(90.0)
    assert report["axes"]["rniah"]["excluded_from_composite"] is True


def test_run_score_skip_gtm_uses_stub(tmp_path, monkeypatch):
    _patch_backends(monkeypatch)
    monkeypatch.setattr(cli, "run_kld", lambda **kw: _kld_result(99.0))
    args = _make_score_args(tmp_path, skip_gtm=True)
    rc = cli._run_score(args)
    assert rc == 0


def test_run_score_skip_kld_uses_stub(tmp_path, monkeypatch):
    _patch_backends(monkeypatch)
    monkeypatch.setattr(cli, "run_trajectory", lambda **kw: _traj_result(95.0))
    args = _make_score_args(tmp_path, skip_kld=True)
    rc = cli._run_score(args)
    assert rc == 0


def test_run_score_thinking_probe_failure_handled(tmp_path, monkeypatch, capsys):
    class _BoomBackend(_FakeBackend):
        def detect_thinking_mode(self, **kw):
            raise RuntimeError("probe blew up")

    import refract.backends as bk_mod

    monkeypatch.setattr(bk_mod, "auto_backend", lambda model: _BoomBackend())
    monkeypatch.setattr(bk_mod, "get_backend", lambda name: _BoomBackend())
    monkeypatch.setattr(cli, "run_trajectory", lambda **kw: _traj_result(95.0))
    monkeypatch.setattr(cli, "run_kld", lambda **kw: _kld_result(99.0))
    args = _make_score_args(tmp_path)
    rc = cli._run_score(args)
    assert rc == 0
    assert "probe failed" in capsys.readouterr().out


def test_run_score_axis_rniah_without_haystack_fails(tmp_path, monkeypatch, capsys):
    _patch_backends(monkeypatch)
    monkeypatch.setattr(cli, "run_trajectory", lambda **kw: _traj_result(95.0))
    monkeypatch.setattr(cli, "run_kld", lambda **kw: _kld_result(99.0))
    # Force _resolve_default_paths to no-op so rniah_haystack stays None
    # and the explicit check inside _run_score fires.
    monkeypatch.setattr(cli, "_resolve_default_paths", lambda *a, **kw: None)
    args = _make_score_args(tmp_path, axis_rniah=True, no_auto_fetch=True)
    rc = cli._run_score(args)
    assert rc == 2
    assert "rniah-haystack" in capsys.readouterr().out.lower()


def test_run_score_writes_json_and_html(tmp_path, monkeypatch):
    _patch_backends(monkeypatch)
    monkeypatch.setattr(cli, "run_trajectory", lambda **kw: _traj_result(95.0))
    monkeypatch.setattr(cli, "run_kld", lambda **kw: _kld_result(99.0))
    json_out = tmp_path / "report.json"
    html_out = tmp_path / "report.html"
    args = _make_score_args(tmp_path, json_out=json_out, html_out=html_out)
    rc = cli._run_score(args)
    assert rc == 0
    assert json_out.exists()
    assert html_out.exists()
    rep = json.loads(json_out.read_text())
    assert rep["schema"].startswith("refract.report")
    assert "<!DOCTYPE html>" in html_out.read_text()


def test_run_score_floor_failure_aborts(tmp_path, monkeypatch, capsys):
    _patch_backends(monkeypatch)
    # Floor's reference-vs-reference must score 100 to pass; force a low
    # GTM score so floor fails.
    monkeypatch.setattr(cli, "run_gtm", lambda **kw: _gtm_result(70.0))
    monkeypatch.setattr(cli, "run_kld", lambda **kw: _kld_result(70.0))
    args = _make_score_args(tmp_path, measure_floor=True, axis_a="gtm")
    rc = cli._run_score(args)
    assert rc == 2
    assert "noise floor" in capsys.readouterr().out.lower()


# --- _run_selftest -------------------------------------------------------


def test_run_selftest_static_only_llamacpp(tmp_path, monkeypatch, capsys):
    """Selftest without --model → static checks only."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("llama-cli", "llama-completion", "llama-tokenize", "llama-perplexity"):
        (bin_dir / tool).write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)

    def fake_run(*a, **kw):
        # Pretend --jinja IS in help text so selftest passes
        return mock.MagicMock(stdout="--jinja flag supported\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    args = argparse.Namespace(backend="llamacpp", model=None)
    rc = cli._run_selftest(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "selftest passed" in out or "warning" in out


def test_run_selftest_missing_binaries_fails(tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / "empty"
    bin_dir.mkdir()
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: mock.MagicMock(stdout=""),
    )
    args = argparse.Namespace(backend="llamacpp", model=None)
    rc = cli._run_selftest(args)
    assert rc == 2  # failures present
    assert "FAILED" in capsys.readouterr().out


def test_run_selftest_vllm_backend_when_unavailable(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "vllm", None)
    args = argparse.Namespace(backend="vllm", model=None)
    rc = cli._run_selftest(args)
    assert rc == 2
    assert "vllm not importable" in capsys.readouterr().out


def test_run_selftest_model_missing_fails(tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("llama-cli", "llama-completion", "llama-tokenize", "llama-perplexity"):
        (bin_dir / tool).write_text("")
    monkeypatch.setattr("refract.runner.DEFAULT_BIN_DIR", bin_dir)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: mock.MagicMock(stdout="--jinja\n"),
    )
    args = argparse.Namespace(
        backend="llamacpp",
        model=tmp_path / "missing.gguf",
    )
    rc = cli._run_selftest(args)
    assert rc == 2
    assert (
        "model missing" in capsys.readouterr().out.lower()
        or "model not found" in capsys.readouterr().out.lower()
    )


# --- _run_repeatability --------------------------------------------------


def test_run_repeatability_rejects_non_positive_runs_before_resolution(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        cli,
        "_resolve_default_prompts",
        lambda args: pytest.fail("invalid run count must fail before resolution"),
    )
    rc = cli._run_repeatability(argparse.Namespace(runs=0))
    assert rc == 2
    assert "at least 1" in capsys.readouterr().out


def test_run_repeatability_aggregates_runs(tmp_path, monkeypatch, capsys):
    _patch_backends(monkeypatch)
    out_dir = tmp_path / "out"

    def fake_run_score(args):
        # Write a fake report each run
        rep = {
            "composite": 92.0 + args.seed * 0.0,  # constant
            "axes": {
                "gtm": {"score": 95.0},
                "kld": {"score": 90.0},
            },
        }
        args.json_out.write_text(json.dumps(rep))
        return 0

    monkeypatch.setattr(cli, "_run_score", fake_run_score)

    args = argparse.Namespace(
        model=tmp_path / "m.gguf",
        reference="ctk=f16,ctv=f16",
        candidate="ctk=q8_0,ctv=q8_0",
        prompts=tmp_path / "p.jsonl",
        corpus=tmp_path / "c.txt",
        runs=3,
        n_predict=8,
        ctx=512,
        chunks=32,
        n_gpu_layers=99,
        seed=42,
        axis_a="trajectory",
        full=False,
        rniah_haystack=None,
        rniah_ctx_max=None,
        backend="auto",
        out_dir=out_dir,
    )
    rc = cli._run_repeatability(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "REPEATABILITY" in out
    assert "HEALTHY" in out  # stdev=0 → healthy


def test_run_repeatability_resolves_defaults_once(tmp_path, monkeypatch):
    _patch_backends(monkeypatch)
    prompts = tmp_path / "bundled.jsonl"
    corpus = tmp_path / "wiki.test.raw"
    haystack = tmp_path / "wiki.train.raw"
    calls = {"prompts": 0, "paths": 0, "runs": []}

    def fake_resolve_prompts(args):
        calls["prompts"] += 1
        args.prompts = prompts
        return True

    def fake_resolve_paths(args, *, need_corpus, need_haystack):
        calls["paths"] += 1
        assert need_corpus is True
        assert need_haystack is True
        args.corpus = corpus
        args.rniah_haystack = haystack

    def fake_run_score(args):
        calls["runs"].append(args)
        args.json_out.write_text(
            json.dumps(
                {
                    "composite": 100.0,
                    "axes": {"gtm": {"score": 100.0}, "kld": {"score": 100.0}},
                }
            )
        )
        return 0

    monkeypatch.setattr(cli, "_resolve_default_prompts", fake_resolve_prompts)
    monkeypatch.setattr(cli, "_resolve_default_paths", fake_resolve_paths)
    monkeypatch.setattr(cli, "_run_score", fake_run_score)
    args = argparse.Namespace(
        model=tmp_path / "m.gguf",
        reference="ctk=f16,ctv=f16",
        candidate="ctk=q8_0,ctv=q8_0",
        prompts=None,
        corpus=None,
        no_auto_fetch=True,
        runs=2,
        n_predict=8,
        ctx=512,
        chunks=32,
        n_gpu_layers=99,
        seed=42,
        axis_a="trajectory",
        full=True,
        rniah_haystack=None,
        rniah_ctx_max=None,
        backend="auto",
        out_dir=tmp_path / "out-defaults",
    )

    assert cli._run_repeatability(args) == 0
    assert calls["prompts"] == 1
    assert calls["paths"] == 1
    assert len(calls["runs"]) == 2
    for run_args in calls["runs"]:
        assert run_args.prompts == prompts
        assert run_args.corpus == corpus
        assert run_args.rniah_haystack == haystack
        assert run_args.no_auto_fetch is True


def test_run_repeatability_real_resolvers_use_offline_cache(tmp_path, monkeypatch):
    _patch_backends(monkeypatch)
    cache = tmp_path / "cache"
    target = cache / "wikitext-2-raw"
    target.mkdir(parents=True)
    corpus = target / "wiki.test.raw"
    haystack = target / "wiki.train.raw"
    corpus.write_text("test")
    haystack.write_text("train")
    monkeypatch.setattr(cli, "_REFRACT_CACHE", cache)
    monkeypatch.setattr(
        cli,
        "_ensure_wikitext_2",
        lambda *a, **kw: pytest.fail("offline cache should not download"),
    )
    captured = []

    def fake_run_score(args):
        captured.append(args)
        args.json_out.write_text(
            json.dumps(
                {
                    "composite": 100.0,
                    "axes": {"gtm": {"score": 100.0}, "kld": {"score": 100.0}},
                }
            )
        )
        return 0

    monkeypatch.setattr(cli, "_run_score", fake_run_score)
    args = argparse.Namespace(
        model=tmp_path / "m.gguf",
        reference="ctk=f16,ctv=f16",
        candidate="ctk=q8_0,ctv=q8_0",
        prompts=None,
        corpus=None,
        no_auto_fetch=True,
        runs=1,
        n_predict=8,
        ctx=512,
        chunks=32,
        n_gpu_layers=99,
        seed=42,
        axis_a="trajectory",
        full=True,
        rniah_haystack=None,
        rniah_ctx_max=None,
        backend="auto",
        out_dir=tmp_path / "out-real-defaults",
    )

    assert cli._run_repeatability(args) == 0
    assert len(captured) == 1
    run_args = captured[0]
    assert run_args.prompts.name == "v0.1.jsonl"
    assert run_args.prompts.is_file()
    assert run_args.corpus == corpus
    assert run_args.rniah_haystack == haystack
    assert run_args.no_auto_fetch is True


def test_run_repeatability_aborts_on_score_failure(tmp_path, monkeypatch, capsys):
    _patch_backends(monkeypatch)
    monkeypatch.setattr(cli, "_run_score", lambda args: 5)
    args = argparse.Namespace(
        model=tmp_path / "m.gguf",
        reference="ctk=f16,ctv=f16",
        candidate="ctk=q8_0,ctv=q8_0",
        prompts=tmp_path / "p.jsonl",
        corpus=tmp_path / "c.txt",
        runs=2,
        n_predict=8,
        ctx=512,
        chunks=32,
        n_gpu_layers=99,
        seed=42,
        axis_a="trajectory",
        full=False,
        rniah_haystack=None,
        rniah_ctx_max=None,
        backend="auto",
        out_dir=tmp_path / "out",
    )
    rc = cli._run_repeatability(args)
    assert rc == 5


def test_run_repeatability_unstable_label(tmp_path, monkeypatch, capsys):
    _patch_backends(monkeypatch)
    seq = iter([10.0, 90.0, 10.0])  # high variance

    def fake_run_score(args):
        rep = {
            "composite": next(seq),
            "axes": {"gtm": {"score": 50.0}, "kld": {"score": 50.0}},
        }
        args.json_out.write_text(json.dumps(rep))
        return 0

    monkeypatch.setattr(cli, "_run_score", fake_run_score)
    args = argparse.Namespace(
        model=tmp_path / "m.gguf",
        reference="ctk=f16,ctv=f16",
        candidate="ctk=q8_0,ctv=q8_0",
        prompts=tmp_path / "p.jsonl",
        corpus=tmp_path / "c.txt",
        runs=3,
        n_predict=8,
        ctx=512,
        chunks=32,
        n_gpu_layers=99,
        seed=42,
        axis_a="trajectory",
        full=False,
        rniah_haystack=None,
        rniah_ctx_max=None,
        backend="auto",
        out_dir=tmp_path / "out",
    )
    rc = cli._run_repeatability(args)
    assert rc == 0
    assert "UNSTABLE" in capsys.readouterr().out
