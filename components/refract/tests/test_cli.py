"""Tests for refract.cli: argparse + subcommand dispatch + helpers.

Heavy I/O paths (real downloads, real subprocesses) are mocked.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest import mock

import pytest

import refract.cli as cli
from refract.cli import (
    _ensure_wikitext_2,
    _resolve_default_paths,
    _run_compare,
    _run_fetch,
    _stub_gtm,
    _stub_kld,
    main,
)


# --- stub helpers --------------------------------------------------------


def test_stub_gtm_returns_perfect_result():
    r = _stub_gtm()
    assert r.score == 100.0
    assert r.full_match_rate == 1.0
    assert r.n_prompts == 0


def test_stub_kld_returns_perfect_result():
    r = _stub_kld(chunks=32, ctx=512)
    assert r.score == 100.0
    assert r.mean_kld == 0.0
    assert r.chunks == 32
    assert r.ctx == 512


# --- main / argparse -----------------------------------------------------


def test_main_no_args_errors():
    with pytest.raises(SystemExit):
        main([])


def test_main_unknown_subcommand_errors():
    with pytest.raises(SystemExit):
        main(["totally-not-a-subcommand"])


def test_score_help_includes_axes_and_acronym(capsys):
    with pytest.raises(SystemExit):
        main(["score", "--help"])
    out = capsys.readouterr().out
    assert "Greedy Trajectory Match" in out
    assert "KL Divergence" in out
    assert "Retrieval Needle-In-A-Haystack" in out
    assert "Perturbation-Locality Aware Drift" in out


def test_top_level_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("score", "selftest", "compare", "fetch", "repeatability"):
        assert cmd in out


# --- _ensure_wikitext_2 (mock urlretrieve + zipfile) ---------------------


def test_ensure_wikitext_2_idempotent_when_cached(tmp_path):
    target = tmp_path / "wikitext-2-raw"
    target.mkdir()
    (target / "wiki.test.raw").write_text("test data")
    (target / "wiki.train.raw").write_text("train data")
    out = _ensure_wikitext_2(cache_dir=tmp_path, silent=True)
    assert out == target


def test_ensure_wikitext_2_downloads_when_missing(tmp_path, monkeypatch):
    """_ensure_wikitext_2 should fetch + extract when cache empty."""
    import zipfile

    def fake_urlretrieve(url, dest):
        # Build a minimal zip with the two expected files.
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("wikitext-2-raw/wiki.test.raw", "test data")
            zf.writestr("wikitext-2-raw/wiki.train.raw", "train data")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
    out = _ensure_wikitext_2(cache_dir=tmp_path, silent=True)
    assert (out / "wiki.test.raw").exists()
    assert (out / "wiki.train.raw").exists()


def test_ensure_wikitext_2_raises_if_unzip_missing_files(tmp_path, monkeypatch):
    import zipfile

    def fake_urlretrieve(url, dest):
        # Empty zip — neither expected file present
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("README.txt", "nothing useful")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
    with pytest.raises(RuntimeError):
        _ensure_wikitext_2(cache_dir=tmp_path, silent=True)


# --- _resolve_default_paths ----------------------------------------------


def test_resolve_default_paths_no_op_when_neither_needed():
    args = argparse.Namespace()
    _resolve_default_paths(args, need_corpus=False, need_haystack=False)
    # No attributes added
    assert not hasattr(args, "corpus")


def test_resolve_default_paths_no_op_when_user_supplied(tmp_path, monkeypatch):
    # User passed both → no auto-resolve.
    args = argparse.Namespace(
        corpus=tmp_path / "user.corpus",
        rniah_haystack=tmp_path / "user.hay",
        no_auto_fetch=False,
    )
    monkeypatch.setattr(cli, "_REFRACT_CACHE", tmp_path / "cache")
    _resolve_default_paths(args, need_corpus=True, need_haystack=True)
    assert args.corpus == tmp_path / "user.corpus"


def test_resolve_default_paths_no_auto_fetch_raises_when_missing(tmp_path, monkeypatch):
    args = argparse.Namespace(corpus=None, no_auto_fetch=True, rniah_haystack=None)
    # Empty cache dir
    monkeypatch.setattr(cli, "_REFRACT_CACHE", tmp_path)
    with pytest.raises(SystemExit):
        _resolve_default_paths(args, need_corpus=True, need_haystack=False)


def test_resolve_default_paths_uses_cache_when_present(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache"
    target = cache / "wikitext-2-raw"
    target.mkdir(parents=True)
    (target / "wiki.test.raw").write_text("t")
    (target / "wiki.train.raw").write_text("t")
    args = argparse.Namespace(corpus=None, rniah_haystack=None, no_auto_fetch=False)
    monkeypatch.setattr(cli, "_REFRACT_CACHE", cache)
    _resolve_default_paths(args, need_corpus=True, need_haystack=True)
    assert args.corpus == target / "wiki.test.raw"
    assert args.rniah_haystack == target / "wiki.train.raw"


# --- _run_fetch ----------------------------------------------------------


def test_run_fetch_idempotent(tmp_path, capsys):
    target = tmp_path / "wikitext-2-raw"
    target.mkdir()
    (target / "wiki.test.raw").write_text("t")
    (target / "wiki.train.raw").write_text("t")
    args = argparse.Namespace(cache_dir=tmp_path)
    rc = _run_fetch(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "wiki.test.raw" in out


# --- _run_compare --------------------------------------------------------


def _write_report(path: Path, **overrides):
    rep = {
        "composite": 92.5,
        "band": "EXCELLENT",
        "summary": "ok",
        "framework_version": "0.3.2",
        "environment": {"backend": "llamacpp"},
        "axes": {
            "gtm": {"score": 95.0, "band": "EXCELLENT"},
            "kld": {"score": 90.0, "band": "EXCELLENT"},
        },
    }
    rep.update(overrides)
    path.write_text(json.dumps(rep))


def test_run_compare_two_reports(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_report(a)
    _write_report(b, composite=72.5, band="DEGRADED")
    rc = _run_compare(argparse.Namespace(reports=[a, b]))
    assert rc == 0
    out = capsys.readouterr().out
    assert "a" in out and "b" in out
    assert "EXCELLENT" in out
    assert "DEGRADED" in out


def test_run_compare_skips_unparseable(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    good = tmp_path / "g.json"
    _write_report(good)
    rc = _run_compare(argparse.Namespace(reports=[bad, good]))
    assert rc == 0
    out = capsys.readouterr().out
    assert "skip" in out


def test_run_compare_no_parseable_returns_1(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    rc = _run_compare(argparse.Namespace(reports=[bad]))
    assert rc == 1


def test_run_compare_handles_missing_axis_score(tmp_path, capsys):
    p = tmp_path / "no_axes.json"
    _write_report(p, axes={})
    rc = _run_compare(argparse.Namespace(reports=[p]))
    assert rc == 0
    out = capsys.readouterr().out
    # Em dash placeholder for missing axes
    assert "—" in out


# --- main dispatch -------------------------------------------------------


def test_main_compare_dispatches(tmp_path, capsys):
    p = tmp_path / "r.json"
    _write_report(p)
    rc = main(["compare", str(p)])
    assert rc == 0


def test_main_fetch_dispatches(tmp_path, capsys):
    target = tmp_path / "wikitext-2-raw"
    target.mkdir()
    (target / "wiki.test.raw").write_text("t")
    (target / "wiki.train.raw").write_text("t")
    rc = main(["fetch", "--cache-dir", str(tmp_path)])
    assert rc == 0
