"""Tests for kv_fidelity.cli: argparse + subcommand dispatch + helpers.

Heavy I/O paths (real downloads, real subprocesses) are mocked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import kv_fidelity.cli as cli
from kv_fidelity import __version__
from kv_fidelity.cli import (
    _ensure_wikitext_2,
    _resolve_default_paths,
    _resolve_default_prompts,
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


def test_top_level_version_reports_runtime_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"kv-fidelity {__version__}"


def test_repeatability_parser_uses_score_input_defaults(monkeypatch):
    captured = {}

    def fake_run_repeatability(args):
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(cli, "_run_repeatability", fake_run_repeatability)
    rc = main(
        [
            "repeatability",
            "--model",
            "model.gguf",
            "--candidate",
            "ctk=q8_0,ctv=q8_0",
            "--no-auto-fetch",
        ]
    )
    assert rc == 0
    assert captured["prompts"] is None
    assert captured["corpus"] is None
    assert captured["rniah_haystack"] is None
    assert captured["no_auto_fetch"] is True


@pytest.mark.parametrize("runs", ["0", "-1"])
def test_repeatability_parser_rejects_non_positive_runs(runs):
    with pytest.raises(SystemExit):
        main(
            [
                "repeatability",
                "--model",
                "model.gguf",
                "--candidate",
                "ctk=q8_0,ctv=q8_0",
                "--runs",
                runs,
            ]
        )


# --- _ensure_wikitext_2 (real verifier with a pinned synthetic download) ---


def test_ensure_wikitext_2_idempotent_when_cached(tmp_path, pinned_wikitext):
    target = tmp_path / "wikitext-2-raw"
    target.mkdir()
    for name, data in pinned_wikitext.data.items():
        (target / name).write_bytes(data)
    out = _ensure_wikitext_2(cache_dir=tmp_path, silent=True)
    assert out == target


def test_ensure_wikitext_2_downloads_when_missing(tmp_path, pinned_wikitext):
    """_ensure_wikitext_2 should fetch + extract when cache empty."""
    out = _ensure_wikitext_2(cache_dir=tmp_path, silent=True)
    assert (out / "wiki.test.raw").exists()
    assert (out / "wiki.train.raw").exists()


def test_ensure_wikitext_2_raises_if_unzip_missing_files(
    tmp_path, monkeypatch, pinned_wikitext
):
    import hashlib
    import io
    import zipfile
    from dataclasses import replace

    import metria.artifacts as artifacts

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("README.txt", "nothing useful")
    data = buffer.getvalue()
    monkeypatch.setattr(
        cli._corpora,
        "WIKITEXT_ARCHIVE",
        replace(
            pinned_wikitext.archive,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        ),
    )
    monkeypatch.setattr(artifacts, "_open_url", lambda *a: io.BytesIO(data))

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
    monkeypatch.setattr(cli, "_KV_FIDELITY_CACHE", tmp_path / "cache")
    _resolve_default_paths(args, need_corpus=True, need_haystack=True)
    assert args.corpus == tmp_path / "user.corpus"


def test_resolve_default_paths_no_auto_fetch_raises_when_missing(tmp_path, monkeypatch):
    args = argparse.Namespace(corpus=None, no_auto_fetch=True, rniah_haystack=None)
    # Empty cache dir
    monkeypatch.setattr(cli, "_KV_FIDELITY_CACHE", tmp_path)
    with pytest.raises(SystemExit):
        _resolve_default_paths(args, need_corpus=True, need_haystack=False)


def test_resolve_default_paths_uses_cache_when_present(
    tmp_path, monkeypatch, capsys, pinned_wikitext
):
    cache = tmp_path / "cache"
    target = cache / "wikitext-2-raw"
    target.mkdir(parents=True)
    (target / "wiki.test.raw").write_bytes(pinned_wikitext.data["wiki.test.raw"])
    (target / "wiki.train.raw").write_bytes(pinned_wikitext.data["wiki.train.raw"])
    args = argparse.Namespace(corpus=None, rniah_haystack=None, no_auto_fetch=False)
    monkeypatch.setattr(cli, "_KV_FIDELITY_CACHE", cache)
    _resolve_default_paths(args, need_corpus=True, need_haystack=True)
    assert args.corpus == target / "wiki.test.raw"
    assert args.rniah_haystack == target / "wiki.train.raw"


def test_resolve_default_paths_fetches_into_active_cache(
    tmp_path, monkeypatch, pinned_wikitext
):
    cache = tmp_path / "cache"
    calls = []

    def fake_ensure(*, cache_dir, silent=False):
        calls.append((cache_dir, silent))
        target = cache_dir / "wikitext-2-raw"
        target.mkdir(parents=True)
        (target / "wiki.test.raw").write_text("test")
        (target / "wiki.train.raw").write_text("train")
        return target

    args = argparse.Namespace(
        corpus=None,
        rniah_haystack=None,
        no_auto_fetch=False,
    )
    monkeypatch.setattr(cli, "_KV_FIDELITY_CACHE", cache)
    monkeypatch.setattr(cli, "_ensure_wikitext_2", fake_ensure)

    _resolve_default_paths(args, need_corpus=True, need_haystack=True)
    assert calls == [(cache, False)]
    assert args.corpus == cache / "wikitext-2-raw" / "wiki.test.raw"
    assert args.rniah_haystack == cache / "wikitext-2-raw" / "wiki.train.raw"


def test_resolve_default_paths_accepts_partial_cache_for_quick_run(
    tmp_path, monkeypatch, pinned_wikitext
):
    cache = tmp_path / "cache"
    target = cache / "wikitext-2-raw"
    target.mkdir(parents=True)
    (target / "wiki.test.raw").write_text("test")
    args = argparse.Namespace(
        corpus=None,
        rniah_haystack=None,
        no_auto_fetch=True,
    )
    monkeypatch.setattr(cli, "_KV_FIDELITY_CACHE", cache)
    monkeypatch.setattr(
        cli,
        "_ensure_wikitext_2",
        lambda *a, **kw: pytest.fail("partial cache should not download"),
    )

    _resolve_default_paths(args, need_corpus=True, need_haystack=False)
    assert args.corpus == target / "wiki.test.raw"
    assert args.rniah_haystack is None


def test_resolve_default_paths_offline_error_names_missing_haystack(
    tmp_path, monkeypatch, pinned_wikitext
):
    cache = tmp_path / "cache"
    target = cache / "wikitext-2-raw"
    target.mkdir(parents=True)
    (target / "wiki.test.raw").write_text("test")
    args = argparse.Namespace(
        corpus=None,
        rniah_haystack=None,
        no_auto_fetch=True,
    )
    monkeypatch.setattr(cli, "_KV_FIDELITY_CACHE", cache)

    with pytest.raises(SystemExit) as exc:
        _resolve_default_paths(args, need_corpus=True, need_haystack=True)
    message = str(exc.value)
    assert "--rniah-haystack" in message
    assert "--corpus" not in message


# --- _resolve_default_prompts --------------------------------------------


def test_resolve_default_prompts_preserves_explicit_path(tmp_path):
    prompts = tmp_path / "custom.jsonl"
    args = argparse.Namespace(prompts=prompts)
    assert _resolve_default_prompts(args) is True
    assert args.prompts == prompts


def test_resolve_default_prompts_uses_packaged_file():
    args = argparse.Namespace(prompts=None)
    assert _resolve_default_prompts(args) is True
    assert args.prompts.name == "v0.1.jsonl"
    assert args.prompts.is_file()


def test_resolve_default_prompts_reports_missing_resource(monkeypatch, capsys):
    import importlib.resources

    monkeypatch.setattr(
        importlib.resources,
        "files",
        lambda package: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    args = argparse.Namespace(prompts=None)
    assert _resolve_default_prompts(args) is False
    assert "Pass --prompts" in capsys.readouterr().out


# --- _run_fetch ----------------------------------------------------------


def test_run_fetch_idempotent(tmp_path, capsys, pinned_wikitext):
    target = tmp_path / "wikitext-2-raw"
    target.mkdir()
    for name, content in pinned_wikitext.data.items():
        (target / name).write_bytes(content)
    args = argparse.Namespace(cache_dir=tmp_path)
    rc = _run_fetch(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "wiki.test.raw" in out
    assert "not auto-discovered" in out


def test_run_fetch_default_cache_reports_auto_discovery(
    tmp_path, monkeypatch, capsys, pinned_wikitext
):
    target = tmp_path / "wikitext-2-raw"
    target.mkdir()
    for name, content in pinned_wikitext.data.items():
        (target / name).write_bytes(content)
    monkeypatch.setattr(cli, "_KV_FIDELITY_CACHE", tmp_path)

    assert _run_fetch(argparse.Namespace(cache_dir=tmp_path)) == 0
    assert "will auto-find" in capsys.readouterr().out


# --- _run_compare --------------------------------------------------------


def _write_report(path: Path, **overrides):
    from ._fixtures import make_comparable_report

    rep = make_comparable_report()
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


def test_run_compare_rejects_unparseable(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    good = tmp_path / "g.json"
    _write_report(good)
    rc = _run_compare(argparse.Namespace(reports=[bad, good]))
    assert rc == 2
    out = capsys.readouterr().out
    assert "NOT_COMPARABLE" in out


def test_run_compare_single_report_rejected(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    rc = _run_compare(argparse.Namespace(reports=[bad]))
    assert rc == 2


def test_run_compare_handles_missing_axis_score(tmp_path, capsys):
    p = tmp_path / "no_axes.json"
    _write_report(p, axes={})
    rc = _run_compare(argparse.Namespace(reports=[p, p]))
    assert rc == 2
    out = capsys.readouterr().out
    # Em dash placeholder for missing axes
    assert "—" in out


# --- main dispatch -------------------------------------------------------


def test_main_compare_dispatches(tmp_path, capsys):
    p = tmp_path / "r.json"
    _write_report(p)
    rc = main(["compare", str(p)])
    assert rc == 2


def test_main_fetch_dispatches(tmp_path, capsys, pinned_wikitext):
    target = tmp_path / "wikitext-2-raw"
    target.mkdir()
    for name, content in pinned_wikitext.data.items():
        (target / name).write_bytes(content)
    rc = main(["fetch", "--cache-dir", str(tmp_path)])
    assert rc == 0
