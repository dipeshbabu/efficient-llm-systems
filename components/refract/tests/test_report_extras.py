"""Extra coverage for refract.report + report_html: notes paths, env_meta,
exceptional argv, hardware metadata branches."""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

from refract.axes.gtm import GTMResult
from refract.axes.kld import KLDResult as AxisKLDResult
from refract.axes.plad import PLADPerPrompt, PLADResult
from refract.axes.rniah import RNIAHCell, RNIAHResult
from refract.report import _sanitize_home_arg, json_report, text_report
from refract.report_html import (
    _hardware_metadata,
    _highlight_repro,
    _model_metadata,
    _repro_command,
    html_report,
)
from refract.score import composite_score
from refract.runner import set_active_backend

from ._fixtures import (
    make_gtm,
    make_kld,
    make_plad,
    make_rniah_high_base,
)


# --- text_report notes branches ------------------------------------------


def test_text_report_renders_gtm_notes():
    gtm = make_gtm()
    gtm.notes = ["inflated retokenize"]
    kld = make_kld()
    comp = composite_score(95, 95)
    out = text_report(
        model="m.gguf", reference_label="r", candidate_label="c",
        composite=comp, gtm=gtm, kld=kld,
    )
    assert "inflated retokenize" in out


def test_text_report_renders_rniah_notes_and_skipped_cells():
    gtm = make_gtm()
    kld = make_kld()
    rniah = make_rniah_high_base()
    rniah.notes = ["suspicious cell"]
    rniah.skipped_cells = [(32768, 0.5)]
    comp = composite_score(gtm.score, kld.score, rniah_score=rniah.score)
    out = text_report(
        model="m.gguf", reference_label="r", candidate_label="c",
        composite=comp, gtm=gtm, kld=kld, rniah=rniah,
    )
    assert "suspicious cell" in out
    assert "cells skipped" in out


def test_text_report_renders_plad_notes():
    gtm = make_gtm()
    kld = make_kld()
    plad = make_plad()
    plad.notes = ["paraphrase did not apply"]
    comp = composite_score(gtm.score, kld.score, plad_score=plad.score)
    out = text_report(
        model="m.gguf", reference_label="r", candidate_label="c",
        composite=comp, gtm=gtm, kld=kld, plad=plad,
    )
    assert "paraphrase did not apply" in out


def test_text_report_renders_composite_notes():
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(95, 95, floor_score=80.0)  # → emits Floor failed note
    out = text_report(
        model="m.gguf", reference_label="r", candidate_label="c",
        composite=comp, gtm=gtm, kld=kld,
    )
    assert "Floor failed" in out


# --- json_report exceptional paths ----------------------------------------


def test_json_report_env_meta_swallows_backend_exception():
    """If active backend raises in model_metadata, env_meta stays empty."""
    class _BadBackend:
        name = "bad"
        def model_metadata(self, **kw):
            raise RuntimeError("boom")
    set_active_backend(_BadBackend())
    try:
        gtm = make_gtm()
        kld = make_kld()
        comp = composite_score(95, 95)
        rep = json_report(
            model="m.gguf", reference_label="r", candidate_label="c",
            composite=comp, gtm=gtm, kld=kld,
        )
        assert rep["environment"] == {}
    finally:
        set_active_backend(None)


def test_json_report_handles_bad_argv(monkeypatch):
    """If sys.argv is something weird that errors out, repro_command is ''."""
    class _BoomArgv:
        def __iter__(self):
            raise RuntimeError("argv broke")
    # We can't easily monkeypatch sys.argv to this; but we can monkeypatch
    # `sys.argv` to a list where ANY iteration over it triggers the os.path
    # branch. Easier: replace os.path.expanduser to raise, which bubbles up
    # the except.
    import os
    orig_expanduser = os.path.expanduser
    def boom(path):
        if path == "~":
            raise RuntimeError("home explosion")
        return orig_expanduser(path)
    monkeypatch.setattr("os.path.expanduser", boom)
    monkeypatch.setattr("sys.argv", ["python3", "-m", "refract.cli", "score"])

    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(95, 95)
    rep = json_report(
        model="m.gguf", reference_label="r", candidate_label="c",
        composite=comp, gtm=gtm, kld=kld,
    )
    assert rep["repro_command"] == ""


def test_sanitize_home_arg_handles_windows_and_posix_separators():
    home = r"C:\Users\alice"
    assert _sanitize_home_arg(r"C:\Users\alice\models\m.gguf", home) == \
        r"~\models\m.gguf"
    assert _sanitize_home_arg("C:/Users/alice/models/m.gguf", home) == \
        "~/models/m.gguf"


# --- hardware metadata (exercised on both Darwin and Linux paths) --------


def test_hardware_metadata_returns_dict():
    info = _hardware_metadata()
    assert "system" in info
    assert "platform" in info
    assert "machine" in info
    assert "python" in info


def test_hardware_metadata_handles_subprocess_failures(monkeypatch):
    """All subprocess probes wrapped in try/except — should never raise."""
    def boom(*a, **kw):
        raise RuntimeError("no sysctl here")
    monkeypatch.setattr("subprocess.run", boom)
    info = _hardware_metadata()
    # Always returns at least the static info
    assert "system" in info


def test_hardware_metadata_linux_branch(monkeypatch, tmp_path):
    """Force Linux branch + mock /proc."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("model name\t: Test CPU 9000\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16384000 kB\n")
    real_open = open

    def fake_open(path, *a, **kw):
        if path == "/proc/cpuinfo":
            return real_open(cpuinfo, *a, **kw)
        if path == "/proc/meminfo":
            return real_open(meminfo, *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()))
    info = _hardware_metadata()
    assert info["system"] == "Linux"
    assert info["chip"] == "Test CPU 9000"
    assert info["ram_gb"] == round(16384000 / 1024 / 1024, 1)


# --- _model_metadata edge case ------------------------------------------


def test_model_metadata_directory_no_config(tmp_path):
    d = tmp_path / "raw_model"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x" * 1024)
    info = _model_metadata(d)
    assert info["format"] == "directory"
    assert "model_type" not in info  # no config.json present


# --- _highlight_repro edge case -----------------------------------------


def test_highlight_repro_short_arg_treated_as_arg():
    out = _highlight_repro("-1")
    # -1 starts with - but next char is digit → treated as arg, not flag.
    assert 'class="arg"' in out


# --- html_report including hardware --------------------------------------


def test_html_report_includes_hardware_section():
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(95, 95)
    html = html_report(
        model="m.gguf", reference_label="r", candidate_label="c",
        composite=comp, gtm=gtm, kld=kld,
    )
    # Run details cards include some hardware tag (chip name or "macOS"/"Linux"
    # marker depending on host).
    assert "hardware" in html.lower() or "machine" in html.lower() or \
           platform.machine() in html
