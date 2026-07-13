"""Tests for refract.report_html: html_report end-to-end + helpers."""

from __future__ import annotations

import re

from refract.report_html import (
    _AXIS_FULL,
    _AXIS_LETTER,
    _AXIS_PROSE,
    _AXIS_SHORT,
    _BAND_CLASS,
    _BAND_PRETTY,
    _axis_letter_chip,
    _axis_row,
    _badge,
    _esc,
    _findings,
    _highlight_repro,
    _kv_pair,
    _meter,
    _mini_meter,
    _model_metadata,
    _plad_table_detail,
    _report_id,
    _repro_command,
    _rniah_low_confidence,
    _rniah_matrix_detail,
    _stat_block,
    _summary_box,
    html_report,
)
from refract.score import composite_score

from ._fixtures import (
    make_gtm,
    make_kld,
    make_plad,
    make_rniah_high_base,
    make_rniah_low_base,
    make_trajectory,
)

# --- mappings consistency -------------------------------------------------


def test_band_pretty_keys_match_band_classes():
    assert (
        set(_BAND_PRETTY.keys())
        == set(_BAND_CLASS.keys())
        == {"EXCELLENT", "PASS", "DEGRADED", "FAIL"}
    )


def test_axis_letter_short_full_have_same_keys():
    assert set(_AXIS_LETTER.keys()) == set(_AXIS_SHORT.keys()) == set(_AXIS_FULL.keys())


def test_axis_prose_keys_in_full_set():
    # PROSE may be a subset; LETTER must cover all axes that get rendered.
    for k in _AXIS_PROSE:
        assert k in _AXIS_LETTER, f"axis {k} present in PROSE but missing LETTER"


# --- escaping + simple helpers --------------------------------------------


def test_esc_escapes_html_special_chars():
    assert _esc("<a>") == "&lt;a&gt;"
    assert _esc(None) == ""
    assert _esc(42) == "42"


def test_badge_known_band_uses_class():
    out = _badge("FAIL")
    assert "red" in out
    assert "Fail" in out


def test_badge_unknown_band_uses_gray():
    out = _badge("???", override_label="Low confidence")
    assert "gray" in out
    assert "Low confidence" in out


def test_meter_renders_width_percentage():
    out = _meter(75.5, "PASS")
    assert "75.5%" in out


def test_meter_clamps_above_100():
    out = _meter(200.0, "EXCELLENT")
    # Should clamp to 100 in the rendered width
    assert "100.0%" in out


def test_meter_empty_band_uses_gray():
    out = _meter(50.0, "")
    assert "var(--gray)" in out


def test_mini_meter_uses_band_class():
    out = _mini_meter(85.0, "PASS")
    assert "var(--green)" in out


def test_axis_letter_chip_wraps_in_span():
    out = _axis_letter_chip("A")
    assert "letter" in out
    assert "A" in out


def test_kv_pair_returns_dt_dd():
    out = _kv_pair("name", "model.gguf")
    assert "<dt>name</dt>" in out
    assert "<dd>model.gguf</dd>" in out


def test_report_id_format():
    rid = _report_id()
    assert re.match(r"^#\d{4}-\d{4}$", rid)


# --- highlight_repro ------------------------------------------------------


def test_highlight_repro_marks_flags():
    out = _highlight_repro("python3 -m refract.cli score --model x.gguf")
    assert 'class="flag"' in out
    assert 'class="arg"' in out


def test_highlight_repro_marks_placeholders():
    out = _highlight_repro("--model <path>")
    # The literal <path> gets HTML-escaped to &lt;path&gt; → placeholder branch
    assert "placeholder" in out


# --- stat_block + axis_row + findings -------------------------------------


def test_stat_block_composite_shows_score():
    out = _stat_block("composite", 87.5, is_composite=True)
    assert "87.50" in out
    assert "Composite score" in out


def test_stat_block_low_confidence_suppresses_score():
    out = _stat_block("rniah", 100.0, low_confidence=True)
    # Score itself is replaced with em-dash "—"
    assert "—" in out
    assert "Low confidence" in out
    # Tooltip preserves the raw number
    assert "100.00" in out


def test_stat_block_normal_axis_renders_score():
    out = _stat_block("kld", 92.34)
    assert "92.34" in out


def test_axis_row_low_confidence_has_dash_and_gray():
    out = _axis_row("rniah", 100.0, low_confidence=True)
    assert "—" in out
    assert "Low confidence" in out


def test_axis_row_normal_renders_band():
    out = _axis_row("kld", 50.0)
    assert "Fail" in out


def test_findings_empty_returns_empty_string():
    assert _findings([]) == ""


def test_findings_renders_numbered_items():
    out = _findings(["First finding. With detail.", "Second one."])
    assert "01" in out
    assert "02" in out
    assert "First finding" in out


# --- rniah / plad detail blocks ------------------------------------------


def test_rniah_low_confidence_true_when_base_acc_low():
    rniah = make_rniah_low_base()
    assert _rniah_low_confidence(rniah) is True


def test_rniah_low_confidence_false_when_high():
    rniah = make_rniah_high_base()
    assert _rniah_low_confidence(rniah) is False


def test_rniah_low_confidence_empty_cells_returns_true():
    rniah = make_rniah_high_base()
    rniah.cells = []
    assert _rniah_low_confidence(rniah) is True


def test_rniah_matrix_detail_empty_cells_returns_empty():
    rniah = make_rniah_high_base()
    rniah.cells = []
    assert _rniah_matrix_detail(rniah) == ""


def test_rniah_matrix_detail_low_conf_emits_warning():
    rniah = make_rniah_low_base()
    out = _rniah_matrix_detail(rniah)
    assert "Low confidence" in out
    # Cells where base_acc==0 → "n/a"
    assert "n/a" in out


def test_rniah_matrix_detail_high_conf_no_warning_block():
    rniah = make_rniah_high_base()
    out = _rniah_matrix_detail(rniah)
    assert "Low confidence" not in out
    assert "<table" in out


def test_plad_table_detail_renders_perturbations():
    plad = make_plad()
    out = _plad_table_detail(plad)
    for k in ("typo", "case", "punct", "paraphrase"):
        assert k in out


def test_plad_table_detail_nan_gets_skipped_label():
    plad = make_plad(with_nan=True)
    out = _plad_table_detail(plad)
    assert "skipped" in out
    assert "N/A" in out


# --- summary box ----------------------------------------------------------


def test_summary_box_excellent_no_failures():
    comp = composite_score(95, 95)
    out = _summary_box(comp)
    assert "Excellent" in out
    assert "All measured axes pass" in out


def test_summary_box_fail_lists_failed_axes():
    comp = composite_score(20, 20)
    out = _summary_box(comp)
    assert "Fail" in out
    # Should mention which axes are below threshold
    assert "trajectory" in out.lower() or "kld" in out.lower()


# --- model metadata ------------------------------------------------------


def test_model_metadata_handles_missing_path(tmp_path):
    p = tmp_path / "missing.gguf"
    info = _model_metadata(p)
    assert info["path"] == str(p)
    assert info["name"] == "missing.gguf"


def test_model_metadata_file_size(tmp_path):
    p = tmp_path / "tiny.gguf"
    p.write_bytes(b"x" * 1024 * 1024)
    info = _model_metadata(p)
    assert info["size_gb"] == round(1024 * 1024 / 1024**3, 2)
    assert info["format"] == "gguf"


def test_model_metadata_directory_with_config(tmp_path):
    d = tmp_path / "model_dir"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x" * 1024)
    (d / "config.json").write_text(
        '{"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32}'
    )
    info = _model_metadata(d)
    assert info["format"] == "directory"
    assert info["model_type"] == "llama"
    assert info["hidden_size"] == 4096


def test_model_metadata_directory_with_bad_config(tmp_path):
    d = tmp_path / "model_dir"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    (d / "config.json").write_text("{ this is not json")
    info = _model_metadata(d)
    # Bad JSON should not crash
    assert info["format"] == "directory"


# --- repro command --------------------------------------------------------


def test_repro_command_uses_json_field_when_present():
    out = _repro_command(
        raw_json={"repro_command": "refract score --model X"},
        model="X.gguf",
        reference_label="ref",
        candidate_label="cand",
        has_rniah=False,
        has_plad=False,
    )
    assert out == "refract score --model X"


def test_repro_command_synthesizes_when_argv_not_refract(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pytest"])
    out = _repro_command(
        raw_json=None,
        model="/path/to/m.gguf",
        reference_label="ctk=f16,ctv=f16",
        candidate_label="ctk=q8_0,ctv=q8_0",
        has_rniah=False,
        has_plad=False,
    )
    assert "refract.cli" in out
    assert "m.gguf" in out
    # Should not include personal absolute path
    assert "/path/to/" not in out


def test_repro_command_full_when_rniah_or_plad(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pytest"])
    out = _repro_command(
        raw_json=None,
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        has_rniah=True,
        has_plad=True,
    )
    assert "--full" in out
    assert "--rniah-up-to" in out


# --- html_report end-to-end ----------------------------------------------


def test_html_report_two_axis_smoke():
    gtm = make_gtm(score=95.0)
    kld = make_kld(score=99.0)
    comp = composite_score(gtm.score, kld.score)
    html = html_report(
        model="my-model.gguf",
        reference_label="ctk=f16,ctv=f16",
        candidate_label="ctk=q8_0,ctv=q8_0",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    # Basic structural sanity
    assert "<!DOCTYPE html>" in html
    assert "</html>" in html
    # Composite + axes labels show up
    assert "Composite score" in html
    assert "GTM" in html or "Trajectory" in html
    assert "KLD" in html
    # Acronym expansion in axis full names
    assert "Greedy Trajectory Match" in html or "decode-time IDs" in html


def test_html_report_full_four_axis():
    traj = make_trajectory(score=95.0)
    kld = make_kld(score=99.0)
    rniah = make_rniah_high_base()
    plad = make_plad()
    comp = composite_score(traj.score, kld.score, rniah.score, plad.score)
    html = html_report(
        model="my-model.gguf",
        reference_label="ref",
        candidate_label="cand",
        composite=comp,
        gtm=traj,
        kld=kld,
        rniah=rniah,
        plad=plad,
    )
    assert "R-NIAH" in html
    assert "PLAD" in html
    assert "Retrieval Needle-In-A-Haystack" in html
    assert "Perturbation-Locality Aware Drift" in html


def test_html_report_low_confidence_rniah_suppresses_headline():
    traj = make_trajectory(score=95.0)
    kld = make_kld(score=99.0)
    rniah = make_rniah_low_base()  # base_acc avg < 0.2
    comp = composite_score(traj.score, kld.score, rniah_score=rniah.score)
    html = html_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=traj,
        kld=kld,
        rniah=rniah,
    )
    assert "Low confidence" in html
    # n/a cells when base_acc==0
    assert "n/a" in html


def test_html_report_includes_acronym_in_footer():
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(gtm.score, kld.score)
    html = html_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    # Footer "What is this?" expands the REFRACT acronym
    assert "REFerence-anchored Robust Acid-test for Compressed Transformers" in html


def test_html_report_embeds_raw_json():
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(gtm.score, kld.score)
    raw = {"composite": 95.0, "schema": "refract.report.v0.3.1"}
    html = html_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
        raw_json=raw,
    )
    assert "refract.report.v0.3.1" in html


def test_html_report_score_display_correct_for_known_band():
    # FAIL band should render a "Fail" badge somewhere.
    gtm = make_gtm(score=10.0)
    kld = make_kld(score=10.0)
    comp = composite_score(10.0, 10.0)
    html = html_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    assert "Fail" in html
