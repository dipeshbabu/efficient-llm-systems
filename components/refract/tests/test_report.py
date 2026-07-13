"""Tests for refract.report: text_report, json_report, score_direction stamps."""

from __future__ import annotations

import json

from refract.report import (
    _AXIS_PROSE,
    _BAND_PROSE,
    _axis_label,
    _band_color,
    _bar,
    _wrap_lines,
    json_report,
    text_report,
    to_json_string,
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

# --- helpers --------------------------------------------------------------


def test_axis_prose_keys_match_known_axes():
    # Description matching: every axis exposed by the framework must have a prose entry.
    expected = {"gtm", "trajectory", "kld", "rniah", "plad"}
    assert expected.issubset(_AXIS_PROSE.keys())


def test_band_prose_keys_match_band_strings():
    assert set(_BAND_PROSE.keys()) == {"EXCELLENT", "PASS", "DEGRADED", "FAIL"}


def test_axis_label_unknown_falls_back_to_name():
    assert _axis_label("unknown_axis") == "unknown_axis"


def test_band_color_unknown_falls_back_to_reset():
    assert _band_color("???") == "0"


def test_bar_clamps_score_to_0_100():
    bar_high = _bar(150.0, width=10)
    bar_low = _bar(-50.0, width=10)
    # ANSI markers + 10-char body present
    assert "[" in bar_high and "]" in bar_high
    assert "[" in bar_low and "]" in bar_low


def test_wrap_lines_handles_empty_input():
    assert _wrap_lines("", indent="") == [""]


def test_wrap_lines_indents_long_text():
    result = _wrap_lines(
        "a b c d e f g h i j k l m n o p q r s t u v w x y", indent=">>", width=10
    )
    assert all(r.startswith(">>") for r in result)


# --- text_report ----------------------------------------------------------


def test_text_report_two_axis_basic():
    gtm = make_gtm(score=95.0)
    kld = make_kld(score=99.0)
    comp = composite_score(95.0, 99.0)
    out = text_report(
        model="my-model.gguf",
        reference_label="ctk=f16,ctv=f16",
        candidate_label="ctk=q8_0,ctv=q8_0",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    assert "REFRACT score" in out
    assert "my-model.gguf" in out
    assert "harmonic mean of 2 axes" in out
    assert "GTM diagnostics" in out
    assert "KLD diagnostics" in out


def test_text_report_uses_trajectory_label_when_trajectory_result():
    traj = make_trajectory(score=95.0)
    kld = make_kld(score=99.0)
    comp = composite_score(95.0, 99.0)
    out = text_report(
        model="m.gguf",
        reference_label="ref",
        candidate_label="cand",
        composite=comp,
        gtm=traj,
        kld=kld,
    )
    assert "Axis A Trajectory" in out


def test_text_report_full_four_axis():
    gtm = make_gtm()
    kld = make_kld()
    rniah = make_rniah_high_base()
    plad = make_plad()
    comp = composite_score(gtm.score, kld.score, rniah.score, plad.score)
    out = text_report(
        model="m.gguf",
        reference_label="ref",
        candidate_label="cand",
        composite=comp,
        gtm=gtm,
        kld=kld,
        rniah=rniah,
        plad=plad,
    )
    assert "harmonic mean of 4 axes" in out
    assert "R-NIAH diagnostics" in out
    assert "PLAD diagnostics" in out


def test_text_report_floor_block_when_present():
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(95, 95, floor_score=99.9)
    out = text_report(
        model="m.gguf",
        reference_label="ref",
        candidate_label="cand",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    assert "Noise floor" in out


def test_text_report_floor_not_measured_warning():
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(95, 95)
    out = text_report(
        model="m.gguf",
        reference_label="ref",
        candidate_label="cand",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    assert "NOT MEASURED" in out


def test_text_report_extras_rendered():
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(95, 95)
    out = text_report(
        model="m.gguf",
        reference_label="ref",
        candidate_label="cand",
        composite=comp,
        gtm=gtm,
        kld=kld,
        extras={"runtime_seconds": 42},
    )
    assert "runtime_seconds" in out
    assert "42" in out


def test_text_report_no_color_env_strips_ansi(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(95, 95)
    out = text_report(
        model="m.gguf",
        reference_label="ref",
        candidate_label="cand",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    assert "\033[" not in out


# --- json_report ----------------------------------------------------------


def test_json_report_schema_stamp():
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(95, 95)
    rep = json_report(
        model="m.gguf",
        reference_label="ref",
        candidate_label="cand",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    assert rep["schema"] == "refract.report.v0.3.2"
    assert "framework_version" in rep
    assert rep["score_direction"] == "higher_is_better"
    assert rep["score_range"] == [0, 100]


def test_json_report_band_summary_matches_band():
    gtm = make_gtm()
    kld = make_kld()
    # Force a specific band by tuning scores
    comp = composite_score(50, 50)  # FAIL
    rep = json_report(
        model="m.gguf",
        reference_label="ref",
        candidate_label="cand",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    assert rep["band"] == "FAIL"
    assert "broken" in rep["summary"].lower()


def test_json_report_axes_have_band_and_description():
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(gtm.score, kld.score)
    rep = json_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    for axis in ("gtm", "kld"):
        assert "band" in rep["axes"][axis]
        assert "description" in rep["axes"][axis]


def test_json_report_per_prompt_can_be_excluded():
    gtm = make_gtm()
    gtm.per_prompt = [{"id": "p1"}]
    kld = make_kld()
    comp = composite_score(gtm.score, kld.score)
    rep = json_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
        include_per_prompt=False,
    )
    assert "per_prompt" not in rep["axes"]["gtm"]


def test_json_report_rniah_confidence_low_when_base_avg_under_02():
    gtm = make_gtm()
    kld = make_kld()
    rniah = make_rniah_low_base()
    comp = composite_score(gtm.score, kld.score, rniah_score=rniah.score)
    rep = json_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
        rniah=rniah,
    )
    assert rep["axes"]["rniah"]["confidence"] == "low"
    assert rep["axes"]["rniah"]["excluded_from_composite"] is True
    assert rep["axes"]["rniah"]["band"] == "unscored"


def test_json_report_rniah_confidence_ok_when_base_avg_high():
    gtm = make_gtm()
    kld = make_kld()
    rniah = make_rniah_high_base()
    comp = composite_score(gtm.score, kld.score, rniah_score=rniah.score)
    rep = json_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
        rniah=rniah,
    )
    assert rep["axes"]["rniah"]["confidence"] == "ok"


def test_json_report_plad_partial_confidence_with_nan():
    gtm = make_gtm()
    kld = make_kld()
    plad = make_plad(with_nan=True)
    comp = composite_score(gtm.score, kld.score, plad_score=plad.score)
    rep = json_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
        plad=plad,
    )
    assert rep["axes"]["plad"]["confidence"] == "partial"
    assert "paraphrase" in rep["axes"]["plad"]["skipped_perturbations"]
    assert rep["axes"]["plad"]["per_perturbation_score"]["paraphrase"] is None


def test_json_report_plad_ok_confidence_no_nan():
    gtm = make_gtm()
    kld = make_kld()
    plad = make_plad(with_nan=False)
    comp = composite_score(gtm.score, kld.score, plad_score=plad.score)
    rep = json_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
        plad=plad,
    )
    assert rep["axes"]["plad"]["confidence"] == "ok"
    assert rep["axes"]["plad"]["skipped_perturbations"] == []


def test_json_report_repro_command_empty_when_not_a_refract_run(monkeypatch):
    # When sys.argv doesn't look like a refract CLI invocation, repro_command
    # is empty (not polluted by render scripts).
    monkeypatch.setattr("sys.argv", ["pytest"])
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(gtm.score, kld.score)
    rep = json_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    assert rep["repro_command"] == ""


def test_json_report_repro_command_populated_when_refract_run(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["python3", "-m", "refract.cli", "score", "--model", "/tmp/m.gguf"],
    )
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(gtm.score, kld.score)
    rep = json_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    assert "refract.cli" in rep["repro_command"]


def test_json_report_repro_strips_home_to_tilde(monkeypatch, tmp_path):
    import os

    home = os.path.expanduser("~")
    monkeypatch.setattr(
        "sys.argv",
        ["python3", "-m", "refract.cli", "score", "--model", f"{home}/secrets/m.gguf"],
    )
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(gtm.score, kld.score)
    rep = json_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    assert "~" in rep["repro_command"]
    assert home not in rep["repro_command"]


def test_to_json_string_is_valid_json():
    gtm = make_gtm()
    kld = make_kld()
    comp = composite_score(gtm.score, kld.score)
    rep = json_report(
        model="m.gguf",
        reference_label="r",
        candidate_label="c",
        composite=comp,
        gtm=gtm,
        kld=kld,
    )
    s = to_json_string(rep)
    parsed = json.loads(s)
    assert parsed["schema"] == rep["schema"]


def test_to_json_string_replaces_non_finite_floats_with_null():
    report = {
        "values": [float("nan"), float("inf"), float("-inf")],
        "nested": {"score": float("nan")},
    }

    def reject_nonstandard_constant(value: str):
        raise AssertionError(f"non-standard JSON constant: {value}")

    serialized = to_json_string(report)
    parsed = json.loads(serialized, parse_constant=reject_nonstandard_constant)

    assert parsed == {"values": [None, None, None], "nested": {"score": None}}
