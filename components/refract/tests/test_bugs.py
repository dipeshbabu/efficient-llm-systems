"""Regression tests for bugs caught during ultrathink coverage hunt.

Each test documents the bug and pins the fix so it doesn't regress.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pytest

from refract.cli import _run_compare
from refract.report import text_report
from refract.score import composite_score

from ._fixtures import make_gtm, make_kld, make_plad


def test_compare_handles_missing_composite_key(tmp_path, capsys):
    """Bug: _run_compare crashed with TypeError when a JSON report omitted the
    'composite' key. Now it renders '—' as the placeholder instead."""
    p = tmp_path / "no_composite.json"
    p.write_text(json.dumps({"band": "PASS", "axes": {}}))
    rc = _run_compare(argparse.Namespace(reports=[p]))
    assert rc == 0
    out = capsys.readouterr().out
    assert "—" in out


def test_compare_handles_missing_band_key(tmp_path, capsys):
    """Bug: _run_compare crashed when the 'band' key was missing or null."""
    p = tmp_path / "no_band.json"
    p.write_text(json.dumps({"composite": 80.0, "axes": {}}))
    rc = _run_compare(argparse.Namespace(reports=[p]))
    assert rc == 0


def test_compare_handles_explicit_null_composite(tmp_path, capsys):
    p = tmp_path / "null_composite.json"
    p.write_text(json.dumps({"composite": None, "band": None, "axes": {}}))
    rc = _run_compare(argparse.Namespace(reports=[p]))
    assert rc == 0


def test_text_report_plad_nan_renders_as_skipped():
    """Bug: text_report displayed 'nan FAIL' (in red) when a PLAD
    per-perturbation score was NaN — a skipped/inapplicable perturbation
    was misread as a failure. Now it renders 'skipped' / 'n/a'."""
    gtm = make_gtm()
    kld = make_kld()
    plad = make_plad(with_nan=True)
    comp = composite_score(gtm.score, kld.score, plad_score=plad.score)
    out = text_report(
        model="m.gguf", reference_label="r", candidate_label="c",
        composite=comp, gtm=gtm, kld=kld, plad=plad,
    )
    paraphrase_lines = [ln for ln in out.splitlines() if "paraphrase" in ln]
    assert paraphrase_lines, "expected a paraphrase line in PLAD diagnostics"
    line = paraphrase_lines[0]
    assert "nan" not in line.lower()
    assert "skipped" in line


def test_band_with_nan_does_not_crash():
    """Edge case: passing NaN to band() falls through to FAIL (all
    comparisons with NaN return False). Behavior pinned so regressions surface."""
    from refract.score import band
    assert band(float("nan")) == "FAIL"


def test_harmonic_mean_with_nan_input():
    """Edge case: NaN propagates through max() but not through sum(1/v)
    cleanly. Pin the current behavior — anything with a single broken axis
    must NOT silently produce a high score."""
    from refract.score import harmonic_mean
    # max(NaN, 0.0) is implementation-defined (returns NaN on CPython).
    # Then 1.0 / NaN = NaN, sum is NaN, n/NaN is NaN. Output goes through
    # min(max(NaN, 0), 100) which returns the last comparison result.
    # The important property: it must NOT silently return 100.
    out = harmonic_mean([100.0, float("nan")])
    assert not (out >= 90.0), \
        "harmonic_mean must not silently report EXCELLENT for a NaN axis"


def test_score_direction_higher_is_better_invariant():
    """Score direction sanity: a perfect cand-vs-ref always scores ≥ a
    degraded cand-vs-ref. If this ever inverts, downstream comparison code
    breaks silently."""
    perfect = composite_score(100, 100).composite
    bad = composite_score(50, 50).composite
    worse = composite_score(10, 10).composite
    assert perfect > bad > worse


def test_min_floor_constant_is_99_5_pinned():
    """If anyone bumps MIN_FLOOR, force them to read this test + the
    paper's bit-exact-zero claim."""
    from refract.score import MIN_FLOOR
    assert MIN_FLOOR == 99.5


def test_axes_band_consistent_with_composite_score_band():
    """Score → band mapping must agree across modules so reports
    don't render different bands for the same number."""
    from refract.score import band as s_band
    for score in (95.0, 85.0, 70.0, 30.0):
        c = composite_score(score, score)
        assert c.band == s_band(score)


def test_kld_score_to_kld_inverse_consistency():
    """exp(-mean_kld) is the score formula. Round-trip a few values."""
    from refract.axes.kld import _kld_to_score
    for kld_nats in (0.0, 0.1, 0.5, 1.0, 2.5):
        s = _kld_to_score(kld_nats)
        assert 0.0 <= s <= 100.0
        # Recover kld from score
        recovered = -math.log(s / 100.0)
        assert recovered == pytest.approx(kld_nats, abs=1e-6)


def test_levenshtein_symmetric():
    """Levenshtein must be symmetric: d(a,b) == d(b,a)."""
    from refract.axes.plad import _levenshtein
    pairs = [
        ([1, 2, 3], [1, 5, 3]),
        ([1, 2, 3], [3, 2, 1]),
        ([], [1, 2, 3]),
        ([1, 2, 3, 4, 5], [1, 2, 3]),
    ]
    for a, b in pairs:
        assert _levenshtein(a, b) == _levenshtein(b, a), \
            f"asymmetric Levenshtein: {a} vs {b}"


def test_kvconfig_label_is_round_trippable():
    """KVConfig.label() output must parse back into an equivalent KVConfig."""
    from refract.runner import KVConfig
    original = KVConfig.parse(
        "ctk=q8_0,ctv=turbo4,attn_rot_v=0,attn_rot_k=1"
    )
    reparsed = KVConfig.parse(original.label())
    assert reparsed.ctk == original.ctk
    assert reparsed.ctv == original.ctv
    assert reparsed.attn_rot_k == original.attn_rot_k
    assert reparsed.attn_rot_v == original.attn_rot_v
