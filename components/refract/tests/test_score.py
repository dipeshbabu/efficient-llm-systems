"""Tests for refract.score: harmonic_mean, band thresholds, composite, diagnosis."""

from __future__ import annotations

import math

import pytest

from refract.score import (
    MIN_FLOOR,
    band,
    composite_score,
    harmonic_mean,
    interpret_pattern,
)


# --- harmonic_mean -------------------------------------------------------


def test_harmonic_mean_empty_returns_zero():
    assert harmonic_mean([]) == 0.0


def test_harmonic_mean_single_value():
    assert harmonic_mean([42.0]) == pytest.approx(42.0)


def test_harmonic_mean_zero_dominates():
    assert harmonic_mean([100.0, 100.0, 0.0]) == 0.0


def test_harmonic_mean_negative_clipped_to_zero_then_dominates():
    # Negative values must be clipped to 0, which then dominates.
    assert harmonic_mean([100.0, -5.0]) == 0.0


def test_harmonic_mean_clamped_to_100():
    # Inputs above 100 don't push HM above 100 (the clip is intentional).
    assert harmonic_mean([200.0, 200.0]) == pytest.approx(100.0)


def test_harmonic_mean_punishes_low_axis_more_than_arithmetic():
    # 90 + 30 → arithmetic 60, harmonic 45. The "fail-loud" property.
    h = harmonic_mean([90, 30])
    assert h < 60.0  # well below arithmetic mean


# --- band ----------------------------------------------------------------


@pytest.mark.parametrize("score,expected", [
    (100.0, "EXCELLENT"),
    (90.0, "EXCELLENT"),     # boundary inclusive
    (89.99, "PASS"),
    (80.0, "PASS"),          # boundary inclusive
    (79.99, "DEGRADED"),
    (60.0, "DEGRADED"),      # boundary inclusive
    (59.99, "FAIL"),
    (0.0, "FAIL"),
])
def test_band_thresholds_inclusive_lower(score, expected):
    assert band(score) == expected


# --- composite_score -----------------------------------------------------


def test_composite_two_axis_only():
    c = composite_score(95.0, 95.0)
    assert c.rniah_score is None
    assert c.plad_score is None
    assert c.band == "EXCELLENT"
    assert c.composite == pytest.approx(95.0)


def test_composite_four_axis():
    c = composite_score(95.0, 95.0, rniah_score=95.0, plad_score=95.0)
    assert c.rniah_score == 95.0
    assert c.plad_score == 95.0


def test_composite_floor_pass_no_notes():
    c = composite_score(95.0, 95.0, floor_score=99.9)
    assert c.floor_ok is True
    assert c.notes == []


def test_composite_floor_fail_emits_note():
    c = composite_score(95.0, 95.0, floor_score=80.0)
    assert c.floor_ok is False
    assert any("Floor failed" in n for n in c.notes)


def test_composite_floor_at_threshold():
    c = composite_score(95.0, 95.0, floor_score=MIN_FLOOR)
    assert c.floor_ok is True


def test_composite_score_direction_higher_is_better_property():
    # Sanity: the band of (100,100) ≥ band of (50,50). Otherwise some
    # downstream consumer could invert direction.
    a = composite_score(100, 100).composite
    b = composite_score(50, 50).composite
    assert a > b


# --- interpret_pattern ---------------------------------------------------


def test_interpret_all_clear_two_axis():
    notes = interpret_pattern(gtm_score=95, kld_score=95)
    assert len(notes) == 1
    assert "All axes intact" in notes[0]


def test_interpret_catastrophic_takes_precedence():
    notes = interpret_pattern(
        gtm_score=10, kld_score=10, rniah_score=10, plad_score=10,
    )
    assert len(notes) == 1
    assert "Catastrophic" in notes[0]


def test_interpret_distribution_break_high_level_intact():
    # Both A + B FAIL, C + D fine → the targeted distribution-break message.
    notes = interpret_pattern(
        gtm_score=20, kld_score=20, rniah_score=95, plad_score=95,
    )
    assert any("distribution is broken" in n.lower() for n in notes)


def test_interpret_mild_short_drift_no_panic():
    # A or B in DEGRADED (not FAIL), no other axis broken.
    notes = interpret_pattern(
        gtm_score=72, kld_score=95, rniah_score=95, plad_score=95,
    )
    assert any("Mild short-context drift" in n for n in notes)


def test_interpret_long_context_only():
    notes = interpret_pattern(
        gtm_score=95, kld_score=95, rniah_score=40, plad_score=95,
    )
    assert any("Long-context retrieval degraded" in n for n in notes)
    # Should NOT label as short-context drift
    assert not any("short-context" in n for n in notes)


def test_interpret_brittle_only():
    notes = interpret_pattern(
        gtm_score=95, kld_score=95, rniah_score=95, plad_score=40,
    )
    assert any("Brittleness" in n for n in notes)


def test_interpret_mixed_short_and_long():
    notes = interpret_pattern(
        gtm_score=20, kld_score=20, rniah_score=20, plad_score=95,
    )
    # Multiple notes — at least one each for short + long
    msgs = " ".join(notes)
    assert "Decode distribution drift" in msgs
    assert "Long-context retrieval" in msgs


def test_interpret_two_axis_call_omits_optional():
    # When rniah/plad are None, only A+B drive the diagnosis.
    notes = interpret_pattern(gtm_score=20, kld_score=20)
    # All measured axes < 60 → catastrophic
    assert any("Catastrophic" in n for n in notes)


# --- MIN_FLOOR constant ---------------------------------------------------


def test_min_floor_is_99_5():
    assert MIN_FLOOR == 99.5


# --- skip-axis composite: regression for false-100 reporting -------------


def test_composite_skip_gtm_not_inflated_to_100():
    """v0.3.2.1: --skip-gtm passes gtm_score=None to composite_score, which
    must drop it from the harmonic mean rather than treat it as a 100 stub.

    Regression: AJ reported (2026-05-02) that --skip-gtm produced a 99.74
    EXCELLENT composite from a single real KLD score of 99.47, because
    the previous wiring fed a stub gtm=100."""
    cs = composite_score(gtm_score=None, kld_score=99.47)
    assert abs(cs.composite - 99.47) < 0.01
    assert cs.gtm_score is None
    assert cs.kld_score == 99.47


def test_composite_skip_kld_not_inflated_to_100():
    cs = composite_score(gtm_score=62.40, kld_score=None)
    assert abs(cs.composite - 62.40) < 0.01
    assert cs.gtm_score == 62.40
    assert cs.kld_score is None


def test_composite_skip_both_returns_zero():
    """If both A and B are skipped and no rniah/plad, don't crash and
    don't claim 100. Return 0 with FAIL band."""
    cs = composite_score(gtm_score=None, kld_score=None)
    assert cs.composite == 0.0
    assert cs.band == "FAIL"


def test_composite_skip_gtm_with_rniah_plad():
    """Skip-gtm + KLD + R-NIAH + PLAD = harmonic mean of 3 axes only."""
    cs = composite_score(
        gtm_score=None, kld_score=99.0, rniah_score=100.0, plad_score=90.0,
    )
    expected = 3 / (1/99 + 1/100 + 1/90)
    assert abs(cs.composite - expected) < 0.01


def test_interpret_pattern_handles_skipped_gtm():
    """interpret_pattern shouldn't blow up on None gtm/kld."""
    notes = interpret_pattern(
        gtm_score=None, kld_score=99.0, rniah_score=100.0, plad_score=95.0,
    )
    assert isinstance(notes, list)


def test_text_report_skip_gtm_renders_n_a_and_one_axis_count():
    """text_report on a skipped axis emits 'n/a' / 'skipped' (not 100/EXCELLENT)
    and counts the harmonic-mean axes correctly."""
    from refract.axes.gtm import GTMResult
    from refract.axes.kld import KLDResult
    from refract.report import text_report
    gtm_stub = GTMResult(
        score=100.0, full_match_rate=1.0,
        median_first_divergence=None,
        mean_prefix_agreement_length=0.0,
        mean_cand_length=0.0, mean_ref_length=0.0,
        n_prompts=0, n_tokens_each=0, per_prompt=[],
    )
    kld_real = KLDResult(
        score=99.47, mean_kld=0.0053, ppl=None,
        rms_dp_pct=None, same_topp_pct=None,
        base_path="", chunks=32, ctx=512,
        is_self_reference=False,
    )
    cs = composite_score(gtm_score=None, kld_score=99.47)
    out = text_report(
        model="m.gguf",
        reference_label="ctk=f16,ctv=f16",
        candidate_label="ctk=q8_0,ctv=q8_0",
        composite=cs, gtm=gtm_stub, kld=kld_real,
    )
    import re as _re
    plain = _re.sub(r"\x1b\[[0-9;]*m", "", out)
    assert "n/a" in plain or "skipped" in plain.lower()
    assert "99.47" in plain
    # gloss must reflect actual axis count after skip, not the old hardcoded 2
    assert "harmonic mean of 1 axes" in plain
