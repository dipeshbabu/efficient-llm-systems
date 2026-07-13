"""v0.1.3 regression tests for report.json_report() schema layout.

Pins the v0.1.1 schema fix (composite is a scalar, not a nested dict at
top level) and the v0.1.3 schema bump.
"""

from __future__ import annotations

from refract.axes.gtm import GTMResult
from refract.axes.kld import KLDResult
from refract.report import json_report
from refract.score import composite_score


def _build_minimal_report() -> dict:
    gtm = GTMResult(
        score=92.0,
        full_match_rate=0.5,
        median_first_divergence=20,
        mean_prefix_agreement_length=44.0,
        mean_cand_length=48.0,
        mean_ref_length=48.0,
        n_prompts=2,
        n_tokens_each=48,
        per_prompt=[
            {"id": "p1", "ref": "x", "cand": "x"},
            {"id": "p2", "ref": "y", "cand": "z"},
        ],
    )
    kld = KLDResult(
        score=99.5,
        mean_kld=0.005,
        ppl=6.0,
        rms_dp_pct=1.1,
        same_topp_pct=98.4,
        base_path="/tmp/base.bin",
        chunks=32,
        ctx=512,
        is_self_reference=False,
        corpus={
            "path": "/tmp/wiki.raw",
            "size_bytes": 1234,
            "sha256_head": "deadbeef",
            "sha256_head_bytes": 1234,
        },
    )
    composite = composite_score(gtm.score, kld.score)
    return json_report(
        model="test.gguf",
        reference_label="ctk=f16,ctv=f16",
        candidate_label="ctk=q8_0,ctv=turbo4",
        composite=composite,
        gtm=gtm,
        kld=kld,
    )


def test_composite_is_scalar_at_top_level():
    """v0.1.1 regression: d['composite'] must be a number, not a nested dict.
    v0.1 had this nested as a dict; aggregator scripts printed '{compos…' as
    if it were a number."""
    rep = _build_minimal_report()
    assert isinstance(rep["composite"], float)
    # Sanity: it's the actual harmonic mean of 92 and 99.5
    assert 90 < rep["composite"] < 100


def test_band_is_string_at_top_level():
    rep = _build_minimal_report()
    assert isinstance(rep["band"], str)
    assert rep["band"] in {"EXCELLENT", "PASS", "DEGRADED", "FAIL"}


def test_composite_detail_is_dict():
    rep = _build_minimal_report()
    assert isinstance(rep["composite_detail"], dict)
    # gtm_score / kld_score live inside composite_detail
    assert "gtm_score" in rep["composite_detail"]
    assert "kld_score" in rep["composite_detail"]


def test_axes_block_present():
    rep = _build_minimal_report()
    assert "axes" in rep
    assert "gtm" in rep["axes"]
    assert "kld" in rep["axes"]
    # v0.1.3: GTM block exposes the new mean_cand/mean_ref fields
    assert rep["axes"]["gtm"]["mean_cand_length"] == 48.0
    assert rep["axes"]["gtm"]["mean_ref_length"] == 48.0


def test_schema_is_v0_3_3():
    rep = _build_minimal_report()
    assert rep["schema"] == "refract.report.v0.3.3"


def test_framework_version_present():
    rep = _build_minimal_report()
    assert "framework_version" in rep
    assert isinstance(rep["framework_version"], str)


def test_environment_block_present():
    rep = _build_minimal_report()
    assert "environment" in rep
    assert isinstance(rep["environment"], dict)


# v0.1.4 additions: layman summary + per-axis bands/descriptions.


def test_summary_is_layman_string():
    """v0.1.4: top-level 'summary' field carries a one-line plain-English
    interpretation of the score band so non-techie consumers can read the
    report without grepping the paper."""
    rep = _build_minimal_report()
    assert isinstance(rep.get("summary"), str)
    assert rep["summary"]  # non-empty
    # The summary's content reflects the band; pin a substring of each band
    # so the prose can be edited but the contract holds.
    band_to_substr = {
        "EXCELLENT": "No material drift",
        "PASS": "Minor measured drift",
        "DEGRADED": "Audit",
        "FAIL": "broken",
    }
    expected = band_to_substr[rep["band"]]
    assert expected.lower() in rep["summary"].lower(), (
        f"summary {rep['summary']!r} does not match band {rep['band']!r}"
    )


def test_axes_carry_per_axis_band():
    """v0.1.4: each axis block carries its own band, not just the
    composite. Lets the layman see which axis caused a DEGRADED."""
    rep = _build_minimal_report()
    assert rep["axes"]["gtm"]["band"] in {"EXCELLENT", "PASS", "DEGRADED", "FAIL"}
    assert rep["axes"]["kld"]["band"] in {"EXCELLENT", "PASS", "DEGRADED", "FAIL"}


def test_axes_carry_description():
    """v0.1.4: each axis block has a one-line 'description' so a layman
    knows what 'KLD: DEGRADED' actually means without the paper."""
    rep = _build_minimal_report()
    for ax in ("gtm", "kld"):
        d = rep["axes"][ax].get("description")
        assert isinstance(d, str) and d, f"missing description on axis {ax}"
