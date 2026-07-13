"""REFRACT v0.1 composite scoring.

Combines per-axis scores into a single 0–100 number using the harmonic mean,
which penalises a single bad axis more aggressively than the arithmetic mean
(matches the "fail-loud" intent of the design).

Bands (tunable; align these with the paper's findings as we collect more data):

    [90, 100]  EXCELLENT — within reference noise / true equivalence
    [80,  90)  PASS      — minor drift, safe to deploy
    [60,  80)  DEGRADED  — visible drift, audit before use
    [ 0,  60)  FAIL      — flag and treat as broken

Floor verification:
    REFRACT(fp16-KV, fp16-KV) must be >= MIN_FLOOR (default 99.5).
    If it is not, the reference itself is non-deterministic on this build
    and KLD deltas cannot be trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# Minimum REFRACT score the reference vs reference must hit. The paper §4.5
# shows bit-exact zero KLD on Metal, so 99.5 leaves headroom for non-Metal
# float-jitter without admitting a broken reference.
MIN_FLOOR = 99.5


def harmonic_mean(values: list[float]) -> float:
    """Harmonic mean clipped to [0, 100]. Returns 0 if any value is 0."""
    clean = [max(v, 0.0) for v in values]
    if not clean:
        return 0.0
    if any(v <= 0.0 for v in clean):
        return 0.0
    n = len(clean)
    h = n / sum(1.0 / v for v in clean)
    return min(max(h, 0.0), 100.0)


def band(score: float) -> str:
    if score >= 90.0:
        return "EXCELLENT"
    if score >= 80.0:
        return "PASS"
    if score >= 60.0:
        return "DEGRADED"
    return "FAIL"


def interpret_pattern(
    *,
    gtm_score: Optional[float],
    kld_score: Optional[float],
    rniah_score: Optional[float] = None,
    plad_score: Optional[float] = None,
) -> list[str]:
    """v0.2.0: turn the per-axis band pattern into plain-English diagnosis.

    Returns 0-3 short sentences a non-techie reader can act on. Each sentence
    names what's broken (in user terms, not "axis A score") and suggests a
    direction. Does not duplicate the band labels — rendering callers should
    print bands AND interpretation; the bands say "what" and the
    interpretation says "what it means".

    Pattern recognised:
      - all axes ≥ 80 (PASS or above)            → "all clear"
      - short-context drift (A or B < 80)        → "decode distribution shift"
      - long-context drift (C < 80)              → "retrieval at long context"
      - brittleness (D < 80)                     → "perturbation brittleness"
      - all axes < 60 (FAIL)                     → "catastrophic"
    Multiple patterns can fire; sentences are emitted in order of severity.
    """
    notes: list[str] = []

    def low(s: Optional[float]) -> bool:
        return s is not None and s < 80.0

    def fail(s: Optional[float]) -> bool:
        return s is not None and s < 60.0

    short_drift = low(gtm_score) or low(kld_score)
    long_drift = low(rniah_score)
    brittle = low(plad_score)

    # Catastrophic case first: every measured axis below FAIL.
    measured = [s for s in (gtm_score, kld_score, rniah_score, plad_score)
                if s is not None]
    if measured and all(s < 60.0 for s in measured):
        notes.append(
            "Catastrophic: every measured surface is broken. Treat as a "
            "non-functional quantization; revert to a higher-bit config."
        )
        return notes

    if not (short_drift or long_drift or brittle):
        notes.append(
            "All axes intact. Quantization is faithful to the fp16 reference "
            "across the surfaces tested."
        )
        return notes

    if short_drift and not long_drift and not brittle:
        if fail(gtm_score) and fail(kld_score):
            notes.append(
                "Per-token distribution is broken but high-level surfaces "
                "(retrieval, robustness) are intact. The model decodes "
                "materially different tokens than fp16; consider a higher-bit "
                "V-cache or a non-rotating V scheme."
            )
        else:
            notes.append(
                "Mild short-context drift; long-context retrieval and "
                "perturbation robustness are intact. Likely safe for typical "
                "use; audit on your specific decoding workload before shipping."
            )
        return notes

    # Mixed cases: name each axis that's affected.
    if short_drift:
        notes.append(
            "Decode distribution drift detected: the candidate generates "
            "different tokens than fp16 on short-context prompts."
        )
    if long_drift:
        notes.append(
            "Long-context retrieval degraded: candidate fails on prompts "
            "where the fp16 reference still retrieves correctly. Inspect the "
            "per-(length, position) cell breakdown to see where it breaks."
        )
    if brittle:
        notes.append(
            "Brittleness to small input changes: candidate's output drifts "
            "more than fp16's under typo / casing / punctuation perturbations."
        )
    return notes


@dataclass
class CompositeScore:
    """REFRACT composite output.

    v0.1 shipped two axes (gtm + kld); v0.2 adds rniah + plad as
    optional axes. The composite is the harmonic mean of *all axes
    actually scored* — None values are dropped before aggregation, so
    a v0.1-style two-axis run still produces the same number it did
    before. Per-axis scores are kept as separate fields so the report
    layer can render bands per axis even when not all axes are run.
    """

    composite: float                 # 0–100 (harmonic_mean of scored axes)
    band: str                        # EXCELLENT / PASS / DEGRADED / FAIL
    gtm_score: Optional[float]       # 0–100 (axis A); None if --skip-gtm
    kld_score: Optional[float]       # 0–100 (axis B); None if --skip-kld
    rniah_score: Optional[float] = None  # 0–100 (axis C; v0.2)
    plad_score: Optional[float] = None   # 0–100 (axis D; v0.2)
    floor_score: Optional[float] = None  # measured floor (ref vs ref)
    floor_ok: Optional[bool] = None
    floor_min: float = MIN_FLOOR
    notes: list[str] = field(default_factory=list)


def composite_score(
    gtm_score: Optional[float],
    kld_score: Optional[float],
    rniah_score: Optional[float] = None,
    plad_score: Optional[float] = None,
    floor_score: Optional[float] = None,
) -> CompositeScore:
    """Combine the per-axis scores into a REFRACT composite.

    Axes that weren't run (passed as ``None``) are dropped before
    aggregation. v0.1 callers passing only gtm + kld get a 2-axis
    harmonic mean (unchanged behaviour); v0.2 callers passing all four
    get a 4-axis harmonic mean. v0.3.2.1+: gtm and kld are also
    Optional so that ``--skip-gtm`` / ``--skip-kld`` exclude their axis
    from the composite rather than feeding a stub 100.
    """
    axes: list[float] = []
    if gtm_score is not None:
        axes.append(gtm_score)
    if kld_score is not None:
        axes.append(kld_score)
    if rniah_score is not None:
        axes.append(rniah_score)
    if plad_score is not None:
        axes.append(plad_score)
    composite = harmonic_mean(axes) if axes else 0.0
    floor_ok: Optional[bool] = None
    notes: list[str] = []
    if floor_score is not None:
        floor_ok = floor_score >= MIN_FLOOR
        if not floor_ok:
            notes.append(
                f"Floor failed: REFRACT(ref, ref) = {floor_score:.2f} < {MIN_FLOOR}. "
                "Reference is non-deterministic on this build; KLD deltas are unreliable."
            )
    return CompositeScore(
        composite=composite,
        band=band(composite),
        gtm_score=gtm_score,
        kld_score=kld_score,
        rniah_score=rniah_score,
        plad_score=plad_score,
        floor_score=floor_score,
        floor_ok=floor_ok,
        notes=notes,
    )
