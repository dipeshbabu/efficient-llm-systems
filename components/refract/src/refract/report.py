"""REFRACT report-card formatter.

Produces:
  - text() : ANSI-coloured human-readable report card with bar charts.
  - json() : machine-readable dict suitable for ML pipelines.

No external dependencies: plain ANSI escapes for colour. Set NO_COLOR=1 in
the env to suppress them.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import os
from dataclasses import asdict
from typing import Optional

from . import __report_schema__, __version__
from .axes.gtm import GTMResult
from .axes.kld import KLDResult
from .axes.plad import PLADResult
from .axes.rniah import RNIAHResult
from .axes.trajectory import TrajectoryResult
from .score import CompositeScore, MIN_FLOOR, band, interpret_pattern


def _use_color() -> bool:
    return not os.environ.get("NO_COLOR")


def _c(code: str, s: str) -> str:
    if not _use_color():
        return s
    return f"\033[{code}m{s}\033[0m"


def _wrap_lines(text: str, indent: str = "", width: int = 68) -> list[str]:
    """Word-wrap a paragraph to ``width`` chars with a leading ``indent``.
    Used by the v0.2.0 Diagnosis block so multi-sentence interpretations
    don't overflow the report card on narrow terminals.
    """
    import textwrap
    return textwrap.wrap(
        text, width=width,
        initial_indent=indent, subsequent_indent=indent,
    ) or [indent.rstrip()]


def _band_color(b: str) -> str:
    return {
        "EXCELLENT": "32",  # green
        "PASS":      "32",
        "DEGRADED":  "33",  # yellow
        "FAIL":      "31",  # red
    }.get(b, "0")


# v0.1.4: layman-readable one-line interpretation per band.
_BAND_PROSE: dict[str, str] = {
    "EXCELLENT": "No material drift detected on the measured surfaces.",
    "PASS":      "Minor measured drift; validate on the target workload.",
    "DEGRADED":  "Visible drift. Audit on your workload before deploying.",
    "FAIL":      "Material quality loss. Treat as broken.",
}


def _sanitize_home_arg(arg: str, home: Optional[str] = None) -> str:
    """Replace a leading home directory with ``~`` on any path style."""
    home = home or os.path.expanduser("~")
    if not home:
        return arg
    variants = {home, home.replace("\\", "/"), home.replace("/", "\\")}
    for prefix in sorted(variants, key=len, reverse=True):
        if arg == prefix:
            return "~"
        if arg.startswith(prefix + "/") or arg.startswith(prefix + "\\"):
            return "~" + arg[len(prefix):]
    return arg


# v0.1.4: short, human description per axis. Used to label what each
# axis actually measures so layman-readers can map "Axis A score is bad"
# to a real-world consequence without reading the paper.
_AXIS_PROSE: dict[str, str] = {
    "gtm":        "Token-level agreement with the fp16 reference.",
    "trajectory": "Token-level agreement with the fp16 reference.",
    "kld":        "Distribution-level divergence from the fp16 reference.",
    "rniah":      "Long-context retrieval quality vs the reference.",
    "plad":       "Robustness to small prompt changes vs the reference.",
}


def _axis_label(name: str) -> str:
    """Map internal axis key → display label in the report card."""
    return {
        "gtm":        "Axis A GTM       ",
        "trajectory": "Axis A Trajectory",
        "kld":        "Axis B KLD       ",
        "rniah":      "Axis C R-NIAH    ",
        "plad":       "Axis D PLAD      ",
    }.get(name, name)


def _axis_line(name: str, score: Optional[float], bar_width: int = 40) -> str:
    """One report line per axis: label, score, bar, band, prose.

    When ``score is None`` the axis was skipped via ``--skip-gtm`` /
    ``--skip-kld``; render it explicitly as ``n/a`` so a reader doesn't
    interpret a missing axis as a perfect 100.
    """
    if score is None:
        empty_bar = " " * bar_width
        skipped = _c("33", "skipped ")
        return (
            f" {_axis_label(name)}:    n/a  "
            f"{empty_bar}  {skipped}  "
            f"{_AXIS_PROSE.get(name, '')}"
        )
    b = band(score)
    band_str = _c(_band_color(b), f"{b:<9}")
    return (
        f" {_axis_label(name)}: {score:6.2f}  "
        f"{_bar(score, bar_width)}  {band_str}  "
        f"{_AXIS_PROSE.get(name, '')}"
    )


def _bar(score: float, width: int = 40) -> str:
    """ANSI bar of length ``width`` representing 0–100."""
    fill = int(round(width * max(0.0, min(score, 100.0)) / 100.0))
    bar = "#" * fill + "-" * (width - fill)
    color = _band_color(band(score))
    return _c(color, f"[{bar}]")


def text_report(
    *,
    model: str,
    reference_label: str,
    candidate_label: str,
    composite: CompositeScore,
    gtm: GTMResult,
    kld: KLDResult,
    rniah: Optional[RNIAHResult] = None,
    plad: Optional[PLADResult] = None,
    extras: Optional[dict] = None,
) -> str:
    """Render the report card as a human-readable string."""
    lines: list[str] = []
    bar_width = 40

    # Header
    lines.append("=" * 72)
    lines.append(_c(
        "1", f" REFRACT v{__version__} — Reference-anchored Robust Acid-test"
    ))
    lines.append(_c("2",
        " Scoring: 0 = broken, 100 = matches the fp16 reference. "
        "Higher is better."))
    lines.append("=" * 72)
    lines.append(f" model     : {model}")
    lines.append(f" reference : {reference_label}")
    lines.append(f" candidate : {candidate_label}")
    lines.append(f" timestamp : {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("-" * 72)

    # Floor
    if composite.floor_score is not None:
        floor_ok = composite.floor_ok
        tag = _c("32", "OK") if floor_ok else _c("31", "FAIL")
        lines.append(
            f" Noise floor (ref vs ref): "
            f"{composite.floor_score:6.2f}  (min {MIN_FLOOR})  [{tag}]"
        )
    else:
        lines.append(
            _c("33", " Noise floor: NOT MEASURED — pass --measure-floor to verify."),
        )
    lines.append("-" * 72)

    # Composite
    band_str = _c(_band_color(composite.band), composite.band)
    # Count axes that contributed to the harmonic mean so the gloss is honest
    # about what produced the number. Skipped axes (--skip-gtm / --skip-kld)
    # contribute None to composite_score and are excluded here too.
    n_axes = (
        (1 if composite.gtm_score is not None else 0)
        + (1 if composite.kld_score is not None else 0)
        + (1 if composite.rniah_score is not None else 0)
        + (1 if composite.plad_score is not None else 0)
    )
    lines.append(
        _c("1", f" REFRACT score    : {composite.composite:6.2f}  "
                f"{_bar(composite.composite, bar_width)}  {band_str}")
    )
    lines.append(f"   (harmonic mean of {n_axes} axes; any single low axis "
                 f"drops the score hard)")
    # v0.1.4: layman-readable interpretation. The headline tells techies a
    # number; this line tells everyone else what the number means.
    lines.append(f" → { _BAND_PROSE.get(composite.band, '') }")
    lines.append("")
    # v0.1.4: per-axis lines now carry per-axis bands and a short prose
    # description of what that axis measures. So a "DEGRADED" composite
    # comes with a per-axis breakdown showing which surface degraded.
    axis_a_key = "trajectory" if isinstance(gtm, TrajectoryResult) else "gtm"
    lines.append(_axis_line(axis_a_key, composite.gtm_score, bar_width))
    lines.append(_axis_line("kld", composite.kld_score, bar_width))
    if composite.rniah_score is not None:
        lines.append(_axis_line("rniah", composite.rniah_score, bar_width))
    elif rniah is not None:
        lines.append(
            f" {_axis_label('rniah')}: {rniah.score:6.2f}  "
            "excluded (low-confidence fp16 baseline)"
        )
    if composite.plad_score is not None:
        lines.append(_axis_line("plad", composite.plad_score, bar_width))
    lines.append("-" * 72)

    # v0.2.0: pattern-matched plain-English diagnosis. Tells the user what
    # the per-axis pattern means in human terms (e.g. "decode distribution
    # broken but reasoning intact") and what to consider next. Distinct from
    # the layman summary line under the composite, which only describes the
    # severity band.
    diagnosis = interpret_pattern(
        gtm_score=composite.gtm_score,
        kld_score=composite.kld_score,
        rniah_score=composite.rniah_score,
        plad_score=composite.plad_score,
    )
    if diagnosis:
        lines.append(_c("1", " Diagnosis"))
        for d in diagnosis:
            for chunk in _wrap_lines(d, indent="   ", width=68):
                lines.append(chunk)
        lines.append("-" * 72)

    # GTM / Trajectory diagnostics — suppress if axis was skipped, since
    # the diagnostic block on a stub result reads as "0 prompts, all matched"
    # which is misleading.
    if composite.gtm_score is not None:
        lines.append(" GTM diagnostics")
        lines.append(f"   prompts                    : {gtm.n_prompts}")
        lines.append(f"   token cap each (n_predict) : {gtm.n_tokens_each}")
        lines.append(f"   full match rate            : {gtm.full_match_rate*100:5.1f} %")
        if gtm.median_first_divergence is not None:
            lines.append(
                f"   median first divergence    : token {gtm.median_first_divergence}"
            )
        else:
            lines.append("   median first divergence    : (all matched)")
        lines.append(
            f"   mean prefix agreement      : {gtm.mean_prefix_agreement_length:5.1f} tokens"
        )
        lines.append(
            f"   mean cand / ref length     : {gtm.mean_cand_length:5.1f} / "
            f"{gtm.mean_ref_length:5.1f} tokens"
        )
        if gtm.notes:
            for n in gtm.notes:
                lines.append(_c("33", f"   NOTE: {n}"))
    else:
        lines.append(_c("33",
            " GTM/Trajectory: SKIPPED via --skip-gtm; not measured."))

    # KLD diagnostics
    if composite.kld_score is not None:
        lines.append("")
        lines.append(" KLD diagnostics")
        lines.append(f"   chunks x ctx               : {kld.chunks} x {kld.ctx}")
        approximate = not kld.metadata.get("full_vocabulary", True)
        metric_name = "approx. KLD (nats)" if approximate else "mean KLD (nats)"
        lines.append(f"   {metric_name:<28}: {kld.mean_kld:.6f}")
        if approximate:
            lines.append(
                "   estimator                  : normalized top-k + omitted-mass bucket"
            )
        if kld.ppl is not None:
            lines.append(f"   candidate PPL              : {kld.ppl:.4f}")
        if kld.rms_dp_pct is not None:
            lines.append(f"   RMS Δp (vs reference)      : {kld.rms_dp_pct:.2f} %")
        if kld.same_topp_pct is not None:
            lines.append(f"   same top-p (vs reference)  : {kld.same_topp_pct:.2f} %")
    else:
        lines.append("")
        lines.append(_c("33",
            " KLD: SKIPPED via --skip-kld; not measured."))

    # R-NIAH diagnostics
    if rniah is not None:
        lines.append("")
        lines.append(" R-NIAH diagnostics")
        lines.append(f"   needle keyword             : {rniah.password_keyword}")
        lines.append(f"   cells run                  : {rniah.n_cells}")
        lines.append(f"   fp16 base accuracy         : {rniah.base_accuracy:.1%}")
        if rniah.confidence == "low":
            lines.append(
                _c("33", "   LOW CONFIDENCE: excluded from composite; fp16 did not "
                   "reliably engage the retrieval task.")
            )
        if rniah.skipped_cells:
            lines.append(
                f"   cells skipped (length>ctx) : {len(rniah.skipped_cells)}"
            )
        if rniah.cells:
            lines.append("   per-cell (length, pos) → base_acc / cand_acc / degradation:")
            for c in rniah.cells:
                lines.append(
                    f"     ({c.length:>5}, {c.position:.2f}) → "
                    f"{c.base_acc:.2f} / {c.cand_acc:.2f} / {c.degradation:.2f}"
                )
        if rniah.notes:
            for n in rniah.notes:
                lines.append(_c("33", f"   NOTE: {n}"))

    # PLAD diagnostics
    if plad is not None:
        lines.append("")
        lines.append(" PLAD diagnostics")
        lines.append(f"   prompts × perturbations    : "
                     f"{plad.n_prompts} × {plad.n_perturbations}")
        import math as _math
        for pert, score in plad.per_perturbation_score.items():
            if not isinstance(score, (int, float)) or _math.isnan(score):
                # Skipped perturbation (no eligible word, no synonym match).
                # Display as "skipped" rather than "nan FAIL" which would
                # misread as a failure rather than an inapplicable test.
                ptag = _c("33", f"{'skipped':>6} {'n/a':<9}")
            else:
                ptag = _c(_band_color(band(score)), f"{score:6.2f} {band(score):<9}")
            lines.append(f"   {pert:<10} : {ptag}")
        if plad.notes:
            for n in plad.notes:
                lines.append(_c("33", f"   NOTE: {n}"))

    if composite.notes:
        lines.append("-" * 72)
        for n in composite.notes:
            lines.append(_c("33", f" NOTE: {n}"))

    if extras:
        lines.append("-" * 72)
        for k, v in extras.items():
            lines.append(f" {k}: {v}")

    lines.append("=" * 72)
    return "\n".join(lines)


def json_report(
    *,
    model: str,
    reference_label: str,
    candidate_label: str,
    composite: CompositeScore,
    gtm: GTMResult,
    kld: KLDResult,
    rniah: Optional[RNIAHResult] = None,
    plad: Optional[PLADResult] = None,
    include_per_prompt: bool = True,
    extras: Optional[dict] = None,
) -> dict:
    """Return a JSON-serialisable dict twin of the text report."""
    gtm_dict = asdict(gtm)
    if not include_per_prompt:
        gtm_dict.pop("per_prompt", None)
    composite_dict = asdict(composite)
    # Flatten the composite scalar to top-level so consumers can read
    # `d['composite']` as a number directly. Keep the full breakdown under
    # `composite_detail` for diagnostics.
    composite_scalar = composite_dict.pop("composite")
    composite_band = composite_dict.pop("band")
    def _band_or_skipped(s):
        return band(s) if s is not None else "skipped"
    # When an axis was skipped, the underlying result dataclass still has a
    # stub score=100. Null out the JSON 'score' field so downstream readers
    # (compare, leaderboards, paper tables) don't pick up the stub as real.
    if composite.gtm_score is None:
        gtm_dict["score"] = None
    kld_dict = asdict(kld)
    if composite.kld_score is None:
        kld_dict["score"] = None
    axes_block: dict = {
        "gtm": {
            **gtm_dict,
            "band": _band_or_skipped(composite.gtm_score),
            "skipped": composite.gtm_score is None,
            "description": _AXIS_PROSE["gtm"],
        },
        "kld": {
            **kld_dict,
            "band": _band_or_skipped(composite.kld_score),
            "skipped": composite.kld_score is None,
            "description": _AXIS_PROSE["kld"],
        },
    }
    if rniah is not None:
        rn_dict = asdict(rniah)
        # v0.3.1: confidence guard. If base_acc averages below 0.2 across
        # cells, the model isn't engaging the retrieval task at all; an
        # R-NIAH score of 100 is then a noise-floor reading rather than
        # real signal.
        base_avg = rniah.base_accuracy
        excluded = rniah.confidence == "low" or composite.rniah_score is None
        rn_dict["confidence"] = rniah.confidence
        rn_dict["base_acc_avg"] = base_avg
        rn_dict["excluded_from_composite"] = excluded
        axes_block["rniah"] = {
            **rn_dict,
            "band": "unscored" if excluded else band(composite.rniah_score),
            "description": _AXIS_PROSE["rniah"],
        }
    if plad is not None and composite.plad_score is not None:
        pl_dict = asdict(plad)
        # v0.3.1: confidence guard. Per-perturbation scores that are NaN
        # indicate the perturbation never fired (typo on prompts with no
        # ≥4-char words, paraphrase with no synonym matches). Mark them
        # as "skipped" so a reader doesn't read NaN as FAIL.
        import math as _math
        skipped = [k for k, v in plad.per_perturbation_score.items()
                   if not isinstance(v, (int, float)) or _math.isnan(v)]
        pl_dict["skipped_perturbations"] = skipped
        pl_dict["confidence"] = (
            "partial" if skipped else "ok"
        )
        axes_block["plad"] = {
            **pl_dict,
            "band": band(composite.plad_score),
            "description": _AXIS_PROSE["plad"],
        }
    # v0.3.1: framework version + environment metadata so cross-person
    # report comparison is reproducible. Backend metadata (llama.cpp commit,
    # mlx-lm version, etc.) is also captured if the active backend can
    # supply it.
    try:
        from . import __version__ as _fv
    except Exception:
        _fv = "unknown"
    env_meta: dict = {}
    try:
        from .runner import get_active_backend
        bk = get_active_backend()
        if bk is not None:
            from pathlib import Path as _P
            env_meta = bk.model_metadata(model=_P(model)) if model else {"backend": bk.name}
    except Exception:
        pass

    # v0.3.2: capture sanitized repro command so the HTML report shows the
    # actual `refract score ...` invocation that produced the result rather
    # than whatever script re-rendered the JSON. Strip /Users/<name>/ to ~/
    # to avoid leaking personal paths in shared reports. Gated on "argv
    # looks like a refract CLI run" so regen/test scripts don't pollute
    # the field.
    try:
        import sys as _sys, shlex as _shlex, os as _os
        argv = [str(a) for a in _sys.argv]
        looks_like_refract = any(
            "refract.cli" in a
            or a.replace("\\", "/").rstrip("/").endswith("/refract")
            or a == "refract"
            for a in argv
        )
        if looks_like_refract:
            home = _os.path.expanduser("~")
            parts = [_shlex.quote(_sanitize_home_arg(a, home)) for a in argv]
            repro_cmd = " ".join(parts)
        else:
            repro_cmd = ""
    except Exception:
        repro_cmd = ""

    return {
        "schema": __report_schema__,
        "framework_version": _fv,
        "environment": env_meta,
        "repro_command": repro_cmd,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        # v0.2.0: be explicit about score direction so machine consumers
        # (and humans converting from PPL where lower is better) don't
        # invert the comparison.
        "score_direction": "higher_is_better",
        "score_range": [0, 100],
        "model": model,
        "reference": reference_label,
        "candidate": candidate_label,
        "composite": composite_scalar,
        "band": composite_band,
        # v0.1.4: layman-readable one-liner alongside the numeric score
        # so non-techies can read the report without grepping the paper.
        "summary": _BAND_PROSE.get(composite_band, ""),
        # v0.2.0: pattern-matched plain-English diagnosis of the per-axis
        # band combination. Empty list means all axes were intact.
        "diagnosis": interpret_pattern(
            gtm_score=composite.gtm_score,
            kld_score=composite.kld_score,
            rniah_score=composite.rniah_score,
            plad_score=composite.plad_score,
        ),
        "composite_detail": composite_dict,
        "axes": axes_block,
        "extras": extras or {},
    }


def to_json_string(report: dict) -> str:
    return _json.dumps(report, indent=2, default=str)
