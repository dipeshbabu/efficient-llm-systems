"""CLI presentation for shared Metria comparisons of KV Fidelity reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .comparison import COMPARISON_SCHEMA, _mapping, _number, compare_reports


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not _number(number):
        raise ValueError("non-finite JSON number")
    return number


def _read_report(path: Path) -> dict[str, Any]:
    limit = 64 * 1024 * 1024
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError("report exceeds the 64 MiB input limit")
    report = json.loads(
        data,
        object_pairs_hook=_pairs,
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    return report


def _score(value: Any) -> str:
    return f"{value:.2f}" if _number(value) else "—"


def run_report_comparison(
    paths: list[Path],
    *,
    override_reason: str | None = None,
    json_out: Path | None = None,
) -> int:
    """Load all requested reports, retain diagnostics, and return a CLI status."""
    if len(paths) < 2:
        print("ERROR: comparison requires at least two valid reports")
        return 2
    if override_reason is not None and not override_reason.strip():
        print("ERROR: --allow-incompatible requires a nonempty reason")
        return 2
    if json_out is not None and json_out.resolve() in {
        path.resolve() for path in paths
    }:
        print("ERROR: comparison output must not overwrite an input report")
        return 2
    reports = []
    errors = []
    for path in paths:
        try:
            reports.append(_read_report(path))
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            errors.append({"report": str(path), "reason": str(exc)})
    if errors:
        result: dict[str, Any] = {
            "schema": COMPARISON_SCHEMA,
            "compatible": False,
            "errors": errors,
            "override": {
                "enabled": override_reason is not None,
                "reason": override_reason,
            },
        }
        print("NOT_COMPARABLE: one or more reports could not be read")
        for error in errors:
            print(f"  {error['report']}: {error['reason']}")
        code = 2
    else:
        try:
            result = compare_reports(reports, override_reason=override_reason)
        except (TypeError, ValueError, RecursionError) as exc:
            print(f"ERROR: invalid report evidence: {exc}")
            return 2
        print("COMPARABLE" if result["compatible"] else "NOT_COMPARABLE")
        if override_reason is not None:
            print(f"Inspection override: {override_reason}")
            print("The compatibility and metric-method results below remain unchanged.")
        labels = [path.stem for path in paths]
        for item in result["input_issues"]:
            for issue in item["issues"]:
                print(f"  {labels[item['report']]}: {issue}")
        result["reports"] = [
            {"label": label, "sha256": digest}
            for label, digest in zip(labels, result["report_sha256"], strict=True)
        ]
        for pair in result["pairs"]:
            if not pair["compatible"]:
                print(f"  {labels[pair['left']]} vs {labels[pair['right']]}:")
                for issue in pair["issues"]:
                    print(
                        f"    {issue['dimension']}: {issue['reason']} (left={json.dumps(issue['left'], ensure_ascii=True)}, right={json.dumps(issue['right'], ensure_ascii=True)})"
                    )
                for name, reason in pair["incompatible_metrics"].items():
                    print(f"    metric {name}: {reason}")
        print()
        print(
            f"{'Report':<32} {'Comp':>7} {'Band':<10} {'Traj/GTM':>8} {'KLD':>7} {'R-NIAH':>7} {'PLAD':>7}"
        )
        print("-" * 84)
        for label, report in zip(labels, reports, strict=True):
            axes = _mapping(report.get("axes"))
            scores = []
            for axis in ("gtm", "kld", "rniah", "plad"):
                data = _mapping(axes.get(axis))
                scores.append(
                    "skip" if data.get("skipped") else _score(data.get("score"))
                )
            band = report.get("band")
            band = band if isinstance(band, str) else "—"
            print(
                f"{label[:32]:<32} {_score(report.get('composite')):>7} {band:<10} {scores[0]:>8} {scores[1]:>7} {scores[2]:>7} {scores[3]:>7}"
            )
        code = 0 if result["compatible"] or override_reason is not None else 2
    if json_out is not None:
        try:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            print(f"ERROR: could not write comparison JSON: {exc}")
            return 2
    return code
