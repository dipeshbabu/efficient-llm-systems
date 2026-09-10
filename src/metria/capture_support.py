"""Fail-closed negotiation of measurement capture requirements with runtimes."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .identity import SupportLevel
from .models import RunSpec
from .protocols import (
    CaptureRequest,
    CaptureSupportProbe,
    RuntimeAdapter,
    SupportReport,
)

_STATUS_PRIORITY = {
    SupportLevel.SUPPORTED: 0,
    SupportLevel.EXPERIMENTAL: 1,
    SupportLevel.UNKNOWN: 2,
    SupportLevel.UNSUPPORTED: 3,
}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_LLAMA_TOKEN_IDS_QUALIFICATION_KEY = "llama_cpp_token_ids_capture_sha256"

# Current first-party probes predate the structured capture capability schema.
# Keep these markers explicit and conservative while adapters migrate toward a
# dedicated probe_captures() implementation.
_KNOWN_CAPTURE_MARKERS: Mapping[str, SupportLevel] = {
    "native_output_token_ids": SupportLevel.SUPPORTED,
    "binary_present_unverified": SupportLevel.UNKNOWN,
    "unavailable": SupportLevel.UNSUPPORTED,
}


def _sha256_file(path: Path) -> str:
    """Hash a candidate capture provider without retaining its contents."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker_status(marker: Any) -> SupportLevel:
    """Interpret one probe-evidence capture marker without guessing support."""

    if isinstance(marker, Mapping):
        raw_status = marker.get("status")
        try:
            return SupportLevel(raw_status)
        except (TypeError, ValueError):
            return SupportLevel.UNKNOWN
    if isinstance(marker, str):
        return _KNOWN_CAPTURE_MARKERS.get(marker, SupportLevel.UNKNOWN)
    return SupportLevel.UNKNOWN


def _qualified_llama_token_ids(
    *,
    marker: Any,
    runtime_support: SupportReport,
    environment: Mapping[str, Any],
) -> tuple[SupportLevel, Mapping[str, Any], str | None]:
    """Promote an unverified llama capture binary only by exact content identity."""

    if (
        marker != "binary_present_unverified"
        or "llama_completion" not in runtime_support.evidence
    ):
        return _marker_status(marker), {}, None

    expected = environment.get(_LLAMA_TOKEN_IDS_QUALIFICATION_KEY)
    provider = runtime_support.evidence.get("llama_completion")
    if expected is None:
        return (
            SupportLevel.UNKNOWN,
            {"qualification": "missing"},
            "llama.cpp token_ids capture binary is present but not qualified",
        )
    if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
        return (
            SupportLevel.UNSUPPORTED,
            {"qualification": "invalid"},
            f"environment.{_LLAMA_TOKEN_IDS_QUALIFICATION_KEY} must be a SHA-256 digest",
        )
    if not isinstance(provider, str) or not provider:
        return (
            SupportLevel.UNKNOWN,
            {"qualification": "provider_path_missing"},
            "llama.cpp token_ids capture provider identity is missing",
        )

    path = Path(provider)
    try:
        actual = _sha256_file(path)
    except OSError:
        return (
            SupportLevel.UNSUPPORTED,
            {"qualification": "provider_unreadable"},
            "qualified llama.cpp token_ids capture provider cannot be read",
        )

    qualification = {
        "qualification": "sha256",
        "expected_sha256": expected.lower(),
        "observed_sha256": actual,
    }
    if actual.lower() != expected.lower():
        return (
            SupportLevel.UNSUPPORTED,
            qualification,
            "llama.cpp token_ids capture provider does not match qualified SHA-256",
        )
    return SupportLevel.SUPPORTED, qualification, None


def _worst_status(statuses: Sequence[SupportLevel]) -> SupportLevel:
    """Return the most conservative status across required captures."""

    if not statuses:
        return SupportLevel.SUPPORTED
    return max(statuses, key=_STATUS_PRIORITY.__getitem__)


def _fallback_from_runtime_probe(
    runtime_support: SupportReport,
    environment: Mapping[str, Any],
    capture: tuple[CaptureRequest, ...],
) -> SupportReport:
    """Translate existing ``<kind>_capture`` probe evidence conservatively."""

    if not capture:
        return SupportReport(
            status=SupportLevel.SUPPORTED,
            evidence={"mechanism": "no_capture_required", "captures": {}},
        )

    conclusions: dict[str, Any] = {}
    reasons: list[str] = []
    statuses: list[SupportLevel] = []
    for request in capture:
        evidence_key = f"{request.kind}_capture"
        marker = runtime_support.evidence.get(evidence_key)
        if request.kind == "token_ids":
            status, qualification, reason = _qualified_llama_token_ids(
                marker=marker,
                runtime_support=runtime_support,
                environment=environment,
            )
        else:
            status = _marker_status(marker)
            qualification = {}
            reason = None

        statuses.append(status)
        conclusions[request.kind] = {
            "status": status.value,
            "probe_evidence_key": evidence_key,
            "probe_marker": marker,
            **qualification,
        }
        if reason is not None:
            reasons.append(reason)
        elif status is SupportLevel.UNSUPPORTED:
            reasons.append(
                f"runtime reports required capture {request.kind!r} unavailable"
            )
        elif status is SupportLevel.UNKNOWN:
            reasons.append(
                f"runtime does not provide verified support for required capture "
                f"{request.kind!r}"
            )
        elif status is SupportLevel.EXPERIMENTAL:
            reasons.append(
                f"runtime reports required capture {request.kind!r} as experimental"
            )

    status = _worst_status(statuses)
    return SupportReport(
        status=status,
        reasons=tuple(reasons),
        evidence={
            "mechanism": "runtime_probe_evidence",
            "captures": conclusions,
        },
    )


def probe_capture_support(
    *,
    adapter: RuntimeAdapter,
    runtime_support: SupportReport,
    spec: RunSpec,
    environment: Mapping[str, Any],
    capture: tuple[CaptureRequest, ...],
) -> SupportReport:
    """Prove whether all selected measurement captures are available pre-launch.

    Adapters can implement the optional, runtime-checkable ``CaptureSupportProbe``
    contract without changing the base runtime lifecycle protocol. This keeps the
    transition source-compatible for existing adapters while ensuring any
    required capture still fails closed. When no dedicated method exists, Metria
    consumes the existing ``<kind>_capture`` evidence emitted by first-party
    runtime probes. Unknown or unrecognized evidence never becomes a support
    claim.

    llama.cpp's existing ``binary_present_unverified`` marker remains unknown
    unless the environment supplies the exact qualified ``llama-completion``
    content digest. That qualification bridge is intentionally narrow and will
    be superseded by the real-engine qualification artifacts tracked in #12.
    """

    if not isinstance(adapter, CaptureSupportProbe):
        return _fallback_from_runtime_probe(runtime_support, environment, capture)

    report = adapter.probe_captures(spec, environment, capture)
    if not isinstance(report, SupportReport):
        raise TypeError("runtime capture probe must return SupportReport")
    return report
