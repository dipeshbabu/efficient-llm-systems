"""Typed observed identity evidence for Metria runtime verification."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from ._freeze import freeze_mapping

_IDENTITY_SCHEMA = "metria.runtime_identity.v1"
_IDENTITY_FIELDS = (
    "model",
    "tokenizer",
    "runtime",
    "chat_template",
    "applied",
    "endpoint",
    "reasons",
)
_IDENTITY_KEYS = ("schema", "status", *_IDENTITY_FIELDS)
_IDENTITY_KEY_SET = frozenset(_IDENTITY_KEYS)
_SAFE_ENDPOINT_FIELDS = frozenset(
    {
        "scheme",
        "host",
        "port",
        "service",
        "instance_id",
        "region",
        "path_sha256",
    }
)


class IdentityStatus(str, Enum):
    """Authority state for observed identity evidence."""

    VERIFIED = "verified"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    MISMATCH = "mismatch"


def _normalize_status(value: IdentityStatus | str, *, name: str) -> IdentityStatus:
    try:
        return IdentityStatus(value)
    except (TypeError, ValueError) as exc:
        supported = ", ".join(status.value for status in IdentityStatus)
        raise ValueError(f"{name} must be one of: {supported}") from exc


def _normalize_component(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} identity must be a mapping")
    frozen = freeze_mapping(value)
    if not frozen:
        return frozen
    if "status" not in frozen:
        raise ValueError(f"{name} identity must declare status")
    status = _normalize_status(frozen["status"], name=f"{name} identity status")
    normalized = dict(frozen)
    normalized["status"] = status.value
    # All caller-owned values were detached above; only the normalized scalar
    # status changed. Re-freezing would copy every nested value a second time.
    return MappingProxyType(normalized)


def _component_status(value: Mapping[str, Any]) -> IdentityStatus | None:
    if not value:
        return None
    return _normalize_status(value["status"], name="identity component status")


def _normalize_reasons(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("identity reasons must be a sequence of strings")
    reasons = tuple(value)
    if any(not isinstance(reason, str) or not reason.strip() for reason in reasons):
        raise ValueError("identity reasons must be non-empty strings")
    return reasons


def aggregate_identity_status(
    statuses: Sequence[IdentityStatus | str],
) -> IdentityStatus:
    """Aggregate component authority without promoting unknown evidence."""

    normalized = tuple(
        _normalize_status(status, name="identity component status")
        for status in statuses
    )
    if not normalized:
        return IdentityStatus.UNKNOWN
    if IdentityStatus.MISMATCH in normalized:
        return IdentityStatus.MISMATCH
    if all(status is IdentityStatus.VERIFIED for status in normalized):
        return IdentityStatus.VERIFIED
    if all(status is IdentityStatus.UNKNOWN for status in normalized):
        return IdentityStatus.UNKNOWN
    return IdentityStatus.PARTIAL


@dataclass(frozen=True)
class RuntimeIdentityEvidence(Mapping[str, Any]):
    """Normalized observed identity for one launched inference runtime.

    Component mappings deliberately retain only observed facts and their
    authority state. Requested intent stays in ``RunRecord.requested`` and
    resolved choices stay in ``RunRecord.resolved``. This prevents a requested
    value from being copied into observation and accidentally treated as proof.

    ``endpoint`` is intentionally constrained to a small safe field set so a
    future server adapter cannot place credentials or authorization headers in
    durable identity evidence by mistake.
    """

    status: IdentityStatus | str
    model: Mapping[str, Any] = field(default_factory=dict)
    tokenizer: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    chat_template: Mapping[str, Any] = field(default_factory=dict)
    applied: Mapping[str, Any] = field(default_factory=dict)
    endpoint: Mapping[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        status = _normalize_status(self.status, name="runtime identity status")
        model = _normalize_component(self.model, name="model")
        tokenizer = _normalize_component(self.tokenizer, name="tokenizer")
        runtime = _normalize_component(self.runtime, name="runtime")
        chat_template = _normalize_component(self.chat_template, name="chat template")
        applied = _normalize_component(self.applied, name="applied configuration")
        if not isinstance(self.endpoint, Mapping):
            raise TypeError("endpoint identity must be a mapping")
        endpoint = freeze_mapping(self.endpoint)
        unknown_endpoint_fields = sorted(set(endpoint) - _SAFE_ENDPOINT_FIELDS)
        if unknown_endpoint_fields:
            raise ValueError(
                "endpoint identity contains unsupported or potentially sensitive fields: "
                + ", ".join(unknown_endpoint_fields)
            )
        reasons = _normalize_reasons(self.reasons)

        component_statuses = tuple(
            component_status
            for component_status in (
                _component_status(model),
                _component_status(tokenizer),
                _component_status(runtime),
                _component_status(chat_template),
                _component_status(applied),
            )
            if component_status is not None
        )
        aggregate = aggregate_identity_status(component_statuses)
        if status is not aggregate:
            raise ValueError(
                "runtime identity status must match aggregate component authority: "
                f"expected {aggregate.value!r}, got {status.value!r}"
            )

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "chat_template", chat_template)
        object.__setattr__(self, "applied", applied)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "reasons", reasons)

    def to_mapping(self) -> Mapping[str, Any]:
        """Expose the already-frozen fields without copying nested evidence."""

        return MappingProxyType(dict(self))

    def __getitem__(self, key: str) -> Any:
        if key not in _IDENTITY_KEY_SET:
            raise KeyError(key)
        if key == "schema":
            return _IDENTITY_SCHEMA
        if key == "status":
            return IdentityStatus(self.status).value
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_IDENTITY_KEYS)

    def __len__(self) -> int:
        return len(_IDENTITY_KEYS)
