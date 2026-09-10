"""Comparison semantics for Metria studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from .models import (
    ComparisonPlan,
    CompatibilityIssue,
    CompatibilityReport,
    RunRecord,
    RunStatus,
)

_MISSING = object()
_MISSING_LABEL = "<missing>"


class _AmbiguousPathError(ValueError):
    """Evidence fields overlap in the comparison's dotted path namespace."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"ambiguous comparison path {path!r}")


_REQUESTED_DIMENSIONS = (
    "model",
    "runtime",
    "scenario",
    "measurements",
    "treatments",
    "trial_policy",
    "environment_selector",
)
_RESOLVED_NON_COMPARISON_ROOTS = frozenset({"support"})
_OBSERVED_NON_COMPARISON_ROOTS = frozenset(
    {
        "invocations",
        "reset_count",
        "closed",
        "cleanup",
    }
)
_INVOCATION_IDENTITY_FIELDS = frozenset(
    {
        "prompt_sha256",
        "rendered_prompt_sha256",
        "system_sha256",
        "generation",
    }
)

# Requested declarations for these stable concepts also govern their direct
# downstream identity/configuration evidence unless a more specific
# resolved/observed declaration is present.
_LIFECYCLE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "model": ("resolved.model", "observed.model"),
    "runtime": (
        "resolved.runtime",
        "observed.runtime",
        "observed.configured.runtime",
    ),
    "scenario": ("resolved.scenario", "observed.scenario"),
}


def _path_covers(declaration: str, path: str) -> bool:
    """Return whether ``declaration`` names ``path`` or one of its ancestors."""

    return path == declaration or path.startswith(f"{declaration}.")


def _mapping_lookup(
    mapping: Mapping[str, Any],
    segments: tuple[str, ...],
    index: int,
) -> tuple[Any, int]:
    """Resolve one mapping key, accepting evidence keys that contain dots."""

    # Some runtime introspection evidence currently stores keys such as
    # ``cache.cache_dtype`` inside one mapping. Prefer the longest matching key
    # so comparison paths can still be written naturally with dots.
    for end in range(len(segments), index, -1):
        candidate = ".".join(segments[index:end])
        if candidate in mapping:
            return mapping[candidate], end
    return _MISSING, index


def _resolve_dimension(record: RunRecord, dimension: str) -> Any:
    """Resolve a nested comparison dimension without conflating missing and None."""

    segments = tuple(dimension.split("."))
    if not segments or any(not segment for segment in segments):
        raise KeyError(dimension)

    root = segments[0]
    if root == "resolved":
        current: Any = record.resolved
        index = 1
    elif root == "observed":
        current = record.observed
        index = 1
    elif root in _REQUESTED_DIMENSIONS:
        current = getattr(record.requested, root)
        index = 1
    else:
        raise KeyError(dimension)

    while index < len(segments):
        if isinstance(current, Mapping):
            current, next_index = _mapping_lookup(current, segments, index)
            if current is _MISSING:
                return _MISSING
            index = next_index
            continue
        if is_dataclass(current):
            name = segments[index]
            if not hasattr(current, name):
                return _MISSING
            current = getattr(current, name)
            index += 1
            continue
        if (
            isinstance(current, Sequence)
            and not isinstance(current, (str, bytes, bytearray))
            and segments[index].isdigit()
        ):
            item_index = int(segments[index])
            if item_index >= len(current):
                return _MISSING
            current = current[item_index]
            index += 1
            continue
        return _MISSING
    return current


def _display_value(value: Any) -> Any:
    """Convert the private missing sentinel into a readable issue value."""

    return _MISSING_LABEL if value is _MISSING else value


def _role_declarations(plan: ComparisonPlan) -> tuple[tuple[str, str], ...]:
    """Return direct role declarations in deterministic specificity order."""

    items: list[tuple[str, str]] = []
    items.extend((dimension, "vary") for dimension in plan.vary)
    items.extend((dimension, "control") for dimension in plan.control)
    items.extend((dimension, "block") for dimension in plan.block_by)
    items.extend((dimension, "waiver") for dimension in plan.waivers)
    return tuple(sorted(items, key=lambda item: (-len(item[0]), item[0], item[1])))


def _role_for_path(
    plan: ComparisonPlan,
    path: str,
    direct_roles: Sequence[tuple[str, str]],
) -> tuple[str, str, bool] | None:
    """Return the most specific direct or lifecycle role governing ``path``."""

    for declaration, role in direct_roles:
        if _path_covers(declaration, path):
            return role, declaration, True

    for requested_dimension, downstream_prefixes in _LIFECYCLE_ALIASES.items():
        inherited_role: str | None = None
        if requested_dimension in plan.vary:
            inherited_role = "vary"
        elif requested_dimension in plan.control:
            inherited_role = "control"
        elif requested_dimension in plan.block_by:
            inherited_role = "block"
        if inherited_role is None:
            continue
        if any(_path_covers(prefix, path) for prefix in downstream_prefixes):
            return inherited_role, requested_dimension, False
    return None


def _flatten(value: Any, path: str) -> dict[str, Any]:
    """Flatten mappings/dataclasses into deterministic leaf comparison paths."""

    if isinstance(value, Mapping):
        if not value:
            return {path: value}
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            segments = key.split(".")
            for index in range(1, len(segments)):
                prefix = ".".join(segments[:index])
                if prefix in value:
                    raise _AmbiguousPathError(f"{path}.{prefix}" if path else prefix)
            child_path = f"{path}.{key}" if path else key
            flattened.update(_flatten(value[key], child_path))
        return flattened

    if is_dataclass(value):
        flattened = {}
        for item in fields(value):
            flattened.update(_flatten(getattr(value, item.name), f"{path}.{item.name}"))
        return flattened or {path: value}

    # Positional recipe collections have stable equality semantics as complete
    # values. Flattening them by index would turn a reorder into noisy
    # pseudo-dimensions.
    return {path: value}


def _invocation_identity(observed: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Project only comparison-relevant prompt/generation identity from invocations."""

    raw = observed.get("invocations")
    if (
        isinstance(raw, (str, bytes, bytearray))
        or not isinstance(raw, Sequence)
        or not raw
    ):
        return None

    projected: dict[str, Any] = {}
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            continue
        identity = {
            key: row[key] for key in sorted(_INVOCATION_IDENTITY_FIELDS) if key in row
        }
        if identity:
            projected[str(index)] = identity
    return projected or None


def _comparison_roots(record: RunRecord) -> dict[str, Any]:
    """Return comparison-relevant requested, resolved, and observed roots."""

    roots = {
        dimension: getattr(record.requested, dimension)
        for dimension in _REQUESTED_DIMENSIONS
    }
    roots.update(
        {
            f"resolved.{key}": value
            for key, value in record.resolved.items()
            if key not in _RESOLVED_NON_COMPARISON_ROOTS
        }
    )
    roots.update(
        {
            f"observed.{key}": value
            for key, value in record.observed.items()
            if key not in _OBSERVED_NON_COMPARISON_ROOTS
        }
    )
    invocation_identity = _invocation_identity(record.observed)
    if invocation_identity is not None:
        roots["observed.invocations"] = invocation_identity
    return roots


def _pair_leaf_values(
    left: RunRecord,
    right: RunRecord,
) -> tuple[tuple[str, Any, Any], ...]:
    """Return all differing comparison-relevant leaves, including missing sides."""

    # Flatten the roots together so overlapping top-level observed/resolved
    # keys are validated just like overlapping keys in nested evidence.
    left_flat = _flatten(_comparison_roots(left), "")
    right_flat = _flatten(_comparison_roots(right), "")
    differences: list[tuple[str, Any, Any]] = []

    for path in sorted(set(left_flat) | set(right_flat)):
        left_leaf = left_flat.get(path, _MISSING)
        right_leaf = right_flat.get(path, _MISSING)
        if left_leaf is not _MISSING and right_leaf is not _MISSING:
            if left_leaf == right_leaf:
                continue
        differences.append((path, left_leaf, right_leaf))
    return tuple(differences)


def _declared_dimension_issue(
    left: RunRecord,
    right: RunRecord,
    dimension: str,
    role: str,
) -> CompatibilityIssue | None:
    """Validate one explicit vary/control/block dimension before leaf checks."""

    try:
        left_value = _resolve_dimension(left, dimension)
        right_value = _resolve_dimension(right, dimension)
    except KeyError:
        return CompatibilityIssue(
            dimension=dimension,
            left=None,
            right=None,
            reason=f"unknown {role} dimension",
        )

    if left_value is _MISSING or right_value is _MISSING:
        return CompatibilityIssue(
            dimension=dimension,
            left=_display_value(left_value),
            right=_display_value(right_value),
            reason=f"required {role} dimension is missing",
        )
    if role in {"vary", "waiver"}:
        return None
    if left_value == right_value:
        return None
    return CompatibilityIssue(
        dimension=dimension,
        left=left_value,
        right=right_value,
        reason=(
            "controlled dimension differs"
            if role == "control"
            else "runs belong to different comparison blocks"
        ),
    )


def compare_runs(
    left: RunRecord,
    right: RunRecord,
    plan: ComparisonPlan,
) -> CompatibilityReport:
    """Determine whether two completed runs support the comparison in ``plan``.

    Comparability is fail-closed. Every comparison-relevant difference must be
    explicitly intentional (``vary``), controlled, blocking, or waived with a
    retained reason. Requested and resolved configuration are comparison
    relevant by default. Observed evidence is also compared except for known
    lifecycle bookkeeping such as cleanup and output-dependent invocation data.

    Direct comparison is strict about lifecycle state. A failed, timed-out, or
    partial record remains useful experiment evidence, but is not analysis-ready
    by default. Both records must therefore be ``COMPLETED``.

    Metric compatibility is stricter than study compatibility: two metrics are
    directly comparable only when their complete metric identities match.
    Waivers never make method-incompatible raw metrics compatible.
    """

    issues: list[CompatibilityIssue] = []
    waived_differences: list[CompatibilityIssue] = []
    if (
        left.status is not RunStatus.COMPLETED
        or right.status is not RunStatus.COMPLETED
    ):
        issues.append(
            CompatibilityIssue(
                dimension="status",
                left=left.status.value,
                right=right.status.value,
                reason="both runs must be completed for direct comparison",
            )
        )

    for dimension in sorted(plan.vary):
        issue = _declared_dimension_issue(left, right, dimension, "vary")
        if issue is not None:
            issues.append(issue)
    for dimension in sorted(plan.control):
        issue = _declared_dimension_issue(left, right, dimension, "control")
        if issue is not None:
            issues.append(issue)
    for dimension in sorted(plan.block_by):
        issue = _declared_dimension_issue(left, right, dimension, "block")
        if issue is not None:
            issues.append(issue)
    for dimension in sorted(plan.waivers):
        issue = _declared_dimension_issue(left, right, dimension, "waiver")
        if issue is not None:
            issues.append(issue)

    direct_roles = _role_declarations(plan)
    try:
        differences = _pair_leaf_values(left, right)
    except _AmbiguousPathError as exc:
        issues.append(
            CompatibilityIssue(
                dimension=exc.path,
                left=None,
                right=None,
                reason=str(exc),
            )
        )
        differences = ()
    for path, left_value, right_value in differences:
        role_match = _role_for_path(plan, path, direct_roles)
        if role_match is None:
            issues.append(
                CompatibilityIssue(
                    dimension=path,
                    left=_display_value(left_value),
                    right=_display_value(right_value),
                    reason="undeclared comparison-relevant difference",
                )
            )
            continue

        role, declaration, direct = role_match
        if left_value is _MISSING or right_value is _MISSING:
            # Direct declarations were already checked as coherent dimensions
            # above. Do not duplicate the same missing subtree as one issue per
            # leaf.
            if direct:
                continue
            issues.append(
                CompatibilityIssue(
                    dimension=path,
                    left=_display_value(left_value),
                    right=_display_value(right_value),
                    reason=f"required {role} evidence is missing",
                )
            )
            continue

        if role == "vary":
            continue
        if role == "waiver":
            waived_differences.append(
                CompatibilityIssue(
                    dimension=path,
                    left=left_value,
                    right=right_value,
                    reason=plan.waivers[declaration],
                )
            )
            continue
        if direct:
            # The explicit control/block declaration was already checked as one
            # coherent dimension above.
            continue

        issues.append(
            CompatibilityIssue(
                dimension=path,
                left=left_value,
                right=right_value,
                reason=(
                    "controlled dimension differs"
                    if role == "control"
                    else "runs belong to different comparison blocks"
                ),
            )
        )

    comparable_metrics: list[str] = []
    incompatible_metrics: dict[str, str] = {}
    shared_metrics = sorted(set(left.metrics) & set(right.metrics))
    for name in shared_metrics:
        left_metric = left.metrics[name]
        right_metric = right.metrics[name]
        if left_metric.definition.identity == right_metric.definition.identity:
            comparable_metrics.append(name)
        else:
            incompatible_metrics[name] = (
                "metric identities differ; compare effect sizes against a "
                "method-matched baseline instead of combining raw values"
            )

    return CompatibilityReport(
        compatible=not issues and not incompatible_metrics,
        issues=tuple(issues),
        comparable_metrics=tuple(comparable_metrics),
        incompatible_metrics=incompatible_metrics,
        waived_differences=tuple(waived_differences),
    )
