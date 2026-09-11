"""Explicit contracts for domain capability checks; no plugin discovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .capabilities import GeometryInspection
from .identity import Capability
from .models import RunSpec


@dataclass(frozen=True)
class CapabilityCheckResult:
    """A domain conclusion and whether this run requires that capability."""

    capability: Capability
    required: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.capability, Capability):
            raise TypeError("capability checks must return a Capability")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")


CapabilityEvaluator = Callable[
    [RunSpec, GeometryInspection, bool], CapabilityCheckResult | None
]


@dataclass(frozen=True)
class CapabilityCheck:
    """A named/versioned rule with explicit experimental-override support.

    The evaluator receives the run, normalized geometry, and an authorized
    override-request flag. It returns None when it does not apply. A rule may
    classify an explicitly overridden case as EXPERIMENTAL, but core never
    upgrades UNSUPPORTED or UNKNOWN results on its behalf.
    """

    name: str
    evaluate: CapabilityEvaluator
    version: str = "1"
    supports_experimental_override: bool = False

    def __post_init__(self) -> None:
        for field in ("name", "version"):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(
                    f"capability check {field} must be a nonempty, trimmed string"
                )
        if not callable(self.evaluate):
            raise TypeError("capability check evaluate must be callable")
        if not isinstance(self.supports_experimental_override, bool):
            raise TypeError("supports_experimental_override must be a boolean")


@dataclass(frozen=True)
class CapabilityCheckRegistry:
    """An immutable, explicitly supplied collection of additional checks.

    Inspection always includes built-in checks. An application registry extends
    them and cannot replace their identities. No import paths or entry points
    are loaded from recipe data.
    """

    checks: tuple[CapabilityCheck, ...] = ()

    def __post_init__(self) -> None:
        checks = tuple(self.checks)
        names = {"model.geometry", "metria.capability_overrides"}
        for check in checks:
            if not isinstance(check, CapabilityCheck):
                raise TypeError("registry entries must be CapabilityCheck values")
            if check.name in names:
                raise ValueError(
                    f"duplicate or reserved capability check name: {check.name}"
                )
            names.add(check.name)
        object.__setattr__(self, "checks", checks)

    def with_check(self, check: CapabilityCheck) -> CapabilityCheckRegistry:
        """Return a registry extended with one explicitly provided rule."""
        return CapabilityCheckRegistry((*self.checks, check))
