"""Explicit built-in integrations, separate from generic evidence policy."""

from __future__ import annotations

from ..capability_checks import CapabilityCheckRegistry
from .turboquant import TURBOQUANT_KV_CHECK


def builtin_capability_checks() -> CapabilityCheckRegistry:
    """Return the bundled checks; no third-party modules are discovered."""
    return CapabilityCheckRegistry((TURBOQUANT_KV_CHECK,))
