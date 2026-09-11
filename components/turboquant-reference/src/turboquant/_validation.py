"""Common boundary checks for the float64 reference implementation."""

from __future__ import annotations

import math
import operator
from numbers import Real
from typing import SupportsIndex, cast

import numpy as np


def integer(
    value: object, name: str, *, minimum: int | None = 0, maximum: int | None = None
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(cast(SupportsIndex, value))
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if (minimum is not None and result < minimum) or (
        maximum is not None and result > maximum
    ):
        bound = (
            f"<= {maximum}"
            if minimum is None
            else (f"{minimum}-{maximum}" if maximum is not None else f">= {minimum}")
        )
        raise ValueError(f"{name} must be {bound}, got {result}")
    return result


def dimension(value: object, name: str = "d") -> int:
    result = integer(value, name, minimum=1)
    if result > np.iinfo(np.intp).max:
        raise ValueError(f"{name} exceeds NumPy's maximum dimension")
    return result


def finite_output(values: np.ndarray, name: str = "reconstruction") -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} exceeds the finite float64 range")
    return values


def real(
    value: object, name: str, *, minimum: float = 0, maximum: float | None = None
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < minimum or (maximum is not None and result > maximum):
        bound = f"{minimum}-{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {bound}")
    return result


def array(
    value: object,
    name: str,
    *,
    kind: str = "real",
    ndim: tuple[int, ...] | None = None,
    d: int | None = None,
) -> np.ndarray:
    result = np.asarray(value)
    if kind == "byte":
        if result.dtype != np.uint8:
            raise TypeError(f"{name} must have dtype uint8")
    elif kind == "integer":
        if result.dtype.kind not in "iu":
            raise TypeError(f"{name} must have an integer dtype")
    else:
        if result.dtype.kind not in "iuf":
            raise TypeError(f"{name} must have a real numeric dtype")
        if kind == "real":
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.asarray(result, dtype=np.float64)
        if result.dtype.kind == "f" and not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain finite float64 values")
    if ndim is not None and result.ndim not in ndim:
        raise ValueError(f"{name} must have dimensionality {ndim}, got {result.ndim}")
    if d is not None and (not result.ndim or result.shape[-1] != d):
        raise ValueError(f"{name} must have final dimension {d}")
    return result


def vectors(value: object, name: str, d: int, *, kind: str = "real") -> np.ndarray:
    return array(value, name, kind=kind, ndim=(1, 2), d=d)


def signs(value: object, name: str, *, d: int | None = None) -> np.ndarray:
    result = array(value, name, kind="numeric", ndim=(1, 2), d=d)
    if not np.all(np.abs(result) == 1):
        raise ValueError(f"{name} must contain only -1 and +1")
    return result


def norms(
    value: object, vector_shape: tuple[int, ...], name: str = "norms"
) -> np.ndarray:
    result = array(value, name)
    if result.shape != vector_shape[:-1]:
        raise ValueError(
            f"{name} must have shape {vector_shape[:-1]}, got {result.shape}"
        )
    if np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def l2_norm(values: np.ndarray) -> np.ndarray:
    """Preserve ordinary NumPy results, with scaling for extreme magnitudes."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.linalg.norm(values, axis=-1)
    bad = ~np.isfinite(result)
    zero = result == 0
    if np.any(zero):
        bad[zero] = np.any(values[zero] != 0, axis=-1)
    if np.any(bad):
        affected = values[bad]
        scale = np.max(np.abs(affected), axis=-1)
        normalized = affected / scale[:, np.newaxis]
        with np.errstate(over="ignore"):
            result[bad] = scale * np.sqrt(np.sum(normalized * normalized, axis=-1))
        if not np.all(np.isfinite(result)):
            raise ValueError("vector norms exceed the finite float64 range")
    return result


def float32_norms(value: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(result)) or np.any((value != 0) & (result == 0)):
        raise ValueError("norms cannot be represented in the packed float32 format")
    return result


def packed_shape(shape: tuple[int, ...], *, d: int) -> tuple[int, ...]:
    if not isinstance(shape, tuple) or len(shape) not in (1, 2):
        raise ValueError("original_shape must describe a vector or a batch")
    result = tuple(integer(item, "original_shape dimension") for item in shape)
    if result[-1] != d:
        raise ValueError(f"original_shape must have final dimension {d}")
    return result


def packed_array(
    value: object, shape: tuple[int, ...], bits: int, name: str
) -> np.ndarray:
    result = array(value, name, kind="byte")
    expected = (*shape[:-1], (shape[-1] * bits + 7) // 8)
    if result.shape != expected:
        raise ValueError(
            f"{name} must have packed shape {expected}, got {result.shape}"
        )
    return result


def packed_norms(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    if np.asarray(value).dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    return norms(value, shape, name)


def finite_ratio(
    numerator: int | float, denominator: int | float, name: str = "compression ratio"
) -> float:
    try:
        result = numerator / denominator
    except (OverflowError, ZeroDivisionError):
        raise ValueError(f"{name} cannot be represented as a finite value") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} cannot be represented as a finite value")
    return result
