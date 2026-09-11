"""Public boundary regressions, including validation under optimized Python."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from turboquant.codebook import nearest_centroid_indices, optimal_centroids
from turboquant.kv_cache import KVCacheCompressor
from turboquant.outlier import OutlierTurboQuant
from turboquant.polar_quant import PolarQuant
from turboquant.qjl import QJL
from turboquant.rotation import apply_fast_rotation, random_rotation_fast
from turboquant.turboquant import TurboQuant, TurboQuantMSE
from turboquant.utils import (
    memory_footprint_bytes,
    pack_bits,
    pack_indices,
    unpack_bits,
    unpack_indices,
)


@pytest.mark.parametrize(
    "factory",
    [
        QJL,
        lambda d: PolarQuant(d, 1),
        lambda d: TurboQuant(d, 2),
        lambda d: TurboQuantMSE(d, 1),
        lambda d: OutlierTurboQuant(d, 2.5),
        KVCacheCompressor,
    ],
)
@pytest.mark.parametrize("dimension", [0, -1])
def test_dimensions_must_be_positive(factory, dimension):
    with pytest.raises(ValueError, match="d|head_dim"):
        factory(dimension)


@pytest.mark.parametrize("dimension", [True, np.bool_(True), 3.0, "3", None])
def test_dimension_kinds_are_checked(dimension):
    with pytest.raises(TypeError, match="integer"):
        QJL(dimension)


@pytest.mark.parametrize("bits", [0, -1, 9])
def test_scalar_codebook_and_polar_bit_range(bits):
    for call in (
        lambda: optimal_centroids(bits, 4),
        lambda: PolarQuant(4, bits),
        lambda: TurboQuantMSE(4, bits),
    ):
        with pytest.raises(ValueError, match="bit_width"):
            call()


@pytest.mark.parametrize("value", [True, 2.0, "2"])
def test_bit_width_must_be_an_integer(value):
    with pytest.raises(TypeError, match="integer"):
        pack_indices(np.array([0], dtype=np.uint8), value)
    with pytest.raises(TypeError, match="integer"):
        PolarQuant(4, value)


@pytest.mark.parametrize("bits", [1, 10])
def test_full_quantizer_bit_range(bits):
    with pytest.raises(ValueError, match="bit_width"):
        TurboQuant(4, bits)


@pytest.mark.parametrize("bits", [1.9, 9.1, float("nan"), float("inf")])
def test_outlier_precision_is_bounded_and_finite(bits):
    with pytest.raises(ValueError, match="target_bits"):
        OutlierTurboQuant(4, bits)


@pytest.mark.parametrize(
    "factory",
    [
        QJL,
        lambda d: PolarQuant(d, 1),
        lambda d: TurboQuant(d, 2),
        lambda d: OutlierTurboQuant(d, 2.5),
    ],
)
@pytest.mark.parametrize("values", [np.ones(3), np.ones((1, 1, 4)), np.array(1.0)])
def test_quantizer_shapes_are_checked(factory, values):
    with pytest.raises(ValueError, match="dimension"):
        factory(4).quantize(values)


@pytest.mark.parametrize(
    "values", [np.array([1, 2, np.nan, 4]), np.array([1, 2, np.inf, 4])]
)
def test_nonfinite_inputs_are_rejected_before_projection(values):
    for quantizer in (QJL(4), PolarQuant(4, 1), OutlierTurboQuant(4, 2.5)):
        with pytest.raises(ValueError, match="finite"):
            quantizer.quantize(values)


@pytest.mark.parametrize("dtype", [np.complex128, np.bool_, object, str])
def test_quantizers_reject_non_real_numeric_dtypes(dtype):
    with pytest.raises(TypeError, match="dtype"):
        QJL(4).quantize(np.ones(4).astype(dtype))


def test_numpy_integer_parameters_and_real_lists_are_supported():
    quantizer = QJL(np.int64(4), seed=np.int64(1))
    signs, norms = quantizer.quantize([1, 2, 3, 4])
    assert signs.shape == (4,)
    assert np.isfinite(norms)


@pytest.mark.parametrize("norm", [-1, float("nan"), float("inf"), np.ones(2)])
def test_reconstruction_norms_are_finite_nonnegative_and_shape_matched(norm):
    with pytest.raises(ValueError):
        QJL(4).dequantize(np.ones(4, dtype=np.int8), norm)
    with pytest.raises(ValueError):
        PolarQuant(4, 1).dequantize(np.zeros(4, dtype=np.uint8), norm)


@pytest.mark.parametrize("signs", [np.array([1, 0, -1, 1]), np.array([1, 2, -1, 1])])
def test_sign_values_cannot_be_silently_coerced(signs):
    with pytest.raises(ValueError, match="only"):
        pack_bits(signs)
    with pytest.raises(ValueError, match="only"):
        QJL(4).dequantize(signs, 1.0)


def test_reconstruction_indices_are_not_negative_or_float_coerced():
    quantizer = PolarQuant(4, 1)
    for indices in (np.array([-1, 0, 0, 0]), np.array([2, 0, 0, 0])):
        with pytest.raises(ValueError, match="indices"):
            quantizer.dequantize(indices, 1.0)
    with pytest.raises(TypeError, match="integer dtype"):
        quantizer.dequantize(np.zeros(4), 1.0)


@pytest.mark.parametrize(
    "centroids", [np.array([]), np.array([1, -1]), np.array([0, np.nan])]
)
def test_centroids_are_nonempty_sorted_and_finite(centroids):
    with pytest.raises(ValueError):
        nearest_centroid_indices(np.array([0.0]), centroids)


def test_midpoint_lookup_does_not_overflow_for_finite_centroids():
    result = nearest_centroid_indices(
        np.array([1.1e308, 1.3e308]), np.array([1e308, 1.4e308])
    )
    np.testing.assert_array_equal(result, [0, 1])


@pytest.mark.parametrize("magnitude", [1e-200, 1e200])
def test_extreme_representable_norms_do_not_become_zero_or_infinity(magnitude):
    vector = np.array([magnitude, 0.0, 0.0, 0.0])
    for quantizer in (QJL(4), PolarQuant(4, 1)):
        data, norm = quantizer.quantize(vector)
        assert norm == pytest.approx(magnitude, rel=1e-14, abs=0)
        assert np.all(np.isfinite(quantizer.dequantize(data, norm)))


@pytest.mark.parametrize("magnitude", [1e-100, 1e100])
def test_packed_norm_precision_limits_fail_explicitly(magnitude):
    with pytest.raises(ValueError, match="float32"):
        TurboQuant(4, 2).quantize_packed(np.full(4, magnitude))


@pytest.mark.parametrize("width", range(1, 9))
@pytest.mark.parametrize("shape", [(0,), (9,), (1, 9), (3, 7), (0, 7), (2, 0)])
def test_bit_packing_round_trip_across_widths_and_empty_batches(width, shape):
    values = np.random.default_rng(17).integers(
        0, 1 << width, size=shape, dtype=np.uint16
    )
    packed = pack_indices(values, width)
    np.testing.assert_array_equal(unpack_indices(packed, width, shape[-1]), values)
    assert packed.nbytes == int(np.prod(shape[:-1])) * ((shape[-1] * width + 7) // 8)


def test_unpacking_checks_byte_dtype_shape_and_capacity():
    with pytest.raises(TypeError, match="uint8"):
        unpack_indices(np.array([257]), 2, 1)
    with pytest.raises(ValueError, match="dimensionality"):
        pack_indices(np.array(1), 2)
    with pytest.raises(ValueError, match="need"):
        unpack_bits(np.zeros(1, dtype=np.uint8), 9)


def test_mixed_compressed_vector_shapes_do_not_broadcast():
    quantizer = TurboQuant(4, 2)
    compressed = quantizer.quantize(np.ones((2, 4)))
    with pytest.raises(ValueError, match="same shape"):
        quantizer.dequantize(replace(compressed, qjl_signs=compressed.qjl_signs[0]))


@pytest.mark.parametrize("quantizer", [TurboQuant(4, 2), TurboQuantMSE(4, 1)])
def test_packed_shape_and_norm_storage_are_validated(quantizer):
    packed = quantizer.quantize_packed(np.ones((2, 4)))
    with pytest.raises(ValueError, match="original_shape"):
        quantizer.dequantize_packed(replace(packed, original_shape=(4, 2)))
    name = "vector_norms" if isinstance(quantizer, TurboQuant) else "norms"
    with pytest.raises(TypeError, match="float32"):
        quantizer.dequantize_packed(
            replace(packed, **{name: getattr(packed, name).astype(np.float64)})
        )


@pytest.mark.parametrize("shape", [(4,), (1, 4), (3, 4)])
def test_outlier_quantizer_preserves_single_and_batch_shapes(shape):
    quantizer = OutlierTurboQuant(4, 2.5)
    values = np.random.default_rng(1).normal(size=shape)
    compressed = quantizer.quantize(values)
    assert compressed.outlier_indices.shape == (*shape[:-1], quantizer.n_outlier)
    assert quantizer.dequantize(compressed).shape == shape


def test_outlier_calibration_rejects_empty_samples_and_handles_large_values():
    quantizer = OutlierTurboQuant(4, 2.5)
    with pytest.raises(ValueError, match="at least one"):
        quantizer.calibrate(np.empty((0, 4)))
    quantizer.calibrate(np.array([[1e308, 9e307, 1, 1]] * 3))
    np.testing.assert_array_equal(quantizer.outlier_idx, [0, 1])


def test_cache_payload_geometry_is_checked_before_allocation():
    compressor = KVCacheCompressor(4, k_bits=2, v_bits=1)
    compressed = compressor.compress(np.ones((1, 1, 2, 4)), np.ones((1, 1, 2, 4)))
    with pytest.raises(ValueError, match="shape"):
        compressor.decompress(replace(compressed, seq_len=10**30))
    with pytest.raises(ValueError, match="counts"):
        compressor.decompress(replace(compressed, num_layers=2))


@pytest.mark.parametrize(
    "args", [(0, 4, 2), (-1, 4, 2), (1, 0, 2), (1, 4, 1), (1, 4, 10)]
)
def test_memory_reports_reject_undefined_or_negative_sizes(args):
    with pytest.raises(ValueError):
        memory_footprint_bytes(*args)


def test_memory_helpers_keep_large_integer_counts_exact():
    count = np.int64(2**53 + 1)
    result = memory_footprint_bytes(count, np.int64(7), np.int64(3))
    assert isinstance(result["total_bytes"], int)
    assert result["total_bytes"] == int(count) * 11
    assert np.isfinite(result["compression_ratio"])
    with pytest.raises(ValueError):
        TurboQuant(4, 2).compressed_size_bits(-1)
    with pytest.raises(ValueError):
        OutlierTurboQuant(4, 2.5).compression_ratio(-16)


@pytest.mark.parametrize(
    "precision,expected", [(2.0, 128 / 80), (2.5, 128 / 116), (3.0, 128 / 88)]
)
def test_outlier_ratio_counts_only_active_norm_fields(precision, expected):
    assert OutlierTurboQuant(8, precision).compression_ratio() == pytest.approx(
        expected
    )


def test_highest_packed_precision_round_trip():
    quantizer = TurboQuant(1, 9)
    original = np.array([1.0])
    packed = quantizer.quantize_packed(original)
    assert packed.mse_indices.dtype == np.uint8
    np.testing.assert_allclose(quantizer.dequantize_packed(packed), original, atol=1e-6)


def test_rotation_geometry_and_signs_are_checked():
    with pytest.raises(ValueError, match="d"):
        random_rotation_fast(0, np.random.default_rng(1))
    with pytest.raises(ValueError, match="power of 2"):
        apply_fast_rotation(np.ones(3), np.ones(3), np.ones(3), 3)
    with pytest.raises(ValueError, match="only"):
        apply_fast_rotation(np.ones(4), np.zeros(4), np.ones(4), 4)


def test_impossible_dimension_fails_before_allocation():
    with pytest.raises(ValueError, match="maximum dimension"):
        QJL(2**100)


def test_nonrepresentable_reconstruction_raises_instead_of_returning_infinity():
    with pytest.raises(ValueError, match="finite float64"):
        QJL(1).dequantize(np.ones(1, dtype=np.int8), np.finfo(np.float64).max)
    with pytest.raises(ValueError, match="finite"):
        TurboQuant(4, 2).compression_ratio(10**1000)


def test_validation_and_orthogonality_checks_survive_python_optimization():
    code = """
import numpy as np
from turboquant.qjl import QJL
from turboquant.utils import memory_footprint_bytes
for call in (lambda: QJL(0), lambda: memory_footprint_bytes(-1, 4, 2)):
    try:
        call()
    except ValueError:
        pass
    else:
        raise RuntimeError('validation disappeared under -O')
np.linalg.qr = lambda matrix: (np.ones_like(matrix), np.eye(matrix.shape[0]))
try:
    QJL(4)
except RuntimeError:
    pass
else:
    raise RuntimeError('orthogonality invariant disappeared under -O')
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", code], capture_output=True, text=True, timeout=45
    )
    assert result.returncode == 0, result.stdout + result.stderr
