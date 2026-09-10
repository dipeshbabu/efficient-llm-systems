"""Tests for bit packing and memory utilities (Issue #8)."""

import numpy as np
import pytest

from turboquant.utils import (
    memory_footprint_bytes,
    pack_bits,
    pack_indices,
    unpack_bits,
    unpack_indices,
)


class TestBitPacking:
    @pytest.mark.parametrize(
        "shape", [(0,), (1,), (7,), (8,), (9,), (2, 0), (0, 9), (3, 9)]
    )
    def test_sign_packing_matches_byte_layout_for_strided_inputs(self, shape):
        rng = np.random.default_rng(42)
        storage = (
            rng.integers(0, 2, size=(*shape[:-1], shape[-1] * 2), dtype=np.int8) * 2 - 1
        )
        signs = storage[..., ::2]
        expected = np.zeros((*shape[:-1], (shape[-1] + 7) // 8), dtype=np.uint8)
        for index in np.ndindex(shape):
            if signs[index] > 0:
                byte_index = (*index[:-1], index[-1] // 8)
                expected[byte_index] |= 1 << (7 - index[-1] % 8)

        packed = pack_bits(signs)

        np.testing.assert_array_equal(packed, expected)
        np.testing.assert_array_equal(unpack_bits(packed, shape[-1]), signs)

    @pytest.mark.parametrize("shape", [(0,), (5,), (0, 5), (2, 0), (2, 7)])
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
    def test_8bit_packing_preserves_strided_rows_and_owned_output(self, shape, dtype):
        rng = np.random.default_rng(42)
        storage = rng.integers(0, 256, size=(*shape[:-1], shape[-1] * 2), dtype=dtype)
        indices = storage[..., ::2]

        packed = pack_indices(indices, 8)
        np.testing.assert_array_equal(packed, indices)
        assert packed.dtype == np.uint8
        assert not np.shares_memory(packed, indices)

        extra = np.full((*shape[:-1], 1), 255, dtype=np.uint8)
        padded = np.concatenate((packed, extra), axis=-1)
        unpacked = unpack_indices(padded, 8, shape[-1])
        np.testing.assert_array_equal(unpacked, indices)
        assert unpacked.dtype == np.uint8
        assert not np.shares_memory(unpacked, padded)

    def test_pack_unpack_round_trip(self):
        signs = np.array([1, -1, 1, -1, 1, 1, -1, 1], dtype=np.int8)
        packed = pack_bits(signs)
        unpacked = unpack_bits(packed, 8)
        np.testing.assert_array_equal(signs, unpacked)

    def test_pack_unpack_non_multiple_of_8(self):
        signs = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        packed = pack_bits(signs)
        unpacked = unpack_bits(packed, 5)
        np.testing.assert_array_equal(signs, unpacked)

    def test_pack_correct_size(self):
        signs = np.random.default_rng(42).choice([-1, 1], size=128).astype(np.int8)
        packed = pack_bits(signs)
        assert packed.shape == (16,)  # 128/8

    def test_batch_pack_unpack(self):
        rng = np.random.default_rng(42)
        signs = rng.choice([-1, 1], size=(5, 64)).astype(np.int8)
        packed = pack_bits(signs)
        assert packed.shape == (5, 8)
        unpacked = unpack_bits(packed, 64)
        np.testing.assert_array_equal(signs, unpacked)

    def test_pack_indices_round_trip_2bit(self):
        indices = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)
        packed = pack_indices(indices, 2)
        assert len(packed) < len(indices)  # should be compressed
        np.testing.assert_array_equal(unpack_indices(packed, 2, len(indices)), indices)

    def test_pack_indices_batch_preserves_rows(self):
        indices = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.uint8)
        packed = pack_indices(indices, 2)
        assert packed.shape == (2, 1)
        np.testing.assert_array_equal(unpack_indices(packed, 2, 3), indices)

    def test_pack_indices_round_trip_3bit_with_padding(self):
        indices = np.array([0, 7, 3, 5, 1], dtype=np.uint8)
        packed = pack_indices(indices, 3)
        np.testing.assert_array_equal(unpack_indices(packed, 3, 5), indices)

    def test_pack_indices_rejects_out_of_range_values(self):
        import pytest

        with pytest.raises(ValueError, match="do not fit"):
            pack_indices(np.array([0, 4]), 2)

    def test_pack_indices_invalid_bit_width(self):
        """Should raise ValueError for bit_width <= 0 or > 8."""
        import pytest

        from turboquant.utils import pack_indices

        with pytest.raises(ValueError, match="bit_width must be 1-8"):
            pack_indices(np.array([0, 1]), 0)
        with pytest.raises(ValueError, match="bit_width must be 1-8"):
            pack_indices(np.array([0, 1]), 9)

    def test_pack_indices_8bit(self):
        indices = np.array([0, 127, 255], dtype=np.uint8)
        packed = pack_indices(indices, 8)
        np.testing.assert_array_equal(packed, indices)


class TestMemoryFootprint:
    def test_compression_ratio_3bit(self):
        result = memory_footprint_bytes(1000, 128, 3)
        assert result["compression_ratio"] > 4.0
        assert result["total_bytes"] < result["original_fp16_bytes"]

    def test_compression_ratio_4bit(self):
        result = memory_footprint_bytes(1000, 128, 4)
        assert result["compression_ratio"] > 3.0

    def test_components_add_up(self):
        result = memory_footprint_bytes(100, 64, 3)
        expected_total = (
            result["mse_indices_bytes"]
            + result["qjl_signs_bytes"]
            + result["norms_bytes"]
        )
        assert result["total_bytes"] == expected_total
        assert result["norms_bytes"] == 100 * 8
