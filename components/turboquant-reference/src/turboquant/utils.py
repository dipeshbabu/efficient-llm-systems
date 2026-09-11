# Copyright 2026 Dipesh Tharu Mahato
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for bit packing and memory measurement."""

import numpy as np

from . import _validation as validate


def pack_bits(signs: np.ndarray) -> np.ndarray:
    """Pack {+1, -1} sign array into uint8 bitfield.

    8 signs per byte. +1 → 1, -1 → 0.

    Args:
        signs: int8 array of shape (d,) or (batch, d) with values {+1, -1}.

    Returns:
        uint8 array of shape (ceil(d/8),) or (batch, ceil(d/8)).
    """
    signs = validate.signs(signs, "signs")
    # NumPy packs boolean values directly and pads the final byte per row.
    return np.packbits(signs > 0, axis=-1)


def unpack_bits(packed: np.ndarray, d: int) -> np.ndarray:
    """Unpack uint8 bitfield back to {+1, -1} signs.

    Args:
        packed: uint8 array from pack_bits.
        d: Original dimension (to truncate padding).

    Returns:
        int8 array of shape (d,) or (batch, d) with values {+1, -1}.
    """
    d = validate.integer(d, "d")
    packed = validate.array(packed, "packed", kind="byte", ndim=(1, 2))
    if packed.shape[-1] * 8 < d:
        raise ValueError(f"packed input has {packed.shape[-1] * 8} bits, need {d}")
    bits = np.unpackbits(packed, axis=-1)[..., :d]
    return bits.astype(np.int8) * 2 - 1


def pack_indices(indices: np.ndarray, bit_width: int) -> np.ndarray:
    """Pack b-bit indices into compact byte array.

    Indices are packed independently along the final axis, so batch
    boundaries are preserved and padding never spills from one vector into
    the next.

    Args:
        indices: Integer indices, shape (d,) or (batch, d).
        bit_width: Bits per index.

    Returns:
        Packed byte array.
    """
    bit_width = validate.integer(bit_width, "bit_width", minimum=1, maximum=8)
    values = validate.array(indices, "indices", kind="integer", ndim=(1, 2))
    if values.size and (np.any(values < 0) or np.any(values >= (1 << bit_width))):
        raise ValueError(f"indices do not fit in {bit_width} bits")

    if not values.size:
        return np.empty(
            (*values.shape[:-1], (values.shape[-1] * bit_width + 7) // 8),
            dtype=np.uint8,
        )
    if bit_width == 8:
        # Each index already occupies one whole byte. Preserve owned output.
        return values.astype(np.uint8, copy=True)
    values = values.astype(np.uint8, copy=False)
    shifts = np.arange(bit_width - 1, -1, -1, dtype=np.uint8)
    bits = ((values[..., np.newaxis] >> shifts) & 1).reshape(
        *values.shape[:-1], values.shape[-1] * bit_width
    )
    return np.packbits(bits, axis=-1)


def unpack_indices(
    packed: np.ndarray,
    bit_width: int,
    n_indices: int,
) -> np.ndarray:
    """Unpack indices produced by :func:`pack_indices`.

    Packing is along the final axis. Padding bits in the final byte are
    discarded using ``n_indices``.
    """
    bit_width = validate.integer(bit_width, "bit_width", minimum=1, maximum=8)
    n_indices = validate.integer(n_indices, "n_indices")
    packed = validate.array(packed, "packed", kind="byte", ndim=(1, 2))
    needed_bits = n_indices * bit_width
    available_bits = packed.shape[-1] * 8
    if available_bits < needed_bits:
        raise ValueError(f"packed input has {available_bits} bits, need {needed_bits}")
    if n_indices == 0:
        return np.empty((*packed.shape[:-1], 0), dtype=np.uint8)
    if bit_width == 8:
        return packed[..., :n_indices].copy()
    bits = np.unpackbits(packed, axis=-1)[..., :needed_bits]
    groups = bits.reshape(*packed.shape[:-1], n_indices, bit_width)
    weights = (1 << np.arange(bit_width - 1, -1, -1)).astype(np.uint16)
    return np.asarray(
        np.sum(groups * weights, axis=-1, dtype=np.uint16), dtype=np.uint8
    )


def memory_footprint_bytes(n_vectors: int, d: int, bit_width: int) -> dict:
    """Calculate memory footprint of compressed KV cache.

    Counts and dimensions must be positive; an empty report has no ratio.

    Returns:
        Dict with breakdown: mse_indices, qjl_signs, norms, total, original_fp16.
    """
    n_vectors = validate.integer(n_vectors, "n_vectors", minimum=1)
    d = validate.dimension(d)
    bit_width = validate.integer(bit_width, "bit_width", minimum=2, maximum=9)
    mse_bits = bit_width - 1  # PolarQuant uses b-1 bits
    qjl_bits = 1

    # Each vector is packed separately, including its own final-byte padding.
    # Keep byte arithmetic integral so large counts do not lose precision.
    mse_bytes = n_vectors * ((d * mse_bits + 7) // 8)
    qjl_bytes = n_vectors * ((d * qjl_bits + 7) // 8)
    # Full TurboQuant stores the original vector norm and QJL residual norm.
    norm_bytes = n_vectors * 8  # two float32 values per vector
    total = mse_bytes + qjl_bytes + norm_bytes
    original = n_vectors * d * 2  # fp16

    return {
        "mse_indices_bytes": mse_bytes,
        "qjl_signs_bytes": qjl_bytes,
        "norms_bytes": norm_bytes,
        "total_bytes": total,
        "original_fp16_bytes": original,
        "compression_ratio": original / total,
    }
