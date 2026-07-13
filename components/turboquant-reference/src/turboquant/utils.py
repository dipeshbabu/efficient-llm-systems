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


def pack_bits(signs: np.ndarray) -> np.ndarray:
    """Pack {+1, -1} sign array into uint8 bitfield.

    8 signs per byte. +1 → 1, -1 → 0.

    Args:
        signs: int8 array of shape (d,) or (batch, d) with values {+1, -1}.

    Returns:
        uint8 array of shape (ceil(d/8),) or (batch, ceil(d/8)).
    """
    # Convert {+1, -1} → {1, 0}
    bits = (signs > 0).astype(np.uint8)

    if bits.ndim == 1:
        # Pad to multiple of 8
        padded_len = (len(bits) + 7) // 8 * 8
        padded = np.zeros(padded_len, dtype=np.uint8)
        padded[:len(bits)] = bits
        # Pack 8 bits into each byte
        packed = np.packbits(padded)
        return packed
    else:
        batch, d = bits.shape
        padded_len = (d + 7) // 8 * 8
        padded = np.zeros((batch, padded_len), dtype=np.uint8)
        padded[:, :d] = bits
        # packbits works on last axis
        packed = np.packbits(padded, axis=1)
        return packed


def unpack_bits(packed: np.ndarray, d: int) -> np.ndarray:
    """Unpack uint8 bitfield back to {+1, -1} signs.

    Args:
        packed: uint8 array from pack_bits.
        d: Original dimension (to truncate padding).

    Returns:
        int8 array of shape (d,) or (batch, d) with values {+1, -1}.
    """
    if packed.ndim == 1:
        bits = np.unpackbits(packed)[:d]
        # Convert {1, 0} → {+1, -1}
        return (bits.astype(np.int8) * 2 - 1)
    else:
        bits = np.unpackbits(packed, axis=1)[:, :d]
        return (bits.astype(np.int8) * 2 - 1)


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
    if bit_width <= 0 or bit_width > 8:
        raise ValueError(f"bit_width must be 1-8, got {bit_width}")

    values = np.asarray(indices)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("indices must have an integer dtype")
    if values.size and (np.any(values < 0) or np.any(values >= (1 << bit_width))):
        raise ValueError(f"indices do not fit in {bit_width} bits")

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
    if bit_width <= 0 or bit_width > 8:
        raise ValueError(f"bit_width must be 1-8, got {bit_width}")
    if n_indices < 0:
        raise ValueError("n_indices must be non-negative")

    packed = np.asarray(packed, dtype=np.uint8)
    needed_bits = n_indices * bit_width
    available_bits = packed.shape[-1] * 8 if packed.ndim else 0
    if available_bits < needed_bits:
        raise ValueError(
            f"packed input has {available_bits} bits, need {needed_bits}"
        )
    bits = np.unpackbits(packed, axis=-1)[..., :needed_bits]
    if n_indices == 0:
        return np.empty((*packed.shape[:-1], 0), dtype=np.uint8)
    groups = bits.reshape(*packed.shape[:-1], n_indices, bit_width)
    weights = (1 << np.arange(bit_width - 1, -1, -1)).astype(np.uint16)
    return np.sum(groups * weights, axis=-1, dtype=np.uint16).astype(np.uint8)


def memory_footprint_bytes(n_vectors: int, d: int, bit_width: int) -> dict:
    """Calculate memory footprint of compressed KV cache.

    Returns:
        Dict with breakdown: mse_indices, qjl_signs, norms, total, original_fp16.
    """
    mse_bits = bit_width - 1  # PolarQuant uses b-1 bits
    qjl_bits = 1

    mse_bytes = int(np.ceil(n_vectors * d * mse_bits / 8))
    qjl_bytes = int(np.ceil(n_vectors * d * qjl_bits / 8))
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
        "compression_ratio": original / total if total > 0 else float("inf"),
    }
