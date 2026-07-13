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

"""TurboQuant: Full algorithm combining PolarQuant + QJL.

Algorithm 2 from the paper — Inner Product TurboQuant.

Two-stage process:
1. PolarQuant at (b-1) bits for MSE-optimal compression
2. QJL at 1 bit on the residual for bias elimination

Total: b bits per coordinate with near-optimal inner product distortion.
"""

from dataclasses import dataclass

import numpy as np

from turboquant.polar_quant import PolarQuant
from turboquant.qjl import QJL
from turboquant.utils import pack_bits, pack_indices, unpack_bits, unpack_indices


@dataclass
class CompressedVector:
    """Container for a TurboQuant-compressed vector."""

    mse_indices: (
        np.ndarray
    )  # (d,) or (batch, d) — PolarQuant indices, (b-1)-bit integers
    vector_norms: np.ndarray  # scalar or (batch,) — original ||x||_2 for rescaling
    qjl_signs: np.ndarray  # (d,) or (batch, d) — QJL sign bits, int8 {+1, -1}
    residual_norms: np.ndarray  # scalar or (batch,) — ||residual||_2
    bit_width: int  # total bits per coordinate


@dataclass
class PackedCompressedVector:
    """Physically packed TurboQuant vector batch.

    Rotation matrices and codebooks are shared quantizer state and are not
    duplicated per vector. Norms use float32, matching the storage accounting.
    """

    mse_indices: np.ndarray
    vector_norms: np.ndarray
    qjl_signs: np.ndarray
    residual_norms: np.ndarray
    original_shape: tuple[int, ...]
    bit_width: int

    @property
    def nbytes(self) -> int:
        return sum(
            int(a.nbytes)
            for a in (
                self.mse_indices,
                self.vector_norms,
                self.qjl_signs,
                self.residual_norms,
            )
        )


@dataclass
class PackedMSEVector:
    """Physically packed PolarQuant-only vector batch."""

    indices: np.ndarray
    norms: np.ndarray
    original_shape: tuple[int, ...]
    bit_width: int

    @property
    def nbytes(self) -> int:
        return int(self.indices.nbytes + self.norms.nbytes)


class TurboQuant:
    """Full TurboQuant quantizer: PolarQuant (b-1 bits) + QJL (1 bit).

    Usage:
        tq = TurboQuant(d=128, bit_width=3, seed=42)
        compressed = tq.quantize(x)
        x_hat = tq.dequantize(compressed)

        # Verify inner product preservation
        ip_original = np.dot(x, y)
        ip_approx = np.dot(tq.dequantize(tq.quantize(x)),
                           tq.dequantize(tq.quantize(y)))
    """

    def __init__(
        self, d: int, bit_width: int, seed: int = 42, norm_correction: bool = True
    ):
        """
        Args:
            d: Vector dimension.
            bit_width: Total bits per coordinate (b). PolarQuant uses b-1, QJL uses 1.
            seed: Random seed for both rotation and projection matrices.
        """
        if bit_width < 2:
            raise ValueError(
                "TurboQuant requires bit_width >= 2 (1 bit PolarQuant + 1 bit QJL). "
                "For 1-bit, use QJL directly."
            )

        self.d = d
        self.bit_width = bit_width

        # Stage 1: PolarQuant at (b-1) bits
        self.polar_quant = PolarQuant(
            d,
            bit_width=bit_width - 1,
            seed=seed,
            norm_correction=norm_correction,
        )

        # Stage 2: QJL for residual (uses different seed)
        self.qjl = QJL(d, seed=seed + 1000)

    def quantize(self, x: np.ndarray) -> CompressedVector:
        """Quantize a vector or batch.

        Args:
            x: Input vector(s), shape (d,) or (batch, d).

        Returns:
            CompressedVector containing indices, signs, and norms.
        """
        # Stage 1: PolarQuant (with norm extraction)
        mse_indices, vector_norms, residual = self.polar_quant.quantize_and_residual(x)

        # Stage 2: QJL on residual
        qjl_signs, residual_norms = self.qjl.quantize(residual)

        return CompressedVector(
            mse_indices=mse_indices,
            vector_norms=vector_norms,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            bit_width=self.bit_width,
        )

    def dequantize(
        self, compressed: CompressedVector, shrinkage: float = 1.0
    ) -> np.ndarray:
        """Dequantize back to approximate vector.

        Args:
            compressed: CompressedVector from quantize().
            shrinkage: Multiplicative factor applied to the QJL stage.
                Default ``1.0`` is the classical unbiased estimator
                (paper-faithful, backward-compatible). MMSE-optimal is
                ``2/np.pi ≈ 0.6366`` — see ``QJL.dequantize`` for the
                derivation.

        Returns:
            Reconstructed vector(s), same shape as original.
        """
        # Stage 1: PolarQuant reconstruction (with norm rescaling)
        x_mse = self.polar_quant.dequantize(
            compressed.mse_indices, compressed.vector_norms
        )

        # Stage 2: QJL residual reconstruction
        x_qjl = self.qjl.dequantize(
            compressed.qjl_signs, compressed.residual_norms, shrinkage=shrinkage
        )

        return x_mse + x_qjl

    def quantize_packed(self, x: np.ndarray) -> PackedCompressedVector:
        """Quantize ``x`` and bit-pack indices and QJL signs."""
        x = np.asarray(x)
        compressed = self.quantize(x)
        return PackedCompressedVector(
            mse_indices=pack_indices(compressed.mse_indices, self.bit_width - 1),
            vector_norms=np.asarray(compressed.vector_norms, dtype=np.float32),
            qjl_signs=pack_bits(compressed.qjl_signs),
            residual_norms=np.asarray(compressed.residual_norms, dtype=np.float32),
            original_shape=tuple(x.shape),
            bit_width=self.bit_width,
        )

    def dequantize_packed(
        self,
        compressed: PackedCompressedVector,
        shrinkage: float = 1.0,
    ) -> np.ndarray:
        """Unpack and reconstruct a :class:`PackedCompressedVector`."""
        if compressed.bit_width != self.bit_width:
            raise ValueError(
                f"packed bit width {compressed.bit_width} does not match "
                f"quantizer bit width {self.bit_width}"
            )
        unpacked = CompressedVector(
            mse_indices=unpack_indices(
                compressed.mse_indices, self.bit_width - 1, self.d
            ),
            vector_norms=compressed.vector_norms,
            qjl_signs=unpack_bits(compressed.qjl_signs, self.d),
            residual_norms=compressed.residual_norms,
            bit_width=self.bit_width,
        )
        return self.dequantize(unpacked, shrinkage=shrinkage).reshape(
            compressed.original_shape
        )

    def compressed_size_bits(self, n_vectors: int) -> int:
        """Compute total storage in bits for n_vectors compressed vectors.

        Includes:
        - PolarQuant indices: (b-1) bits per coordinate per vector
        - QJL signs: 1 bit per coordinate per vector
        - Original vector norms: 32 bits (float32) per vector
        - Residual norms: 32 bits (float32) per vector
        """
        per_vector = self.d * self.bit_width  # (b-1) + 1 bits per coordinate
        norms = 64  # original vector norm + QJL residual norm, both float32
        return n_vectors * (per_vector + norms)

    def compression_ratio(self, original_bits_per_value: int = 16) -> float:
        """Compute compression ratio vs original precision.

        Args:
            original_bits_per_value: Bits per value in the original cache (16 for fp16).

        Returns:
            Compression ratio (e.g., 4.0 means 4× smaller).
        """
        original_per_vector = self.d * original_bits_per_value
        compressed_per_vector = self.d * self.bit_width + 64
        return original_per_vector / compressed_per_vector


class TurboQuantMSE:
    """MSE-only TurboQuant (Algorithm 1) — no QJL stage.

    Use for V cache compression where MSE matters more than inner product.
    Simpler, slightly less storage overhead (no QJL signs needed).
    """

    def __init__(
        self, d: int, bit_width: int, seed: int = 42, norm_correction: bool = True
    ):
        self.d = d
        self.bit_width = bit_width
        self.polar_quant = PolarQuant(
            d,
            bit_width=bit_width,
            seed=seed,
            norm_correction=norm_correction,
        )

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (indices, norms)."""
        return self.polar_quant.quantize(x)

    def dequantize(self, indices: np.ndarray, norms: np.ndarray) -> np.ndarray:
        return self.polar_quant.dequantize(indices, norms)

    def quantize_packed(self, x: np.ndarray) -> PackedMSEVector:
        """Quantize ``x`` and pack its centroid indices."""
        x = np.asarray(x)
        indices, norms = self.quantize(x)
        return PackedMSEVector(
            indices=pack_indices(indices, self.bit_width),
            norms=np.asarray(norms, dtype=np.float32),
            original_shape=tuple(x.shape),
            bit_width=self.bit_width,
        )

    def dequantize_packed(self, compressed: PackedMSEVector) -> np.ndarray:
        """Unpack and reconstruct a :class:`PackedMSEVector`."""
        if compressed.bit_width != self.bit_width:
            raise ValueError(
                f"packed bit width {compressed.bit_width} does not match "
                f"quantizer bit width {self.bit_width}"
            )
        indices = unpack_indices(compressed.indices, self.bit_width, self.d)
        return self.dequantize(indices, compressed.norms).reshape(
            compressed.original_shape
        )
