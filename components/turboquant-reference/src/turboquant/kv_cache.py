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

"""KV Cache integration layer for TurboQuant.

Compresses transformer KV cache tensors using TurboQuant (for K cache, inner product
preservation) and PolarQuant MSE-only (for V cache, MSE preservation).

KV cache shape: (num_layers, num_heads, seq_len, head_dim)
Quantization is along head_dim — each (head_dim,) vector is quantized independently.
"""

from dataclasses import dataclass, field

import numpy as np

from turboquant.turboquant import (
    PackedCompressedVector,
    PackedMSEVector,
    TurboQuant,
    TurboQuantMSE,
)


@dataclass
class CompressedKVCache:
    """Container for a compressed KV cache."""

    # Per-layer, per-head compressed K vectors
    k_compressed: list[list[PackedCompressedVector]] = field(default_factory=list)
    # Per-layer, per-head physically packed V vectors.
    v_compressed: list[list[PackedMSEVector]] = field(default_factory=list)

    num_layers: int = 0
    num_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0
    k_bit_width: int = 0
    v_bit_width: int = 0

    @property
    def packed_nbytes(self) -> int:
        """Actual bytes occupied by packed payload arrays and norms."""
        k_bytes = sum(item.nbytes for layer in self.k_compressed for item in layer)
        v_bytes = sum(item.nbytes for layer in self.v_compressed for item in layer)
        return k_bytes + v_bytes


class KVCacheCompressor:
    """Compress and decompress transformer KV cache tensors.

    Uses:
    - TurboQuant (Algorithm 2) for K cache — inner product preservation matters
      for attention score computation (Q @ K^T)
    - TurboQuantMSE (Algorithm 1) for V cache — MSE preservation matters
      for value reconstruction (attn_weights @ V)

    Usage:
        compressor = KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3)

        # Compress
        compressed = compressor.compress(k_cache, v_cache)

        # Decompress
        k_hat, v_hat = compressor.decompress(compressed)

        # Or compress streaming (one token at a time)
        compressor.compress_token(k_vec, v_vec, layer=0, head=0)
    """

    def __init__(
        self,
        head_dim: int,
        k_bits: int = 3,
        v_bits: int = 3,
        seed: int = 42,
        norm_correction: bool = True,
    ):
        """
        Args:
            head_dim: Dimension of each attention head vector.
            k_bits: Bit-width for K cache (TurboQuant, inner product).
            v_bits: Bit-width for V cache (PolarQuant MSE-only).
            seed: Random seed.
        """
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits

        # K cache uses full TurboQuant (inner product preservation)
        self.k_quantizer = TurboQuant(
            head_dim,
            bit_width=k_bits,
            seed=seed,
            norm_correction=norm_correction,
        )

        # V cache uses MSE-only PolarQuant (value reconstruction)
        self.v_quantizer = TurboQuantMSE(
            head_dim,
            bit_width=v_bits,
            seed=seed + 500,
            norm_correction=norm_correction,
        )

    def compress(self, k_cache: np.ndarray, v_cache: np.ndarray) -> CompressedKVCache:
        """Compress full KV cache tensors.

        Args:
            k_cache: Key cache, shape (num_layers, num_heads, seq_len, head_dim).
            v_cache: Value cache, same shape.

        Returns:
            CompressedKVCache with compressed K and V.
        """
        if k_cache.ndim != 4:
            raise ValueError(
                "k_cache must have shape (layers, heads, sequence, head_dim)"
            )
        if v_cache.shape != k_cache.shape:
            raise ValueError("v_cache must have the same shape as k_cache")
        num_layers, num_heads, seq_len, head_dim = k_cache.shape
        if head_dim != self.head_dim:
            raise ValueError(
                f"cache head_dim {head_dim} does not match compressor "
                f"head_dim {self.head_dim}"
            )

        result = CompressedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            k_bit_width=self.k_bits,
            v_bit_width=self.v_bits,
        )

        for layer in range(num_layers):
            k_layer = []
            v_layer = []
            for head in range(num_heads):
                # K: batch quantize all seq positions for this layer/head
                k_vecs = k_cache[layer, head]  # (seq_len, head_dim)
                k_compressed = self.k_quantizer.quantize_packed(k_vecs)
                k_layer.append(k_compressed)

                # V: MSE quantize and physically pack centroid indices.
                v_vecs = v_cache[layer, head]  # (seq_len, head_dim)
                v_layer.append(self.v_quantizer.quantize_packed(v_vecs))

            result.k_compressed.append(k_layer)
            result.v_compressed.append(v_layer)

        return result

    def decompress(
        self, compressed: CompressedKVCache
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompress back to full KV cache tensors.

        Returns:
            (k_cache, v_cache) both shape (num_layers, num_heads, seq_len, head_dim).
        """
        k_cache = np.zeros(
            (
                compressed.num_layers,
                compressed.num_heads,
                compressed.seq_len,
                compressed.head_dim,
            )
        )
        v_cache = np.zeros_like(k_cache)

        for layer in range(compressed.num_layers):
            for head in range(compressed.num_heads):
                k_cache[layer, head] = self.k_quantizer.dequantize_packed(
                    compressed.k_compressed[layer][head]
                )
                v_cache[layer, head] = self.v_quantizer.dequantize_packed(
                    compressed.v_compressed[layer][head]
                )

        return k_cache, v_cache

    def memory_stats(self, seq_len: int, num_layers: int, num_heads: int) -> dict:
        """Compute memory usage statistics.

        Returns dict with original_mb, compressed_mb, ratio.
        """
        if seq_len <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("seq_len, num_layers, and num_heads must be positive")
        n_vectors = num_layers * num_heads * seq_len
        # A KV cache contains both a key and a value fp16 vector.
        original_bytes = n_vectors * self.head_dim * 2 * 2

        # Packing is per vector, so include final-byte padding exactly.
        k_index_bytes = int(np.ceil(self.head_dim * (self.k_bits - 1) / 8))
        k_sign_bytes = int(np.ceil(self.head_dim / 8))
        k_bytes_per_vector = k_index_bytes + k_sign_bytes + 8
        v_index_bytes = int(np.ceil(self.head_dim * self.v_bits / 8))
        v_bytes_per_vector = v_index_bytes + 4
        compressed_bytes = n_vectors * (k_bytes_per_vector + v_bytes_per_vector)

        return {
            "original_mb": original_bytes / 1024 / 1024,
            "compressed_mb": compressed_bytes / 1024 / 1024,
            "compression_ratio": original_bytes / compressed_bytes,
            "k_bits_per_value": self.k_bits,
            "v_bits_per_value": self.v_bits,
            "k_metadata_bytes_per_vector": 8,
            "v_metadata_bytes_per_vector": 4,
            "storage_model": "bit_packed_payloads_with_float32_norms",
        }
