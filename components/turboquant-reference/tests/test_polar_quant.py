"""Tests for PolarQuant (Algorithm 1)."""

import numpy as np
import pytest

from turboquant.polar_quant import PolarQuant


class TestPolarQuantRoundTrip:
    """Quantize → dequantize should produce bounded MSE."""

    @pytest.mark.parametrize("bit_width", [1, 2, 3])
    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_round_trip_mse_within_bounds(self, bit_width, d):
        """MSE distortion should be within paper's bounds (Table 2)."""
        expected_mse = {1: 0.36, 2: 0.117, 3: 0.03}

        pq = PolarQuant(d=d, bit_width=bit_width, seed=42)
        rng = np.random.default_rng(99)

        n_samples = 500
        mse_total = 0.0
        for _ in range(n_samples):
            x = rng.standard_normal(d)
            x = x / np.linalg.norm(x)

            indices, norms = pq.quantize(x)
            x_hat = pq.dequantize(indices, norms)
            mse_total += np.mean((x - x_hat) ** 2)

        avg_mse = mse_total / n_samples
        assert avg_mse < expected_mse[bit_width] * 2.0, (
            f"MSE {avg_mse:.4f} exceeds 2× paper bound {expected_mse[bit_width]} "
            f"at d={d}, b={bit_width}"
        )

    def test_zero_vector(self):
        """Zero vector should reconstruct to zero (norm=0 → rescale to 0)."""
        pq = PolarQuant(d=128, bit_width=2, seed=42)
        x = np.zeros(128)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        np.testing.assert_allclose(x_hat, 0.0, atol=1e-15)

    def test_non_unit_norm_vectors(self):
        """CRITICAL: PolarQuant must work on non-unit-norm vectors (real KV cache).

        Roast review found this was the biggest bug — codebook is calibrated for
        unit norm, but real KV tensors have norms ~10-50.
        """
        d = 128
        pq = PolarQuant(d=d, bit_width=3, seed=42)
        rng = np.random.default_rng(42)

        for scale in [0.01, 1.0, 10.0, 50.0, 100.0]:
            x = rng.standard_normal(d) * scale

            indices, norms = pq.quantize(x)
            x_hat = pq.dequantize(indices, norms)

            # Relative MSE should be bounded regardless of scale
            # (same as unit-norm MSE since we extract/rescale norms)
            norm_sq_per_d = np.linalg.norm(x) ** 2 / d
            if norm_sq_per_d > 1e-10:
                relative_mse = np.mean((x - x_hat) ** 2) / norm_sq_per_d
                assert relative_mse < 0.5, (
                    f"Relative MSE {relative_mse:.4f} too high at scale={scale}"
                )

    def test_deterministic(self):
        """Same seed, same input → same output."""
        d = 128
        x = np.random.default_rng(1).standard_normal(d)

        pq1 = PolarQuant(d=d, bit_width=2, seed=42)
        pq2 = PolarQuant(d=d, bit_width=2, seed=42)

        idx1, n1 = pq1.quantize(x)
        idx2, n2 = pq2.quantize(x)
        np.testing.assert_array_equal(idx1, idx2)
        assert n1 == n2

    def test_batch_matches_single(self):
        """Batch quantization should match single-vector quantization."""
        d = 128
        pq = PolarQuant(d=d, bit_width=2, seed=42)
        rng = np.random.default_rng(7)

        X = rng.standard_normal((10, d))

        batch_indices, batch_norms = pq.quantize(X)
        batch_recon = pq.dequantize(batch_indices, batch_norms)

        for i in range(10):
            single_idx, single_norm = pq.quantize(X[i])
            single_recon = pq.dequantize(single_idx, single_norm)
            np.testing.assert_array_equal(batch_indices[i], single_idx)
            np.testing.assert_allclose(batch_recon[i], single_recon, atol=1e-12)

    def test_indices_in_range(self):
        """All indices should be in [0, 2^bit_width)."""
        d = 256
        pq = PolarQuant(d=d, bit_width=3, seed=42)
        rng = np.random.default_rng(5)
        x = rng.standard_normal(d)

        indices, _ = pq.quantize(x)
        assert indices.min() >= 0
        assert indices.max() < (1 << 3)

    def test_norm_correction_preserves_unit_norm_better(self):
        """Norm correction should bring reconstructed unit vectors closer to norm 1."""
        d = 128
        rng = np.random.default_rng(11)
        x = rng.standard_normal(d)
        x = x / np.linalg.norm(x)

        pq_uncorrected = PolarQuant(d=d, bit_width=3, seed=42, norm_correction=False)
        pq_corrected = PolarQuant(d=d, bit_width=3, seed=42, norm_correction=True)

        idx_u, norm_u = pq_uncorrected.quantize(x)
        idx_c, norm_c = pq_corrected.quantize(x)

        x_hat_u = pq_uncorrected.dequantize(idx_u, norm_u)
        x_hat_c = pq_corrected.dequantize(idx_c, norm_c)

        err_u = abs(np.linalg.norm(x_hat_u) - 1.0)
        err_c = abs(np.linalg.norm(x_hat_c) - 1.0)
        assert err_c < err_u


class TestPolarQuantResidual:
    """Test quantize_and_residual method."""

    def test_residual_identity(self):
        """residual = x - dequantize(quantize(x))."""
        d = 128
        pq = PolarQuant(d=d, bit_width=2, seed=42)
        rng = np.random.default_rng(3)
        x = rng.standard_normal(d)

        indices, norms, residual = pq.quantize_and_residual(x)
        x_hat = pq.dequantize(indices, norms)

        np.testing.assert_allclose(residual, x - x_hat, atol=1e-12)
