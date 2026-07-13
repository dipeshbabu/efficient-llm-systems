"""Quick demo: compress and decompress a simulated KV cache with TurboQuant."""

import time
import numpy as np
from turboquant import TurboQuant, KVCacheCompressor


def demo_single_vector():
    """Compress a single attention head vector."""
    print("=" * 60)
    print("SINGLE VECTOR DEMO")
    print("=" * 60)

    d = 128  # typical head_dim
    rng = np.random.default_rng(42)

    for bit_width in [2, 3, 4]:
        tq = TurboQuant(d=d, bit_width=bit_width, seed=42)

        # Random unit vector (simulating a K/V vector)
        x = rng.standard_normal(d)
        x = x / np.linalg.norm(x)

        # Compress
        t0 = time.perf_counter()
        compressed = tq.quantize(x)
        t_quant = time.perf_counter() - t0

        # Decompress
        t0 = time.perf_counter()
        x_hat = tq.dequantize(compressed)
        t_dequant = time.perf_counter() - t0

        # Measure quality
        mse = np.mean((x - x_hat) ** 2)
        cosine = np.dot(x, x_hat) / (np.linalg.norm(x) * np.linalg.norm(x_hat))
        ratio = tq.compression_ratio()

        print(f"\n  {bit_width}-bit TurboQuant (d={d}):")
        print(f"    MSE:              {mse:.6f}")
        print(f"    Cosine similarity: {cosine:.6f}")
        print(f"    Compression ratio: {ratio:.1f}×")
        print(f"    Quantize time:     {t_quant*1000:.2f} ms")
        print(f"    Dequantize time:   {t_dequant*1000:.2f} ms")


def demo_kv_cache():
    """Compress a full KV cache tensor."""
    print("\n" + "=" * 60)
    print("KV CACHE DEMO")
    print("=" * 60)

    # Simulate a small transformer's KV cache
    num_layers = 4
    num_heads = 8
    seq_len = 512
    head_dim = 128

    rng = np.random.default_rng(42)
    k_cache = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))
    v_cache = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))

    for k_bits, v_bits in [(3, 3), (4, 3), (4, 4)]:
        compressor = KVCacheCompressor(head_dim=head_dim, k_bits=k_bits, v_bits=v_bits)

        # Compress
        t0 = time.perf_counter()
        compressed = compressor.compress(k_cache, v_cache)
        t_compress = time.perf_counter() - t0

        # Decompress
        t0 = time.perf_counter()
        k_hat, v_hat = compressor.decompress(compressed)
        t_decompress = time.perf_counter() - t0

        # Quality
        k_mse = np.mean((k_cache - k_hat) ** 2)
        v_mse = np.mean((v_cache - v_hat) ** 2)

        # Memory
        stats = compressor.memory_stats(seq_len, num_layers, num_heads)

        print(f"\n  K={k_bits}-bit, V={v_bits}-bit (layers={num_layers}, heads={num_heads}, seq={seq_len}, d={head_dim}):")
        print(f"    K cache MSE:       {k_mse:.6f}")
        print(f"    V cache MSE:       {v_mse:.6f}")
        print(f"    Original size:     {stats['original_mb']:.1f} MB")
        print(f"    Compressed size:   {stats['compressed_mb']:.1f} MB")
        print(f"    Compression ratio: {stats['compression_ratio']:.1f}×")
        print(f"    Compress time:     {t_compress:.2f}s")
        print(f"    Decompress time:   {t_decompress:.2f}s")


def demo_inner_product():
    """Show inner product preservation."""
    print("\n" + "=" * 60)
    print("INNER PRODUCT PRESERVATION DEMO")
    print("=" * 60)

    d = 256
    rng = np.random.default_rng(42)

    for bit_width in [2, 3, 4]:
        tq = TurboQuant(d=d, bit_width=bit_width, seed=42)

        errors = []
        for _ in range(1000):
            x = rng.standard_normal(d)
            y = rng.standard_normal(d)
            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)

            # Single-side quantization (paper's intended use for attention)
            x_hat = tq.dequantize(tq.quantize(x))
            ip_orig = np.dot(y, x)
            ip_approx = np.dot(y, x_hat)
            errors.append(abs(ip_orig - ip_approx))

        print(f"\n  {bit_width}-bit TurboQuant (d={d}, 1000 random pairs, single-side):")
        print(f"    Mean |IP error|:  {np.mean(errors):.6f}")
        print(f"    Max |IP error|:   {np.max(errors):.6f}")
        print(f"    Std |IP error|:   {np.std(errors):.6f}")


if __name__ == "__main__":
    demo_single_vector()
    demo_kv_cache()
    demo_inner_product()
    print("\n✅ All demos complete.")
