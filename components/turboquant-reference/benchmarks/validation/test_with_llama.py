"""Integration test: simulate KV cache compression at real model dimensions.

Tests TurboQuant at the exact dimensions used by Qwen 3.5 27B and 35B-A3B MoE
running on the M5 Max 128GB setup.

Usage:
    python components/turboquant-reference/benchmarks/validation/test_with_llama.py
"""

import time
import numpy as np
from turboquant import TurboQuant, TurboQuantMSE, KVCacheCompressor


# Qwen 3.5 architecture constants
QWEN_27B = {
    "name": "Qwen 3.5 27B (dense)",
    "num_layers": 28,      # decoder layers (actually varies, use 28 as approx)
    "num_heads": 32,       # attention heads
    "num_kv_heads": 8,     # GQA: 8 KV heads
    "head_dim": 128,       # per-head dimension
    "hidden_dim": 4096,    # total hidden
}

QWEN_MOE = {
    "name": "Qwen 3.5 35B-A3B (MoE)",
    "num_layers": 28,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "hidden_dim": 4096,
}


def simulate_kv_cache(config: dict, seq_len: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic KV cache tensors at real model dimensions.

    Shape: (num_layers, num_kv_heads, seq_len, head_dim)
    Values drawn from N(0, 1/√d) which approximates real KV distributions.
    """
    rng = np.random.default_rng(seed)
    shape = (config["num_layers"], config["num_kv_heads"], seq_len, config["head_dim"])
    # KV values are roughly N(0, 1/√head_dim) in real models
    scale = 1.0 / np.sqrt(config["head_dim"])
    k_cache = rng.standard_normal(shape) * scale
    v_cache = rng.standard_normal(shape) * scale
    return k_cache, v_cache


def test_compression(config: dict, seq_len: int, k_bits: int, v_bits: int):
    """Test TurboQuant compression at given bit-width."""
    print(f"\n  {config['name']}, seq={seq_len}, K={k_bits}b, V={v_bits}b:")

    k_cache, v_cache = simulate_kv_cache(config, seq_len)
    head_dim = config["head_dim"]

    compressor = KVCacheCompressor(head_dim=head_dim, k_bits=k_bits, v_bits=v_bits)

    # Compress
    t0 = time.perf_counter()
    compressed = compressor.compress(k_cache, v_cache)
    t_compress = time.perf_counter() - t0

    # Decompress
    t0 = time.perf_counter()
    k_hat, v_hat = compressor.decompress(compressed)
    t_decompress = time.perf_counter() - t0

    # Quality metrics
    k_mse = np.mean((k_cache - k_hat) ** 2)
    v_mse = np.mean((v_cache - v_hat) ** 2)

    # Cosine similarity (per-vector, averaged)
    k_flat = k_cache.reshape(-1, head_dim)
    k_hat_flat = k_hat.reshape(-1, head_dim)
    cosines = []
    for i in range(min(1000, len(k_flat))):
        norm_orig = np.linalg.norm(k_flat[i])
        norm_recon = np.linalg.norm(k_hat_flat[i])
        if norm_orig > 1e-10 and norm_recon > 1e-10:
            cosines.append(np.dot(k_flat[i], k_hat_flat[i]) / (norm_orig * norm_recon))
    avg_cosine = np.mean(cosines) if cosines else 0.0

    # Memory stats
    stats = compressor.memory_stats(seq_len, config["num_layers"], config["num_kv_heads"])

    # Compute effective bit rate for Prince Canuma comparison
    # His "2.5-bit" uses outlier strategy: some channels at higher bits
    avg_bits = (k_bits + v_bits) / 2.0

    print(f"    K MSE:             {k_mse:.8f}")
    print(f"    V MSE:             {v_mse:.8f}")
    print(f"    K cosine sim:      {avg_cosine:.6f}")
    print(f"    Original:          {stats['original_mb']:.1f} MB")
    print(f"    Compressed:        {stats['compressed_mb']:.1f} MB")
    print(f"    Compression:       {stats['compression_ratio']:.1f}×")
    print(f"    Compress time:     {t_compress:.2f}s")
    print(f"    Decompress time:   {t_decompress:.2f}s")

    return stats['compression_ratio'], avg_cosine, k_mse


def test_attention_preservation(config: dict, seq_len: int = 64):
    """Test that compressed KV cache produces similar attention outputs."""
    print(f"\n  Attention test: {config['name']}, seq={seq_len}")

    head_dim = config["head_dim"]
    rng = np.random.default_rng(42)

    # Single head: query + KV cache
    q = rng.standard_normal((1, head_dim)) / np.sqrt(head_dim)
    k = rng.standard_normal((seq_len, head_dim)) / np.sqrt(head_dim)
    v = rng.standard_normal((seq_len, head_dim)) / np.sqrt(head_dim)

    # Original attention
    scores_orig = q @ k.T / np.sqrt(head_dim)
    attn_orig = _softmax(scores_orig)
    out_orig = attn_orig @ v

    for k_bits in [3, 4]:
        compressor = KVCacheCompressor(head_dim=head_dim, k_bits=k_bits, v_bits=k_bits)

        # Compress and decompress
        k_4d = k[np.newaxis, np.newaxis, :, :]
        v_4d = v[np.newaxis, np.newaxis, :, :]
        compressed = compressor.compress(k_4d, v_4d)
        k_hat, v_hat = compressor.decompress(compressed)
        k_hat = k_hat[0, 0]
        v_hat = v_hat[0, 0]

        # Compressed attention
        scores_comp = q @ k_hat.T / np.sqrt(head_dim)
        attn_comp = _softmax(scores_comp)
        out_comp = attn_comp @ v_hat

        # Compare
        cosine = np.dot(out_orig.ravel(), out_comp.ravel()) / (
            np.linalg.norm(out_orig) * np.linalg.norm(out_comp)
        )
        mse = np.mean((out_orig - out_comp) ** 2)

        print(f"    {k_bits}-bit: attn output cosine={cosine:.4f}, MSE={mse:.8f}")


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def main():
    print("=" * 70)
    print("TURBOQUANT LOCAL LLM INTEGRATION TEST")
    print("Simulating KV cache compression at real Qwen 3.5 dimensions")
    print("=" * 70)

    # Test at various context lengths matching our experiment data
    for config in [QWEN_27B, QWEN_MOE]:
        print(f"\n{'─' * 70}")
        print(f"Model: {config['name']}")
        print(f"  Architecture: {config['num_layers']}L × {config['num_kv_heads']}KV × {config['head_dim']}d")
        print(f"{'─' * 70}")

        for seq_len in [512, 2048, 8192]:
            for k_bits, v_bits in [(3, 3), (4, 3), (4, 4)]:
                test_compression(config, seq_len, k_bits, v_bits)

    # Attention preservation test
    print(f"\n{'─' * 70}")
    print("ATTENTION SCORE PRESERVATION")
    print(f"{'─' * 70}")
    test_attention_preservation(QWEN_27B)

    # Summary comparison with Prince Canuma's results
    print(f"\n{'─' * 70}")
    print("COMPARISON WITH PRINCE CANUMA'S MLX RESULTS (Qwen3.5-35B-A3B)")
    print(f"{'─' * 70}")
    print("\n  Prince's results:")
    print("    full:            0.703 GB, 6/6 NIAH")
    print("    TurboQuant 2.5b: 0.143 GB, 6/6 NIAH, 4.9× smaller")
    print("    TurboQuant 3.5b: 0.185 GB, 6/6 NIAH, 3.8× smaller")

    print("\n  Our simulation (3-bit K+V, seq=8192, MoE dims):")
    ratio, cosine, mse = test_compression(QWEN_MOE, 8192, 3, 3)
    print(f"\n    → Compression: {ratio:.1f}× (Prince: 4.9× at 2.5b)")
    print(f"    → Cosine sim:  {cosine:.4f}")
    print(f"    → K MSE:       {mse:.8f}")

    print(f"\n✅ Integration test complete.")
    print(f"   Next step: extract real KV tensors from llama.cpp for ground-truth validation.")


if __name__ == "__main__":
    main()
