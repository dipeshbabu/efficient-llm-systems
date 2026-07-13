#!/usr/bin/env python3
"""Temporal Decay Prototype — validate 3→2 bit requantization quality.

Tests whether progressive requantization (turbo3 → effective 2-bit) preserves
enough quality for old KV cache tokens. This validates the approach BEFORE
implementing it in the Metal shader.

The idea: old tokens get requantized to fewer effective bits, saving memory
while keeping recent tokens at full precision. If cosine similarity stays
above 0.80 and the attention score error is bounded, temporal decay is viable.

Approach:
  1. Generate turbo3-quantized vectors (rotate → quantize → pack)
  2. Requantize to 2-bit (dequant → find nearest 2-bit centroid → repack)
  3. Measure quality: cosine sim, MSE, max error
  4. Compare against direct 2-bit quantization (the theoretical best)
  5. Test on real Qwen3 KV tensors if available

References:
  - TurboQuant paper (ICLR 2026): Algorithm 1 (PolarQuant)
  - Codex review: "just remap centroid indices is too optimistic unless
    codebooks are explicitly nested"
  - buun's KVTuner pointer: retrieval heads sensitive, streaming heads robust
"""

import numpy as np
from turboquant.rotation import (
    random_rotation_fast,
    apply_fast_rotation,
    apply_fast_rotation_transpose,
)

# 3-bit centroids (from the paper, d=128)
CENTROIDS_3BIT = np.array([
    -0.190685, -0.117832, -0.065717, -0.021460,
     0.021460,  0.065717,  0.117832,  0.190685
])

# 2-bit centroids (from the paper, d=128)
CENTROIDS_2BIT = np.array([
    -0.133462, -0.039994, 0.039994, 0.133462
])

# Midpoints for nearest-centroid lookup
MIDPOINTS_3BIT = np.array([
    -0.154259, -0.091775, -0.043589, 0.0, 0.043589, 0.091775, 0.154259
])

MIDPOINTS_2BIT = np.array([
    -0.086728, 0.0, 0.086728
])


def quantize_3bit(x_normalized: np.ndarray) -> np.ndarray:
    """Quantize normalized values to 3-bit centroid indices (0-7)."""
    indices = np.digitize(x_normalized, MIDPOINTS_3BIT)
    return indices.astype(np.uint8)


def quantize_2bit(x_normalized: np.ndarray) -> np.ndarray:
    """Quantize normalized values to 2-bit centroid indices (0-3)."""
    indices = np.digitize(x_normalized, MIDPOINTS_2BIT)
    return indices.astype(np.uint8)


def dequantize_3bit(indices: np.ndarray, norm: float) -> np.ndarray:
    """Dequantize 3-bit indices back to float values."""
    centroids = CENTROIDS_3BIT[indices]
    # Norm correction: reconstruct with corrected norm
    recon_norm = np.sqrt(np.sum(centroids ** 2))
    if recon_norm > 1e-10:
        corrected_norm = norm / recon_norm
    else:
        corrected_norm = norm
    return centroids * corrected_norm


def dequantize_2bit(indices: np.ndarray, norm: float) -> np.ndarray:
    """Dequantize 2-bit indices back to float values."""
    centroids = CENTROIDS_2BIT[indices]
    recon_norm = np.sqrt(np.sum(centroids ** 2))
    if recon_norm > 1e-10:
        corrected_norm = norm / recon_norm
    else:
        corrected_norm = norm
    return centroids * corrected_norm


def requantize_3to2(indices_3bit: np.ndarray, norm_3bit: float) -> tuple[np.ndarray, float]:
    """Requantize 3-bit block to effective 2-bit.

    This is the core of temporal decay:
    1. Dequant from 3-bit (get float values)
    2. Requantize to nearest 2-bit centroid
    3. Recompute norm correction

    Returns (2-bit indices, corrected norm for 2-bit).
    """
    # Step 1: dequant 3-bit
    values = dequantize_3bit(indices_3bit, norm_3bit)

    # Step 2: normalize for 2-bit requant
    recon_norm = np.linalg.norm(values)
    if recon_norm > 1e-10:
        normalized = values / recon_norm
    else:
        normalized = values
        recon_norm = 0.0

    # Step 3: quantize to 2-bit
    indices_2bit = quantize_2bit(normalized)

    # Step 4: norm correction for 2-bit
    centroids_2bit = CENTROIDS_2BIT[indices_2bit]
    centroid_norm = np.sqrt(np.sum(centroids_2bit ** 2))
    if centroid_norm > 1e-10:
        corrected_norm = recon_norm / centroid_norm
    else:
        corrected_norm = recon_norm

    return indices_2bit, corrected_norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def run_synthetic_test(d: int = 128, n_vectors: int = 1000, seed: int = 42):
    """Test temporal decay on synthetic Gaussian vectors."""
    print(f"\n{'='*60}")
    print(f"SYNTHETIC TEST: d={d}, n_vectors={n_vectors}")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(seed)
    signs1, signs2, padded_d = random_rotation_fast(d, rng)

    cos_sims_3bit = []
    cos_sims_2bit_direct = []
    cos_sims_decay = []  # 3bit → requant to 2bit
    mse_3bit = []
    mse_2bit = []
    mse_decay = []

    for i in range(n_vectors):
        # Generate random vector
        x = rng.standard_normal(d).astype(np.float32)
        norm = np.linalg.norm(x)
        if norm < 1e-10:
            continue
        x_normalized = x / norm

        # Rotate
        x_rotated = apply_fast_rotation(x_normalized, signs1, signs2, padded_d)

        # --- Path A: 3-bit quantization (current turbo3) ---
        indices_3bit = quantize_3bit(x_rotated)
        recon_3bit_rotated = dequantize_3bit(indices_3bit, norm)
        recon_3bit = apply_fast_rotation_transpose(recon_3bit_rotated / norm, signs1, signs2, padded_d) * norm  # un-rotate

        # --- Path B: Direct 2-bit quantization (theoretical best for 2-bit) ---
        indices_2bit_direct = quantize_2bit(x_rotated)
        recon_2bit_rotated = dequantize_2bit(indices_2bit_direct, norm)
        recon_2bit = apply_fast_rotation_transpose(recon_2bit_rotated / norm, signs1, signs2, padded_d) * norm

        # --- Path C: Temporal decay (3-bit → requant to 2-bit) ---
        indices_decay, norm_decay = requantize_3to2(indices_3bit, norm)
        recon_decay_rotated = dequantize_2bit(indices_decay, norm_decay)
        recon_decay = apply_fast_rotation_transpose(recon_decay_rotated / norm_decay, signs1, signs2, padded_d) * norm_decay

        # Measure quality
        cos_sims_3bit.append(cosine_similarity(x, recon_3bit))
        cos_sims_2bit_direct.append(cosine_similarity(x, recon_2bit))
        cos_sims_decay.append(cosine_similarity(x, recon_decay))

        mse_3bit.append(np.mean((x - recon_3bit) ** 2))
        mse_2bit.append(np.mean((x - recon_2bit) ** 2))
        mse_decay.append(np.mean((x - recon_decay) ** 2))

    # Report
    print(f"{'Method':<25} {'Cosine Sim':>12} {'MSE':>12} {'vs 3-bit':>10}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    cs3 = np.mean(cos_sims_3bit)
    cs2 = np.mean(cos_sims_2bit_direct)
    csd = np.mean(cos_sims_decay)
    m3 = np.mean(mse_3bit)
    m2 = np.mean(mse_2bit)
    md = np.mean(mse_decay)

    print(f"{'turbo3 (3-bit)':<25} {cs3:>12.6f} {m3:>12.6f} {'baseline':>10}")
    print(f"{'Direct 2-bit':<25} {cs2:>12.6f} {m2:>12.6f} {m2/m3:>10.2f}x")
    print(f"{'Decay 3→2 (requant)':<25} {csd:>12.6f} {md:>12.6f} {md/m3:>10.2f}x")
    print()

    # Quality assessment
    print("Quality Assessment:")
    if csd > 0.80:
        print(f"  ✅ Decay cosine sim {csd:.4f} > 0.80 threshold — VIABLE")
    else:
        print(f"  ❌ Decay cosine sim {csd:.4f} < 0.80 threshold — TOO LOSSY")

    if csd > cs2 * 0.95:
        print(f"  ✅ Decay within 5% of direct 2-bit — requant doesn't add much error")
    else:
        gap = (1 - csd/cs2) * 100
        print(f"  ⚠️  Decay {gap:.1f}% worse than direct 2-bit — double-quant error")

    # Inner product preservation (critical for attention scores)
    print(f"\nInner Product Preservation (attention score proxy):")
    ip_errors_3bit = []
    ip_errors_decay = []
    for i in range(min(100, n_vectors)):
        x = rng.standard_normal(d).astype(np.float32)
        q = rng.standard_normal(d).astype(np.float32)  # query vector

        norm = np.linalg.norm(x)
        if norm < 1e-10:
            continue
        x_rot = apply_fast_rotation(x / norm, signs1, signs2, padded_d)

        # True inner product
        ip_true = np.dot(x, q)

        # 3-bit inner product
        idx3 = quantize_3bit(x_rot)
        r3 = dequantize_3bit(idx3, norm)
        r3_unrot = apply_fast_rotation_transpose(r3 / norm, signs1, signs2, padded_d) * norm
        ip_3bit = np.dot(r3_unrot, q)

        # Decay inner product
        idx_d, norm_d = requantize_3to2(idx3, norm)
        rd = dequantize_2bit(idx_d, norm_d)
        rd_unrot = apply_fast_rotation_transpose(rd / norm_d, signs1, signs2, padded_d) * norm_d
        ip_decay = np.dot(rd_unrot, q)

        ip_errors_3bit.append(abs(ip_3bit - ip_true) / (abs(ip_true) + 1e-10))
        ip_errors_decay.append(abs(ip_decay - ip_true) / (abs(ip_true) + 1e-10))

    print(f"  3-bit mean relative error: {np.mean(ip_errors_3bit):.4f}")
    print(f"  Decay mean relative error: {np.mean(ip_errors_decay):.4f}")
    print(f"  Decay/3-bit error ratio:   {np.mean(ip_errors_decay)/np.mean(ip_errors_3bit):.2f}x")

    return {
        "cosine_3bit": cs3, "cosine_2bit": cs2, "cosine_decay": csd,
        "mse_3bit": m3, "mse_2bit": m2, "mse_decay": md,
        "ip_error_3bit": np.mean(ip_errors_3bit),
        "ip_error_decay": np.mean(ip_errors_decay),
    }


def run_real_model_test():
    """Test temporal decay on real Qwen3 KV tensors."""
    print(f"\n{'='*60}")
    print(f"REAL MODEL TEST: Qwen3-1.7B KV tensors")
    print(f"{'='*60}\n")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("SKIPPED: torch/transformers not available")
        print("Install: pip install torch transformers accelerate")
        return None

    print("Loading model...")
    model_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.float32)

    text = "The quick brown fox jumps over the lazy dog. " * 50
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    rng_model = np.random.default_rng(42)
    signs1_m, signs2_m, padded_d_m = random_rotation_fast(128, rng_model)

    all_results = []
    # Extract KV — handle DynamicCache (new HF) and tuple (old HF)
    past_kv = outputs.past_key_values
    try:
        kv_keys = [past_kv.key_cache[i] for i in range(min(6, len(past_kv.key_cache)))]
    except (AttributeError, TypeError):
        kv_keys = [layer_kv[0] for layer_kv in list(past_kv)[:6]]

    for layer_idx, k_tensor in enumerate(kv_keys):
        k = k_tensor.squeeze(0)  # (heads, seq, dim)
        n_heads, seq_len, dim = k.shape

        if dim != 128:
            print(f"  Layer {layer_idx}: dim={dim}, skipping (need 128)")
            continue

        layer_cos_3bit = []
        layer_cos_decay = []

        for head in range(n_heads):
            for pos in range(0, seq_len, 10):  # sample every 10th position
                x = k[head, pos].numpy()
                norm = np.linalg.norm(x)
                if norm < 1e-10:
                    continue

                x_rot = apply_fast_rotation(x / norm, signs1_m, signs2_m, padded_d_m)

                # 3-bit
                idx3 = quantize_3bit(x_rot)
                r3 = dequantize_3bit(idx3, norm)
                r3_unrot = apply_fast_rotation_transpose(r3 / norm, signs1_m, signs2_m, padded_d_m) * norm

                # Decay 3→2
                idx_d, norm_d = requantize_3to2(idx3, norm)
                rd = dequantize_2bit(idx_d, norm_d)
                rd_unrot = apply_fast_rotation_transpose(rd / norm_d, signs1_m, signs2_m, padded_d_m) * norm_d

                layer_cos_3bit.append(cosine_similarity(x, r3_unrot))
                layer_cos_decay.append(cosine_similarity(x, rd_unrot))

        cs3 = np.mean(layer_cos_3bit)
        csd = np.mean(layer_cos_decay)
        print(f"  Layer {layer_idx}: 3-bit cos={cs3:.4f}, decay cos={csd:.4f}, "
              f"delta={csd-cs3:.4f}")
        all_results.append({"layer": layer_idx, "cos_3bit": cs3, "cos_decay": csd})

    if all_results:
        avg_3bit = np.mean([r["cos_3bit"] for r in all_results])
        avg_decay = np.mean([r["cos_decay"] for r in all_results])
        print(f"\n  Average: 3-bit={avg_3bit:.4f}, decay={avg_decay:.4f}")

        if avg_decay > 0.75:
            print(f"  ✅ Real model decay cosine {avg_decay:.4f} > 0.75 — VIABLE for old tokens")
        else:
            print(f"  ❌ Real model decay cosine {avg_decay:.4f} < 0.75 — needs investigation")

    return all_results


def run_memory_savings_estimate():
    """Estimate memory savings from temporal decay."""
    print(f"\n{'='*60}")
    print(f"MEMORY SAVINGS ESTIMATE")
    print(f"{'='*60}\n")

    # Qwen3.5-35B-A3B: 10 attention layers, 2 KV heads, d=512 (2 groups of 128)
    n_layers = 10
    n_kv_heads = 2
    d_head = 256  # 2 groups of 128

    turbo3_bytes_per_element = 3.5 / 8  # 3.5 bits = 0.4375 bytes
    turbo2_bytes_per_element = 2.0 / 8  # 2 bits = 0.25 bytes
    q8_bytes_per_element = 1.0           # 8 bits = 1 byte

    print(f"Model: Qwen3.5-35B-A3B (10 attn layers, {n_kv_heads} KV heads, d={d_head})")
    print()

    for context in [32768, 65536, 131072, 262144]:
        # Without decay: all turbo3
        kv_no_decay = context * n_layers * n_kv_heads * d_head * 2 * turbo3_bytes_per_element

        # With decay: recent 4K at turbo3, rest at turbo2
        recent = min(4096, context)
        old = context - recent

        # Layer-aware: first 8 layers (of 10 attn) decay, last 2 don't
        decay_layers = 8
        no_decay_layers = 2

        kv_decay = (
            # Recent tokens: all turbo3
            recent * n_layers * n_kv_heads * d_head * 2 * turbo3_bytes_per_element +
            # Old tokens in decay layers: turbo2
            old * decay_layers * n_kv_heads * d_head * 2 * turbo2_bytes_per_element +
            # Old tokens in non-decay layers: turbo3
            old * no_decay_layers * n_kv_heads * d_head * 2 * turbo3_bytes_per_element
        )

        savings_pct = (1 - kv_decay / kv_no_decay) * 100
        print(f"  {context//1024:>4}K context: "
              f"no decay={kv_no_decay/1024/1024:.1f} MB, "
              f"with decay={kv_decay/1024/1024:.1f} MB, "
              f"savings={savings_pct:.0f}%")

    print()
    print("  Note: savings increase with context because the fraction of 'old' tokens grows.")
    print("  At 262K context, ~98% of tokens are 'old' and get decayed in 8/10 layers.")


if __name__ == "__main__":
    print("TurboQuant Temporal Decay Prototype")
    print("=" * 60)

    # Synthetic test (always runs, no deps)
    synthetic = run_synthetic_test(d=128, n_vectors=1000)

    # Memory savings estimate
    run_memory_savings_estimate()

    # Real model test (optional, needs torch)
    real = run_real_model_test()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Synthetic cosine sim: 3-bit={synthetic['cosine_3bit']:.4f}, "
          f"decay={synthetic['cosine_decay']:.4f}")
    print(f"  Inner product error:  3-bit={synthetic['ip_error_3bit']:.4f}, "
          f"decay={synthetic['ip_error_decay']:.4f} "
          f"({synthetic['ip_error_decay']/synthetic['ip_error_3bit']:.1f}x)")
    print()

    if synthetic['cosine_decay'] > 0.80:
        print("  ✅ TEMPORAL DECAY IS VIABLE — proceed to Metal implementation")
    elif synthetic['cosine_decay'] > 0.70:
        print("  ⚠️  MARGINAL — may work for non-critical layers, test with PPL")
    else:
        print("  ❌ TOO LOSSY — need a different approach")
