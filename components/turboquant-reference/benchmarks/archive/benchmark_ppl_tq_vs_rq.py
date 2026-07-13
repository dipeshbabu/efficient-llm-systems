#!/usr/bin/env python3
"""Head-to-head perplexity comparison: TurboQuant vs RotorQuant on real model K tensors.

Approach:
1. Load Qwen2.5-0.5B on MPS
2. Load wikitext-2-test
3. Run forward passes, monkey-patching the KV cache to quantize-dequantize K tensors
4. Measure PPL for: fp16 baseline, TQ 3-bit, RQ 3-bit, TQ 4-bit, RQ 4-bit

Usage: python components/turboquant-reference/benchmarks/archive/benchmark_ppl_tq_vs_rq.py
"""

import os
import time
import math
import numpy as np
import torch
from torch import nn
from functools import partial

from turboquant.turboquant import TurboQuantMSE
try:
    from turboquant.rotorquant_numpy import RotorQuantMSENp, IsoQuantMSENp
except ImportError:
    RotorQuantMSENp = None
    IsoQuantMSENp = None
from transformers import AutoModelForCausalLM, AutoTokenizer
# datasets import optional — we prefer local wikitext file
WIKITEXT_LOCAL = "/Users/dipesh/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw"


# ---------------------------------------------------------------------------
# Quantize-dequantize helper
# ---------------------------------------------------------------------------

def quant_dequant_keys(key_states: torch.Tensor, quantizer_factory) -> torch.Tensor:
    """Quantize-dequantize K tensor through a given quantizer.

    key_states: (batch, n_heads, seq_len, head_dim) float16/float32
    quantizer_factory: callable(head_dim, seed) -> quantizer with .quantize/.dequantize
    """
    B, H, S, D = key_states.shape
    device = key_states.device
    dtype = key_states.dtype

    # Process each head independently (quantizers are per-head-dim)
    out = torch.empty_like(key_states)
    for b in range(B):
        for h in range(H):
            # (seq_len, head_dim) -> numpy
            k_np = key_states[b, h].float().cpu().numpy()  # (S, D)

            q = quantizer_factory(D, seed=h)  # seed per head for diversity
            indices, norms = q.quantize(k_np)
            k_hat = q.dequantize(indices, norms)

            out[b, h] = torch.from_numpy(k_hat).to(dtype=dtype, device=device)
    return out


def make_tq_factory(bits):
    def factory(d, seed=42):
        return TurboQuantMSE(d=d, bit_width=bits, seed=seed)
    return factory


def make_rq_factory(bits):
    def factory(d, seed=42):
        return RotorQuantMSENp(d=d, bit_width=bits, seed=seed)
    return factory


def make_iq_factory(bits, mode='full'):
    def factory(d, seed=42):
        return IsoQuantMSENp(d=d, bit_width=bits, seed=seed, mode=mode)
    return factory


# ---------------------------------------------------------------------------
# Monkey-patch approach: wrap the model's attention to intercept K cache
# ---------------------------------------------------------------------------

def patch_model_attention(model, quant_factory):
    """Patch all attention layers to quantize K before caching.

    Works with Qwen2 architecture (Qwen2Attention / Qwen2SdpaAttention).
    """
    hooks = []
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads

    for layer in model.model.layers:
        attn = layer.self_attn
        original_k_proj = attn.k_proj

        class QuantizedKProj(nn.Module):
            def __init__(self, orig, qf, hd, nkv):
                super().__init__()
                self.orig = orig
                self.qf = qf
                self.hd = hd
                self.nkv = nkv

            def forward(self, x):
                k = self.orig(x)
                B, S, _ = k.shape
                k_reshaped = k.view(B, S, self.nkv, self.hd)
                k_reshaped = k_reshaped.permute(0, 2, 1, 3)  # (B, H, S, D)
                k_quant = quant_dequant_keys(k_reshaped, self.qf)
                k_quant = k_quant.permute(0, 2, 1, 3).reshape(B, S, -1)
                return k_quant

        attn.k_proj = QuantizedKProj(original_k_proj, quant_factory, head_dim, num_kv_heads)
        hooks.append((attn, original_k_proj))

    return hooks


def unpatch_model(hooks):
    for attn, orig in hooks:
        attn.k_proj = orig


# ---------------------------------------------------------------------------
# Perplexity evaluation (sliding window)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, tokenizer, encodings, device, max_length=1024, stride=512):
    """Sliding window perplexity on tokenized text."""
    seq_len = encodings.size(1)
    nlls = []
    n_tokens = 0

    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        input_ids = encodings[:, begin:end].to(device)

        target_len = end - prev_end if begin > 0 else end - begin
        target_ids = input_ids.clone()
        # Mask out tokens we've already scored
        if begin > 0:
            target_ids[:, :-target_len] = -100
        else:
            # First window: skip first token (no context)
            target_ids[:, 0] = -100
            target_len -= 1

        outputs = model(input_ids, labels=target_ids)
        # Loss is averaged over non-ignored tokens
        neg_log_likelihood = outputs.loss * target_len
        nlls.append(neg_log_likelihood.item())
        n_tokens += target_len

        prev_end = end
        if end == seq_len:
            break

    ppl = math.exp(sum(nlls) / n_tokens)
    return ppl, n_tokens


# ---------------------------------------------------------------------------
# Also measure MSE on real K tensors (simpler, more robust)
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_k_mse(model, tokenizer, encodings, device, quant_factory,
                  max_chunks=10, chunk_size=256):
    """Run forward passes and measure MSE of quantized K vs original K."""
    total_mse = 0.0
    total_count = 0
    seq_len = encodings.size(1)

    for i in range(min(max_chunks, seq_len // chunk_size)):
        begin = i * chunk_size
        end = begin + chunk_size
        input_ids = encodings[:, begin:end].to(device)

        outputs = model(input_ids, output_attentions=False, use_cache=True)
        past_kv = outputs.past_key_values

        for layer_idx, kv in enumerate(past_kv):
            k = kv[0]  # (batch, n_heads, seq_len, head_dim)
            k_quant = quant_dequant_keys(k, quant_factory)
            mse = ((k.float().cpu() - k_quant.float().cpu()) ** 2).mean().item()
            total_mse += mse
            total_count += 1

    return total_mse / total_count if total_count > 0 else float('inf')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if RotorQuantMSENp is None or IsoQuantMSENp is None:
        raise SystemExit(
            "This historical head-to-head benchmark requires the optional "
            "RotorQuant/IsoQuant NumPy adapter, which is not distributed in "
            "this repo. Install or vendor a compatible adapter from "
            "https://github.com/scrya-com/rotorquant before running it."
        )
    model_name = os.environ.get("PPL_MODEL", "Qwen/Qwen2.5-3B")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {model_name}")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Load wikitext-2
    print(f"Loading wikitext-2 from {WIKITEXT_LOCAL}...")
    with open(WIKITEXT_LOCAL, "r") as f:
        text = f.read()
    encodings = tokenizer(text, return_tensors="pt")["input_ids"]
    print(f"  {encodings.size(1)} tokens")

    # Trim to manageable size for PPL eval
    max_tokens = 4096  # keep it fast
    if encodings.size(1) > max_tokens:
        encodings = encodings[:, :max_tokens]
        print(f"  Trimmed to {max_tokens} tokens for speed")

    # ------------------------------------------------------------------
    # Phase 1: MSE on real K tensors (fast, always works)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: MSE on Real Model K Tensors")
    print("=" * 70)

    configs = [
        ("TQ 3-bit",     make_tq_factory(3)),
        ("RQ 3-bit",     make_rq_factory(3)),
        ("IQ-F 3-bit",   make_iq_factory(3, mode='fast')),
        ("IQ 3-bit",     make_iq_factory(3, mode='full')),
        ("TQ 4-bit",     make_tq_factory(4)),
        ("RQ 4-bit",     make_rq_factory(4)),
        ("IQ-F 4-bit",   make_iq_factory(4, mode='fast')),
        ("IQ 4-bit",     make_iq_factory(4, mode='full')),
    ]

    mse_results = {}
    for name, factory in configs:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        mse = measure_k_mse(model, tokenizer, encodings, device, factory,
                           max_chunks=8, chunk_size=256)
        elapsed = time.time() - t0
        mse_results[name] = mse
        print(f"MSE={mse:.6f}  ({elapsed:.1f}s)")

    print(f"\n  {'Method':<14s}  {'K-cache MSE':>12s}")
    print(f"  {'─'*14}  {'─'*12}")
    for name, mse in mse_results.items():
        print(f"  {name:<14s}  {mse:>12.6f}")

    # ------------------------------------------------------------------
    # Phase 2: Full perplexity with monkey-patched K projection
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2: Perplexity (monkey-patched K quantization)")
    print("=" * 70)

    ppl_results = {}

    # Baseline
    print("  fp16 baseline...", end=" ", flush=True)
    t0 = time.time()
    ppl, n_tok = evaluate_ppl(model, tokenizer, encodings, device,
                               max_length=1024, stride=512)
    elapsed = time.time() - t0
    ppl_results["fp16"] = ppl
    print(f"PPL={ppl:.2f}  ({n_tok} tokens, {elapsed:.1f}s)")

    # Quantized variants
    for name, factory in configs:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        hooks = patch_model_attention(model, factory)
        try:
            ppl, n_tok = evaluate_ppl(model, tokenizer, encodings, device,
                                       max_length=1024, stride=512)
        finally:
            unpatch_model(hooks)
        elapsed = time.time() - t0
        ppl_results[name] = ppl
        print(f"PPL={ppl:.2f}  ({n_tok} tokens, {elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'Method':<14s}  {'PPL':>8s}  {'K MSE':>12s}  {'PPL Δ':>8s}")
    print(f"  {'─'*14}  {'─'*8}  {'─'*12}  {'─'*8}")

    baseline_ppl = ppl_results.get("fp16", 0)
    for name in ["fp16", "TQ 3-bit", "RQ 3-bit", "IQ-F 3-bit", "IQ 3-bit",
                  "TQ 4-bit", "RQ 4-bit", "IQ-F 4-bit", "IQ 4-bit"]:
        ppl_val = ppl_results.get(name, float('nan'))
        mse_val = mse_results.get(name, 0.0)
        delta = ppl_val - baseline_ppl if name != "fp16" else 0.0
        mse_str = f"{mse_val:.6f}" if mse_val > 0 else "—"
        delta_str = f"+{delta:.2f}" if name != "fp16" else "—"
        print(f"  {name:<14s}  {ppl_val:>8.2f}  {mse_str:>12s}  {delta_str:>8s}")


if __name__ == "__main__":
    main()
