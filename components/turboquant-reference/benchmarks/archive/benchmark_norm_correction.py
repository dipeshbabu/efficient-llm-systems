#!/usr/bin/env python3
"""Norm correction experiment: does re-normalizing after dequant fix TQ on Qwen2.5?

Hypothesis: The Python TQ prototype lacks the norm correction that the production
C/Metal implementation has (grp_norm / recon_norm). This causes catastrophic PPL
on Qwen2.5 at 3-bit. IsoQuant accidentally avoids the issue because block-local
rotation produces smaller norm distortion.

The fix: after dequantizing rotated coordinates y_hat, re-normalize to unit norm
before applying the inverse rotation. Since the original rotated vector y has
||y|| = 1 (orthogonal rotation of a unit vector), this restores the correct magnitude.

This script tests the shared PolarQuant / TurboQuant implementation with
norm_correction enabled vs disabled on the same model, same data. Compares
against IQ Full as reference.

Usage:
    python components/turboquant-reference/benchmarks/archive/benchmark_norm_correction.py
    PPL_MODEL='TinyLlama/TinyLlama-1.1B-Chat-v1.0' python components/turboquant-reference/benchmarks/archive/benchmark_norm_correction.py
"""

import math
import os
import time

import torch
from torch import nn

from turboquant.turboquant import TurboQuantMSE

try:
    from turboquant.rotorquant_numpy import IsoQuantMSENp
except ImportError:
    IsoQuantMSENp = None
from transformers import AutoModelForCausalLM, AutoTokenizer

WIKITEXT_LOCAL = os.environ.get(
    "WIKITEXT_PATH", "/Users/dipesh/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw"
)


# ---------------------------------------------------------------------------
# Quantize-dequantize helper (same as main benchmark)
# ---------------------------------------------------------------------------


def quant_dequant_keys(key_states: torch.Tensor, quantizer_factory) -> torch.Tensor:
    B, H, S, D = key_states.shape
    device = key_states.device
    dtype = key_states.dtype

    out = torch.empty_like(key_states)
    for b in range(B):
        for h in range(H):
            k_np = key_states[b, h].float().cpu().numpy()
            q = quantizer_factory(D, seed=h)
            indices, norms = q.quantize(k_np)
            k_hat = q.dequantize(indices, norms)
            out[b, h] = torch.from_numpy(k_hat).to(dtype=dtype, device=device)
    return out


def patch_model_attention(model, quant_factory):
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
                k_reshaped = k_reshaped.permute(0, 2, 1, 3)
                k_quant = quant_dequant_keys(k_reshaped, self.qf)
                k_quant = k_quant.permute(0, 2, 1, 3).reshape(B, S, -1)
                return k_quant

        attn.k_proj = QuantizedKProj(
            original_k_proj, quant_factory, head_dim, num_kv_heads
        )
        hooks.append((attn, original_k_proj))

    return hooks


def unpatch_model(hooks):
    for attn, orig in hooks:
        attn.k_proj = orig


@torch.no_grad()
def evaluate_ppl(model, tokenizer, encodings, device, max_length=1024, stride=512):
    seq_len = encodings.size(1)
    nlls = []
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        input_ids = encodings[:, begin:end].to(device)

        target_len = end - prev_end if begin > 0 else end - begin
        target_ids = input_ids.clone()
        if begin > 0:
            target_ids[:, :-target_len] = -100
        else:
            target_ids[:, 0] = -100
            target_len -= 1

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * target_len
        nlls.append(neg_log_likelihood.item())
        n_tokens += target_len

        prev_end = end
        if end == seq_len:
            break

    return math.exp(sum(nlls) / n_tokens), n_tokens


@torch.no_grad()
def measure_k_mse(
    model, tokenizer, encodings, device, quant_factory, max_chunks=8, chunk_size=256
):
    total_mse = 0.0
    total_count = 0
    seq_len = encodings.size(1)

    for i in range(min(max_chunks, seq_len // chunk_size)):
        begin = i * chunk_size
        end = begin + chunk_size
        input_ids = encodings[:, begin:end].to(device)

        outputs = model(input_ids, output_attentions=False, use_cache=True)
        past_kv = outputs.past_key_values

        for _layer_idx, kv in enumerate(past_kv):
            k = kv[0]
            k_quant = quant_dequant_keys(k, quant_factory)
            mse = ((k.float().cpu() - k_quant.float().cpu()) ** 2).mean().item()
            total_mse += mse
            total_count += 1

    return total_mse / total_count if total_count > 0 else float("inf")


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_tq_factory(bits):
    def factory(d, seed=42):
        return TurboQuantMSE(d=d, bit_width=bits, seed=seed, norm_correction=False)

    return factory


def make_tqnc_factory(bits):
    def factory(d, seed=42):
        return TurboQuantMSE(d=d, bit_width=bits, seed=seed, norm_correction=True)

    return factory


def make_iq_factory(bits):
    def factory(d, seed=42):
        return IsoQuantMSENp(d=d, bit_width=bits, seed=seed, mode="full")

    return factory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if IsoQuantMSENp is None:
        raise SystemExit(
            "This historical comparison requires the optional RotorQuant/"
            "IsoQuant NumPy adapter, which is not distributed in this repo. "
            "Install or vendor a compatible adapter from "
            "https://github.com/scrya-com/rotorquant before running it."
        )
    model_name = os.environ.get("PPL_MODEL", "Qwen/Qwen2.5-3B")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print("Experiment: Norm correction effect on TQ")

    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    wikitext_path = WIKITEXT_LOCAL
    if not os.path.exists(wikitext_path):
        # Try M2 path
        wikitext_path = os.path.expanduser(
            "~/dev/turbo_test/wikitext-2-raw/wiki.test.raw"
        )
    print(f"Loading wikitext-2 from {wikitext_path}...")
    with open(wikitext_path) as f:
        text = f.read()
    encodings = tokenizer(text, return_tensors="pt")["input_ids"]
    print(f"  {encodings.size(1)} tokens")

    max_tokens = 4096
    if encodings.size(1) > max_tokens:
        encodings = encodings[:, :max_tokens]
        print(f"  Trimmed to {max_tokens} tokens")

    # Focused config: only TQ, TQ+NC, and IQ Full as reference
    configs = [
        ("TQ 3-bit", make_tq_factory(3)),
        ("TQ+NC 3-bit", make_tqnc_factory(3)),
        ("TQ 4-bit", make_tq_factory(4)),
        ("TQ+NC 4-bit", make_tqnc_factory(4)),
        ("IQ Full 3-bit", make_iq_factory(3)),
        ("IQ Full 4-bit", make_iq_factory(4)),
    ]

    # Phase 1: MSE
    print("\n" + "=" * 70)
    print("PHASE 1: K-cache MSE")
    print("=" * 70)

    mse_results = {}
    for name, factory in configs:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        mse = measure_k_mse(model, tokenizer, encodings, device, factory)
        elapsed = time.time() - t0
        mse_results[name] = mse
        print(f"MSE={mse:.6f}  ({elapsed:.1f}s)")

    # Phase 2: PPL
    print("\n" + "=" * 70)
    print("PHASE 2: Perplexity")
    print("=" * 70)

    ppl_results = {}

    print("  fp16 baseline...", end=" ", flush=True)
    t0 = time.time()
    ppl, n_tok = evaluate_ppl(model, tokenizer, encodings, device)
    elapsed = time.time() - t0
    ppl_results["fp16"] = ppl
    print(f"PPL={ppl:.2f}  ({n_tok} tokens, {elapsed:.1f}s)")

    for name, factory in configs:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        hooks = patch_model_attention(model, factory)
        try:
            ppl, n_tok = evaluate_ppl(model, tokenizer, encodings, device)
        finally:
            unpatch_model(hooks)
        elapsed = time.time() - t0
        ppl_results[name] = ppl
        print(f"PPL={ppl:.2f}  ({n_tok} tokens, {elapsed:.1f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Norm Correction Experiment")
    print("=" * 70)

    baseline = ppl_results.get("fp16", 0)
    all_names = ["fp16"] + [n for n, _ in configs]

    print(f"\n  {'Method':<16s}  {'PPL':>8s}  {'K MSE':>12s}  {'vs fp16':>8s}")
    print(f"  {'─' * 16}  {'─' * 8}  {'─' * 12}  {'─' * 8}")

    for name in all_names:
        ppl_val = ppl_results.get(name, float("nan"))
        mse_val = mse_results.get(name, 0.0)
        delta = ppl_val - baseline if name != "fp16" else 0.0
        mse_str = f"{mse_val:.6f}" if mse_val > 0 else "—"
        delta_str = f"+{delta:.2f}" if name != "fp16" else "—"
        print(f"  {name:<16s}  {ppl_val:>8.2f}  {mse_str:>12s}  {delta_str:>8s}")

    # Before/after comparison
    print("\n  BEFORE vs AFTER norm correction:")
    for bits in [3, 4]:
        old = ppl_results.get(f"TQ {bits}-bit", float("nan"))
        new = ppl_results.get(f"TQ+NC {bits}-bit", float("nan"))
        iq = ppl_results.get(f"IQ Full {bits}-bit", float("nan"))
        print(f"    {bits}-bit: TQ {old:.2f} → TQ+NC {new:.2f}  (IQ Full: {iq:.2f})")


if __name__ == "__main__":
    main()
