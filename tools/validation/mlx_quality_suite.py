#!/usr/bin/env python3
"""MLX TurboQuant Quality Suite — NIAH, KLD, and Context-Size Scaling

Runs quality and performance tests for TurboKVCache using mlx-lm as the backend.
Does NOT modify any existing llama.cpp test scripts.

Tests:
  niah      Needle-In-A-Haystack retrieval at various depths and context sizes
  kld       KL Divergence between baseline and turbo logit distributions
  context   Decode speed and memory at different context lengths

Usage:
    python3 tools/validation/mlx_quality_suite.py --model mlx-community/Qwen3.5-2B-8bit
    python3 tools/validation/mlx_quality_suite.py --model mlx-community/Qwen3.5-2B-8bit --test niah
    python3 tools/validation/mlx_quality_suite.py --model mlx-community/Qwen3.5-2B-8bit --test kld
    python3 tools/validation/mlx_quality_suite.py --model mlx-community/Qwen3.5-2B-8bit --test context
    python3 tools/validation/mlx_quality_suite.py --model mlx-community/Qwen3.5-2B-8bit --bits 4 --asymmetric

Requirements: Python 3.10+, mlx, mlx-lm (no other pip deps).
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx.nn.layers.turbo_kv_cache import TurboKVCache
from mlx_lm.models.cache import make_prompt_cache

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42

NEEDLE_TEXT = "The special magic verification code is BLUE TIGER 42."
NEEDLE_QUERY = (
    "What is the special magic verification code mentioned in the text above? "
    "Reply with ONLY the code, nothing else."
)
NEEDLE_ANSWER = "BLUE TIGER 42"

# 12 filler paragraphs (~150 words each, ~750 chars).  Enough variety to fill
# up to ~8K tokens when repeated/shuffled without being obviously repetitive.
FILLER_PARAGRAPHS = [
    (
        "The observable universe spans roughly 93 billion light-years in diameter, "
        "a figure that continues to grow as space itself expands. Within this volume "
        "lie an estimated two trillion galaxies, each hosting hundreds of billions of "
        "stars. Our own Milky Way is a barred spiral galaxy approximately 100,000 "
        "light-years across. The Sun orbits the galactic center at about 230 km/s, "
        "completing one full revolution every 225 to 250 million years. Despite the "
        "staggering numbers the universe is overwhelmingly empty."
    ),
    (
        "Roman engineers perfected concrete construction over two thousand years ago. "
        "The Pantheon in Rome, completed around 125 AD, features an unreinforced "
        "concrete dome that remains the world's largest of its kind at 43.3 meters. "
        "The secret lay in their mixture: volcanic ash from Pozzuoli combined with "
        "lime and seawater created pozzolanic concrete. Modern researchers found that "
        "seawater strengthened the material over time by promoting interlocking mineral "
        "crystals within the matrix."
    ),
    (
        "The hadal zone, comprising ocean trenches deeper than 6,000 meters, represents "
        "one of the least explored environments on Earth. Despite crushing pressures "
        "exceeding 1,000 atmospheres, thriving ecosystems exist in these depths. "
        "Snailfish have been observed at nearly 8,200 meters in the Mariana Trench. "
        "Hydrothermal vents, first discovered in 1977, support dense communities of "
        "tube worms and shrimp that derive energy from chemosynthesis rather than "
        "photosynthesis."
    ),
    (
        "The invention of movable type by Gutenberg around 1440 triggered a revolution "
        "in knowledge dissemination. Before the press a single book could take months "
        "to copy by hand. Within fifty years an estimated twenty million volumes had "
        "been printed. Literacy rates climbed as printed material became affordable. "
        "Luther's Ninety-Five Theses spread across Germany in weeks thanks to the "
        "press, accelerating the Protestant Reformation."
    ),
    (
        "Plate tectonics explains the large-scale motions of Earth's lithosphere. "
        "Roughly fifteen major plates float on the semi-fluid asthenosphere below, "
        "moving at rates of two to ten centimeters per year. Where plates diverge, "
        "magma creates new ocean floor along the Mid-Atlantic Ridge. Where they "
        "converge, subduction generates deep trenches and volcanic arcs. Transform "
        "boundaries like the San Andreas Fault produce devastating earthquakes."
    ),
    (
        "Coffee cultivation began in the highlands of Ethiopia where Coffea arabica "
        "still grows wild. By the fifteenth century coffee was being roasted and brewed "
        "in Yemen, and coffeehouses had become centers of intellectual exchange. The "
        "Dutch smuggled plants to Java in the late 1600s breaking the Arab monopoly. "
        "Today Brazil produces roughly a third of the world's coffee. Climate change "
        "threatens arabica production as the plant requires specific conditions."
    ),
    (
        "Western music theory traces its roots to Pythagoras, who discovered that "
        "harmonious intervals correspond to simple numerical ratios of string lengths. "
        "The twelve-tone equal temperament system divides the octave into twelve equal "
        "semitones. Bach's Well-Tempered Clavier demonstrated the versatility of this "
        "system with preludes and fugues in all twenty-four major and minor keys. "
        "Modern electronic music explores microtonal scales and algorithmic composition."
    ),
    (
        "Glaciers cover approximately ten percent of Earth's land surface and hold "
        "about 69 percent of the world's fresh water. The Antarctic Ice Sheet alone "
        "contains enough ice to raise sea levels by roughly 58 meters. Glaciers move "
        "through internal deformation and basal sliding. The retreat of mountain "
        "glaciers worldwide serves as one of the most visible indicators of climate "
        "change, affecting water supplies for billions of people."
    ),
    (
        "The Silk Road connected China to the Mediterranean for over fifteen hundred "
        "years, facilitating trade in silk, spices, precious metals, and ideas. Along "
        "these routes Buddhism spread from India to China and Central Asia. Paper and "
        "gunpowder traveled westward while glass-making techniques moved east. The "
        "network encompassed maritime routes as well, with Indian Ocean trade linking "
        "East Africa, Arabia, India, and Southeast Asia."
    ),
    (
        "Honeybees exhibit remarkable collective intelligence through their waggle "
        "dance, a figure-eight movement that communicates the direction and distance "
        "of food sources relative to the sun. A single hive contains up to 60,000 "
        "workers, all daughters of the same queen. Bees maintain hive temperature at "
        "precisely 35 degrees Celsius by fanning their wings or clustering together. "
        "They produce about 25 kilograms of honey per year per hive."
    ),
    (
        "The human brain contains roughly 86 billion neurons connected by an estimated "
        "100 trillion synapses. Each neuron can fire up to 200 times per second, "
        "creating electrochemical signals that travel at speeds between 1 and 120 "
        "meters per second. The brain consumes about 20 percent of the body's energy "
        "despite representing only 2 percent of its mass. Neuroplasticity allows the "
        "brain to reorganize itself throughout life."
    ),
    (
        "The history of timekeeping stretches back to ancient sundials and water "
        "clocks. Mechanical clocks appeared in European monasteries in the 13th "
        "century, driven by falling weights and regulated by verge escapements. "
        "Pendulum clocks, invented by Huygens in 1656, achieved accuracy of about "
        "15 seconds per day. Modern atomic clocks based on cesium-133 are accurate "
        "to one second in 300 million years."
    ),
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _model_short_name(model_path: str) -> str:
    """Extract a short name from a model path/HF id for filenames."""
    return model_path.rstrip("/").split("/")[-1]


def _n_layers(model: nn.Module) -> int:
    """Return the number of transformer layers in a model."""
    return len(model.layers) if hasattr(model, "layers") else len(model.model.layers)


def _make_baseline_cache(model: nn.Module) -> list:
    """Create a baseline (FP16) KVCache list."""
    return make_prompt_cache(model)


def _make_turbo_cache(
    model: nn.Module, bits: int, asymmetric: bool, min_compress_tokens: int = 256
) -> list:
    """Create a cache list with TurboKVCache replacing standard KVCache layers.

    For hybrid models (e.g., Qwen3.5) that mix linear-attention (ArraysCache)
    and full-attention (KVCache) layers, only the KVCache layers are swapped
    to TurboKVCache. Non-KVCache layers keep their default cache type.

    Boundary protection: first ``boundary`` and last ``boundary`` KVCache layers
    stay at FP16 to avoid NaN from extreme V norms on boundary layers.
    """
    key_bits = 0 if asymmetric else None  # 0 = keep keys FP16

    # Get the model's default cache to discover per-layer types
    baseline = make_prompt_cache(model)
    kv_indices = [i for i, c in enumerate(baseline) if type(c).__name__ == "KVCache"]
    n_kv = len(kv_indices)
    boundary = 2  # first/last 2 KV layers at FP16

    turbo_cache = list(baseline)
    for rank, idx in enumerate(kv_indices):
        if rank < boundary or rank >= n_kv - boundary:
            continue  # Keep boundary layers at FP16
        turbo_cache[idx] = TurboKVCache(
            bits=bits,
            key_bits=key_bits,
            min_compress_tokens=min_compress_tokens,
        )
    return turbo_cache


def _build_haystack(
    target_tokens: int, tokenizer, rng: random.Random, needle_depth_pct: float = 0.5
) -> str:
    """Build filler text with a needle inserted at the given depth.

    Returns the full prompt string (haystack + query) ready for tokenization.
    The target_tokens count is approximate — we iterate filler paragraphs until
    we hit the target, then insert the needle at the right spot.
    """
    # Estimate chars per token (~3.5 for English on most tokenizers)
    target_chars = int(target_tokens * 3.5)

    paragraphs: list[str] = []
    total_chars = 0
    pool = list(FILLER_PARAGRAPHS)
    rng.shuffle(pool)
    idx = 0
    while total_chars < target_chars:
        p = pool[idx % len(pool)]
        paragraphs.append(p)
        total_chars += len(p)
        idx += 1

    # Insert needle at depth
    insert_idx = max(0, int(len(paragraphs) * needle_depth_pct))
    insert_idx = min(insert_idx, len(paragraphs))
    paragraphs.insert(insert_idx, NEEDLE_TEXT)

    haystack = "\n\n".join(paragraphs)
    return f"{haystack}\n\n{NEEDLE_QUERY}"


def _generate_text(
    model, tokenizer, prompt: str, max_tokens: int = 64, cache: list | None = None
) -> tuple[str, float, float]:
    """Generate text and return (text, prompt_tps, gen_tps).

    If cache is provided it's used as prompt_cache (and mutated in place).
    """
    text = ""
    prompt_tps = 0.0
    gen_tps = 0.0
    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        prompt_cache=cache,
    ):
        text += resp.text
        prompt_tps = resp.prompt_tps
        gen_tps = resp.generation_tps
    return text.strip(), prompt_tps, gen_tps


def _forward_logits(model, tokenizer, text: str, cache: list | None = None) -> mx.array:
    """Run a forward pass and collect logits at every token position.

    Returns shape (seq_len - 1, vocab_size) — logits predicting token[i+1]
    from prefix[:i+1].

    We process the full prompt through the model in one prefill pass, then
    step token-by-token during decode to ensure the cache compresses.
    """
    tokens = tokenizer.encode(text)
    input_ids = mx.array(tokens)

    if cache is None:
        cache = _make_baseline_cache(model)

    # Prefill: feed first N-1 tokens to populate cache
    prefill_len = min(32, len(tokens) - 1)
    if prefill_len > 0:
        prefill_input = input_ids[:prefill_len][None]  # (1, prefill_len)
        logits = model(prefill_input, cache=cache)
        mx.eval(logits)
        mx.eval([c.state for c in cache])
        all_logits = [logits[0]]  # (prefill_len, vocab)

    # Decode: one token at a time so compression kicks in
    for i in range(prefill_len, len(tokens) - 1):
        step_input = input_ids[i : i + 1][None]  # (1, 1)
        logits = model(step_input, cache=cache)
        mx.eval(logits)
        mx.eval([c.state for c in cache])
        all_logits.append(logits[0])  # (1, vocab)

    return mx.concatenate(all_logits, axis=0)  # (seq_len - 1, vocab)


# ---------------------------------------------------------------------------
# Test: NIAH
# ---------------------------------------------------------------------------


def run_niah(
    model, tokenizer, bits: int, asymmetric: bool, verbose: bool
) -> list[dict]:
    """Run Needle-In-A-Haystack tests at various depths and context sizes."""
    depths = [0.0, 0.25, 0.50, 0.75, 0.90]
    context_sizes = [1024, 2048, 4096]
    configs = [
        ("baseline", lambda: _make_baseline_cache(model)),
        ("turbo", lambda: _make_turbo_cache(model, bits, asymmetric)),
    ]

    results: list[dict] = []
    rng = random.Random(SEED)

    total = len(depths) * len(context_sizes) * len(configs)
    done = 0

    for ctx_size in context_sizes:
        for depth in depths:
            prompt = _build_haystack(ctx_size, tokenizer, rng, depth)
            for config_name, cache_fn in configs:
                done += 1
                cache = cache_fn()
                print(
                    f"  [{done}/{total}] ctx={ctx_size} depth={depth:.0%} "
                    f"config={config_name} ...",
                    end=" ",
                    flush=True,
                )

                text, p_tps, g_tps = _generate_text(
                    model, tokenizer, prompt, max_tokens=32, cache=cache
                )
                passed = NEEDLE_ANSWER.lower() in text.lower()
                status = "PASS" if passed else "FAIL"
                print(f"{status}  (prompt {p_tps:.0f} t/s, gen {g_tps:.1f} t/s)")

                if verbose and not passed:
                    print(f"         got: {text[:120]}")

                results.append(
                    {
                        "test": "niah",
                        "context_tokens": ctx_size,
                        "depth_pct": depth,
                        "config": config_name,
                        "passed": passed,
                        "response": text[:200],
                        "prompt_tps": p_tps,
                        "gen_tps": g_tps,
                    }
                )

    return results


# ---------------------------------------------------------------------------
# Test: KLD
# ---------------------------------------------------------------------------

# Fixed text for KLD comparison (~500+ tokens). Multi-topic so we exercise
# varied token distributions. Must exceed min_compress_tokens (default 256)
# so that TurboKVCache actually compresses.
KLD_TEXT = (
    "The development of the transistor at Bell Labs in 1947 marked the beginning "
    "of the semiconductor revolution. William Shockley, John Bardeen, and Walter "
    "Brattain demonstrated that a solid-state device could amplify electrical "
    "signals, replacing bulky vacuum tubes. By 1958, Jack Kilby at Texas "
    "Instruments had created the first integrated circuit, combining multiple "
    "transistors on a single germanium chip. Robert Noyce independently developed "
    "a silicon-based version at Fairchild Semiconductor, which proved more "
    "practical for mass production. Gordon Moore observed in 1965 that the number "
    "of transistors on a chip doubled approximately every two years, a prediction "
    "that held remarkably well for over five decades. Modern processors contain "
    "billions of transistors etched at scales of just a few nanometers, pushing "
    "against fundamental physical limits. Quantum tunneling effects become "
    "significant below about 5 nanometers, forcing engineers to explore novel "
    "architectures including gate-all-around transistors and chiplet designs. "
    "The industry continues to find ways to extend performance through packaging "
    "innovations, heterogeneous integration, and specialized accelerators for "
    "tasks like machine learning inference. The original transistor that started "
    "it all now sits in the Smithsonian Institution.\n\n"
    "The history of cryptography stretches back thousands of years, from simple "
    "substitution ciphers used by Julius Caesar to the sophisticated public-key "
    "systems that protect modern internet communications. The Enigma machine, "
    "used by Nazi Germany during World War Two, employed a series of rotating "
    "electrical rotors to create a cipher with an astronomical number of possible "
    "settings. Breaking Enigma required the combined efforts of Polish and British "
    "mathematicians, most notably Alan Turing at Bletchley Park, who designed "
    "electromechanical devices called bombes to systematically test key settings. "
    "The success of this effort shortened the war by an estimated two years and "
    "saved millions of lives. After the war, Claude Shannon published his "
    "groundbreaking paper on communication theory, establishing the mathematical "
    "foundations of information theory and proving that a one-time pad provides "
    "perfect secrecy. The invention of public-key cryptography by Diffie and "
    "Hellman in 1976 revolutionized the field by allowing two parties to establish "
    "a shared secret over an insecure channel without any prior arrangement. "
    "Today, RSA and elliptic curve cryptography protect trillions of dollars in "
    "daily electronic transactions, while quantum computing threatens to upend "
    "these systems entirely within the coming decades."
)


def _kl_divergence(p_logits: mx.array, q_logits: mx.array) -> mx.array:
    """Compute KL(P || Q) per position from raw logits.

    p_logits, q_logits: (seq_len, vocab_size)
    Returns: (seq_len,) array of KL divergences.
    """
    # Log-softmax for numerical stability
    p_log = p_logits - mx.logsumexp(p_logits, axis=-1, keepdims=True)
    q_log = q_logits - mx.logsumexp(q_logits, axis=-1, keepdims=True)

    p = mx.softmax(p_logits, axis=-1)
    # KL(P||Q) = sum_x P(x) * (log P(x) - log Q(x))
    kl = mx.sum(p * (p_log - q_log), axis=-1)
    return kl


def run_kld(model, tokenizer, bits: int, asymmetric: bool, verbose: bool) -> list[dict]:
    """Compare token-level KL divergence between baseline and turbo caches."""
    print("  Computing baseline logits ...", flush=True)
    baseline_cache = _make_baseline_cache(model)
    baseline_logits = _forward_logits(model, tokenizer, KLD_TEXT, cache=baseline_cache)
    mx.eval(baseline_logits)

    print("  Computing turbo logits ...", flush=True)
    # Use min_compress_tokens=32 so compression kicks in during the forward pass
    # (default 256 would skip compression on short texts)
    turbo_cache = _make_turbo_cache(model, bits, asymmetric, min_compress_tokens=32)
    turbo_logits = _forward_logits(model, tokenizer, KLD_TEXT, cache=turbo_cache)
    mx.eval(turbo_logits)

    # Align lengths (should be same, but be safe)
    min_len = min(baseline_logits.shape[0], turbo_logits.shape[0])
    baseline_logits = baseline_logits[:min_len]
    turbo_logits = turbo_logits[:min_len]

    kl = _kl_divergence(baseline_logits, turbo_logits)
    mx.eval(kl)

    kl_np = kl.tolist()
    mean_kl = sum(kl_np) / len(kl_np)
    max_kl = max(kl_np)

    # Top-1 match rate
    baseline_top1 = mx.argmax(baseline_logits, axis=-1)
    turbo_top1 = mx.argmax(turbo_logits, axis=-1)
    match_rate = float(mx.mean(baseline_top1 == turbo_top1).item())

    # Config label
    config_label = f"turbo{bits}"
    if asymmetric:
        config_label += "_asymmetric"

    print(f"  Mean KLD:       {mean_kl:.6f}")
    print(f"  Max KLD:        {max_kl:.6f}")
    print(f"  Top-1 match:    {match_rate:.1%}")
    print(f"  Positions:      {min_len}")

    if verbose:
        # Show worst 5 positions
        sorted_positions = sorted(enumerate(kl_np), key=lambda x: x[1], reverse=True)
        print("  Worst positions:")
        for pos, val in sorted_positions[:5]:
            print(f"    pos {pos}: KLD = {val:.6f}")

    return [
        {
            "test": "kld",
            "config": config_label,
            "positions": min_len,
            "mean_kld": mean_kl,
            "max_kld": max_kl,
            "top1_match_rate": match_rate,
        }
    ]


# ---------------------------------------------------------------------------
# Test: Context Size Scaling
# ---------------------------------------------------------------------------


def _measure_generation(
    model, tokenizer, prompt_tokens: int, gen_tokens: int, cache: list
) -> dict:
    """Generate gen_tokens from a prompt of prompt_tokens length.

    Returns dict with timing and memory info.
    """
    # Build a prompt of the right length from filler text
    rng = random.Random(SEED)
    filler = _build_haystack(prompt_tokens, tokenizer, rng, needle_depth_pct=0.5)
    # Truncate/pad to exact token count
    tokens = tokenizer.encode(filler)[:prompt_tokens]
    prompt_text = tokenizer.decode(tokens)

    mx.reset_peak_memory()

    text, prompt_tps, gen_tps = _generate_text(
        model, tokenizer, prompt_text, max_tokens=gen_tokens, cache=cache
    )
    peak_mem = mx.get_peak_memory() / 1e9

    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "peak_memory_gb": peak_mem,
    }


def run_context(
    model, tokenizer, bits: int, asymmetric: bool, verbose: bool
) -> list[dict]:
    """Measure decode speed and memory at various context lengths."""
    context_sizes = [128, 512, 1024, 2048, 4096]
    gen_tokens = 32  # Generate a fixed number of tokens at each context size

    config_label = f"turbo{bits}"
    if asymmetric:
        config_label += "_asymmetric"

    configs = [
        ("baseline", lambda: _make_baseline_cache(model)),
        (config_label, lambda: _make_turbo_cache(model, bits, asymmetric)),
    ]

    results: list[dict] = []
    total = len(context_sizes) * len(configs)
    done = 0

    for ctx_size in context_sizes:
        for config_name, cache_fn in configs:
            done += 1
            print(
                f"  [{done}/{total}] ctx={ctx_size} config={config_name} ...",
                end=" ",
                flush=True,
            )

            cache = cache_fn()
            info = _measure_generation(model, tokenizer, ctx_size, gen_tokens, cache)
            info["config"] = config_name
            info["test"] = "context"

            print(
                f"prompt {info['prompt_tps']:.0f} t/s, "
                f"gen {info['gen_tps']:.1f} t/s, "
                f"mem {info['peak_memory_gb']:.2f} GB"
            )

            results.append(info)

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _write_report(
    results: list[dict], model_name: str, bits: int, asymmetric: bool, output_dir: Path
) -> Path:
    """Write a markdown report of all test results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    short = _model_short_name(model_name)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mlx_quality_{short}_{date_str}.md"
    path = output_dir / filename

    config_label = f"turbo{bits}"
    if asymmetric:
        config_label += "_asymmetric"

    lines = [
        f"# MLX Quality Suite — {short}",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** `{model_name}`",
        f"**Config:** {config_label}",
        "",
    ]

    # NIAH results
    niah_results = [r for r in results if r["test"] == "niah"]
    if niah_results:
        lines.append("## NIAH (Needle In A Haystack)")
        lines.append("")
        lines.append("| Context | Depth | Config | Result | Prompt t/s | Gen t/s |")
        lines.append("|---------|-------|--------|--------|------------|---------|")
        for r in niah_results:
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(
                f"| {r['context_tokens']} "
                f"| {r['depth_pct']:.0%} "
                f"| {r['config']} "
                f"| {status} "
                f"| {r['prompt_tps']:.0f} "
                f"| {r['gen_tps']:.1f} |"
            )

        # Summary
        baseline_pass = sum(
            1 for r in niah_results if r["config"] == "baseline" and r["passed"]
        )
        baseline_total = sum(1 for r in niah_results if r["config"] == "baseline")
        turbo_pass = sum(
            1 for r in niah_results if r["config"] != "baseline" and r["passed"]
        )
        turbo_total = sum(1 for r in niah_results if r["config"] != "baseline")
        lines.append("")
        lines.append(f"**Baseline:** {baseline_pass}/{baseline_total} passed  ")
        lines.append(f"**Turbo:** {turbo_pass}/{turbo_total} passed")
        lines.append("")

    # KLD results
    kld_results = [r for r in results if r["test"] == "kld"]
    if kld_results:
        lines.append("## KL Divergence")
        lines.append("")
        for r in kld_results:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Config | {r['config']} |")
            lines.append(f"| Positions | {r['positions']} |")
            lines.append(f"| Mean KLD | {r['mean_kld']:.6f} |")
            lines.append(f"| Max KLD | {r['max_kld']:.6f} |")
            lines.append(f"| Top-1 Match | {r['top1_match_rate']:.1%} |")
        lines.append("")

    # Context scaling results
    ctx_results = [r for r in results if r["test"] == "context"]
    if ctx_results:
        lines.append("## Context Size Scaling")
        lines.append("")
        lines.append("| Context | Config | Prompt t/s | Gen t/s | Peak Memory (GB) |")
        lines.append("|---------|--------|------------|---------|-------------------|")
        for r in ctx_results:
            lines.append(
                f"| {r['prompt_tokens']} "
                f"| {r['config']} "
                f"| {r['prompt_tps']:.0f} "
                f"| {r['gen_tps']:.1f} "
                f"| {r['peak_memory_gb']:.2f} |"
            )
        lines.append("")

    report = "\n".join(lines)
    path.write_text(report, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MLX TurboQuant Quality Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument(
        "--test",
        choices=["niah", "kld", "context"],
        default=None,
        help="Run a specific test (default: all)",
    )
    parser.add_argument(
        "--bits", type=int, default=4, help="Turbo compression bits (default: 4)"
    )
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="Keep keys at FP16, compress only values",
    )
    parser.add_argument("--verbose", action="store_true", help="Show extra debug info")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: artifacts/mlx/)",
    )
    args = parser.parse_args()

    # Determine output dir
    repository_root = Path(__file__).resolve().parents[2]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repository_root / "artifacts" / "mlx"
    )

    tests_to_run = [args.test] if args.test else ["kld", "niah", "context"]

    config_label = f"turbo{args.bits}"
    if args.asymmetric:
        config_label += "_asymmetric"

    print("=== MLX Quality Suite ===")
    print(f"Model:  {args.model}")
    print(f"Config: {config_label}")
    print(f"Tests:  {', '.join(tests_to_run)}")
    print()

    # Load model once
    print("Loading model ...", flush=True)
    model, tokenizer = mlx_lm.load(args.model)
    print(f"Loaded. Layers: {_n_layers(model)}")
    print()

    all_results: list[dict] = []

    if "kld" in tests_to_run:
        print("--- KL Divergence ---")
        results = run_kld(model, tokenizer, args.bits, args.asymmetric, args.verbose)
        all_results.extend(results)
        print()

    if "niah" in tests_to_run:
        print("--- NIAH ---")
        results = run_niah(model, tokenizer, args.bits, args.asymmetric, args.verbose)
        all_results.extend(results)
        print()

    if "context" in tests_to_run:
        print("--- Context Size Scaling ---")
        results = run_context(
            model, tokenizer, args.bits, args.asymmetric, args.verbose
        )
        all_results.extend(results)
        print()

    # Write report
    if all_results:
        report_path = _write_report(
            all_results, args.model, args.bits, args.asymmetric, output_dir
        )
        print(f"Report saved to: {report_path}")

    print("Done.")


if __name__ == "__main__":
    main()
