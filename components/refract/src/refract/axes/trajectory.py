"""Axis A (v0.1.4): Trajectory Match.

The v0.1.x GTM axis decoded text under both configs, then *retokenized* the
text with ``llama-tokenize`` to compare in model-token space. Detokenize→
retokenize is lossy: a 50-token greedy generation can come back as 60–137
tokens after retokenization (gemma-4 31B v0.1.2 hit a 2.87× inflation),
which v0.1.3 worked around by normalizing the score by ``mean_cand_length``
instead of ``n_predict``. The score stayed bounded but the *signal* was
still polluted by the round-trip.

v0.1.4 closes that gap by capturing token IDs *at decode time* via a
patched ``llama-completion`` (REFRACT_TRAJECTORY env var → JSONL stream of
``{"step":N,"token_id":ID}`` records). No detokenize, no retokenize, no
whitespace-vs-model-token unit mismatch.

Score mapping (v0.1.4):
    Trajectory_score = 100 * mean_prefix_agreement_steps / mean_cand_steps

where both quantities are in true model-token units. Bounded in [0, 1] by
construction.

Diagnostics returned alongside the score:
    full_match_rate                    binary; both sequences identical
    median_first_divergence_position   token step where ref/cand diverge
    mean_prefix_agreement_length
    mean_cand_length / mean_ref_length
    per_prompt                         full per-prompt diagnostics

This module's signature mirrors :mod:`refract.axes.gtm.run_gtm` so callers
can swap the axis with a one-line import change. The composite-score code
in :mod:`refract.score` is unchanged: it accepts any 0–100 score for
"axis A".

v0.2 plan: extend the patched binary to dump per-step logits as well, so
this same forward pass produces both trajectory-match (token IDs) AND
trajectory-KLD (per-step distribution divergence). KLD@D in v0.1 uses
corpus statistics as a proxy; the v0.2 trajectory-KLD would catch
generation-only drift that corpus KLD misses.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..runner import KVConfig, run_completion_trajectory


@dataclass
class TrajectoryResult:
    """Output of run_trajectory()."""

    score: float                              # 0–100
    full_match_rate: float                    # 0–1
    median_first_divergence: Optional[int]    # token step; None if all match
    mean_prefix_agreement_length: float
    mean_cand_length: float                   # decoded steps, candidate
    mean_ref_length: float                    # decoded steps, reference
    n_prompts: int
    n_tokens_each: int
    per_prompt: list[dict]
    notes: list[str] = field(default_factory=list)


def _load_prompts(path: Path) -> list[dict]:
    """Load a JSONL prompts file. Each line: {id, category, prompt, ...}."""
    out = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            out.append(json.loads(ln))
    return out


def _diff(ref: list, cand: list) -> tuple[Optional[int], int]:
    """Return (first_divergence_position, prefix_agreement_length).

    ``first_divergence_position`` is None iff sequences are identical
    (treating cand as a candidate to match ref to length min(len)). When
    one sequence is a strict prefix of the other, divergence is reported
    at the boundary.
    """
    n = min(len(ref), len(cand))
    for i in range(n):
        if ref[i] != cand[i]:
            return i, i
    if len(ref) == len(cand):
        return None, n
    return n, n


def run_trajectory(
    model: Path,
    reference_kv: KVConfig,
    candidate_kv: KVConfig,
    prompts_path: Path,
    n_predict: int = 128,
    ctx: int = 512,
    n_gpu_layers: int = 99,
    seed: int = 42,
    progress: bool = True,
) -> TrajectoryResult:
    """Run the v0.1.4 Trajectory axis end to end.

    Greedy-decodes ``n_predict`` tokens from each prompt under both KV
    configs using the patched ``llama-completion`` binary, capturing the
    model's actual sampled token IDs (no detokenize round-trip). Returns
    match statistics with the same shape as :class:`GTMResult` so the
    composite scorer can consume either.

    Raises ``RuntimeError`` if the trajectory file comes back empty for a
    prompt; that means the patched binary isn't installed or the env-var
    plumbing didn't reach the subprocess.
    """
    prompts = _load_prompts(prompts_path)
    if not prompts:
        raise ValueError(f"No prompts loaded from {prompts_path}")

    per_prompt: list[dict] = []
    matches = 0
    first_divs: list[int] = []
    prefix_lens: list[int] = []
    cand_lens: list[int] = []
    ref_lens: list[int] = []

    # Backends with per-LLM memory pressure (vLLM on hybrid models) can't
    # afford to swap KV configs every prompt. Batch all ref calls first,
    # then all cand calls, so each backend loads each KV config exactly once.
    ref_traj: list[list[int]] = []
    cand_traj: list[list[int]] = []
    for i, p in enumerate(prompts):
        if progress:
            print(f"  ref [{i + 1}/{len(prompts)}] {p['id']:<10} ({p.get('category', '?')}) ...",
                  flush=True)
        toks, _ = run_completion_trajectory(
            model=model, prompt=p["prompt"], kv=reference_kv,
            n_predict=n_predict, ctx=ctx, n_gpu_layers=n_gpu_layers, seed=seed,
        )
        ref_traj.append(toks)
    for i, p in enumerate(prompts):
        if progress:
            print(f"  cand [{i + 1}/{len(prompts)}] {p['id']:<10} ({p.get('category', '?')}) ...",
                  flush=True)
        toks, _ = run_completion_trajectory(
            model=model, prompt=p["prompt"], kv=candidate_kv,
            n_predict=n_predict, ctx=ctx, n_gpu_layers=n_gpu_layers, seed=seed,
        )
        cand_traj.append(toks)

    for i, p in enumerate(prompts):
        ref_toks = ref_traj[i]
        cand_toks = cand_traj[i]

        if not ref_toks and not cand_toks:
            raise RuntimeError(
                f"Both ref and cand trajectories were empty for prompt "
                f"{p.get('id', i)!r}. The patched llama-completion binary "
                f"likely isn't present at LLAMA_CPP_BIN_DIR. Build it from "
                f"tools/completion/completion.cpp (REFRACT v0.1.4 patch)."
            )

        first_div, prefix_len = _diff(ref_toks, cand_toks)
        is_match = first_div is None

        if is_match:
            matches += 1
        else:
            first_divs.append(first_div)
        prefix_lens.append(prefix_len)
        cand_lens.append(len(cand_toks))
        ref_lens.append(len(ref_toks))

        per_prompt.append({
            "id": p["id"],
            "category": p.get("category"),
            "prompt": p["prompt"],
            "ref_token_ids": ref_toks,
            "cand_token_ids": cand_toks,
            "first_divergence": first_div,
            "prefix_agreement_length": prefix_len,
            "cand_length": len(cand_toks),
            "ref_length": len(ref_toks),
            "matched": is_match,
        })

    n = len(prompts)
    full_match_rate = matches / n
    median_first_div = statistics.median(first_divs) if first_divs else None
    mean_prefix = sum(prefix_lens) / n if n else 0.0
    mean_cand = sum(cand_lens) / n if n else 0.0
    mean_ref = sum(ref_lens) / n if n else 0.0
    if mean_cand > 0:
        score = 100.0 * (mean_prefix / mean_cand)
    else:
        score = 0.0
    score = max(0.0, min(100.0, score))

    notes: list[str] = []
    # No detokenize→retokenize is involved, so the inflation note from GTM
    # v0.1.3 is no longer relevant. We instead flag the case where the
    # candidate stopped early (EOS), which makes the divergence-position
    # diagnostic harder to interpret.
    short_cands = sum(1 for c in cand_lens if c < n_predict)
    if short_cands > 0:
        notes.append(
            f"{short_cands}/{n} candidates stopped before n_predict={n_predict} "
            f"(EOS or other stop condition). Per-prompt cand_length records "
            f"the actual decoded length."
        )

    return TrajectoryResult(
        score=score,
        full_match_rate=full_match_rate,
        median_first_divergence=median_first_div,
        mean_prefix_agreement_length=mean_prefix,
        mean_cand_length=mean_cand,
        mean_ref_length=mean_ref,
        n_prompts=n,
        n_tokens_each=n_predict,
        per_prompt=per_prompt,
        notes=notes,
    )
