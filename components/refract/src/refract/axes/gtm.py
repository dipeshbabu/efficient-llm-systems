"""Axis A: GTM — Greedy Trajectory Match.

For each prompt in the dataset, greedy-decode N tokens from both the
reference (fp16-KV) and the candidate config. Compare token sequences via
the model's own tokenizer (``llama-tokenize``) so the units match
``--n-predict``.

Score mapping (v0.1.3):
    GTM_score = 100 * mean_prefix_agreement_length / mean_cand_length

i.e. "fraction of the candidate's generated text that matches the
reference," bounded in [0, 1] regardless of how detokenize→retokenize
inflates token counts. Earlier versions divided by ``n_predict``, which
broke when re-tokenizing chain-of-thought outputs returned more tokens than
the model decoded (gemma-31B v0.1.2 hit a 2.87× inflation factor).

We additionally report:
    full_match_rate                    (binary, diagnostic only)
    median_first_divergence_position
    mean_prefix_agreement_length
    mean_cand_length / mean_ref_length (so users can spot retokenize blow-up)
    per_prompt                         (list of per-prompt diagnostics)

v0.1.3 fail-loud change: the previous whitespace-tokenizer fallback was
removed. If ``tokenize_to_ids`` raises, the whole axis aborts with the
original tokenizer error rather than silently mixing whitespace-token
counts with model-token expectations.

v0.2 plan: capture token IDs at decode time via a custom binary so we
don't have to detokenize→re-tokenize at all.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..runner import KVConfig, run_completion, tokenize_to_ids


@dataclass
class GTMResult:
    """Output of run_gtm()."""

    score: float                              # 0–100
    full_match_rate: float                    # 0–1
    median_first_divergence: Optional[int]    # token position; None if all match
    mean_prefix_agreement_length: float
    mean_cand_length: float                   # v0.1.3: retokenized cand length
    mean_ref_length: float                    # v0.1.3: retokenized ref length
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


def _tokenize_words(text: str) -> list[str]:
    """v0.1 token proxy: whitespace split.

    DEPRECATED in v0.1.2 and REMOVED from the diff path in v0.1.3 — see
    tokenize_to_ids() in runner.py for true model-token tokenization.
    Kept ONLY for unit tests as a stable utility; do NOT call from the
    GTM diff path or scores will mix units (whitespace vs model tokens).
    """
    return text.split()


def _diff(ref: list, cand: list) -> tuple[Optional[int], int]:
    """Return (first_divergence_position, prefix_agreement_length).

    first_divergence_position is None iff sequences are identical (treating
    cand as a candidate to match ref to length min(len)).
    """
    n = min(len(ref), len(cand))
    for i in range(n):
        if ref[i] != cand[i]:
            return i, i
    if len(ref) == len(cand):
        return None, n
    # one is prefix of the other — divergence is at the boundary
    return n, n


def run_gtm(
    model: Path,
    reference_kv: KVConfig,
    candidate_kv: KVConfig,
    prompts_path: Path,
    n_predict: int = 128,
    ctx: int = 512,
    n_gpu_layers: int = 99,
    seed: int = 42,
    progress: bool = True,
) -> GTMResult:
    """Run the GTM axis end to end.

    Greedy-decodes ``n_predict`` tokens from each prompt under both configs,
    then computes match statistics.
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

    for i, p in enumerate(prompts):
        if progress:
            print(f"  [{i+1}/{len(prompts)}] {p['id']:<10} ({p.get('category','?')}) ...",
                  flush=True)

        ref_text, _ = run_completion(
            model=model, prompt=p["prompt"], kv=reference_kv,
            n_predict=n_predict, ctx=ctx, n_gpu_layers=n_gpu_layers, seed=seed,
        )
        cand_text, _ = run_completion(
            model=model, prompt=p["prompt"], kv=candidate_kv,
            n_predict=n_predict, ctx=ctx, n_gpu_layers=n_gpu_layers, seed=seed,
        )

        # v0.1.2: tokenize via the model's own vocab (llama-tokenize) instead
        # of whitespace. Whitespace tokenization can over-count (a 48-token
        # generation can produce 60+ whitespace tokens when generations
        # contain short words separated by spaces).
        # v0.1.3: REMOVED the silent whitespace fallback. If the tokenizer
        # subprocess fails, raise with a clear message — falling back to
        # whitespace mixes units (whitespace tokens vs model tokens) and
        # produces wrong-unit scores that aren't comparable across prompts.
        try:
            ref_toks = tokenize_to_ids(model, ref_text)
            cand_toks = tokenize_to_ids(model, cand_text)
        except Exception as e:
            raise RuntimeError(
                f"tokenize_to_ids failed for prompt {p.get('id', i)!r}; "
                f"refusing to fall back to whitespace which would mix units. "
                f"Original error: {e!r}"
            ) from e
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
            "ref": ref_text,
            "cand": cand_text,
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
    # v0.1.3 score: divide by mean_cand_length (the candidate's actual
    # retokenized length), NOT by n_predict. This is "fraction of the
    # candidate's generated text that matches the reference," bounded in
    # [0, 1] regardless of detokenize→retokenize inflation. Earlier versions
    # divided by n_predict and tripped a unit mismatch when retokenize
    # produced more tokens than the model decoded.
    if mean_cand > 0:
        score = 100.0 * (mean_prefix / mean_cand)
    else:
        score = 0.0
    score = max(0.0, min(100.0, score))

    notes: list[str] = []
    # Diagnostic note when retokenize materially inflates candidate length
    # past n_predict — this is information, not failure. The new normalization
    # keeps the score bounded, but users should still know their tokenizer
    # is inflating.
    if n_predict > 0 and mean_cand > 1.5 * n_predict:
        notes.append(
            f"detokenize→retokenize inflated candidate length: "
            f"mean_cand_length={mean_cand:.1f} vs n_predict={n_predict} "
            f"(ratio {mean_cand / n_predict:.2f}). Score normalized by "
            f"mean_cand_length to stay bounded."
        )

    return GTMResult(
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
