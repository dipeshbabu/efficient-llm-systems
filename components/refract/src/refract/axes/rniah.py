"""Axis C (v0.2): R-NIAH — Retrieval Needle-In-A-Haystack.

Probes long-context retrieval degradation. GTM/Trajectory and KLD@D both
operate on short windows (typically 512 tokens) and are blind to the
"scores 99 on KLD@D and still fails at 32K context" failure mode that
quants frequently exhibit. R-NIAH catches it directly.

Protocol (v0.2.0)
-----------------

For each (length, position) cell:

    1. Slice the haystack corpus to a chunk of approximately ``length``
       tokens. Approximation is char-based (chars-per-token estimated
       from a one-time full tokenization on a small head sample of the
       corpus). The exact token count is not load-bearing; the position
       fraction is what controls test difficulty.

    2. Insert the needle at ``position`` (a fraction in [0, 1]) of the
       haystack chunk, snapped to the nearest sentence boundary so the
       insertion doesn't break a word.

    3. Append a retrieval question.

    4. Run ``n_trials`` greedy completions under both reference and
       candidate KV configs. Score = 1.0 if the completion contains the
       needle keyword (case-insensitive substring), else 0.0.

    5. Per-cell aggregate:
           base_acc[cell]    = mean(reference trial scores)
           cand_acc[cell]    = mean(candidate trial scores)
           degradation[cell] = max(0, base_acc - cand_acc)

R-NIAH score:
    R-NIAH = 100 * (1 - mean_cell_degradation)

so cand matching base in every cell → 100, cand failing every cell that
base passes → 0.

Cells where ``length > ctx_max`` are skipped and recorded in
``skipped_cells`` so the user can see what coverage they actually got.

Implementation notes
--------------------

  - The haystack is read once at ``run_rniah`` entry. We tokenize a
    small head (the first ~32K chars) to estimate chars-per-token, then
    use char-slicing for the full set of cells. This avoids tokenizing
    the entire corpus (which can be hundreds of MB).
  - ``n_trials=1`` is the default. Increase for variance reduction at
    linear cost. The protocol's "5 trials with phrase variants" remains
    a v0.3 target; v0.2.0 ships with one phrasing.
  - Cost: O(n_lengths × n_positions × n_trials × 2) generations.
    A 5 × 3 × 1 × 2 = 30-call sweep at ~30 sec each ≈ 15 min per model.
  - The needle keyword is auto-extracted as the longest all-caps
    hyphenated token in ``needle`` unless ``needle_keyword``
    is supplied explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..runner import KVConfig, run_completion, tokenize_to_ids

DEFAULT_LENGTHS = (4096, 8192, 16384, 32768, 65536)
DEFAULT_POSITIONS = (0.10, 0.50, 0.90)
DEFAULT_N_TRIALS = 1
# v0.2.1: defaults switched away from "secret password" wording. The
# "secret" + "password" framing reliably triggers safety/refusal training
# on RLHF'd instruct models (gemma-4 family, smaller qwen instructs), so
# even the fp16 reference scored base_acc = 0 across every cell. That made
# R-NIAH = 100 a noise-floor reading rather than real signal. Neutral
# "rare paint color" framing has zero credential association while keeping
# the APRICOT-7-BLUE distinctive token for clean substring matching.
DEFAULT_NEEDLE = (
    "Note: APRICOT-7-BLUE is the rare paint color featured in this article."
)
DEFAULT_QUESTION = "Q: What is the rare paint color featured in this article?\nA:"

# Internal: how many chars to tokenize when estimating chars-per-token.
# 32K chars ≈ 8K tokens, plenty for a stable ratio without a huge tokenize call.
_CHARS_PER_TOKEN_HEAD = 32_768

# Internal: search window (chars) when snapping insertion to a sentence
# boundary. Wide enough to find a period in dense text, narrow enough to
# keep position semantics meaningful.
_BOUNDARY_WINDOW = 200


@dataclass
class RNIAHCell:
    """Per-(length, position) bucket result."""

    length: int
    position: float
    n_trials: int
    base_acc: float
    cand_acc: float
    degradation: float
    base_hits: int = 0
    cand_hits: int = 0


@dataclass
class RNIAHResult:
    score: float
    n_cells: int
    cells: list[RNIAHCell]
    skipped_cells: list[tuple[int, float]]
    needle: str
    needle_keyword: str
    notes: list[str] = field(default_factory=list)

    @property
    def base_accuracy(self) -> float:
        """Mean fp16-reference retrieval accuracy across measured cells."""
        if not self.cells:
            return 0.0
        return sum(cell.base_acc for cell in self.cells) / len(self.cells)

    @property
    def confidence(self) -> str:
        """Whether the reference engaged the retrieval task reliably."""
        return "ok" if self.base_accuracy >= 0.2 else "low"


def _extract_needle_keyword(needle: str) -> str:
    """Pull the distinctive retrieval token out of a needle phrase.

    Picks the longest run of [A-Z0-9] characters with optional hyphens —
    e.g. ``"APRICOT-7-BLUE"`` from the default needle. Falls back to the
    last whitespace-token if no all-caps run exists. Users can override
    via the ``needle_keyword`` parameter on ``run_rniah``.
    """
    candidates = re.findall(r"[A-Z][A-Z0-9\-]{2,}", needle)
    if candidates:
        return max(candidates, key=len)
    parts = needle.split()
    return parts[-1].rstrip(".") if parts else needle


def _nearest_sentence_boundary(text: str, target_char: int) -> int:
    """Snap an insertion point to a sentence boundary.

    Walks outward from ``target_char`` looking for a ``". "`` pattern
    within ±_BOUNDARY_WINDOW chars. Returns the position immediately
    after the period+space so the inserted needle starts a new sentence.
    Falls back to the original target if no boundary is in range.
    """
    if target_char <= 0:
        return 0
    if target_char >= len(text):
        return len(text)
    lo = max(0, target_char - _BOUNDARY_WINDOW)
    hi = min(len(text), target_char + _BOUNDARY_WINDOW)
    # Prefer the closest boundary, scanning outward in alternating directions.
    for delta in range(0, _BOUNDARY_WINDOW):
        for cand in (target_char - delta, target_char + delta):
            if cand < lo or cand + 1 >= hi:
                continue
            if text[cand] == "." and text[cand + 1] in (" ", "\n"):
                return cand + 2
    return target_char


def _estimate_chars_per_token(model: Path, head: str) -> float:
    """Tokenize a small head of text to estimate chars-per-token."""
    if not head:
        return 4.0
    tokens = tokenize_to_ids(model, head)
    if not tokens:
        return 4.0
    return len(head) / len(tokens)


def _build_prompt(
    haystack_chunk: str,
    needle: str,
    question: str,
    position: float,
) -> tuple[str, str]:
    """Build the (system, user) chat split for an R-NIAH cell.

    v0.3.0: returns the split rather than a single string. The haystack
    (with needle inserted at ``position``) becomes the system message —
    that's where chat templates put long-form context — and the retrieval
    question becomes the user message. llama-cli's ``--jinja -sys ... -p
    ...`` then renders the model's own chat template around both.

    Falls back to a single concatenated string when the caller still
    expects v0.2.x behaviour (no chat template); see the legacy adapter
    below.
    """
    target_char = int(len(haystack_chunk) * position)
    insertion = _nearest_sentence_boundary(haystack_chunk, target_char)
    pre = haystack_chunk[:insertion]
    post = haystack_chunk[insertion:]
    system_msg = f"{pre} {needle} {post}"
    return system_msg, question


def _scored(text: str, needle_keyword: str) -> int:
    """1 if the completion contains the needle keyword, else 0."""
    return int(needle_keyword.lower() in text.lower())


def run_rniah(
    model: Path,
    haystack_corpus: Path,
    reference_kv: KVConfig,
    candidate_kv: KVConfig,
    ctx_max: int,
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
    positions: tuple[float, ...] = DEFAULT_POSITIONS,
    n_trials: int = DEFAULT_N_TRIALS,
    needle: str = DEFAULT_NEEDLE,
    question: str = DEFAULT_QUESTION,
    needle_keyword: Optional[str] = None,
    # v0.2.1: bumped from 32 to 256. Modern instruct models (qwen3.5,
    # gemma-4) emit a thinking trace before the answer; 32 tokens runs
    # out before the answer lands. 256 gives space to think + answer.
    # Substring scoring catches the keyword wherever it appears in the
    # response (thinking trace or final answer).
    n_predict: int = 256,
    n_gpu_layers: int = 99,
    seed: int = 42,
    progress: bool = True,
) -> RNIAHResult:
    """Run R-NIAH end to end.

    Returns an :class:`RNIAHResult` with per-cell breakdown. Cells where
    ``length > ctx_max`` are skipped and reported in
    ``RNIAHResult.skipped_cells`` rather than failed silently.
    """
    haystack_text = haystack_corpus.read_text(errors="replace")
    head = haystack_text[:_CHARS_PER_TOKEN_HEAD]
    chars_per_token = _estimate_chars_per_token(model, head)
    if needle_keyword is None:
        needle_keyword = _extract_needle_keyword(needle)

    cells: list[RNIAHCell] = []
    skipped: list[tuple[int, float]] = []
    notes: list[str] = []

    # Reserve enough context for needle + question + a generation budget.
    needle_tok_estimate = max(1, int(len(needle) / max(chars_per_token, 1.0)))
    question_tok_estimate = max(1, int(len(question) / max(chars_per_token, 1.0)))
    overhead = needle_tok_estimate + question_tok_estimate + n_predict + 16

    for length in lengths:
        if length > ctx_max:
            for pos in positions:
                skipped.append((length, pos))
            continue
        usable_tokens = length - overhead
        if usable_tokens <= 0:
            for pos in positions:
                skipped.append((length, pos))
            notes.append(
                f"Skipped length={length}: ≤0 tokens available for haystack "
                f"after needle/question/n_predict overhead ({overhead})."
            )
            continue
        usable_chars = int(usable_tokens * chars_per_token)
        if usable_chars > len(haystack_text):
            notes.append(
                f"Haystack corpus only has {len(haystack_text)} chars; "
                f"length={length} would need ~{usable_chars}. Cell may run "
                f"with a shorter context than nominal."
            )
            usable_chars = len(haystack_text)
        haystack_chunk = haystack_text[:usable_chars]

        for pos in positions:
            system_msg, user_msg = _build_prompt(haystack_chunk, needle, question, pos)
            base_hits = 0
            cand_hits = 0
            for trial in range(n_trials):
                if progress:
                    print(
                        f"  cell length={length} pos={pos:.2f} "
                        f"trial {trial + 1}/{n_trials} ...",
                        flush=True,
                    )
                # v0.3.0: haystack goes into system, question into user.
                # llama-cli applies the model's chat template via --jinja.
                ref_text, _ = run_completion(
                    model=model,
                    prompt=user_msg,
                    kv=reference_kv,
                    n_predict=n_predict,
                    ctx=length + n_predict + 32,
                    n_gpu_layers=n_gpu_layers,
                    seed=seed + trial,
                    apply_chat_template=True,
                    system=system_msg,
                )
                cand_text, _ = run_completion(
                    model=model,
                    prompt=user_msg,
                    kv=candidate_kv,
                    n_predict=n_predict,
                    ctx=length + n_predict + 32,
                    n_gpu_layers=n_gpu_layers,
                    seed=seed + trial,
                    apply_chat_template=True,
                    system=system_msg,
                )
                base_hits += _scored(ref_text, needle_keyword)
                cand_hits += _scored(cand_text, needle_keyword)

            base_acc = base_hits / n_trials
            cand_acc = cand_hits / n_trials
            degradation = max(0.0, base_acc - cand_acc)
            cells.append(
                RNIAHCell(
                    length=length,
                    position=pos,
                    n_trials=n_trials,
                    base_acc=base_acc,
                    cand_acc=cand_acc,
                    degradation=degradation,
                    base_hits=base_hits,
                    cand_hits=cand_hits,
                )
            )

    if not cells:
        score = 0.0
        notes.append(
            "No R-NIAH cells were run — every (length, position) pair was "
            "skipped (typically because length > ctx_max for every cell). "
            "Consider lowering --rniah-lengths."
        )
    else:
        mean_deg = sum(c.degradation for c in cells) / len(cells)
        score = 100.0 * (1.0 - mean_deg)

    return RNIAHResult(
        score=score,
        n_cells=len(cells),
        cells=cells,
        skipped_cells=skipped,
        needle=needle,
        needle_keyword=needle_keyword,
        notes=notes,
    )
