"""Decode-time trajectory evidence and pairwise agreement analysis.

This module bridges KV Fidelity's v0.3.4 trajectory methodology into Metria's
run/evidence model without importing KV Fidelity's CLI or module-global backend
state. Individual runs retain decode-time token trajectories. Agreement is
computed only when a reference and candidate result are compared.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from ..models import MetricDefinition, MetricDirection, MetricSample, MetricSummary
from ..protocols import (
    CaptureRequest,
    InferenceRequest,
    MeasurementResult,
    RuntimeSession,
)

_CAPTURE_SCHEMA = "metria.trajectory_capture.v1"
_CAPTURE_METHOD = "kv_fidelity.decode_time_trajectory"
_CAPTURE_VERSION = "0.3.4"
_COMPARISON_METHOD = "kv_fidelity.trajectory_match"
_COMPARISON_VERSION = "0.3.4"
_ALLOWED_CONFIG_KEYS = frozenset({"prompts", "generation"})
_ALLOWED_PROMPT_KEYS = frozenset({"id", "prompt", "category", "generation"})


def _sha256_text(value: str) -> str:
    """Return a stable prompt fingerprint without retaining prompt contents."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Require a string-keyed mapping at a trajectory configuration boundary."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
    return value


def _prompt_rows(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Validate prompts and detach generation settings from prompt text."""

    unknown = sorted(set(config) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValueError("unsupported trajectory config keys: " + ", ".join(unknown))

    raw_prompts = config.get("prompts")
    if isinstance(raw_prompts, (str, bytes)) or not isinstance(raw_prompts, Sequence):
        raise TypeError("trajectory config.prompts must be a sequence of mappings")
    if not raw_prompts:
        raise ValueError("trajectory config.prompts must contain at least one prompt")

    shared_generation = _as_mapping(
        config.get("generation", {}),
        name="trajectory config.generation",
    )
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_row in enumerate(raw_prompts):
        row = _as_mapping(raw_row, name=f"trajectory prompt[{index}]")
        unknown_row = sorted(set(row) - _ALLOWED_PROMPT_KEYS)
        if unknown_row:
            raise ValueError(
                f"unsupported trajectory prompt[{index}] keys: "
                + ", ".join(unknown_row)
            )

        prompt_id = row.get("id")
        prompt = row.get("prompt")
        category = row.get("category")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise TypeError(f"trajectory prompt[{index}].id must be a non-empty string")
        if prompt_id in seen_ids:
            raise ValueError(f"duplicate trajectory prompt id: {prompt_id!r}")
        seen_ids.add(prompt_id)
        if not isinstance(prompt, str):
            raise TypeError(f"trajectory prompt[{index}].prompt must be a string")
        if category is not None and not isinstance(category, str):
            raise TypeError(
                f"trajectory prompt[{index}].category must be a string or null"
            )

        prompt_generation = _as_mapping(
            row.get("generation", {}),
            name=f"trajectory prompt[{index}].generation",
        )
        generation = dict(shared_generation)
        generation.update(prompt_generation)
        rows.append(
            {
                "id": prompt_id,
                "prompt": prompt,
                "category": category,
                "generation": generation,
            }
        )
    return tuple(rows)


def _token_batches(
    result: Any,
    *,
    expected: int,
) -> tuple[tuple[int, ...], ...]:
    """Validate runtime token capture shape before treating it as evidence."""

    if isinstance(result, (str, bytes)) or not isinstance(result, Sequence):
        raise RuntimeError("runtime token_ids capture must be a sequence of sequences")
    if len(result) != expected:
        raise RuntimeError(
            "runtime token_ids capture count does not match prompt count: "
            f"expected {expected}, got {len(result)}"
        )

    batches: list[tuple[int, ...]] = []
    for index, raw_tokens in enumerate(result):
        if isinstance(raw_tokens, (str, bytes)) or not isinstance(raw_tokens, Sequence):
            raise RuntimeError(f"token_ids capture[{index}] must be a token sequence")
        tokens: list[int] = []
        for token in raw_tokens:
            if isinstance(token, bool) or not isinstance(token, int):
                raise RuntimeError(
                    f"token_ids capture[{index}] contains a non-integer token id"
                )
            tokens.append(token)
        batches.append(tuple(tokens))
    return tuple(batches)


def _first_divergence(
    reference: Sequence[int],
    candidate: Sequence[int],
) -> tuple[int | None, int]:
    """Return first divergence and shared prefix length in model-token steps."""

    shared = min(len(reference), len(candidate))
    for index in range(shared):
        if reference[index] != candidate[index]:
            return index, index
    if len(reference) == len(candidate):
        return None, shared
    return shared, shared


def _capture_rows(result: MeasurementResult) -> tuple[Mapping[str, Any], ...]:
    """Validate trajectory measurement evidence before pairwise analysis."""

    evidence = result.evidence
    if evidence.get("schema") != _CAPTURE_SCHEMA:
        raise ValueError("measurement result is not Metria trajectory capture evidence")
    if evidence.get("method") != _CAPTURE_METHOD:
        raise ValueError("trajectory capture method identity does not match")
    if evidence.get("method_version") != _CAPTURE_VERSION:
        raise ValueError("trajectory capture method version does not match")

    raw_rows = evidence.get("prompts")
    if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
        raise ValueError("trajectory evidence prompts must be a sequence")
    if not raw_rows:
        raise ValueError("trajectory evidence must contain at least one prompt")
    declared_prompts = evidence.get("n_prompts")
    if declared_prompts != len(raw_rows):
        raise ValueError(
            "trajectory evidence n_prompts does not match retained prompt rows"
        )

    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(raw_rows):
        row = _as_mapping(raw_row, name=f"trajectory evidence prompt[{index}]")
        prompt_id = row.get("id")
        prompt_hash = row.get("prompt_sha256")
        tokens = row.get("token_ids")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"trajectory evidence prompt[{index}] has invalid id")
        if prompt_id in seen:
            raise ValueError(f"duplicate trajectory evidence prompt id: {prompt_id!r}")
        seen.add(prompt_id)
        if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
            raise ValueError(
                f"trajectory evidence prompt[{index}] has invalid prompt fingerprint"
            )
        validated_tokens = _token_batches((tokens,), expected=1)[0]
        token_count = row.get("token_count")
        if token_count != len(validated_tokens):
            raise ValueError(
                f"trajectory evidence prompt[{index}] token_count does not match token_ids"
            )
        rows.append(row)
    return tuple(rows)


class TokenTrajectoryProtocol:
    """Capture true decode-time model token IDs for a configured prompt set."""

    name = _CAPTURE_METHOD
    version = _CAPTURE_VERSION

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]:
        """Require decode-time token IDs after validating the prompt configuration."""

        _prompt_rows(config)
        return (CaptureRequest(kind="token_ids"),)

    def execute(
        self,
        session: RuntimeSession,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        """Run configured prompts and retain token trajectories without prompt text."""

        del scenario  # the prepared runtime already owns the resolved scenario
        rows = _prompt_rows(config)
        requests = tuple(
            InferenceRequest(prompt=row["prompt"], generation=row["generation"])
            for row in rows
        )
        batch = session.infer(
            requests,
            capture=self.requirements(config),
        )
        if "token_ids" not in batch.captures:
            raise RuntimeError("runtime did not return the required token_ids capture")
        token_batches = _token_batches(batch.captures["token_ids"], expected=len(rows))

        evidence_rows: list[dict[str, Any]] = []
        length_samples: list[MetricSample] = []
        coverage_samples: list[MetricSample] = []
        nonempty = 0
        for row, tokens in zip(rows, token_batches, strict=True):
            prompt_hash = _sha256_text(row["prompt"])
            token_count = len(tokens)
            if token_count:
                nonempty += 1
            metadata = {
                "prompt_id": row["id"],
                "prompt_sha256": prompt_hash,
            }
            length_samples.append(MetricSample(float(token_count), metadata=metadata))
            coverage_samples.append(
                MetricSample(1.0 if token_count else 0.0, metadata=metadata)
            )
            evidence_rows.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "prompt_sha256": prompt_hash,
                    "token_ids": tokens,
                    "token_count": token_count,
                }
            )

        n_prompts = len(rows)
        mean_steps = sum(sample.value for sample in length_samples) / n_prompts
        capture_coverage = nonempty / n_prompts
        metrics = {
            "trajectory_mean_steps": MetricSummary(
                definition=MetricDefinition(
                    name="trajectory_mean_steps",
                    unit="tokens",
                    direction=MetricDirection.DESCRIPTIVE,
                    method=_CAPTURE_METHOD,
                    version=_CAPTURE_VERSION,
                ),
                value=mean_steps,
                samples=tuple(length_samples),
                aggregation="mean",
                coverage=1.0,
            ),
            "trajectory_nonempty_capture_rate": MetricSummary(
                definition=MetricDefinition(
                    name="trajectory_nonempty_capture_rate",
                    unit="fraction",
                    direction=MetricDirection.DESCRIPTIVE,
                    method=_CAPTURE_METHOD,
                    version=_CAPTURE_VERSION,
                ),
                value=capture_coverage,
                samples=tuple(coverage_samples),
                aggregation="mean",
                coverage=1.0,
            ),
        }
        return MeasurementResult(
            metrics=metrics,
            evidence={
                "schema": _CAPTURE_SCHEMA,
                "method": _CAPTURE_METHOD,
                "method_version": _CAPTURE_VERSION,
                "n_prompts": n_prompts,
                "prompts": tuple(evidence_rows),
            },
        )


def compare_trajectory_results(
    reference: MeasurementResult,
    candidate: MeasurementResult,
) -> MeasurementResult:
    """Reproduce KV Fidelity v0.3.4 trajectory agreement from run evidence.

    The score is 100 times the sum of prefix-agreement steps divided by the
    sum of the longer reference/candidate trajectory length for each prompt.
    This symmetrically penalizes unilateral early stopping and continued
    generation while preserving a score of 100 for identical non-empty
    trajectories that end together.
    """

    reference_rows = _capture_rows(reference)
    candidate_rows = _capture_rows(candidate)
    reference_by_id = {str(row["id"]): row for row in reference_rows}
    candidate_by_id = {str(row["id"]): row for row in candidate_rows}
    if set(reference_by_id) != set(candidate_by_id):
        missing_candidate = sorted(set(reference_by_id) - set(candidate_by_id))
        missing_reference = sorted(set(candidate_by_id) - set(reference_by_id))
        raise ValueError(
            "trajectory prompt sets differ; "
            f"missing_from_candidate={missing_candidate}, "
            f"missing_from_reference={missing_reference}"
        )

    comparison_rows: list[dict[str, Any]] = []
    score_samples: list[MetricSample] = []
    match_samples: list[MetricSample] = []
    prefix_samples: list[MetricSample] = []
    first_divergences: list[int] = []
    prefix_steps_total = 0
    comparison_steps_total = 0
    reference_lengths: list[int] = []
    candidate_lengths: list[int] = []
    full_matches = 0

    for reference_row in reference_rows:
        prompt_id = str(reference_row["id"])
        candidate_row = candidate_by_id[prompt_id]
        if reference_row["prompt_sha256"] != candidate_row["prompt_sha256"]:
            raise ValueError(
                f"trajectory prompt fingerprint differs for prompt id {prompt_id!r}"
            )

        reference_tokens = _token_batches(
            (reference_row["token_ids"],),
            expected=1,
        )[0]
        candidate_tokens = _token_batches(
            (candidate_row["token_ids"],),
            expected=1,
        )[0]
        if not reference_tokens and not candidate_tokens:
            raise RuntimeError(
                f"both trajectories are empty for prompt id {prompt_id!r}; "
                "decode-time token capture is not sufficient to score this pair"
            )

        first_divergence, prefix_steps = _first_divergence(
            reference_tokens,
            candidate_tokens,
        )
        matched = first_divergence is None
        if first_divergence is None:
            full_matches += 1
        else:
            first_divergences.append(first_divergence)

        reference_length = len(reference_tokens)
        candidate_length = len(candidate_tokens)
        denominator = max(reference_length, candidate_length)
        prompt_score = 100.0 * prefix_steps / denominator
        prefix_steps_total += prefix_steps
        comparison_steps_total += denominator
        reference_lengths.append(reference_length)
        candidate_lengths.append(candidate_length)

        sample_metadata = {
            "prompt_id": prompt_id,
            "prompt_sha256": reference_row["prompt_sha256"],
            "comparison_steps": denominator,
            "prefix_steps": prefix_steps,
        }
        score_samples.append(MetricSample(prompt_score, metadata=sample_metadata))
        match_samples.append(
            MetricSample(1.0 if matched else 0.0, metadata=sample_metadata)
        )
        prefix_samples.append(
            MetricSample(float(prefix_steps), metadata=sample_metadata)
        )
        comparison_rows.append(
            {
                "id": prompt_id,
                "prompt_sha256": reference_row["prompt_sha256"],
                "first_divergence": first_divergence,
                "prefix_agreement_steps": prefix_steps,
                "reference_steps": reference_length,
                "candidate_steps": candidate_length,
                "matched": matched,
            }
        )

    n_prompts = len(reference_rows)
    score = 100.0 * prefix_steps_total / comparison_steps_total
    score = max(0.0, min(100.0, score))
    full_match_rate = full_matches / n_prompts
    mean_prefix_steps = prefix_steps_total / n_prompts
    median_first_divergence = (
        statistics.median(first_divergences) if first_divergences else None
    )

    notes: list[str] = []
    length_mismatches = sum(
        1
        for reference_length, candidate_length in zip(
            reference_lengths,
            candidate_lengths,
            strict=True,
        )
        if reference_length != candidate_length
    )
    if length_mismatches:
        notes.append(
            f"{length_mismatches}/{n_prompts} reference/candidate trajectory "
            "lengths differed; each prompt is normalized by the longer trajectory"
        )

    metrics = {
        "trajectory_agreement_score": MetricSummary(
            definition=MetricDefinition(
                name="trajectory_agreement_score",
                unit="score_0_100",
                direction=MetricDirection.HIGHER_IS_BETTER,
                method=_COMPARISON_METHOD,
                version=_COMPARISON_VERSION,
            ),
            value=score,
            samples=tuple(score_samples),
            aggregation="weighted_by_max_trajectory_length",
            coverage=1.0,
        ),
        "trajectory_full_match_rate": MetricSummary(
            definition=MetricDefinition(
                name="trajectory_full_match_rate",
                unit="fraction",
                direction=MetricDirection.HIGHER_IS_BETTER,
                method=_COMPARISON_METHOD,
                version=_COMPARISON_VERSION,
            ),
            value=full_match_rate,
            samples=tuple(match_samples),
            aggregation="mean",
            coverage=1.0,
        ),
        "trajectory_mean_prefix_steps": MetricSummary(
            definition=MetricDefinition(
                name="trajectory_mean_prefix_steps",
                unit="tokens",
                direction=MetricDirection.DESCRIPTIVE,
                method=_COMPARISON_METHOD,
                version=_COMPARISON_VERSION,
            ),
            value=mean_prefix_steps,
            samples=tuple(prefix_samples),
            aggregation="mean",
            coverage=1.0,
        ),
    }
    return MeasurementResult(
        metrics=metrics,
        evidence={
            "schema": "metria.trajectory_comparison.v1",
            "method": _COMPARISON_METHOD,
            "method_version": _COMPARISON_VERSION,
            "n_prompts": n_prompts,
            "median_first_divergence": median_first_divergence,
            "all_trajectories_match": full_matches == n_prompts,
            "mean_reference_steps": sum(reference_lengths) / n_prompts,
            "mean_candidate_steps": sum(candidate_lengths) / n_prompts,
            "comparison_steps": comparison_steps_total,
            "prefix_agreement_steps": prefix_steps_total,
            "per_prompt": tuple(comparison_rows),
            "notes": tuple(notes),
        },
    )
