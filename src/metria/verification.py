"""One local reference/candidate verification with durable evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from ._freeze import freeze_mapping
from .capability_checks import CapabilityCheckRegistry
from .comparison import compare_runs
from .hardware import capture_hardware_fingerprint
from .inspection import resolve_capability_checks
from .measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from .models import CompatibilityReport, RunRecord, RunStatus
from .policies import PolicyDecision, evaluate_policy, render_policy_evaluation
from .protocols import (
    InferenceRequest,
    MeasurementProtocol,
    MeasurementResult,
    PairwiseAnalysis,
    RuntimeAdapter,
)
from .recipes import StudyRecipe, _json_value, study_recipe_digest
from .records import (
    _metric_summary_to_data,
    run_evidence_digest,
    run_record_digest,
    run_record_to_json,
)
from .runtimes.llamacpp import (
    LlamaCppAdapter,
    _generation_options,
    _requested_model_sha256,
    _runtime_config,
)
from .study_execution import PairwiseAnalysisStatus, StudyPairAnalysis, execute_study

VERIFICATION_SCHEMA = "metria.verification.v1"
VERIFICATION_SCOPE = "local_llamacpp_cpu_threads.v1"
_ROLES = ("reference", "candidate")
_CAPTURE_KEY = "llama_cpp_token_ids_capture_sha256"


class VerificationVerdict(str, Enum):
    """Distinct outcomes; VERIFIED denotes completed comparison, not task quality."""

    VERIFIED = "VERIFIED"
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    EXECUTION_FAILED = "EXECUTION_FAILED"


@dataclass(frozen=True)
class VerificationResult:
    """The saved verification result and its process exit status."""

    output_dir: Path
    manifest: Mapping[str, Any]
    exit_code: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))

    def to_data(self) -> dict[str, Any]:
        return dict(_json_value(self.manifest, path="verification"))


@dataclass(frozen=True)
class _Registries:
    adapters: Mapping[str, RuntimeAdapter]
    measurements: Mapping[str, MeasurementProtocol]
    analyses: Mapping[str, PairwiseAnalysis]


def _builtin_registries() -> _Registries:
    measurement = TokenTrajectoryProtocol()
    analysis = TrajectoryAgreementAnalysis()
    return _Registries(
        {"llamacpp": LlamaCppAdapter()},
        {measurement.name: measurement},
        {analysis.name: analysis},
    )


def _validate(recipe: StudyRecipe, registries: _Registries) -> None:
    if len(recipe.study.runs) != 2:
        raise ValueError("verify requires exactly two runs: reference, then candidate")
    if not recipe.study.comparison.vary:
        raise ValueError(
            "verify requires an explicit intended change in comparison.vary"
        )
    if recipe.study.comparison.analyses != (TrajectoryAgreementAnalysis.name,):
        raise ValueError(
            "local verify requires the kv_fidelity.trajectory_match analysis"
        )
    if any(
        name not in registries.analyses for name in recipe.study.comparison.analyses
    ):
        raise ValueError("verification analysis is not registered")
    if set(recipe.environment) - {_CAPTURE_KEY, "llama_cpp_bin_dir"}:
        raise ValueError(
            "local verify environment accepts only the binary directory and qualified capture digest"
        )
    digest = recipe.environment.get(_CAPTURE_KEY)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in digest)
    ):
        raise ValueError(f"local verify requires environment.{_CAPTURE_KEY}")
    if set(recipe.measurement_configs) != {TokenTrajectoryProtocol.name}:
        raise ValueError(
            "local verify requires exactly one trajectory measurement configuration"
        )
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    pins = []
    for index, run in enumerate(recipe.study.runs):
        name = run.runtime.get("name")
        if name not in registries.adapters:
            raise ValueError(f"run[{index}] runtime is not registered for verification")
        if name != "llamacpp":
            raise ValueError("the first verify workflow supports local llama.cpp only")
        if (
            run.measurements != (TokenTrajectoryProtocol.name,)
            or TokenTrajectoryProtocol.name not in registries.measurements
        ):
            raise ValueError(
                "local verify requires the registered decode-time trajectory measurement"
            )
        if run.treatments or run.trial_policy or run.environment_selector:
            raise ValueError(
                "local CPU verification does not yet apply treatments, trial policies, or environment selectors"
            )
        if set(run.runtime) - {
            "name",
            "bin_dir",
            "n_gpu_layers",
            "flash_attention",
            "threads",
            "threads_batch",
            "extra_args",
        }:
            raise ValueError("unsupported local verification runtime fields")
        if set(run.model) - {"path", "sha256", "id", "revision", "geometry"}:
            raise ValueError("unsupported local verification model fields")
        if "system" in run.scenario:
            raise ValueError(
                "the plain-completion verifier does not accept system prompts"
            )
        if set(run.scenario) - {
            "name",
            "context",
            "max_tokens",
            "seed",
            "temperature",
            "timeout",
            "chat_template",
            "reasoning",
        }:
            raise ValueError("unsupported local verification scenario fields")
        runtime = _runtime_config(run)
        if runtime["n_gpu_layers"] != 0 or runtime["extra_args"]:
            raise ValueError("local verify requires n_gpu_layers=0 and no extra_args")
        if runtime["threads"] is None or runtime["threads_batch"] is None:
            raise ValueError("local verify requires explicit threads and threads_batch")
        pin = _requested_model_sha256(run)
        if pin is None:
            raise ValueError(
                "local verify requires model.sha256 from a trusted artifact manifest"
            )
        pins.append(pin)
        registries.measurements[TokenTrajectoryProtocol.name].requirements(config)
        for row in config["prompts"]:
            generation = {**config.get("generation", {}), **row.get("generation", {})}
            options = _generation_options(
                InferenceRequest(row["prompt"], generation), run.scenario
            )
            if (
                options["temperature"] != 0.0
                or options["chat_template"]
                or options["system"]
            ):
                raise ValueError(
                    "local verify requires greedy plain completion: temperature=0 and chat_template=false"
                )
            if options["max_tokens"] <= 0:
                raise ValueError("local verify requires a positive generation length")
    if pins[0] != pins[1]:
        raise ValueError(
            "the first local verifier requires the same pinned model for both runs"
        )
    left, right = recipe.study.runs
    if left.model != right.model or left.scenario != right.scenario:
        raise ValueError(
            "local CPU thread verification requires identical model and scenario inputs"
        )
    left_runtime = {
        key: value for key, value in left.runtime.items() if key != "threads"
    }
    right_runtime = {
        key: value for key, value in right.runtime.items() if key != "threads"
    }
    if left_runtime != right_runtime:
        raise ValueError(
            "the first local verifier permits only runtime.threads to change"
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _invocations(record: RunRecord) -> Sequence[Any]:
    value = record.observed.get("invocations", ())
    return (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else ()
    )


def _observed_facts(record: RunRecord) -> dict[str, Any]:
    identity = _mapping(record.observed.get("identity"))
    fields = _mapping(_mapping(identity.get("applied")).get("fields"))
    return {
        "model_sha256": _mapping(identity.get("model")).get("sha256"),
        "provider_sha256": _mapping(identity.get("runtime")).get("completion_sha256"),
        **{
            name: fields.get(name)
            for name in (
                "threads",
                "threads_batch",
                "context",
                "vocab_size",
                "chat_template_applied",
            )
        },
    }


def _evidence_gaps(record: RunRecord, expected_provider: str) -> tuple[str, ...]:
    gaps: list[str] = []
    identity = _mapping(record.observed.get("identity"))
    model = _mapping(identity.get("model"))
    if identity.get("status") == "mismatch":
        gaps.append("runtime identity is inconsistent or mismatched")
    expected_model = record.requested.model.get("sha256", "")
    if (
        model.get("status") != "verified"
        or model.get("sha256") != str(expected_model).lower()
    ):
        gaps.append("pinned model content was not verified")
    runtime = _mapping(identity.get("runtime"))
    if runtime.get("completion_sha256") != expected_provider.lower():
        gaps.append("runtime does not match the qualified capture provider")
    invocations = _invocations(record)
    if not invocations:
        gaps.append("no runtime invocation evidence was retained")
    for invocation in invocations:
        capture = (
            invocation.get("runtime_capture")
            if isinstance(invocation, Mapping)
            else None
        )
        if not isinstance(capture, Mapping):
            gaps.append("structured runtime readback is missing")
            continue
        for name in ("threads", "threads_batch"):
            if capture.get(name) != record.requested.runtime.get(name):
                gaps.append(f"runtime-applied {name} does not match the request")
        generation = _mapping(invocation.get("generation"))
        if capture.get("context") != generation.get("context"):
            gaps.append("runtime-applied context does not match the request")
        if capture.get("chat_template_applied") is not False:
            gaps.append("plain completion was not confirmed by the runtime")
    coverage = record.metrics.get("trajectory_nonempty_capture_rate")
    if coverage is None or isinstance(coverage.value, bool) or coverage.value != 1.0:
        gaps.append("every prompt must retain non-empty sampled token IDs")
    return tuple(dict.fromkeys(gaps))


class _CheckedAnalysis:
    def __init__(self, analysis: PairwiseAnalysis, provider: str) -> None:
        self.name, self.version = analysis.name, analysis.version
        self._analysis, self._provider = analysis, provider

    def analyze(self, left: RunRecord, right: RunRecord) -> MeasurementResult:
        if _evidence_gaps(left, self._provider) or _evidence_gaps(
            right, self._provider
        ):
            raise ValueError(
                "verification evidence is insufficient for behavioral analysis"
            )
        return self._analysis.analyze(left, right)


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    created = False
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(text)
        if path.exists():
            raise FileExistsError(f"verification artifact already exists: {path.name}")
        temporary.replace(path)
    finally:
        if created:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _comparison_data(report: CompatibilityReport) -> dict[str, Any]:
    # Values may contain private configuration. Keep paths and reasons only.
    return {
        "compatible": report.compatible,
        "issues": [
            {"dimension": issue.dimension, "reason": issue.reason}
            for issue in report.issues
        ],
        "waived_differences": [
            {"dimension": issue.dimension, "reason": issue.reason}
            for issue in report.waived_differences
        ],
        "comparable_metrics": list(report.comparable_metrics),
        "incompatible_metrics": dict(report.incompatible_metrics),
    }


def _analysis_data(outcome: StudyPairAnalysis) -> dict[str, Any]:
    result = outcome.result
    metrics: dict[str, Any] = {}
    metric_errors: dict[str, Any] = {}
    if result is not None:
        for key, value in result.metrics.items():
            try:
                metrics[key] = _metric_summary_to_data(value, key=key)
            except (TypeError, ValueError, AttributeError) as exc:
                metric_errors[key] = {
                    "error_type": type(exc).__name__,
                    "message_sha256": hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                }
    return {
        "name": outcome.name,
        "version": outcome.version,
        "status": outcome.status.value,
        "reason": outcome.reason,
        "error_type": outcome.error_type,
        "message_sha256": outcome.message_sha256,
        "metrics": metrics,
        "metric_errors": metric_errors,
        "diagnostics": {}
        if result is None
        else _json_value(result.evidence, path="analysis.evidence"),
    }


def _wall_time(record: RunRecord) -> dict[str, Any]:
    values: list[float] = []
    for row in _invocations(record):
        value = _mapping(row).get("duration_seconds")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            return {"available": False}
        values.append(float(value))
    if not values:
        return {"available": False}
    return {
        "available": True,
        "method": "llamacpp.subprocess_monotonic.v1",
        "includes": "process startup, model loading, prompt evaluation and generation",
        "samples_seconds": values,
        "mean_seconds": sum(values) / len(values),
    }


def render_verification(manifest: Mapping[str, Any]) -> str:
    """Render a concise report without raw prompts or configuration values."""
    lines = [
        f"Metria verification: {manifest['verdict']}",
        "",
        f"Study: {manifest['study']}",
        "Scope: local llama.cpp CPU thread comparison",
        "",
        "Change:",
        f"  CPU threads: {manifest['change']['reference_threads']} -> {manifest['change']['candidate_threads']}",
        "",
        "Evidence:",
    ]
    for role, record in manifest["records"].items():
        lines.append(f"  {role}: {record['status']} ({record['path']})")
        facts = record["observed"]
        lines.append(
            f"    Observed threads: {facts['threads']}; context: {facts['context']}"
        )
        for gap in record["evidence_gaps"]:
            lines.append(f"    {gap}")
    for issue in manifest["comparison"]["issues"]:
        lines.append(f"  {issue['dimension']}: {issue['reason']}")
    lines.extend(("", "Impact:"))
    for analysis in manifest["analyses"]:
        lines.append(f"  {analysis['name']}: {analysis['status']}")
        metrics = analysis["metrics"]
        for key, error in analysis.get("metric_errors", {}).items():
            lines.append(f"    {key}: invalid metric evidence ({error['error_type']})")
        for key, unit, label, scale, suffix in (
            (
                "trajectory_agreement_score",
                "score_0_100",
                "Token prefix agreement",
                1,
                "/100",
            ),
            (
                "trajectory_full_match_rate",
                "fraction",
                "Exact token-sequence matches",
                100,
                "%",
            ),
        ):
            if key not in metrics:
                continue
            metric = metrics[key]
            expected = {
                "name": key,
                "unit": unit,
                "direction": "higher_is_better",
                "method": TrajectoryAgreementAnalysis.name,
                "version": TrajectoryAgreementAnalysis.version,
            }
            if metric["definition"] != expected:
                lines.append(
                    f"    {key}: unavailable for interpretation (unexpected metric identity)"
                )
            else:
                lines.append(f"    {label}: {metric['value'] * scale:.6g}{suffix}")
        if "per_prompt" in analysis["diagnostics"]:
            diverged = [
                row
                for row in analysis["diagnostics"]["per_prompt"]
                if not row.get("matched")
            ]
            lines.append(f"    Divergent prompts: {len(diverged)}")
            for row in diverged[:10]:
                lines.append(
                    f"      {row['id']}: first divergence at token {row['first_divergence']}"
                )
    for role, timing in manifest["systems"].items():
        if timing.get("available"):
            lines.append(
                f"  {role} mean process wall time: {timing['mean_seconds']:.6g}s (includes startup and model loading)"
            )
    lines.extend(("", "Verdict:", f"  {manifest['verdict']}"))
    if "policy" in manifest:
        lines.extend(render_policy_evaluation(manifest["policy"]))
    else:
        lines.extend(
            (
                "  VERIFIED means comparison and analysis completed within the stated scope.",
                "  No task-quality or performance acceptance policy was evaluated.",
            )
        )
    lines.append("")
    return "\n".join(lines)


def verify_recipe(
    recipe: StudyRecipe,
    output_dir: str | Path = "verification",
    *,
    capability_checks: CapabilityCheckRegistry | None = None,
) -> VerificationResult:
    """Execute the first local CPU verifier and persist each run immediately.

    The output directory must be new. Completed files survive later execution
    interruption or write failure; the manifest is written last and is never
    presented as complete when persistence failed.
    """
    resolve_capability_checks(capability_checks)
    registries = _builtin_registries()
    _validate(recipe, registries)
    recipe_digest = study_recipe_digest(recipe)
    hardware = capture_hardware_fingerprint().to_mapping()
    from . import __version__

    context = {
        "recipe_digest": recipe_digest,
        "scope": VERIFICATION_SCOPE,
        "implementation": {"name": "metria.verify", "version": __version__},
        "hardware": hardware,
    }
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=False)
    saved: list[RunRecord] = []

    def save_record(record: RunRecord) -> None:
        decorated = replace(
            record, provenance={**record.provenance, "verification": context}
        )
        _write_atomic(
            output / f"{_ROLES[len(saved)]}.run.json",
            run_record_to_json(decorated) + "\n",
        )
        saved.append(decorated)

    provider = str(recipe.environment[_CAPTURE_KEY])
    analyses = {
        name: _CheckedAnalysis(analysis, provider)
        for name, analysis in registries.analyses.items()
    }
    interrupted = False
    try:
        execution = execute_study(
            recipe.study,
            adapters=registries.adapters,
            measurements=registries.measurements,
            measurement_configs=recipe.measurement_configs,
            environment=recipe.environment,
            analyses=analyses,
            record_sink=save_record,
            capability_checks=capability_checks,
        )
        comparison = execution.comparisons[0].report
        outcomes = execution.comparisons[0].analyses
    except KeyboardInterrupt:
        interrupted = True
        first_missing = len(saved)
        for index in range(first_missing, 2):
            save_record(
                RunRecord(
                    study_name=recipe.study.name,
                    run_id=f"run-{index:04d}",
                    requested=recipe.study.runs[index],
                    resolved={},
                    observed={},
                    status=RunStatus.INTERRUPTED
                    if index == first_missing
                    else RunStatus.CANCELLED,
                    events=({"stage": "verification", "kind": "interrupted"},),
                )
            )
        comparison = compare_runs(saved[0], saved[1], recipe.study.comparison)
        outcomes = ()

    analysis_data = [_analysis_data(outcome) for outcome in outcomes]
    gaps = [_evidence_gaps(record, provider) for record in saved]
    failed = {
        RunStatus.FAILED,
        RunStatus.TIMED_OUT,
        RunStatus.INTERRUPTED,
        RunStatus.CANCELLED,
    }
    if interrupted or any(record.status in failed for record in saved):
        verdict = VerificationVerdict.EXECUTION_FAILED
    elif any(record.status is not RunStatus.COMPLETED for record in saved) or any(gaps):
        verdict = VerificationVerdict.INSUFFICIENT_EVIDENCE
    elif not comparison.compatible:
        verdict = VerificationVerdict.NOT_COMPARABLE
    elif not outcomes or any(
        outcome.status is not PairwiseAnalysisStatus.COMPLETED for outcome in outcomes
    ):
        verdict = VerificationVerdict.EXECUTION_FAILED
    elif any(analysis["metric_errors"] for analysis in analysis_data):
        verdict = VerificationVerdict.INSUFFICIENT_EVIDENCE
    else:
        verdict = VerificationVerdict.VERIFIED
    policy_result = None
    if recipe.policy is not None:
        policy_result = evaluate_policy(
            recipe.policy, outcomes, verification_status=verdict.value
        )
        if policy_result.status is not PolicyDecision.NOT_EVALUATED:
            verdict = VerificationVerdict(policy_result.status.value)
    manifest = {
        "schema": VERIFICATION_SCHEMA,
        "scope": VERIFICATION_SCOPE,
        "study": recipe.study.name,
        "recipe_digest": recipe_digest,
        "implementation": context["implementation"],
        "hardware": hardware,
        "verdict": verdict.value,
        "acceptance_policy_evaluated": policy_result is not None
        and policy_result.status is not PolicyDecision.NOT_EVALUATED,
        "change": {
            "reference_threads": recipe.study.runs[0].runtime["threads"],
            "candidate_threads": recipe.study.runs[1].runtime["threads"],
        },
        "comparison_plan": {
            "vary": sorted(recipe.study.comparison.vary),
            "control": sorted(recipe.study.comparison.control),
            "block_by": sorted(recipe.study.comparison.block_by),
        },
        "records": {
            role: {
                "path": f"{role}.run.json",
                "run_id": record.run_id,
                "status": record.status.value,
                "record_digest": run_record_digest(record),
                "evidence_digest": run_evidence_digest(record),
                "evidence_gaps": list(gaps[index]),
                "observed": _observed_facts(record),
            }
            for index, (role, record) in enumerate(zip(_ROLES, saved, strict=True))
        },
        "comparison": _comparison_data(comparison),
        "analyses": analysis_data,
        "systems": {
            role: _wall_time(record) for role, record in zip(_ROLES, saved, strict=True)
        },
    }
    if policy_result is not None:
        manifest["policy"] = policy_result.to_data()
    _write_atomic(output / "report.md", render_verification(manifest))
    _write_atomic(
        output / "manifest.json",
        json.dumps(
            _json_value(manifest, path="verification"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    return VerificationResult(
        output,
        manifest,
        130
        if interrupted
        else (
            0
            if verdict in {VerificationVerdict.VERIFIED, VerificationVerdict.PASS}
            else 1
        ),
    )
