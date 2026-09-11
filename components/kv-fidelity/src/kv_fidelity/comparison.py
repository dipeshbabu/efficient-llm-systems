"""Adapt focused KV reports to Metria's shared comparison semantics."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, TypeGuard

from metria import (
    ComparisonPlan,
    MetricDefinition,
    MetricDirection,
    MetricSummary,
    RunRecord,
    RunSpec,
    RunStatus,
    compare_runs,
)

from . import __report_schema__
from .runner import KVConfig

EVIDENCE_SCHEMA = "kv_fidelity.comparison_evidence.v1"
COMPARISON_SCHEMA = "kv_fidelity.comparison.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_AXES = ("gtm", "kld", "rniah", "plad")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip()) and value != "unknown"


def _number(value: Any) -> TypeGuard[int | float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def report_digest(report: Mapping[str, Any]) -> str:
    """Identify report contents; this digest makes no comparability claim."""
    return hashlib.sha256(_canonical(report).encode("ascii")).hexdigest()


def _artifact(value: Any) -> dict[str, Any]:
    data = _mapping(value)
    digest = data.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        return {}
    result: dict[str, Any] = {"sha256": digest}
    size = data.get("size_bytes")
    if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
        result["size_bytes"] = size
    if _string(data.get("kind")):
        result["kind"] = data["kind"]
    return result


def _metric(axis: str, data: Mapping[str, Any], version: str) -> MetricSummary | None:
    score = data.get("score")
    method = data.get("method")
    method_version = data.get("method_version")
    if (
        not _number(score)
        or not 0 <= score <= 100
        or not _string(method)
        or not _string(method_version)
    ):
        return None
    identity: dict[str, Any] = {"name": method}
    if axis == "kld":
        metadata = _mapping(data.get("metadata"))
        if not _string(metadata.get("kld_estimator")) or not isinstance(
            metadata.get("full_vocabulary"), bool
        ):
            return None
        identity.update(
            {
                "estimator": metadata["kld_estimator"],
                "full_vocabulary": metadata["full_vocabulary"],
            }
        )
        if not metadata["full_vocabulary"]:
            topk = metadata.get("topk")
            if not isinstance(topk, int) or isinstance(topk, bool) or topk <= 0:
                return None
            identity["topk"] = topk
        # These parameters change the estimator, rather than its observed value.
        for key in ("log_floor", "position_alignment", "normalization", "other_bucket"):
            if key in metadata:
                identity[key] = metadata[key]
    return MetricSummary(
        definition=MetricDefinition(
            name=axis,
            unit="score_0_100",
            direction=MetricDirection.HIGHER_IS_BETTER,
            method=_canonical(identity),
            version=f"{version}/{method_version}",
        ),
        value=float(score),
    )


def report_to_run_record(report: Mapping[str, Any]) -> RunRecord:
    """Project retained evidence without promoting requested state to observed.

    Legacy reports remain representable. Missing identities are omitted so the
    shared comparison plan rejects them, including when both reports omit them.
    The original report is retained in evidence for schema migration/auditing.
    """
    if not isinstance(report, Mapping):
        raise ValueError("a KV Fidelity report must be a JSON object")
    digest = report_digest(report)
    extras = _mapping(report.get("extras"))
    capture = _mapping(extras.get("comparison_evidence"))
    capture_known = capture.get("schema") == EVIDENCE_SCHEMA
    input_issues = []
    if not capture_known:
        input_issues.append("comparison evidence schema is missing or unsupported")
    if not isinstance(capture.get("changed_inputs"), list):
        input_issues.append("input identity capture was not finalized")
    elif capture["changed_inputs"]:
        input_issues.append(
            "selected inputs changed during scoring: "
            + ", ".join(map(str, capture["changed_inputs"]))
        )
    requested = _mapping(capture.get("requested")) if capture_known else {}
    resolved_input = _mapping(capture.get("resolved")) if capture_known else {}
    observed = _mapping(capture.get("observed")) if capture_known else {}
    environment = _mapping(report.get("environment"))
    # Existing backend metadata is retained at its actual reporting level.
    if environment:
        observed["backend_metadata"] = {
            key: value
            for key, value in environment.items()
            if key not in {"model", "llama_cpp_bin_dir"}
        }

    hardware = _mapping(observed.get("hardware"))
    host = _mapping(hardware.get("host"))
    platform = _mapping(hardware.get("platform"))
    hostname_digest = host.get("hostname_sha256")
    if (
        not isinstance(hostname_digest, str)
        or not _SHA256.fullmatch(hostname_digest)
        or not _string(platform.get("system"))
        or not _string(platform.get("machine"))
    ):
        observed.pop("hardware", None)
    elif not hardware.get("accelerators"):
        hardware.pop("accelerators", None)
        observed["hardware"] = hardware

    model = {"locator": report["model"]} if _string(report.get("model")) else {}
    runtime = _mapping(requested.get("runtime"))
    if runtime.get("unverified_argument_overrides"):
        input_issues.append(
            "KV_FIDELITY_LLAMA_EXTRA_FLAGS prevents verification of the selected inputs; remove it or use an inspection override"
        )
    settings = _mapping(runtime.get("settings"))
    gpu_layers = settings.get("n_gpu_layers")
    if not isinstance(gpu_layers, int) or isinstance(gpu_layers, bool):
        settings.pop("n_gpu_layers", None)
    runtime["settings"] = settings
    if not isinstance(runtime.get("environment_overrides"), Mapping):
        runtime.pop("environment_overrides", None)
    if _string(environment.get("backend")):
        runtime.setdefault("backend", environment["backend"])
        if runtime.get("backend") != environment["backend"]:
            input_issues.append("requested and reported backend identities differ")
    else:
        input_issues.append("reported backend identity is missing")
    for role in ("reference", "candidate"):
        value = report.get(role)
        try:
            if not _string(value):
                raise ValueError("missing KV configuration")
            keys = {part.split("=", 1)[0].strip() for part in value.split(",")}
            if not {"ctk", "ctv"} <= keys:
                raise ValueError("missing cache type")
            config = KVConfig.parse(value)
            if not config.ctk or not config.ctv:
                raise ValueError("empty cache type")
            known = asdict(config)
            known.pop("extras")
            runtime[f"{role}_kv"] = known
            if config.extras:
                input_issues.append(
                    f"{role} KV configuration contains unverified extra command arguments"
                )
        except (TypeError, ValueError):
            input_issues.append(f"{role} KV configuration is missing or invalid")
            runtime.pop(f"{role}_kv", None)

    scenario = _mapping(requested.get("scenario"))
    generation = _mapping(scenario.get("generation"))
    for key in ("ctx", "n_predict", "chunks"):
        value = generation.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            generation.pop(key, None)
    if not _number(generation.get("temperature")):
        generation.pop("temperature", None)
    scenario["generation"] = generation
    for name, value in (
        ("schema", report.get("schema")),
        ("suite_version", report.get("framework_version")),
        ("evidence_schema", capture.get("schema")),
    ):
        if _string(value):
            scenario[name] = value
    resolved: dict[str, Any] = {}
    for name in ("model", "tokenizer", "suite"):
        identity = _artifact(resolved_input.get(name))
        if identity:
            resolved[name] = identity
    artifacts = _mapping(_mapping(resolved_input.get("runtime")).get("artifacts"))
    if artifacts and all(_artifact(value) for value in artifacts.values()):
        resolved["runtime"] = {
            "artifacts": {
                name: _artifact(value) for name, value in sorted(artifacts.items())
            }
        }
    inputs = {
        name: _artifact(value)
        for name, value in _mapping(resolved_input.get("inputs")).items()
    }
    resolved["inputs"] = {name: value for name, value in inputs.items() if value}

    axes = _mapping(report.get("axes"))
    metrics: dict[str, MetricSummary] = {}
    active = []
    methods: dict[str, Any] = {}
    contexts: dict[str, Any] = {}
    version = str(report.get("framework_version", "unknown"))
    version += "/" + resolved.get("suite", {}).get("sha256", "unknown-build")
    for axis in _AXES:
        data = _mapping(axes.get(axis))
        if data and not data.get("skipped", False) and data.get("score") is not None:
            active.append(axis)
            contexts[axis] = {
                key: data[key]
                for key in (
                    "n_prompts",
                    "n_tokens_each",
                    "ctx",
                    "chunks",
                    "n_cells",
                    "skipped_cells",
                    "confidence",
                    "excluded_from_composite",
                    "skipped_perturbations",
                )
                if key in data
            }
            metadata = _mapping(data.get("metadata"))
            coverage = {
                key: metadata[key]
                for key in (
                    "n_positions_total",
                    "n_positions_scored",
                    "n_positions_skipped",
                    "position_coverage",
                    "partial_positions",
                )
                if key in metadata
            }
            if coverage:
                contexts[axis]["coverage"] = coverage
            metric = _metric(axis, data, version)
            if metric is not None:
                metrics[axis] = metric
                methods[axis] = {
                    "method": metric.definition.method,
                    "version": metric.definition.version,
                }
    scenario["metric_methods"] = methods
    scenario["metric_context"] = contexts
    # A composite depends on both the scoring method and the actual axis mask.
    contributors = [
        axis
        for axis in active
        if not _mapping(axes[axis]).get("excluded_from_composite", False)
    ]
    composite = report.get("composite")
    if (
        contributors
        and all(axis in metrics for axis in contributors)
        and _number(composite)
        and 0 <= composite <= 100
    ):
        metrics["composite"] = MetricSummary(
            definition=MetricDefinition(
                name="composite",
                unit="score_0_100",
                direction=MetricDirection.HIGHER_IS_BETTER,
                method=_canonical(
                    {
                        "name": "kv_fidelity.harmonic_mean",
                        "axes": {axis: methods[axis] for axis in contributors},
                    }
                ),
                version=version,
            ),
            value=float(composite),
        )
        scenario["composite_method"] = metrics["composite"].definition.method
    scenario["composite_axes"] = contributors
    trial_policy = _mapping(requested.get("trial_policy"))
    if not isinstance(trial_policy.get("seed"), int) or isinstance(
        trial_policy.get("seed"), bool
    ):
        trial_policy.pop("seed", None)
    status = RunStatus.COMPLETED
    if report.get("schema") != __report_schema__:
        input_issues.append("report schema is missing or unsupported")
    if any(
        _mapping(axes.get(axis)).get("confidence") in {"low", "partial"}
        for axis in active
    ):
        input_issues.append("one or more axes contain incomplete or excluded evidence")
    if input_issues or not active or set(axes) - set(_AXES):
        status = RunStatus.PARTIAL
    return RunRecord(
        study_name="kv-fidelity-reports",
        run_id=digest,
        requested=RunSpec(
            model=model,
            runtime=runtime,
            scenario=scenario,
            measurements=tuple(active),
            trial_policy=trial_policy,
        ),
        resolved=resolved,
        observed=observed,
        status=status,
        metrics=metrics,
        evidence={"kv_fidelity_report": dict(report), "input_issues": input_issues},
        provenance={"adapter_schema": COMPARISON_SCHEMA, "report_sha256": digest},
    )


def _plan(left: RunRecord, right: RunRecord) -> ComparisonPlan:
    control = {
        "runtime.backend",
        "runtime.reference_kv",
        "runtime.settings.n_gpu_layers",
        "runtime.environment_overrides",
        "scenario.schema",
        "scenario.suite_version",
        "scenario.evidence_schema",
        "scenario.generation.ctx",
        "scenario.generation.n_predict",
        "scenario.generation.chunks",
        "scenario.generation.temperature",
        "scenario.composite_axes",
        "scenario.composite_method",
        "scenario.metric_context",
        "trial_policy.seed",
        "measurements",
        "resolved.model.sha256",
        "resolved.tokenizer.sha256",
        "resolved.suite.sha256",
        "resolved.runtime.artifacts",
    }
    axes = set(left.requested.measurements) | set(right.requested.measurements)
    for axis in axes:
        control.add(f"scenario.metric_methods.{axis}")
    roles = set()
    if axes & {"gtm", "plad"}:
        roles.add("prompts")
    if "kld" in axes:
        roles.add("corpus")
    if "rniah" in axes:
        roles.add("rniah_haystack")
    control.update(f"resolved.inputs.{role}.sha256" for role in roles)
    for record in (left, right):
        if record.requested.runtime.get("backend") == "llamacpp":
            required = set()
            selected = set(record.requested.measurements)
            if "kld" in selected:
                required.add("llama-perplexity")
            if selected & {"rniah", "plad"}:
                required.update({"llama-cli", "llama-tokenize"})
            if "gtm" in selected:
                metric = record.metrics.get("gtm")
                method = (
                    json.loads(metric.definition.method).get("name")
                    if metric is not None
                    else None
                )
                if method == "trajectory_prefix_agreement":
                    required.add("llama-completion")
                else:
                    required.update({"llama-cli", "llama-tokenize"})
            control.update(
                f"resolved.runtime.artifacts.{name}.sha256" for name in required
            )
    if any(
        record.requested.runtime.get("settings", {}).get("n_gpu_layers") != 0
        for record in (left, right)
    ):
        control.add("observed.hardware.accelerators")
    # Raw output-dependent metadata never enters the comparison roots. Host and
    # reported backend identity do; differences remain explicit blocking facts.
    return ComparisonPlan(
        vary=frozenset({"runtime.candidate_kv"}),
        control=frozenset(control),
        block_by=frozenset(
            {
                "observed.hardware.host",
                "observed.hardware.platform",
                "observed.hardware.software",
                "observed.backend_metadata",
            }
        ),
        waivers={
            "model.locator": "file locations may differ; resolved model and tokenizer content must match"
        },
    )


def compare_reports(
    reports: Sequence[Mapping[str, Any]], *, override_reason: str | None = None
) -> dict[str, Any]:
    """Compare all pairs through Metria; an override never changes compatibility."""
    if len(reports) < 2:
        raise ValueError("comparison requires at least two valid reports")
    if override_reason is not None and not override_reason.strip():
        raise ValueError("an incompatible comparison override requires a reason")
    records = [report_to_run_record(report) for report in reports]
    pairs = []
    for (left_index, left), (right_index, right) in itertools.combinations(
        enumerate(records), 2
    ):
        result = compare_runs(left, right, _plan(left, right))
        pairs.append(
            {
                "left": left_index,
                "right": right_index,
                "compatible": result.compatible,
                "issues": [
                    {
                        "dimension": issue.dimension,
                        "left": _plain(issue.left),
                        "right": _plain(issue.right),
                        "reason": issue.reason,
                    }
                    for issue in result.issues
                ],
                "method_compatible_metrics": list(result.comparable_metrics),
                "incompatible_metrics": dict(result.incompatible_metrics),
                "waived_differences": [
                    {
                        "dimension": issue.dimension,
                        "left": _plain(issue.left),
                        "right": _plain(issue.right),
                        "reason": issue.reason,
                    }
                    for issue in result.waived_differences
                ],
            }
        )
    return {
        "schema": COMPARISON_SCHEMA,
        "compatible": all(pair["compatible"] for pair in pairs),
        "override": {"enabled": override_reason is not None, "reason": override_reason},
        "report_sha256": [record.run_id for record in records],
        "input_issues": [
            {"report": index, "issues": list(record.evidence["input_issues"])}
            for index, record in enumerate(records)
            if record.evidence["input_issues"]
        ],
        "pairs": pairs,
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value
