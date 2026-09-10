"""Observed identity verification for the first-party vLLM adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from ..identity_evidence import (
    IdentityStatus,
    RuntimeIdentityEvidence,
    aggregate_identity_status,
)


def _scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _tokenizer_revision(tokenizer: Any, model_config: Any) -> Any:
    if model_config is not None and hasattr(model_config, "tokenizer_revision"):
        value = model_config.tokenizer_revision
        if value is not None:
            return _scalar(value)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        for key in ("_commit_hash", "revision"):
            value = init_kwargs.get(key)
            if value is not None:
                return _scalar(value)
    return None


def _tokenizer_identifier(tokenizer: Any, model_config: Any) -> Any:
    if model_config is not None and hasattr(model_config, "tokenizer"):
        value = model_config.tokenizer
        if value is not None:
            return _scalar(value)
    value = getattr(tokenizer, "name_or_path", None)
    return _scalar(value)


def _chat_template_bytes(tokenizer: Any) -> bytes | None:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        return template.encode("utf-8")
    if isinstance(template, Mapping):
        try:
            canonical = json.dumps(
                dict(template),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except (TypeError, ValueError):
            return None
        return canonical.encode("utf-8")
    return None


def _match_status(
    *,
    expected_id: Any,
    observed_id: Any,
    expected_revision: Any,
    observed_revision: Any,
    label: str,
) -> tuple[IdentityStatus, tuple[str, ...]]:
    reasons: list[str] = []
    if observed_id is None:
        return IdentityStatus.UNKNOWN, (f"{label} identifier is not observable",)
    if expected_id is not None and str(observed_id) != str(expected_id):
        reasons.append(
            f"{label} identifier mismatch: expected {expected_id!r}, observed {observed_id!r}"
        )
    if (
        expected_revision is not None
        and observed_revision is not None
        and str(observed_revision) != str(expected_revision)
    ):
        reasons.append(
            f"{label} revision mismatch: expected {expected_revision!r}, "
            f"observed {observed_revision!r}"
        )
    if reasons:
        return IdentityStatus.MISMATCH, tuple(reasons)
    if expected_revision is not None and observed_revision is None:
        return (
            IdentityStatus.PARTIAL,
            (f"{label} revision was requested but is not observable",),
        )
    return IdentityStatus.VERIFIED, ()


def _expected_applied_fields(
    resolved: Mapping[str, Any],
) -> Mapping[str, Any]:
    runtime = resolved.get("runtime")
    kv_cache = resolved.get("kv_cache")
    if not isinstance(runtime, Mapping):
        return {}
    settings = runtime.get("settings")
    if not isinstance(settings, Mapping):
        return {}
    kv = kv_cache if isinstance(kv_cache, Mapping) else {}
    return {
        "model.max_model_len": settings.get("max_model_len"),
        "cache.cache_dtype": kv.get("dtype"),
        "cache.gpu_memory_utilization": settings.get("gpu_memory_utilization"),
        "cache.enable_prefix_caching": settings.get("enable_prefix_caching"),
        "parallel.tensor_parallel_size": settings.get("tensor_parallel_size"),
    }


def _applied_identity(
    *,
    resolved: Mapping[str, Any],
    applied: Mapping[str, Any],
) -> tuple[IdentityStatus, Mapping[str, Any], tuple[str, ...]]:
    raw_status = applied.get("status")
    fields = applied.get("fields")
    if raw_status != "introspected" or not isinstance(fields, Mapping):
        return (
            IdentityStatus.UNKNOWN,
            {
                "status": IdentityStatus.UNKNOWN.value,
                "fields": {},
                "source": "vllm_engine_introspection",
            },
            ("vLLM applied configuration is not introspectable",),
        )

    expected = _expected_applied_fields(resolved)
    mismatches: list[str] = []
    checked: list[str] = []
    missing: list[str] = []
    for field, expected_value in expected.items():
        if expected_value is None:
            continue
        if field not in fields:
            missing.append(field)
            continue
        checked.append(field)
        if fields[field] != expected_value:
            mismatches.append(
                f"applied configuration mismatch for {field}: "
                f"expected {expected_value!r}, observed {fields[field]!r}"
            )

    if mismatches:
        status = IdentityStatus.MISMATCH
        reasons = tuple(mismatches)
    else:
        status = IdentityStatus.PARTIAL
        reasons_list = [
            "applied configuration contains only fields exposed by the vLLM engine"
        ]
        if missing:
            reasons_list.append(
                "vLLM did not expose requested applied fields: " + ", ".join(missing)
            )
        reasons = tuple(reasons_list)

    identity = {
        "status": status.value,
        "fields": fields,
        "checked_fields": tuple(checked),
        "missing_fields": tuple(missing),
        "source": "vllm_engine_introspection",
    }
    return status, identity, reasons


def inspect_vllm_identity(
    *,
    resolved: Mapping[str, Any],
    module: Any,
    llm: Any,
    tokenizer: Any,
    applied: Mapping[str, Any],
) -> RuntimeIdentityEvidence:
    """Build and verify observed vLLM model/tokenizer/runtime identity.

    The function runs immediately after ``vllm.LLM`` construction and before a
    measurement receives the session. A concrete mismatch therefore stops the
    run before prompt execution. Missing upstream metadata remains explicit as
    partial/unknown evidence instead of being copied from the resolved request.
    """

    resolved_runtime = resolved.get("runtime")
    resolved_model = resolved.get("model")
    if not isinstance(resolved_runtime, Mapping) or not isinstance(
        resolved_model, Mapping
    ):
        raise ValueError("resolved vLLM identity inputs are missing")

    model_config = getattr(llm, "model_config", None)
    observed_model = (
        _scalar(model_config.model)
        if model_config is not None and hasattr(model_config, "model")
        else None
    )
    observed_model_revision = (
        _scalar(model_config.revision)
        if model_config is not None and hasattr(model_config, "revision")
        else None
    )
    model_status, model_reasons = _match_status(
        expected_id=resolved_model.get("model"),
        observed_id=observed_model,
        expected_revision=resolved_model.get("revision"),
        observed_revision=observed_model_revision,
        label="model",
    )
    model_identity = {
        "status": model_status.value,
        "identifier": observed_model,
        "revision": observed_model_revision,
        "source": "vllm.model_config",
    }

    observed_tokenizer = _tokenizer_identifier(tokenizer, model_config)
    observed_tokenizer_revision = _tokenizer_revision(tokenizer, model_config)
    tokenizer_status, tokenizer_reasons = _match_status(
        expected_id=resolved_model.get("tokenizer"),
        observed_id=observed_tokenizer,
        expected_revision=resolved_model.get("tokenizer_revision"),
        observed_revision=observed_tokenizer_revision,
        label="tokenizer",
    )
    tokenizer_identity = {
        "status": tokenizer_status.value,
        "identifier": observed_tokenizer,
        "revision": observed_tokenizer_revision,
        "source": "vllm.model_config_or_tokenizer",
    }

    package_version = resolved_runtime.get("version")
    module_version = _scalar(getattr(module, "__version__", None))
    observed_runtime_version = module_version or package_version
    requested_runtime_version = resolved_runtime.get("requested_version")
    runtime_status, runtime_reasons = _match_status(
        expected_id="vllm",
        observed_id="vllm",
        expected_revision=requested_runtime_version,
        observed_revision=observed_runtime_version,
        label="runtime",
    )
    if (
        module_version is not None
        and package_version is not None
        and str(module_version) != str(package_version)
    ):
        runtime_status = IdentityStatus.MISMATCH
        runtime_reasons = (
            *runtime_reasons,
            "loaded vLLM module version does not match installed distribution metadata: "
            f"module {module_version!r}, distribution {package_version!r}",
        )
    elif module_version is None and runtime_status is IdentityStatus.VERIFIED:
        runtime_status = IdentityStatus.PARTIAL
        runtime_reasons = (
            "vLLM module does not expose __version__; distribution metadata is retained",
        )
    runtime_identity = {
        "status": runtime_status.value,
        "name": "vllm",
        "version": observed_runtime_version,
        "distribution_version": package_version,
        "module_version": module_version,
        "source": "loaded_module_and_distribution",
    }

    template_bytes = _chat_template_bytes(tokenizer)
    template_reasons: tuple[str, ...]
    if template_bytes is None:
        template_status = IdentityStatus.UNKNOWN
        template_identity = {
            "status": template_status.value,
            "source": "tokenizer.chat_template",
        }
        template_reasons = ("tokenizer chat-template identity is not observable",)
    else:
        template_status = IdentityStatus.VERIFIED
        template_identity = {
            "status": template_status.value,
            "sha256": hashlib.sha256(template_bytes).hexdigest(),
            "source": "tokenizer.chat_template",
        }
        template_reasons = ()

    applied_status, applied_identity, applied_reasons = _applied_identity(
        resolved=resolved,
        applied=applied,
    )

    component_statuses = (
        model_status,
        tokenizer_status,
        runtime_status,
        template_status,
        applied_status,
    )
    reasons = (
        *model_reasons,
        *tokenizer_reasons,
        *runtime_reasons,
        *template_reasons,
        *applied_reasons,
    )
    return RuntimeIdentityEvidence(
        status=aggregate_identity_status(component_statuses),
        model=model_identity,
        tokenizer=tokenizer_identity,
        runtime=runtime_identity,
        chat_template=template_identity,
        applied=applied_identity,
        reasons=reasons,
    )


def require_matching_vllm_identity(identity: RuntimeIdentityEvidence) -> None:
    """Reject a concrete vLLM identity mismatch before measurement execution."""

    if IdentityStatus(identity.status) is IdentityStatus.MISMATCH:
        detail = "; ".join(identity.reasons) or "observed vLLM identity mismatch"
        raise RuntimeError(detail)
