"""Observed identity evidence for the first-party llama.cpp adapter."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..identity_evidence import (
    IdentityStatus,
    RuntimeIdentityEvidence,
    aggregate_identity_status,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _binary_digest(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    digest = value.get("sha256")
    if not isinstance(digest, str):
        return None
    normalized = digest.lower()
    return normalized if _SHA256_RE.fullmatch(normalized) else None


def inspect_llamacpp_identity(
    *,
    resolved: Mapping[str, Any],
    invocations: Sequence[Mapping[str, Any]],
) -> RuntimeIdentityEvidence:
    """Build conservative observed identity for one llama.cpp session.

    The executable digests are content-based and therefore authoritative for
    the runtime binaries Metria resolved. The current adapter does not hash the
    GGUF model unless an expected SHA-256 was supplied. Embedded tokenizer and
    template metadata and applied KV-cache state remain partial/unknown until
    independently observable evidence is available.
    """

    runtime = resolved.get("runtime")
    model = resolved.get("model")
    if not isinstance(runtime, Mapping) or not isinstance(model, Mapping):
        raise ValueError("resolved llama.cpp identity inputs are missing")

    cli_sha256 = _binary_digest(runtime.get("cli"))
    completion_sha256 = _binary_digest(runtime.get("completion"))
    runtime_reasons: tuple[str, ...]
    if cli_sha256 is None:
        runtime_status = IdentityStatus.UNKNOWN
        runtime_reasons = ("llama.cpp CLI content identity is unavailable",)
    else:
        runtime_status = IdentityStatus.VERIFIED
        runtime_reasons = ()
    runtime_identity = {
        "status": runtime_status.value,
        "name": "llamacpp",
        "cli_sha256": cli_sha256,
        "completion_sha256": completion_sha256,
        "source": "resolved_binary_content_hash",
    }

    model_path = model.get("path")
    model_sha256 = _binary_digest(model)
    model_reasons: tuple[str, ...]
    if isinstance(model_path, str) and model_path:
        model_status = (
            IdentityStatus.VERIFIED if model_sha256 else IdentityStatus.PARTIAL
        )
        model_identity = {
            "status": model_status.value,
            "path": model_path,
            "size_bytes": model.get("size_bytes"),
            "mtime_ns": model.get("mtime_ns"),
            "source": "verified_local_file_sha256"
            if model_sha256
            else "local_file_metadata",
        }
        if model_sha256:
            model_identity["sha256"] = model_sha256
            model_reasons = ()
        else:
            model_reasons = (
                "llama.cpp model file has not yet been bound to verified content identity",
            )
    else:
        model_status = IdentityStatus.UNKNOWN
        model_identity = {
            "status": model_status.value,
            "source": "local_file_metadata",
        }
        model_reasons = ("llama.cpp model identity is unavailable",)

    tokenizer_status = IdentityStatus.UNKNOWN
    tokenizer_identity = {
        "status": tokenizer_status.value,
        "source": "embedded_in_model_uninspected",
    }
    tokenizer_reasons = (
        "llama.cpp tokenizer identity is embedded in the model and not independently inspected",
    )

    template_status = IdentityStatus.UNKNOWN
    template_identity = {
        "status": template_status.value,
        "source": "embedded_in_model_uninspected",
    }
    template_reasons: tuple[str, ...] = (
        "llama.cpp chat-template identity is not independently inspected",
    )

    applied_identity: dict[str, Any]
    if invocations:
        applied_status = IdentityStatus.PARTIAL
        applied_identity = {
            "status": applied_status.value,
            "source": "recorded_command_invocation",
            "invocation_count": len(invocations),
        }
        applied_reasons = (
            "recorded llama.cpp command flags do not prove runtime-applied internal state",
        )
        captures = [row.get("runtime_capture") for row in invocations]
        if all(isinstance(capture, Mapping) for capture in captures):
            first = captures[0]
            assert isinstance(first, Mapping)
            fields = {key: value for key, value in first.items() if key != "schema"}
            if all(capture == first for capture in captures):
                applied_identity["fields"] = fields
                applied_identity["source"] = "qualified_runtime_capture"
                applied_reasons = (
                    "only context, thread counts, vocabulary size and template mode were read back",
                )
                if first.get("chat_template_applied") is False:
                    template_status = IdentityStatus.VERIFIED
                    template_identity = {
                        "status": template_status.value,
                        "mode": "disabled",
                        "source": "qualified_runtime_capture",
                    }
                    template_reasons = ()
            else:
                applied_status = IdentityStatus.MISMATCH
                applied_identity["status"] = applied_status.value
                applied_reasons = ("runtime capture facts changed between requests",)
    else:
        applied_status = IdentityStatus.UNKNOWN
        applied_identity = {
            "status": applied_status.value,
            "source": "recorded_command_invocation",
            "invocation_count": 0,
        }
        applied_reasons = ("no llama.cpp invocation has been observed",)

    statuses = (
        model_status,
        tokenizer_status,
        runtime_status,
        template_status,
        applied_status,
    )
    return RuntimeIdentityEvidence(
        status=aggregate_identity_status(statuses),
        model=model_identity,
        tokenizer=tokenizer_identity,
        runtime=runtime_identity,
        chat_template=template_identity,
        applied=applied_identity,
        reasons=(
            *model_reasons,
            *tokenizer_reasons,
            *runtime_reasons,
            *template_reasons,
            *applied_reasons,
        ),
    )
