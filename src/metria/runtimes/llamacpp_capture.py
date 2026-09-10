"""Strict readers for the qualified local llama.cpp capture provider."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..recipes import _unique_json_object

CAPTURE_SCHEMA = "metria.llamacpp_capture.v1"
_INTEGER_FIELDS = frozenset({"context", "threads", "threads_batch", "vocab_size"})


def read_runtime_capture(path: Path) -> Mapping[str, Any] | None:
    """Read bounded runtime facts; older token-only providers remain explicit."""
    try:
        if path.stat().st_size > 4096:
            raise ValueError("llama.cpp runtime capture exceeds the supported size")
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    value = json.loads(text, object_pairs_hook=_unique_json_object)
    expected = {"schema", "chat_template_applied", *_INTEGER_FIELDS}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid llama.cpp runtime capture fields")
    if value["schema"] != CAPTURE_SCHEMA:
        raise ValueError("unsupported llama.cpp runtime capture schema")
    for field in _INTEGER_FIELDS:
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(
                f"llama.cpp runtime capture {field} must be a positive integer"
            )
    if not isinstance(value["chat_template_applied"], bool):
        raise ValueError(
            "llama.cpp runtime capture chat_template_applied must be boolean"
        )
    return value


def read_token_trajectory(path: Path) -> list[int]:
    """Read sampled token IDs without coercing malformed capture data."""
    tokens: list[int] = []
    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError:
        return tokens
    with stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line, object_pairs_hook=_unique_json_object)
            if not isinstance(row, dict) or not set(row) <= {"step", "token_id"}:
                raise ValueError("invalid llama.cpp token capture fields")
            token = row.get("token_id")
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError(
                    "llama.cpp token capture requires non-negative integer IDs"
                )
            if "step" in row:
                step = row["step"]
                if (
                    isinstance(step, bool)
                    or not isinstance(step, int)
                    or step != len(tokens)
                ):
                    raise ValueError(
                        "llama.cpp token capture steps must be contiguous from zero"
                    )
            tokens.append(token)
    return tokens
