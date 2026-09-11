"""Capture report inputs without confusing selected files with runtime readback."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from metria import capture_hardware_fingerprint

from .comparison import EVIDENCE_SCHEMA, _plain


def _stamp(path: Path) -> tuple[int, int, int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError(
            "an identity requires a regular file without a symlink or reparse point"
        )
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _file_identity(path: Path) -> tuple[dict[str, Any], tuple[int, int, int, int]]:
    stamp = _stamp(path)
    digest = hashlib.sha256()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(os.open(path, flags), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_size) != stamp[:3]
        ):
            raise ValueError("input changed while opening")
        remaining = stamp[2]
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("input shrank while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise ValueError("input grew while hashing")
        after = os.fstat(stream.fileno())
        if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("input changed while hashing")
    if _stamp(path) != stamp:
        raise ValueError("input changed while hashing")
    return {"sha256": digest.hexdigest(), "size_bytes": stamp[2]}, stamp


@dataclass
class ReportEvidenceCapture:
    """Selected input identities, with a stat check after the scoring run.

    The check detects ordinary replacement/mutation during scoring. It is not
    runtime attestation or protection against a hostile local filesystem.
    """

    data: dict[str, Any]
    watched: list[tuple[str, Path, tuple[int, int, int, int]]] = field(
        default_factory=list
    )

    def add_file(
        self, name: str, path: Any, destination: dict[str, Any], key: str
    ) -> None:
        if path is None:
            return
        try:
            local = Path(path)
            identity, stamp = _file_identity(local)
        except (OSError, TypeError, ValueError):
            return  # Missing evidence remains missing; it never becomes equality.
        destination[key] = identity
        self.watched.append((name, local, stamp))

    def finish(self) -> dict[str, Any]:
        changed = []
        for name, path, before in self.watched:
            try:
                if _stamp(path) != before:
                    changed.append(name)
            except (OSError, ValueError):
                changed.append(name)
        self.data["changed_inputs"] = sorted(changed)
        self.data["observed"] = {
            "hardware": _plain(capture_hardware_fingerprint().to_mapping())
        }
        return self.data


def begin_report_capture(
    settings: Mapping[str, Any], *, backend: str
) -> ReportEvidenceCapture:
    """Capture the known local portion of a scoring run's evidence.

    Remote model names, package version strings, and requested GPU placement do
    not establish model/tokenizer/build/device identity. Those remain missing
    unless the backend actually supplies stronger evidence in a future capture.
    """
    generation = {
        key: settings[key]
        for key in (
            "n_predict",
            "ctx",
            "chunks",
            "axis_a",
            "measure_floor",
            "axis_rniah",
            "axis_plad",
            "rniah_ctx_max",
            "rniah_up_to",
            "rniah_lengths",
            "rniah_positions",
            "rniah_trials",
        )
        if key in settings
    }
    generation.update(
        {"temperature": 0.0, "apply_chat_template": True, "reasoning": "off"}
    )
    overrides = {
        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for key, value in sorted(os.environ.items())
        if key.startswith(("KV_FIDELITY_", "LLAMA_", "GGML_"))
        and key != "LLAMA_CPP_BIN_DIR"  # selected executable content is captured below
    }
    requested: dict[str, Any] = {
        "runtime": {
            "backend": backend,
            "settings": {"n_gpu_layers": settings.get("n_gpu_layers")},
            "environment_overrides": overrides,
        },
        "scenario": {"generation": generation},
        "trial_policy": {"seed": settings.get("seed")},
    }
    resolved: dict[str, Any] = {"inputs": {}}
    if (
        backend == "llamacpp"
        and os.environ.get("KV_FIDELITY_LLAMA_EXTRA_FLAGS", "").strip()
    ):
        # Arbitrary trailing flags may replace the model, prompt, or placement.
        # The original CLI arguments then cannot establish selected identities.
        requested["runtime"]["unverified_argument_overrides"] = True
    capture = ReportEvidenceCapture(
        {"schema": EVIDENCE_SCHEMA, "requested": requested, "resolved": resolved}
    )
    # Development builds can share a version string while their scoring code
    # differs. Identify the selected Python implementation as a file manifest.
    package = Path(__file__).parent
    suite_files: dict[str, Any] = {}
    sources = sorted(package.rglob("*.py"))
    for source in sources:
        name = source.relative_to(package).as_posix()
        capture.add_file(f"suite.{name}", source, suite_files, name)
    if sources and len(suite_files) == len(sources):
        payload = json.dumps(suite_files, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        resolved["suite"] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "files": suite_files,
        }
    if not settings.get("skip_gtm") or settings.get("axis_plad"):
        capture.add_file(
            "prompts", settings.get("prompts"), resolved["inputs"], "prompts"
        )
    if not settings.get("skip_kld"):
        capture.add_file("corpus", settings.get("corpus"), resolved["inputs"], "corpus")
    if settings.get("axis_rniah"):
        capture.add_file(
            "rniah_haystack",
            settings.get("rniah_haystack"),
            resolved["inputs"],
            "rniah_haystack",
        )
    if backend == "llamacpp":
        from .runner import DEFAULT_BIN_DIR

        model = settings.get("model")
        if model is not None and Path(model).suffix.lower() == ".gguf":
            capture.add_file("model", model, resolved, "model")
            if "model" in resolved:
                # llama.cpp's tokenizer and chat template are part of this GGUF.
                resolved["tokenizer"] = {
                    **resolved["model"],
                    "kind": "gguf_embedded_tokenizer",
                }
        binaries = set()
        if not settings.get("skip_gtm"):
            binaries.add(
                "llama-completion"
                if settings.get("axis_a") == "trajectory"
                else "llama-cli"
            )
            if settings.get("axis_a") != "trajectory":
                binaries.add("llama-tokenize")
        if not settings.get("skip_kld"):
            binaries.add("llama-perplexity")
        if settings.get("axis_plad") or settings.get("axis_rniah"):
            binaries.update({"llama-cli", "llama-tokenize"})
        artifacts: dict[str, Any] = {}
        for name in sorted(binaries):
            capture.add_file(f"runtime.{name}", DEFAULT_BIN_DIR / name, artifacts, name)
        if binaries and len(artifacts) == len(binaries):
            resolved["runtime"] = {"artifacts": artifacts}
    return capture
