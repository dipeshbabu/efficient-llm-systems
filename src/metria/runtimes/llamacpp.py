"""Instance-scoped llama.cpp runtime adapter for Metria.

The adapter intentionally drives llama.cpp command-line binaries directly. It
shares command-line conventions with the existing KV Fidelity implementation,
but does not use KV Fidelity's module-global backend dispatch. Each Metria
session owns its resolved configuration and invocation evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .._freeze import freeze_mapping
from ..models import RunSpec, TreatmentSpec, TreatmentType
from ..protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    RuntimeSession,
    SupportReport,
)
from .llamacpp_identity import inspect_llamacpp_identity

_DEFAULT_BIN_DIR = Path(
    os.path.expanduser(
        os.environ.get(
            "LLAMA_CPP_BIN_DIR",
            "~/local_llms/llama.cpp/build-test/bin",
        )
    )
)

_MANAGED_EXTRA_FLAGS = frozenset(
    {
        "-m",
        "--model",
        "-p",
        "--prompt",
        "-n",
        "--n-predict",
        "-c",
        "--ctx-size",
        "-ngl",
        "--n-gpu-layers",
        "--seed",
        "--temp",
        "--temperature",
        "-ctk",
        "-ctv",
        "-fa",
        "--flash-attn",
        "--single-turn",
        "--no-display-prompt",
        "--jinja",
        "-sys",
        "--system-prompt",
        "-rea",
        "-no-cnv",
    }
)

_SUPPORTED_GENERATION_KEYS = frozenset(
    {
        "max_tokens",
        "context",
        "seed",
        "temperature",
        "timeout",
        "chat_template",
        "system",
        "reasoning",
    }
)

_SUPPORTED_KV_KEYS = frozenset(
    {
        "key_dtype",
        "value_dtype",
        "attention_rotation_k",
        "attention_rotation_v",
        "attention_rotation_disable",
    }
)

_NOISE_PATTERNS = (
    re.compile(r"^\[End thinking\].*$", re.MULTILINE),
    re.compile(r"^\[ Prompt:.*\]$", re.MULTILINE),
    re.compile(r"^Exiting\.\.\..*$", re.MULTILINE),
    re.compile(r"^llama_perf_.*$", re.MULTILINE),
    re.compile(r"^Log end$", re.MULTILINE),
    re.compile(r"^Loading model\.\.\..*$", re.MULTILINE),
    re.compile(r"^>\s.*$", re.MULTILINE),
)
_BLOCK_CHARS_RE = re.compile(r"^[\s\u2580-\u259F]+$", re.MULTILINE)
_GEN_LINE_RE = re.compile(r"^\|\s.*", re.MULTILINE)


def _extract_completion(text: str) -> str:
    """Remove known llama.cpp UI noise without detokenizing or retokenizing."""

    out = text.replace("\x08", "")
    for pattern in _NOISE_PATTERNS:
        out = pattern.sub("", out)
    matches = list(_GEN_LINE_RE.finditer(out))
    if matches:
        out = out[matches[0].start() :]
        out = re.sub(r"^\|\s?", "", out, flags=re.MULTILINE)
    out = _BLOCK_CHARS_RE.sub("", out)
    return out.strip()


def _sha256_file(path: Path) -> str:
    """Hash a runtime binary so observed runtime identity is content-based."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, *, include_hash: bool) -> dict[str, Any]:
    """Return stable file metadata for a binary or model artifact."""

    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    identity: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        identity["sha256"] = _sha256_file(resolved)
        after = resolved.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise RuntimeError("runtime artifact changed while its content was hashed")
    return identity


def _requested_model_sha256(spec: RunSpec) -> str | None:
    value = spec.model.get("sha256")
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("model.sha256 must be a 64-character SHA-256 digest")
    return value.lower()


def _check_pinned_model(model: Mapping[str, Any], *, hash_content: bool) -> None:
    expected = model.get("sha256")
    if expected is None:
        return
    actual = _file_identity(Path(model["path"]), include_hash=hash_content)
    if hash_content and actual["sha256"] != expected:
        raise ValueError("model content changed after resolution")
    if any(actual[key] != model[key] for key in ("size_bytes", "mtime_ns")):
        raise ValueError("model file changed after resolution")


def _find_binary(bin_dir: Path, stem: str) -> Path | None:
    """Find a llama.cpp binary using native Unix or Windows naming."""

    for name in (stem, f"{stem}.exe"):
        candidate = bin_dir / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _flag_name(value: str) -> str:
    """Return a command-line flag name without an inline assignment."""

    return value.split("=", 1)[0]


def _extra_args(runtime: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate expert llama.cpp flags without allowing managed overrides."""

    raw = runtime.get("extra_args", ())
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise TypeError("runtime.extra_args must be a sequence of argument strings")
    args = tuple(raw)
    if not all(isinstance(arg, str) for arg in args):
        raise TypeError("runtime.extra_args must contain only strings")
    collisions = sorted(
        {_flag_name(arg) for arg in args if _flag_name(arg) in _MANAGED_EXTRA_FLAGS}
    )
    if collisions:
        joined = ", ".join(collisions)
        raise ValueError(
            "runtime.extra_args cannot override Metria-managed flags: " + joined
        )
    return args


def _kv_treatment(treatments: Sequence[TreatmentSpec]) -> dict[str, Any]:
    """Resolve the single KV-cache runtime treatment supported by this adapter."""

    kv: dict[str, Any] = {
        "key_dtype": "f16",
        "value_dtype": "f16",
        "attention_rotation_k": None,
        "attention_rotation_v": None,
        "attention_rotation_disable": None,
    }
    seen = False
    for treatment in treatments:
        if treatment.name not in {"kv_cache", "llamacpp.kv_cache"}:
            raise ValueError(
                "llama.cpp adapter does not yet apply treatment "
                f"{treatment.name!r}; materialize or adapt it before runtime launch"
            )
        if treatment.kind is not TreatmentType.RUNTIME_FEATURE:
            raise ValueError(
                "llama.cpp KV-cache treatment must use kind='runtime_feature'"
            )
        if seen:
            raise ValueError("only one llama.cpp KV-cache treatment may be applied")
        unknown = sorted(set(treatment.config) - _SUPPORTED_KV_KEYS)
        if unknown:
            raise ValueError(
                "unsupported llama.cpp KV-cache treatment keys: " + ", ".join(unknown)
            )
        seen = True
        kv.update(treatment.config)

    for key in ("key_dtype", "value_dtype"):
        value = kv[key]
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{key} must be a non-empty string")
    for key in (
        "attention_rotation_k",
        "attention_rotation_v",
        "attention_rotation_disable",
    ):
        value = kv[key]
        if value is not None and value not in (0, 1, False, True):
            raise ValueError(f"{key} must be 0, 1, true, false, or null")
        if value is not None:
            kv[key] = int(value)
    return kv


def _bin_dir(spec: RunSpec, environment: Mapping[str, Any]) -> Path:
    """Resolve the binary directory from explicit recipe, environment, or default."""

    raw = spec.runtime.get("bin_dir")
    if raw is None:
        raw = environment.get("llama_cpp_bin_dir")
    if raw is None:
        return _DEFAULT_BIN_DIR.expanduser().resolve()
    if not isinstance(raw, (str, os.PathLike)):
        raise TypeError("runtime.bin_dir must be a filesystem path")
    return Path(raw).expanduser().resolve()


def _model_path(spec: RunSpec) -> Path:
    """Resolve the local model artifact required by the llama.cpp adapter."""

    raw = spec.model.get("path")
    if raw is None:
        raise ValueError("llama.cpp requires model.path pointing to a local model file")
    if not isinstance(raw, (str, os.PathLike)):
        raise TypeError("model.path must be a filesystem path")
    return Path(raw).expanduser().resolve()


def _runtime_config(spec: RunSpec) -> dict[str, Any]:
    """Validate llama.cpp-specific runtime settings used by every request."""

    name = spec.runtime.get("name")
    if name != "llamacpp":
        raise ValueError("runtime.name must be 'llamacpp' for LlamaCppAdapter")
    n_gpu_layers = spec.runtime.get("n_gpu_layers", 99)
    if isinstance(n_gpu_layers, bool) or not isinstance(n_gpu_layers, int):
        raise TypeError("runtime.n_gpu_layers must be an integer")
    flash_attention = spec.runtime.get("flash_attention", True)
    if not isinstance(flash_attention, bool):
        raise TypeError("runtime.flash_attention must be a boolean")
    return {
        "name": "llamacpp",
        "n_gpu_layers": n_gpu_layers,
        "flash_attention": flash_attention,
        "extra_args": _extra_args(spec.runtime),
    }


def _generation_options(
    request: InferenceRequest,
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one request's generation policy from scenario defaults and overrides."""

    unknown = sorted(set(request.generation) - _SUPPORTED_GENERATION_KEYS)
    if unknown:
        raise ValueError("unsupported generation keys: " + ", ".join(unknown))

    def choose(key: str, default: Any) -> Any:
        return request.generation.get(key, scenario.get(key, default))

    max_tokens = choose("max_tokens", 128)
    context = choose("context", 512)
    seed = choose("seed", 42)
    temperature = choose("temperature", 0.0)
    timeout = choose("timeout", 300.0)
    chat_template = choose("chat_template", True)
    system = choose("system", None)
    reasoning = choose("reasoning", "off")

    for key, value in (
        ("max_tokens", max_tokens),
        ("context", context),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"generation {key} must be an integer")
    if max_tokens < 0:
        raise ValueError("generation max_tokens must be non-negative")
    if context <= 0:
        raise ValueError("generation context must be positive")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("generation temperature must be numeric")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        raise TypeError("generation timeout must be numeric")
    if timeout <= 0:
        raise ValueError("generation timeout must be positive")
    if not isinstance(chat_template, bool):
        raise TypeError("generation chat_template must be boolean")
    if system is not None and not isinstance(system, str):
        raise TypeError("generation system must be a string or null")
    if not isinstance(reasoning, str):
        raise TypeError("generation reasoning must be a string")

    return {
        "max_tokens": max_tokens,
        "context": context,
        "seed": seed,
        "temperature": float(temperature),
        "timeout": float(timeout),
        "chat_template": chat_template,
        "system": system,
        "reasoning": reasoning,
    }


def _kv_args(kv: Mapping[str, Any]) -> list[str]:
    """Translate resolved KV-cache state into llama.cpp command-line flags."""

    return ["-ctk", str(kv["key_dtype"]), "-ctv", str(kv["value_dtype"])]


def _kv_env(kv: Mapping[str, Any]) -> dict[str, str]:
    """Translate explicit attention-rotation settings into the patched runtime ABI."""

    names = {
        "attention_rotation_k": "LLAMA_ATTN_ROT_K_OVERRIDE",
        "attention_rotation_v": "LLAMA_ATTN_ROT_V_OVERRIDE",
        "attention_rotation_disable": "LLAMA_ATTN_ROT_DISABLE",
    }
    return {
        env_name: str(kv[key])
        for key, env_name in names.items()
        if kv.get(key) is not None
    }


def _redacted_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Remove prompt/system contents while retaining exact executable flags."""

    redacted: list[str] = []
    hide_next = False
    for item in argv:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        redacted.append(item)
        if item in {"-p", "--prompt", "-sys", "--system-prompt"}:
            hide_next = True
    return tuple(redacted)


def _text_hash(value: str | None) -> str | None:
    """Hash sensitive prompt text so invocation identity survives redaction."""

    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class LlamaCppSession:
    """One-shot llama.cpp CLI session with instance-local invocation evidence."""

    def __init__(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> None:
        self._resolved = freeze_mapping(resolved)
        self._environment = freeze_mapping(environment)
        self._closed = False
        self._reset_count = 0
        self._invocations: list[Mapping[str, Any]] = []

    @property
    def resolved(self) -> Mapping[str, Any]:
        """Return the immutable resolved runtime specification."""

        return self._resolved

    @property
    def invocations(self) -> tuple[Mapping[str, Any], ...]:
        """Return recorded invocations without exposing mutable internal state."""

        return tuple(self._invocations)

    @property
    def closed(self) -> bool:
        """Whether this session has been explicitly closed."""

        return self._closed

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        """Run requests sequentially through llama-cli or llama-completion."""

        if self._closed:
            raise RuntimeError("llama.cpp session is closed")
        capture_kinds = tuple(item.kind for item in capture)
        unknown = sorted(set(capture_kinds) - {"token_ids"})
        if unknown:
            raise ValueError("unsupported llama.cpp captures: " + ", ".join(unknown))
        if len(capture_kinds) != len(set(capture_kinds)):
            raise ValueError("duplicate capture requests are not allowed")
        token_capture = "token_ids" in capture_kinds
        if token_capture and self._resolved["runtime"].get("completion") is None:
            raise RuntimeError(
                "token_ids capture requires a llama-completion binary with the "
                "KV_FIDELITY_TRAJECTORY patch"
            )

        outputs: list[str] = []
        token_batches: list[tuple[int, ...]] = []
        batch_invocations: list[Mapping[str, Any]] = []
        for request in requests:
            if not isinstance(request, InferenceRequest):
                raise TypeError("requests must contain InferenceRequest objects")
            output, token_ids, evidence = self._infer_one(
                request,
                token_capture=token_capture,
            )
            outputs.append(output)
            if token_capture:
                token_batches.append(tuple(token_ids))
            frozen_evidence = freeze_mapping(evidence)
            self._invocations.append(frozen_evidence)
            batch_invocations.append(frozen_evidence)

        captures: dict[str, Any] = {}
        if token_capture:
            captures["token_ids"] = tuple(token_batches)
        return InferenceBatch(
            outputs=tuple(outputs),
            captures=captures,
            metadata={
                "runtime": "llamacpp",
                "invocations": tuple(batch_invocations),
            },
        )

    def reset(self, scope: str = "measurement") -> None:
        """Record a reset boundary; one-shot llama.cpp has no warm engine state."""

        if self._closed:
            raise RuntimeError("llama.cpp session is closed")
        if not scope:
            raise ValueError("reset scope must not be empty")
        self._reset_count += 1

    def close(self) -> None:
        """Close the logical session; individual subprocesses are already reaped."""

        self._closed = True

    def observation(self) -> Mapping[str, Any]:
        """Return runtime identity plus exact redacted commands that actually ran."""

        identity = inspect_llamacpp_identity(
            resolved=self._resolved,
            invocations=self._invocations,
        )
        return freeze_mapping(
            {
                "runtime": self._resolved["runtime"],
                "model": self._resolved["model"],
                "identity": identity.to_mapping(),
                "configured": {
                    "kv_cache": self._resolved["kv_cache"],
                },
                "invocations": tuple(self._invocations),
                "reset_count": self._reset_count,
                "closed": self._closed,
                "environment": self._environment,
            }
        )

    def _infer_one(
        self,
        request: InferenceRequest,
        *,
        token_capture: bool,
    ) -> tuple[str, list[int], dict[str, Any]]:
        runtime = self._resolved["runtime"]
        model = self._resolved["model"]
        kv = self._resolved["kv_cache"]
        scenario = self._resolved["scenario"]
        _check_pinned_model(model, hash_content=False)
        generation = _generation_options(request, scenario)
        executable = (
            runtime["completion"]["path"] if token_capture else runtime["cli"]["path"]
        )
        argv: list[str] = [
            str(executable),
            "-m",
            str(model["path"]),
            "-p",
            request.prompt,
            "-n",
            str(generation["max_tokens"]),
            "-c",
            str(generation["context"]),
            "-ngl",
            str(runtime["n_gpu_layers"]),
            "--seed",
            str(generation["seed"]),
            "--temp",
            str(generation["temperature"]),
        ]
        if token_capture:
            argv.extend(["-no-cnv", "--no-display-prompt"])
        else:
            argv.extend(["--single-turn", "--no-display-prompt"])
        argv.extend(["-fa", "on" if runtime["flash_attention"] else "off"])
        if generation["chat_template"]:
            argv.append("--jinja")
            if not token_capture:
                argv.extend(["-rea", str(generation["reasoning"])])
            if generation["system"]:
                argv.extend(["-sys", str(generation["system"])])
        argv.extend(_kv_args(kv))
        argv.extend(runtime["extra_args"])

        process_env = os.environ.copy()
        managed_env = _kv_env(kv)
        process_env.update(managed_env)
        trajectory_path: str | None = None
        if token_capture:
            fd, trajectory_path = tempfile.mkstemp(
                prefix="metria-llama-trajectory-",
                suffix=".jsonl",
            )
            os.close(fd)
            os.unlink(trajectory_path)
            process_env["KV_FIDELITY_TRAJECTORY"] = trajectory_path

        started = time.monotonic()
        try:
            try:
                completed = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    env=process_env,
                    timeout=generation["timeout"],
                    text=True,
                    errors="replace",
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "llama.cpp request timed out after "
                    f"{generation['timeout']:.3f}s using {Path(executable).name}"
                ) from exc
            duration = time.monotonic() - started
            if completed.returncode != 0:
                stderr_tail = completed.stderr[-2000:] if completed.stderr else ""
                raise RuntimeError(
                    f"{Path(executable).name} exited {completed.returncode}; "
                    f"stderr tail: {stderr_tail}"
                )

            token_ids = (
                _read_trajectory(Path(trajectory_path))
                if trajectory_path is not None
                else []
            )
            evidence = {
                "binary": str(executable),
                "argv": _redacted_argv(argv),
                "prompt_sha256": _text_hash(request.prompt),
                "system_sha256": _text_hash(generation["system"]),
                "managed_env": managed_env,
                "returncode": completed.returncode,
                "duration_seconds": duration,
                "capture": "token_ids" if token_capture else "text",
                "token_capture_observed": bool(token_ids) if token_capture else None,
            }
            return _extract_completion(completed.stdout), token_ids, evidence
        finally:
            if trajectory_path is not None:
                try:
                    os.unlink(trajectory_path)
                except OSError:
                    pass


def _read_trajectory(path: Path) -> list[int]:
    """Read token IDs emitted by the existing patched llama-completion ABI."""

    token_ids: list[int] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                token_ids.append(int(record["token_id"]))
    except FileNotFoundError:
        return []
    return token_ids


class LlamaCppAdapter:
    """Resolve and launch local llama.cpp CLI sessions without global state."""

    name = "llamacpp"

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        """Check local model, binary, runtime-option, and treatment availability."""

        reasons: list[str] = []
        evidence: dict[str, Any] = {"runtime": "llamacpp"}
        try:
            runtime = _runtime_config(spec)
            model = _model_path(spec)
            _requested_model_sha256(spec)
            bin_dir = _bin_dir(spec, environment)
            _kv_treatment(spec.treatments)
        except (TypeError, ValueError) as exc:
            return SupportReport(
                status="unsupported", reasons=(str(exc),), evidence=evidence
            )

        cli = _find_binary(bin_dir, "llama-cli")
        completion = _find_binary(bin_dir, "llama-completion")
        if not model.is_file():
            reasons.append(f"model file not found: {model}")
        if cli is None:
            reasons.append(f"llama-cli not found in {bin_dir}")
        evidence.update(
            {
                "bin_dir": str(bin_dir),
                "model": str(model),
                "llama_cli": str(cli) if cli else None,
                "llama_completion": str(completion) if completion else None,
                "token_ids_capture": (
                    "binary_present_unverified" if completion else "unavailable"
                ),
                "n_gpu_layers": runtime["n_gpu_layers"],
            }
        )
        return SupportReport(
            status="unsupported" if reasons else "supported",
            reasons=tuple(reasons),
            evidence=evidence,
        )

    def resolve(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Resolve local files and managed runtime treatment into exact evidence."""

        support = self.probe(spec, environment)
        if support.status != "supported":
            raise ValueError(
                "; ".join(support.reasons) or "llama.cpp run is unsupported"
            )
        runtime = _runtime_config(spec)
        model_path = _model_path(spec)
        bin_dir = _bin_dir(spec, environment)
        cli = _find_binary(bin_dir, "llama-cli")
        if cli is None:  # probe guarantees this, keep fail-loud for races
            raise FileNotFoundError(f"llama-cli disappeared from {bin_dir}")
        completion = _find_binary(bin_dir, "llama-completion")
        expected_sha256 = _requested_model_sha256(spec)
        model_identity = _file_identity(
            model_path, include_hash=expected_sha256 is not None
        )
        if expected_sha256 is not None and model_identity["sha256"] != expected_sha256:
            raise ValueError("model SHA-256 does not match model.sha256")
        for key in ("id", "revision", "sha256"):
            if key in spec.model:
                model_identity[f"requested_{key}"] = spec.model[key]

        resolved_runtime: dict[str, Any] = {
            **runtime,
            "bin_dir": str(bin_dir),
            "cli": _file_identity(cli, include_hash=True),
            "completion": (
                _file_identity(completion, include_hash=True)
                if completion is not None
                else None
            ),
        }
        return freeze_mapping(
            {
                "runtime": resolved_runtime,
                "model": model_identity,
                "scenario": spec.scenario,
                "kv_cache": _kv_treatment(spec.treatments),
                "support": support.evidence,
            }
        )

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> RuntimeSession:
        """Create an instance-scoped logical session for one-shot CLI execution."""

        runtime = resolved.get("runtime")
        if not isinstance(runtime, Mapping) or runtime.get("name") != "llamacpp":
            raise ValueError("resolved runtime is not a llama.cpp specification")
        _check_pinned_model(resolved["model"], hash_content=True)
        return LlamaCppSession(resolved, environment)

    def observe(self, session: RuntimeSession) -> Mapping[str, Any]:
        """Return binary/model identity and redacted invocation evidence."""

        if not isinstance(session, LlamaCppSession):
            raise TypeError("LlamaCppAdapter can only observe LlamaCppSession")
        return session.observation()
