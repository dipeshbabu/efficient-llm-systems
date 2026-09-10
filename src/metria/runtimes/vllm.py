"""Instance-scoped vLLM offline runtime adapter for Metria.

The adapter keeps vLLM optional: importing :mod:`metria` never imports vLLM.
Each launched session owns one ``vllm.LLM`` instance and records requested,
resolved, and introspected runtime evidence without using the module-global LLM
cache retained by the standalone KV Fidelity backend.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
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
from .vllm_identity import inspect_vllm_identity, require_matching_vllm_identity

_SUPPORTED_GENERATION_KEYS = frozenset(
    {"max_tokens", "seed", "temperature", "chat_template", "system"}
)
_SUPPORTED_RUNTIME_KEYS = frozenset(
    {
        "name",
        "version",
        "dtype",
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_model_len",
        "tensor_parallel_size",
        "enforce_eager",
        "enable_prefix_caching",
        "trust_remote_code",
    }
)
_SUPPORTED_KV_DTYPES = frozenset({"auto", "fp8", "fp8_e4m3", "fp8_e5m2"})
_SUPPORTED_KV_KEYS = frozenset({"dtype"})


def _vllm_available() -> bool:
    """Return whether the optional vLLM distribution is importable."""

    try:
        return importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):
        return False


def _vllm_version() -> str | None:
    """Return installed vLLM distribution version without importing the runtime."""

    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_vllm() -> Any:
    """Import vLLM only when a session is actually launched."""

    return importlib.import_module("vllm")


def _text_hash(value: str | None) -> str | None:
    """Hash sensitive text so invocation identity survives evidence redaction."""

    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _requested_runtime_version(spec: RunSpec) -> str | None:
    value = spec.runtime.get("version")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError("runtime.version must be a non-empty string or null")
    return value


def _model_source(spec: RunSpec) -> dict[str, Any]:
    """Resolve a local model path or registry identifier for vLLM."""

    raw_path = spec.model.get("path")
    raw_id = spec.model.get("id")
    if raw_path is not None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise TypeError("model.path must be a non-empty string")
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"model.path does not exist: {path}")
        model = str(path)
        source_kind = "local"
    elif raw_id is not None:
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise TypeError("model.id must be a non-empty string")
        model = raw_id
        source_kind = "registry"
    else:
        raise ValueError("vLLM requires model.id or model.path")

    revision = spec.model.get("revision")
    raw_tokenizer_id = spec.model.get("tokenizer_id")
    tokenizer_revision = spec.model.get("tokenizer_revision")
    if tokenizer_revision is None and raw_tokenizer_id is None:
        tokenizer_revision = revision
    if revision is not None and not isinstance(revision, str):
        raise TypeError("model.revision must be a string or null")
    if raw_tokenizer_id is not None and (
        not isinstance(raw_tokenizer_id, str) or not raw_tokenizer_id.strip()
    ):
        raise TypeError("model.tokenizer_id must be a non-empty string or null")
    if tokenizer_revision is not None and not isinstance(tokenizer_revision, str):
        raise TypeError("model.tokenizer_revision must be a string or null")
    tokenizer = raw_tokenizer_id or model
    return {
        "model": model,
        "source_kind": source_kind,
        "requested_id": raw_id,
        "requested_path": str(raw_path) if raw_path is not None else None,
        "revision": revision,
        "tokenizer": tokenizer,
        "requested_tokenizer_id": raw_tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
    }


def _runtime_config(spec: RunSpec) -> dict[str, Any]:
    """Validate the small stable vLLM runtime surface owned by Metria."""

    unknown = sorted(set(spec.runtime) - _SUPPORTED_RUNTIME_KEYS)
    if unknown:
        raise ValueError("unsupported vLLM runtime keys: " + ", ".join(unknown))
    if spec.runtime.get("name") != "vllm":
        raise ValueError("runtime.name must be 'vllm' for VLLMAdapter")
    _requested_runtime_version(spec)

    dtype = spec.runtime.get("dtype", "auto")
    if not isinstance(dtype, str) or not dtype:
        raise TypeError("runtime.dtype must be a non-empty string")

    gpu_memory_utilization = spec.runtime.get("gpu_memory_utilization", 0.85)
    if isinstance(gpu_memory_utilization, bool) or not isinstance(
        gpu_memory_utilization, (int, float)
    ):
        raise TypeError("runtime.gpu_memory_utilization must be numeric")
    gpu_memory_utilization = float(gpu_memory_utilization)
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("runtime.gpu_memory_utilization must be in (0, 1]")

    max_num_seqs = spec.runtime.get("max_num_seqs", 32)
    tensor_parallel_size = spec.runtime.get("tensor_parallel_size", 1)
    for key, value in (
        ("max_num_seqs", max_num_seqs),
        ("tensor_parallel_size", tensor_parallel_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"runtime.{key} must be an integer")
        if value <= 0:
            raise ValueError(f"runtime.{key} must be positive")

    max_model_len = spec.runtime.get("max_model_len")
    if max_model_len is None:
        context = spec.scenario.get("context", 512)
        max_tokens = spec.scenario.get("max_tokens", 128)
        for key, value in (("context", context), ("max_tokens", max_tokens)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"scenario.{key} must be an integer")
        if context <= 0 or max_tokens < 0:
            raise ValueError("scenario context/max_tokens are invalid")
        max_model_len = context + max_tokens + 32
    elif isinstance(max_model_len, bool) or not isinstance(max_model_len, int):
        raise TypeError("runtime.max_model_len must be an integer or null")
    if max_model_len <= 0:
        raise ValueError("runtime.max_model_len must be positive")

    enforce_eager = spec.runtime.get("enforce_eager", False)
    enable_prefix_caching = spec.runtime.get("enable_prefix_caching", False)
    trust_remote_code = spec.runtime.get("trust_remote_code", False)
    for key, value in (
        ("enforce_eager", enforce_eager),
        ("enable_prefix_caching", enable_prefix_caching),
        ("trust_remote_code", trust_remote_code),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"runtime.{key} must be a boolean")

    return {
        "name": "vllm",
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "enforce_eager": enforce_eager,
        "enable_prefix_caching": enable_prefix_caching,
        "trust_remote_code": trust_remote_code,
    }


def _kv_treatment(treatments: Sequence[TreatmentSpec]) -> dict[str, Any]:
    """Resolve the vLLM-native KV-cache treatment for one run."""

    dtype = "auto"
    seen = False
    for treatment in treatments:
        if treatment.name not in {"kv_cache", "vllm.kv_cache"}:
            raise ValueError(
                "vLLM adapter does not yet apply treatment "
                f"{treatment.name!r}; materialize or adapt it before runtime launch"
            )
        if treatment.kind is not TreatmentType.RUNTIME_FEATURE:
            raise ValueError("vLLM KV-cache treatment must use kind='runtime_feature'")
        if seen:
            raise ValueError("only one vLLM KV-cache treatment may be applied")
        unknown = sorted(set(treatment.config) - _SUPPORTED_KV_KEYS)
        if unknown:
            raise ValueError(
                "unsupported vLLM KV-cache treatment keys: " + ", ".join(unknown)
            )
        requested = treatment.config.get("dtype", "auto")
        if not isinstance(requested, str):
            raise TypeError("vLLM KV-cache dtype must be a string")
        if requested not in _SUPPORTED_KV_DTYPES:
            raise ValueError(
                f"unsupported vLLM KV-cache dtype {requested!r}; "
                f"supported values: {', '.join(sorted(_SUPPORTED_KV_DTYPES))}"
            )
        dtype = requested
        seen = True
    return {"dtype": dtype}


def _generation_options(
    request: InferenceRequest,
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve per-request sampling settings without silently ignoring fields."""

    unknown = sorted(set(request.generation) - _SUPPORTED_GENERATION_KEYS)
    if unknown:
        raise ValueError("unsupported vLLM generation keys: " + ", ".join(unknown))

    def choose(key: str, default: Any) -> Any:
        return request.generation.get(key, scenario.get(key, default))

    max_tokens = choose("max_tokens", 128)
    seed = choose("seed", 42)
    temperature = choose("temperature", 0.0)
    chat_template = choose("chat_template", True)
    system = choose("system", None)

    for key, value in (("max_tokens", max_tokens), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"generation {key} must be an integer")
    if max_tokens < 0:
        raise ValueError("generation max_tokens must be non-negative")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("generation temperature must be numeric")
    if temperature < 0:
        raise ValueError("generation temperature must be non-negative")
    if not isinstance(chat_template, bool):
        raise TypeError("generation chat_template must be boolean")
    if system is not None and not isinstance(system, str):
        raise TypeError("generation system must be a string or null")
    return {
        "max_tokens": max_tokens,
        "seed": seed,
        "temperature": float(temperature),
        "chat_template": chat_template,
        "system": system,
    }


def _format_prompt(tokenizer: Any, prompt: str, options: Mapping[str, Any]) -> str:
    """Apply the model chat template when requested; fail rather than fallback."""

    if not options["chat_template"]:
        return prompt
    messages: list[dict[str, str]] = []
    if options["system"]:
        messages.append({"role": "system", "content": str(options["system"])})
    messages.append({"role": "user", "content": prompt})
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise RuntimeError("vLLM tokenizer does not expose apply_chat_template")
    rendered = apply_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str):
        raise RuntimeError("vLLM tokenizer returned a non-string chat template result")
    return rendered


def _evidence_scalar(value: Any) -> Any:
    """Normalize introspected runtime fields into immutable evidence scalars."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _introspect_applied_config(llm: Any) -> dict[str, Any]:
    """Best-effort introspection of fields vLLM exposes after engine creation."""

    fields: dict[str, Any] = {}
    model_config = getattr(llm, "model_config", None)
    if model_config is not None:
        for attr in (
            "model",
            "dtype",
            "max_model_len",
            "revision",
            "tokenizer",
            "tokenizer_revision",
        ):
            if hasattr(model_config, attr):
                fields[f"model.{attr}"] = _evidence_scalar(getattr(model_config, attr))

    engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        cache_config = getattr(engine, "cache_config", None)
    if cache_config is not None:
        for attr in ("cache_dtype", "gpu_memory_utilization", "enable_prefix_caching"):
            if hasattr(cache_config, attr):
                fields[f"cache.{attr}"] = _evidence_scalar(getattr(cache_config, attr))

    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is not None and hasattr(parallel_config, "tensor_parallel_size"):
        fields["parallel.tensor_parallel_size"] = _evidence_scalar(
            parallel_config.tensor_parallel_size
        )

    return {
        "status": "introspected" if fields else "unverified",
        "fields": fields,
    }


def _cleanup_failed_llm(llm: Any) -> None:
    """Best-effort release of a just-created engine after identity failure."""

    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass
    gc.collect()


class VLLMSession:
    """One instance-scoped vLLM offline engine plus immutable invocation evidence."""

    def __init__(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
        module: Any,
        llm: Any,
    ) -> None:
        self._resolved = freeze_mapping(resolved)
        self._environment = freeze_mapping(environment)
        self._module = module
        self._llm = llm
        self._tokenizer = llm.get_tokenizer()
        self._applied = freeze_mapping(_introspect_applied_config(llm))
        self._identity = inspect_vllm_identity(
            resolved=self._resolved,
            module=module,
            llm=llm,
            tokenizer=self._tokenizer,
            applied=self._applied,
        )
        require_matching_vllm_identity(self._identity)
        self._invocations: list[Mapping[str, Any]] = []
        self._closed = False
        self._reset_count = 0
        self._cleanup: Mapping[str, Any] = {}

    @property
    def closed(self) -> bool:
        """Whether the logical session has been closed."""

        return self._closed

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        """Batch requests through vLLM while preserving true output token IDs."""

        if self._closed or self._llm is None:
            raise RuntimeError("vLLM session is closed")
        capture_kinds = tuple(item.kind for item in capture)
        unknown = sorted(set(capture_kinds) - {"token_ids"})
        if unknown:
            raise ValueError("unsupported vLLM captures: " + ", ".join(unknown))
        if len(capture_kinds) != len(set(capture_kinds)):
            raise ValueError("duplicate capture requests are not allowed")
        token_capture = "token_ids" in capture_kinds

        prompts: list[str] = []
        sampling_params: list[Any] = []
        invocation_rows: list[dict[str, Any]] = []
        for request in requests:
            if not isinstance(request, InferenceRequest):
                raise TypeError("requests must contain InferenceRequest objects")
            options = _generation_options(request, self._resolved["scenario"])
            rendered = _format_prompt(self._tokenizer, request.prompt, options)
            prompts.append(rendered)
            sampling_params.append(
                self._module.SamplingParams(
                    max_tokens=options["max_tokens"],
                    temperature=options["temperature"],
                    seed=options["seed"],
                )
            )
            invocation_rows.append(
                {
                    "prompt_sha256": _text_hash(request.prompt),
                    "rendered_prompt_sha256": _text_hash(rendered),
                    "system_sha256": _text_hash(options["system"]),
                    "generation": {
                        "max_tokens": options["max_tokens"],
                        "temperature": options["temperature"],
                        "seed": options["seed"],
                        "chat_template": options["chat_template"],
                    },
                }
            )

        if not prompts:
            return InferenceBatch(
                outputs=(),
                captures={"token_ids": ()} if token_capture else {},
                metadata={"runtime": "vllm", "invocations": ()},
            )

        responses = self._llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if len(responses) != len(prompts):
            raise RuntimeError(
                "vLLM returned a different number of responses than requests"
            )

        outputs: list[str] = []
        token_batches: list[tuple[int, ...]] = []
        frozen_rows: list[Mapping[str, Any]] = []
        for index, response in enumerate(responses):
            candidates = getattr(response, "outputs", None)
            if not candidates:
                raise RuntimeError(f"vLLM response[{index}] has no generated output")
            output = candidates[0]
            text = getattr(output, "text", None)
            token_ids = getattr(output, "token_ids", None)
            if not isinstance(text, str):
                raise RuntimeError(f"vLLM response[{index}] text is not a string")
            if isinstance(token_ids, (str, bytes)) or not isinstance(
                token_ids, Sequence
            ):
                raise RuntimeError(
                    f"vLLM response[{index}] does not expose output token IDs"
                )
            tokens: list[int] = []
            for token in token_ids:
                if isinstance(token, bool) or not isinstance(token, int):
                    raise RuntimeError(
                        f"vLLM response[{index}] contains a non-integer token ID"
                    )
                tokens.append(token)
            outputs.append(text)
            token_batches.append(tuple(tokens))
            row = {
                **invocation_rows[index],
                "output_tokens": len(tokens),
            }
            frozen = freeze_mapping(row)
            self._invocations.append(frozen)
            frozen_rows.append(frozen)

        captures: dict[str, Any] = {}
        if token_capture:
            captures["token_ids"] = tuple(token_batches)
        return InferenceBatch(
            outputs=tuple(outputs),
            captures=captures,
            metadata={
                "runtime": "vllm",
                "invocations": tuple(frozen_rows),
            },
        )

    def reset(self, scope: str = "measurement") -> None:
        """Record an isolation boundary; the offline engine remains loaded."""

        if self._closed:
            raise RuntimeError("vLLM session is closed")
        if not scope:
            raise ValueError("reset scope must not be empty")
        self._reset_count += 1

    def close(self) -> None:
        """Release session-owned engine references without claiming upstream shutdown."""

        if self._closed:
            return
        cleanup: dict[str, Any] = {
            "mode": "reference_release",
            "explicit_shutdown_called": False,
        }
        shutdown = (
            getattr(self._llm, "shutdown", None) if self._llm is not None else None
        )
        if callable(shutdown):
            shutdown()
            cleanup["mode"] = "public_shutdown"
            cleanup["explicit_shutdown_called"] = True
        self._tokenizer = None
        self._llm = None
        self._module = None
        gc.collect()
        self._cleanup = freeze_mapping(cleanup)
        self._closed = True

    def observation(self) -> Mapping[str, Any]:
        """Return configured state separately from authoritative identity evidence."""

        return freeze_mapping(
            {
                "runtime": {
                    "name": "vllm",
                    "version": self._resolved["runtime"]["version"],
                },
                "model": self._resolved["model"],
                "identity": self._identity.to_mapping(),
                "configured": {
                    "runtime": self._resolved["runtime"]["settings"],
                    "kv_cache": self._resolved["kv_cache"],
                },
                "applied": self._applied,
                "invocations": tuple(self._invocations),
                "reset_count": self._reset_count,
                "closed": self._closed,
                "cleanup": self._cleanup,
                "environment": self._environment,
            }
        )


class VLLMAdapter:
    """Resolve and launch optional in-process vLLM offline inference sessions."""

    name = "vllm"

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        """Validate recipe semantics and report optional dependency availability."""

        del environment
        installed_version = _vllm_version()
        evidence: dict[str, Any] = {
            "runtime": "vllm",
            "runtime_version": installed_version,
            "vllm_version": installed_version,
            "token_ids_capture": "native_output_token_ids",
        }
        reasons: list[str] = []
        try:
            _model_source(spec)
            runtime = _runtime_config(spec)
            kv = _kv_treatment(spec.treatments)
            requested_version = _requested_runtime_version(spec)
            if requested_version is not None and installed_version != requested_version:
                reasons.append(
                    "requested vLLM runtime version does not match installed distribution: "
                    f"requested {requested_version!r}, installed {installed_version!r}"
                )
            evidence.update(
                {
                    "requested_runtime_version": requested_version,
                    "max_model_len": runtime["max_model_len"],
                    "kv_cache_dtype": kv["dtype"],
                }
            )
        except (TypeError, ValueError) as exc:
            reasons.append(str(exc))

        if not _vllm_available():
            reasons.append(
                "vLLM is not installed in this environment; install it only in the "
                "runtime-specific environment used for this adapter"
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
        """Resolve exact model/runtime/KV settings without launching the engine."""

        support = self.probe(spec, environment)
        if support.status != "supported":
            raise ValueError("; ".join(support.reasons) or "vLLM run is unsupported")
        model = _model_source(spec)
        runtime = _runtime_config(spec)
        kv = _kv_treatment(spec.treatments)
        return freeze_mapping(
            {
                "runtime": {
                    "name": "vllm",
                    "version": _vllm_version(),
                    "requested_version": _requested_runtime_version(spec),
                    "settings": runtime,
                },
                "model": model,
                "scenario": spec.scenario,
                "kv_cache": kv,
                "support": support.evidence,
            }
        )

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> RuntimeSession:
        """Create one vLLM LLM instance owned exclusively by the returned session."""

        runtime = resolved.get("runtime")
        if not isinstance(runtime, Mapping) or runtime.get("name") != "vllm":
            raise ValueError("resolved runtime is not a vLLM specification")
        settings = runtime.get("settings")
        model = resolved.get("model")
        kv = resolved.get("kv_cache")
        if not isinstance(settings, Mapping):
            raise ValueError("resolved vLLM runtime settings are missing")
        if not isinstance(model, Mapping) or not isinstance(model.get("model"), str):
            raise ValueError("resolved vLLM model identity is missing")
        if not isinstance(kv, Mapping) or not isinstance(kv.get("dtype"), str):
            raise ValueError("resolved vLLM KV-cache configuration is missing")

        module = _load_vllm()
        llm_cls = getattr(module, "LLM", None)
        if not callable(llm_cls):
            raise RuntimeError("installed vLLM module does not expose LLM")
        kwargs: dict[str, Any] = {
            "model": model["model"],
            "dtype": settings["dtype"],
            "kv_cache_dtype": kv["dtype"],
            "max_model_len": settings["max_model_len"],
            "gpu_memory_utilization": settings["gpu_memory_utilization"],
            "max_num_seqs": settings["max_num_seqs"],
            "tensor_parallel_size": settings["tensor_parallel_size"],
            "enforce_eager": settings["enforce_eager"],
            "enable_prefix_caching": settings["enable_prefix_caching"],
            "trust_remote_code": settings["trust_remote_code"],
        }
        if model.get("revision") is not None:
            kwargs["revision"] = model["revision"]
        if model.get("requested_tokenizer_id") is not None:
            kwargs["tokenizer"] = model["tokenizer"]
        if model.get("tokenizer_revision") is not None:
            kwargs["tokenizer_revision"] = model["tokenizer_revision"]
        llm = llm_cls(**kwargs)
        try:
            return VLLMSession(resolved, environment, module, llm)
        except Exception:
            _cleanup_failed_llm(llm)
            raise

    def observe(self, session: RuntimeSession) -> Mapping[str, Any]:
        """Return configured state and independent applied-config introspection."""

        if not isinstance(session, VLLMSession):
            raise TypeError("VLLMAdapter can only observe VLLMSession")
        return session.observation()
