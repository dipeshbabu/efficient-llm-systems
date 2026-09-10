from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

import metria.runtimes.vllm as vllm_module
from metria import MeasurementResult, RunSpec, RunStatus, execute_run
from metria.protocols import CaptureRequest, InferenceBatch, InferenceRequest
from metria.runtimes.vllm import VLLMAdapter


class _SamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _Tokenizer:
    def __init__(
        self,
        *,
        name: str = "example/model",
        revision: str | None = "abc123",
        chat_template: str | None = "{{ messages }}",
    ) -> None:
        self.name_or_path = name
        self.init_kwargs = {} if revision is None else {"_commit_hash": revision}
        if chat_template is not None:
            self.chat_template = chat_template

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return "CHAT:" + "|".join(
            f"{message['role']}={message['content']}" for message in messages
        )


class _LLM:
    instances: list[_LLM] = []
    observed_model_override: str | None = None
    observed_revision_override: str | None = None
    observed_tokenizer_override: str | None = None
    observed_tokenizer_revision_override: str | None = None
    observed_cache_dtype_override: str | None = None
    omit_tokenizer_revision = False
    omit_chat_template = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        tokenizer_name = kwargs.get("tokenizer", kwargs["model"])
        tokenizer_revision = kwargs.get("tokenizer_revision", kwargs.get("revision"))
        self.tokenizer = _Tokenizer(
            name=self.observed_tokenizer_override or tokenizer_name,
            revision=(
                None
                if self.omit_tokenizer_revision
                else self.observed_tokenizer_revision_override or tokenizer_revision
            ),
            chat_template=None if self.omit_chat_template else "{{ messages }}",
        )
        model_kwargs: dict[str, Any] = {
            "model": self.observed_model_override or kwargs["model"],
            "dtype": kwargs["dtype"],
            "max_model_len": kwargs["max_model_len"],
            "revision": self.observed_revision_override or kwargs.get("revision"),
            "tokenizer": self.observed_tokenizer_override or tokenizer_name,
        }
        if not self.omit_tokenizer_revision:
            model_kwargs["tokenizer_revision"] = (
                self.observed_tokenizer_revision_override or tokenizer_revision
            )
        self.model_config = SimpleNamespace(**model_kwargs)
        cache_config = SimpleNamespace(
            cache_dtype=self.observed_cache_dtype_override or kwargs["kv_cache_dtype"],
            gpu_memory_utilization=kwargs["gpu_memory_utilization"],
            enable_prefix_caching=kwargs["enable_prefix_caching"],
        )
        parallel_config = SimpleNamespace(
            tensor_parallel_size=kwargs["tensor_parallel_size"]
        )
        self.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                cache_config=cache_config,
                parallel_config=parallel_config,
            )
        )
        self.shutdown_calls = 0
        self.generate_calls = 0
        self.__class__.instances.append(self)

    def get_tokenizer(self) -> _Tokenizer:
        return self.tokenizer

    def generate(
        self,
        prompts: list[str],
        *,
        sampling_params: list[_SamplingParams],
        use_tqdm: bool,
    ) -> list[Any]:
        assert use_tqdm is False
        assert len(prompts) == len(sampling_params)
        self.generate_calls += 1
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(text="ok", token_ids=[1, 2])]
            )
            for _ in prompts
        ]

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _Module:
    __version__ = "0.test"
    LLM = _LLM
    SamplingParams = _SamplingParams


def _reset_llm() -> None:
    _LLM.instances.clear()
    _LLM.observed_model_override = None
    _LLM.observed_revision_override = None
    _LLM.observed_tokenizer_override = None
    _LLM.observed_tokenizer_revision_override = None
    _LLM.observed_cache_dtype_override = None
    _LLM.omit_tokenizer_revision = False
    _LLM.omit_chat_template = False


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module: Any = _Module,
) -> None:
    _reset_llm()
    monkeypatch.setattr(vllm_module, "_vllm_available", lambda: True)
    monkeypatch.setattr(vllm_module, "_vllm_version", lambda: "0.test")
    monkeypatch.setattr(vllm_module, "_load_vllm", lambda: module)


def _spec(
    *,
    runtime_version: str | None = "0.test",
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
) -> RunSpec:
    model: dict[str, Any] = {"id": "example/model", "revision": "abc123"}
    if tokenizer_id is not None:
        model["tokenizer_id"] = tokenizer_id
    if tokenizer_revision is not None:
        model["tokenizer_revision"] = tokenizer_revision
    runtime: dict[str, Any] = {
        "name": "vllm",
        "dtype": "bfloat16",
        "max_model_len": 1024,
    }
    if runtime_version is not None:
        runtime["version"] = runtime_version
    return RunSpec(
        model=model,
        runtime=runtime,
        scenario={"max_tokens": 8},
        measurements=("identity.measurement",),
    )


class _Measurement:
    name = "identity.measurement"
    version = "1"

    def __init__(self) -> None:
        self.execute_calls = 0

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]:
        del config
        return ()

    def execute(
        self,
        session: Any,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        del scenario, config
        self.execute_calls += 1
        batch = session.infer((InferenceRequest(prompt="private prompt"),))
        assert isinstance(batch, InferenceBatch)
        return MeasurementResult()


def _execute(measurement: _Measurement) -> Any:
    return execute_run(
        study_name="identity-gate",
        run_id="candidate",
        spec=_spec(),
        adapter=VLLMAdapter(),
        measurement=measurement,
        measurement_config={},
        environment={},
    )


def test_requested_runtime_version_mismatch_fails_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch)
    adapter = VLLMAdapter()

    report = adapter.probe(_spec(runtime_version="different"), {})

    assert report.status.value == "unsupported"
    assert "does not match installed distribution" in report.reasons[0]
    assert report.evidence["runtime_version"] == "0.test"


def test_explicit_tokenizer_is_resolved_and_passed_to_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch)
    adapter = VLLMAdapter()
    spec = _spec(tokenizer_id="example/tokenizer", tokenizer_revision="tok456")

    resolved = adapter.resolve(spec, {})
    session = adapter.launch(resolved, {})

    llm = _LLM.instances[-1]
    assert resolved["model"]["tokenizer"] == "example/tokenizer"
    assert resolved["model"]["tokenizer_revision"] == "tok456"
    assert llm.kwargs["tokenizer"] == "example/tokenizer"
    assert llm.kwargs["tokenizer_revision"] == "tok456"
    session.close()


def test_explicit_tokenizer_without_revision_does_not_inherit_model_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch)
    adapter = VLLMAdapter()
    spec = _spec(tokenizer_id="example/tokenizer")

    resolved = adapter.resolve(spec, {})

    assert resolved["model"]["tokenizer_revision"] is None


def test_vllm_observation_retains_authority_and_chat_template_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch)
    adapter = VLLMAdapter()
    session = adapter.launch(adapter.resolve(_spec(), {}), {})

    observed = adapter.observe(session)
    identity = observed["identity"]

    assert identity["schema"] == "metria.runtime_identity.v1"
    assert identity["status"] == "partial"
    assert identity["model"]["status"] == "verified"
    assert identity["model"]["identifier"] == "example/model"
    assert identity["tokenizer"]["status"] == "verified"
    assert identity["runtime"]["status"] == "verified"
    assert identity["runtime"]["version"] == "0.test"
    assert identity["chat_template"]["status"] == "verified"
    assert identity["chat_template"]["sha256"] == hashlib.sha256(
        b"{{ messages }}"
    ).hexdigest()
    assert identity["applied"]["status"] == "partial"
    assert "cache.cache_dtype" in identity["applied"]["checked_fields"]
    assert "{{ messages }}" not in repr(identity)
    session.close()


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("observed_model_override", "wrong/model"),
        ("observed_revision_override", "wrong-revision"),
        ("observed_tokenizer_override", "wrong/tokenizer"),
        ("observed_tokenizer_revision_override", "wrong-tokenizer-revision"),
        ("observed_cache_dtype_override", "fp8"),
    ],
)
def test_vllm_identity_mismatch_fails_before_measurement_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    wrong_value: str,
) -> None:
    _patch_runtime(monkeypatch)
    setattr(_LLM, field, wrong_value)
    measurement = _Measurement()

    record = _execute(measurement)

    assert record.status is RunStatus.FAILED
    assert measurement.execute_calls == 0
    assert _LLM.instances[-1].generate_calls == 0
    assert _LLM.instances[-1].shutdown_calls == 1
    assert record.events[-1]["stage"] == "launch"


def test_loaded_module_version_mismatch_fails_before_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_module = SimpleNamespace(
        __version__="wrong-version",
        LLM=_LLM,
        SamplingParams=_SamplingParams,
    )
    _patch_runtime(monkeypatch, module=wrong_module)
    measurement = _Measurement()

    record = _execute(measurement)

    assert record.status is RunStatus.FAILED
    assert measurement.execute_calls == 0
    assert _LLM.instances[-1].shutdown_calls == 1


def test_missing_tokenizer_revision_and_template_remain_unknown_not_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch)
    _LLM.omit_tokenizer_revision = True
    _LLM.omit_chat_template = True
    adapter = VLLMAdapter()

    session = adapter.launch(adapter.resolve(_spec(), {}), {})
    identity = adapter.observe(session)["identity"]

    assert identity["status"] == "partial"
    assert identity["tokenizer"]["status"] == "partial"
    assert identity["tokenizer"]["revision"] is None
    assert identity["chat_template"]["status"] == "unknown"
    session.close()
