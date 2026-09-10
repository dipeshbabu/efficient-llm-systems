from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import metria.runtimes.llamacpp as llamacpp_module
import metria.runtimes.vllm as vllm_module
from metria import RunSpec, RunStatus, TreatmentSpec, TreatmentType, execute_run
from metria.measurements import TokenTrajectoryProtocol, compare_trajectory_results
from metria.protocols import CaptureRequest, InferenceRequest, MeasurementResult
from metria.runtimes.llamacpp import LlamaCppAdapter
from metria.runtimes.vllm import VLLMAdapter, VLLMSession


class FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeTokenizer:
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


class FakeLLM:
    instances: list[FakeLLM] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.tokenizer = FakeTokenizer()
        self.generate_calls: list[tuple[list[str], list[FakeSamplingParams]]] = []
        self.model_config = SimpleNamespace(
            model=kwargs["model"],
            dtype=kwargs["dtype"],
            max_model_len=kwargs["max_model_len"],
            revision=kwargs.get("revision"),
            tokenizer=kwargs["model"],
        )
        cache_config = SimpleNamespace(
            cache_dtype=kwargs["kv_cache_dtype"],
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
        self.__class__.instances.append(self)

    def get_tokenizer(self) -> FakeTokenizer:
        return self.tokenizer

    def generate(
        self,
        prompts: list[str],
        *,
        sampling_params: list[FakeSamplingParams],
        use_tqdm: bool,
    ) -> list[Any]:
        assert use_tqdm is False
        self.generate_calls.append((prompts, sampling_params))
        responses = []
        for index, _prompt in enumerate(prompts):
            tokens = [10, 20 + index]
            output = SimpleNamespace(text=f"answer-{index}", token_ids=tokens)
            responses.append(SimpleNamespace(outputs=[output]))
        return responses


class FakeLLMWithShutdown(FakeLLM):
    def shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeVLLMModule:
    SamplingParams = FakeSamplingParams
    LLM = FakeLLM


def _patch_vllm(monkeypatch: pytest.MonkeyPatch, module: Any = None) -> Any:
    fake_module = module or FakeVLLMModule
    monkeypatch.setattr(vllm_module, "_vllm_available", lambda: True)
    monkeypatch.setattr(vllm_module, "_vllm_version", lambda: "0.test")
    monkeypatch.setattr(vllm_module, "_load_vllm", lambda: fake_module)
    return fake_module


def _spec(
    *,
    kv_dtype: str = "auto",
    model: str = "example/model",
    measurement: str = "kv_fidelity.decode_time_trajectory",
) -> RunSpec:
    return RunSpec(
        model={"id": model, "revision": "abc123"},
        runtime={
            "name": "vllm",
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.8,
            "max_num_seqs": 8,
            "tensor_parallel_size": 1,
            "enable_prefix_caching": False,
        },
        scenario={"context": 1024, "max_tokens": 32},
        measurements=(measurement,),
        treatments=(
            TreatmentSpec(
                name="vllm.kv_cache",
                kind=TreatmentType.RUNTIME_FEATURE,
                config={"dtype": kv_dtype},
            ),
        ),
    )


def test_probe_reports_optional_dependency_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vllm_module, "_vllm_available", lambda: False)
    monkeypatch.setattr(vllm_module, "_vllm_version", lambda: None)

    report = VLLMAdapter().probe(_spec(), {})

    assert report.status == "unsupported"
    assert any("not installed" in reason for reason in report.reasons)
    assert report.evidence["token_ids_capture"] == "native_output_token_ids"


def test_resolve_records_exact_settings_without_importing_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch)
    imported = False

    def should_not_load() -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("resolve must not import vLLM")

    monkeypatch.setattr(vllm_module, "_load_vllm", should_not_load)

    resolved = VLLMAdapter().resolve(_spec(kv_dtype="fp8_e4m3"), {})

    assert imported is False
    assert resolved["runtime"]["version"] == "0.test"
    assert resolved["runtime"]["settings"]["max_model_len"] == 1088
    assert resolved["kv_cache"]["dtype"] == "fp8_e4m3"
    assert resolved["model"]["revision"] == "abc123"


def test_vllm_kv_treatment_is_native_and_fail_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch)
    bad = _spec(kv_dtype="q4_0")

    report = VLLMAdapter().probe(bad, {})

    assert report.status == "unsupported"
    assert any("unsupported vLLM KV-cache dtype" in reason for reason in report.reasons)


def test_launch_passes_resolved_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch)
    FakeLLM.instances.clear()
    adapter = VLLMAdapter()
    resolved = adapter.resolve(_spec(kv_dtype="fp8"), {})

    session = adapter.launch(resolved, {"hardware_class": "test-gpu"})

    assert isinstance(session, VLLMSession)
    llm = FakeLLM.instances[-1]
    assert llm.kwargs["model"] == "example/model"
    assert llm.kwargs["revision"] == "abc123"
    assert llm.kwargs["kv_cache_dtype"] == "fp8"
    assert llm.kwargs["max_model_len"] == 1088
    assert llm.kwargs["gpu_memory_utilization"] == 0.8
    assert llm.kwargs["max_num_seqs"] == 8
    assert llm.kwargs["enable_prefix_caching"] is False


def test_infer_batches_requests_and_redacts_prompt_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch)
    FakeLLM.instances.clear()
    adapter = VLLMAdapter()
    session = adapter.launch(adapter.resolve(_spec(), {}), {})

    batch = session.infer(
        (
            InferenceRequest(
                prompt="private prompt one",
                generation={"temperature": 0.0, "system": "private system"},
            ),
            InferenceRequest(prompt="private prompt two", generation={"max_tokens": 7}),
        ),
        capture=(CaptureRequest(kind="token_ids"),),
    )
    observed = adapter.observe(session)

    assert batch.outputs == ("answer-0", "answer-1")
    assert batch.captures["token_ids"] == ((10, 20), (10, 21))
    llm = FakeLLM.instances[-1]
    assert len(llm.generate_calls) == 1
    prompts, params = llm.generate_calls[0]
    assert prompts[0].startswith("CHAT:system=private system|user=private prompt one")
    assert params[1].kwargs["max_tokens"] == 7
    assert "private prompt one" not in repr(observed)
    assert "private prompt two" not in repr(observed)
    assert "private system" not in repr(observed)
    assert observed["invocations"][0]["prompt_sha256"]
    assert observed["invocations"][0]["rendered_prompt_sha256"]


def test_observation_separates_configured_from_introspected_applied_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch)
    adapter = VLLMAdapter()
    session = adapter.launch(adapter.resolve(_spec(kv_dtype="fp8"), {}), {})

    observed = adapter.observe(session)

    assert observed["configured"]["kv_cache"]["dtype"] == "fp8"
    assert observed["applied"]["status"] == "introspected"
    assert observed["applied"]["fields"]["cache.cache_dtype"] == "fp8"
    assert observed["applied"]["fields"]["parallel.tensor_parallel_size"] == 1


def test_unsupported_generation_and_capture_fields_fail_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch)
    adapter = VLLMAdapter()
    session = adapter.launch(adapter.resolve(_spec(), {}), {})

    with pytest.raises(ValueError, match="unsupported vLLM generation keys"):
        session.infer((InferenceRequest(prompt="x", generation={"timeout": 1.0}),))
    with pytest.raises(ValueError, match="unsupported vLLM captures"):
        session.infer(
            (InferenceRequest(prompt="x"),),
            capture=(CaptureRequest(kind="logprobs"),),
        )


def test_close_releases_instance_without_claiming_unavailable_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch)
    adapter = VLLMAdapter()
    session = adapter.launch(adapter.resolve(_spec(), {}), {})

    session.close()
    observed = adapter.observe(session)

    assert session.closed
    assert observed["cleanup"]["mode"] == "reference_release"
    assert observed["cleanup"]["explicit_shutdown_called"] is False
    with pytest.raises(RuntimeError, match="session is closed"):
        session.infer((InferenceRequest(prompt="x"),))


def test_close_uses_public_shutdown_if_future_runtime_exposes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = SimpleNamespace(
        LLM=FakeLLMWithShutdown,
        SamplingParams=FakeSamplingParams,
    )
    _patch_vllm(monkeypatch, fake_module)
    FakeLLMWithShutdown.instances.clear()
    adapter = VLLMAdapter()
    session = adapter.launch(adapter.resolve(_spec(), {}), {})
    llm = FakeLLMWithShutdown.instances[-1]

    session.close()

    assert llm.shutdown_calls == 1
    assert adapter.observe(session)["cleanup"]["mode"] == "public_shutdown"


def test_execute_run_with_vllm_and_trajectory_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch)
    protocol = TokenTrajectoryProtocol()

    record = execute_run(
        study_name="vllm-study",
        run_id="vllm-run",
        spec=_spec(measurement=protocol.name),
        adapter=VLLMAdapter(),
        measurement=protocol,
        measurement_config={
            "prompts": (
                {"id": "p1", "prompt": "private one"},
                {"id": "p2", "prompt": "private two"},
            )
        },
        environment={"hardware_class": "fake-gpu"},
    )

    assert record.status is RunStatus.COMPLETED
    assert record.observed["runtime"]["name"] == "vllm"
    assert record.metrics["trajectory_mean_steps"].value == 2.0
    assert record.provenance["preflight"]["captures"]["status"] == "supported"
    evidence = record.evidence["measurements"][protocol.name]
    assert evidence["prompts"][0]["token_ids"] == (10, 20)
    assert "private one" not in repr(evidence)


def _llamacpp_spec(tmp_path: Path, protocol: TokenTrajectoryProtocol) -> RunSpec:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("llama-cli", "llama-completion"):
        (bin_dir / name).write_bytes(b"fake")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake model")
    return RunSpec(
        model={"path": str(model)},
        runtime={"name": "llamacpp", "bin_dir": str(bin_dir)},
        scenario={"context": 512, "max_tokens": 8},
        measurements=(protocol.name,),
    )


def test_same_trajectory_protocol_compares_llamacpp_and_vllm_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch)
    protocol = TokenTrajectoryProtocol()

    def fake_llama_run(
        argv: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        trajectory_path = kwargs["env"].get("KV_FIDELITY_TRAJECTORY")
        if trajectory_path:
            Path(trajectory_path).write_text(
                json.dumps({"step": 0, "token_id": 10})
                + "\n"
                + json.dumps({"step": 1, "token_id": 20})
                + "\n",
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="| llama\n", stderr="")

    monkeypatch.setattr(llamacpp_module.subprocess, "run", fake_llama_run)
    measurement_config = {"prompts": ({"id": "p1", "prompt": "same prompt"},)}
    llama_record = execute_run(
        study_name="cross-runtime",
        run_id="llama",
        spec=_llamacpp_spec(tmp_path, protocol),
        adapter=LlamaCppAdapter(),
        measurement=protocol,
        measurement_config=measurement_config,
        environment={
            "hardware_class": "fake-gpu",
            "llama_cpp_token_ids_capture_sha256": hashlib.sha256(b"fake").hexdigest(),
        },
    )
    vllm_record = execute_run(
        study_name="cross-runtime",
        run_id="vllm",
        spec=_spec(measurement=protocol.name),
        adapter=VLLMAdapter(),
        measurement=protocol,
        measurement_config=measurement_config,
        environment={"hardware_class": "fake-gpu"},
    )

    llama_result = MeasurementResult(
        evidence=llama_record.evidence["measurements"][protocol.name]
    )
    vllm_result = MeasurementResult(
        evidence=vllm_record.evidence["measurements"][protocol.name]
    )
    comparison = compare_trajectory_results(llama_result, vllm_result)

    assert llama_record.status is RunStatus.COMPLETED
    assert vllm_record.status is RunStatus.COMPLETED
    assert llama_record.provenance["preflight"]["captures"]["status"] == "supported"
    assert comparison.metrics["trajectory_agreement_score"].value == 100.0
