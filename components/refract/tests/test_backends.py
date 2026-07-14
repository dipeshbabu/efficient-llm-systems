"""Tests for refract.backends: registry, base ABC defaults, llamacpp adapter,
mlx KV translation + import-gate, vllm + sglang surface checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from refract.backends import auto_backend, get_backend
from refract.backends.base import (
    Backend,
    BackendCapabilityError,
    CompletionResult,
    KLDResult,
    TrajectoryResult,
    _full_token_chunks,
    approximate_topk_kl,
)
from refract.backends.llamacpp import LlamaCppBackend
from refract.backends.sglang import SGLangBackend, _validate_kv_str
from refract.backends.vllm import VLLMBackend, _kv_str_to_vllm_dtype

# --- shared KLD chunking --------------------------------------------------


@pytest.mark.parametrize(
    ("token_ids", "chunk_len", "max_chunks", "expected"),
    [
        ([0, 1, 2], 4, 3, []),
        ([0, 1, 2, 3], 4, 3, [[0, 1, 2, 3]]),
        (list(range(8)), 4, 3, [list(range(4)), list(range(4, 8))]),
        (list(range(10)), 4, 3, [list(range(4)), list(range(4, 8))]),
        (list(range(12)), 4, 2, [list(range(4)), list(range(4, 8))]),
    ],
)
def test_full_token_chunks(
    token_ids: list[int],
    chunk_len: int,
    max_chunks: int,
    expected: list[list[int]],
):
    assert (
        _full_token_chunks(token_ids, chunk_len=chunk_len, max_chunks=max_chunks)
        == expected
    )


def test_full_token_chunks_rejects_non_positive_chunk_length():
    with pytest.raises(ValueError, match="chunk_len must be positive"):
        _full_token_chunks([0], chunk_len=0, max_chunks=1)


# --- registry -------------------------------------------------------------


def test_get_backend_llamacpp():
    assert isinstance(get_backend("llamacpp"), LlamaCppBackend)


def test_get_backend_case_insensitive():
    assert isinstance(get_backend("LLAMACPP"), LlamaCppBackend)


def test_get_backend_vllm():
    assert isinstance(get_backend("vllm"), VLLMBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("nothing")


def test_auto_backend_gguf_picks_llamacpp(tmp_path, monkeypatch):
    monkeypatch.delenv("REFRACT_BACKEND", raising=False)
    p = tmp_path / "model.gguf"
    p.write_text("x")
    bk = auto_backend(p)
    assert bk.name == "llamacpp"


def test_auto_backend_directory_with_config_picks_mlx(tmp_path, monkeypatch):
    monkeypatch.delenv("REFRACT_BACKEND", raising=False)
    d = tmp_path / "mlx_model"
    d.mkdir()
    (d / "config.json").write_text(
        '{"quantization": {"bits": 4, "group_size": 64}}',
        encoding="utf-8",
    )
    bk = auto_backend(d)
    assert bk.name == "mlx"


def test_auto_backend_plain_hf_directory_picks_vllm(tmp_path, monkeypatch):
    monkeypatch.delenv("REFRACT_BACKEND", raising=False)
    d = tmp_path / "hf_model"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "qwen2"}', encoding="utf-8")
    assert auto_backend(d).name == "vllm"


def test_auto_backend_unknown_falls_back_to_vllm(tmp_path, monkeypatch):
    monkeypatch.delenv("REFRACT_BACKEND", raising=False)
    p = tmp_path / "weird.bin"
    p.write_text("x")
    bk = auto_backend(p)
    assert bk.name == "vllm"


def test_auto_backend_env_override_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("REFRACT_BACKEND", "vllm")
    p = tmp_path / "model.gguf"  # would normally be llamacpp
    p.write_text("x")
    bk = auto_backend(p)
    assert bk.name == "vllm"


# --- vllm backend (production v0.3.2.1) -----------------------------------


def test_vllm_kv_str_to_dtype_known_mappings():
    """Each KV config the score axes will pass should round-trip through the
    vllm backend's translation table."""
    assert _kv_str_to_vllm_dtype("ctk=f16,ctv=f16") == "auto"
    assert _kv_str_to_vllm_dtype("ctk=bf16,ctv=bf16") == "auto"
    assert _kv_str_to_vllm_dtype("ctk=q8_0,ctv=q8_0") == "fp8_e4m3"


def test_vllm_kv_str_to_dtype_turboquant_presets():
    """TurboQuant + TQ+ presets resolve to the matching vllm kv_cache_dtype."""
    assert _kv_str_to_vllm_dtype("ctk=q8_0,ctv=turbo4") == "turboquant_k8v4"
    assert _kv_str_to_vllm_dtype("ctk=turbo4,ctv=turbo4") == "turboquant_4bit_nc"
    assert _kv_str_to_vllm_dtype("ctk=turbo3,ctv=turbo3") == "turboquant_3bit_nc"


def test_vllm_kv_str_unknown_raises_capability():
    with pytest.raises(BackendCapabilityError) as exc:
        _kv_str_to_vllm_dtype("ctk=somethingweird,ctv=alsoweird")
    assert "no mapping" in str(exc.value).lower()


def test_vllm_run_completion_signature_takes_kv_config():
    """v0.3.2.1+: run_completion takes the same kw-only contract as the
    base class. We don't actually invoke vllm here (no GPU in test); we
    just confirm the method accepts the contract args."""
    import inspect

    sig = inspect.signature(VLLMBackend.run_completion)
    expected = {"model", "prompt", "kv_config_str"}
    assert expected.issubset(set(sig.parameters)), (
        f"VLLMBackend.run_completion missing required kw args: "
        f"{expected - set(sig.parameters)}"
    )


def test_vllm_methods_present():
    bk = VLLMBackend()
    for meth in (
        "run_completion",
        "run_completion_trajectory",
        "run_kld",
        "tokenize_to_ids",
        "model_metadata",
    ):
        assert callable(getattr(bk, meth)), f"VLLMBackend missing {meth}"


# --- sglang backend (production v0.3.2.1) ---------------------------------


def test_sglang_get_backend_registers():
    assert isinstance(get_backend("sglang"), SGLangBackend)


def test_sglang_kv_str_supported_configs():
    """SGLang accepts BF16 and fp8_e4m3 KV. f16/bf16 maps to bf16."""
    assert _validate_kv_str("ctk=f16,ctv=f16") == ("f16", "f16")
    assert _validate_kv_str("ctk=bf16,ctv=bf16") == ("bf16", "bf16")
    assert _validate_kv_str("ctk=q8_0,ctv=q8_0") == ("q8_0", "q8_0")


def test_sglang_kv_str_rejects_turboquant():
    """SGLang has no TurboQuant KV path; the backend rejects loudly."""
    with pytest.raises(BackendCapabilityError) as exc:
        _validate_kv_str("ctk=turbo4,ctv=turbo4")
    assert "turboquant" in str(exc.value).lower()


def test_sglang_run_kld_requires_dual_urls(monkeypatch):
    """run_kld needs two simultaneous SGLang servers (KV dtype is fixed
    at server launch); raise loudly if env vars aren't set."""
    monkeypatch.delenv("REFRACT_SGLANG_REF_URL", raising=False)
    monkeypatch.delenv("REFRACT_SGLANG_CAND_URL", raising=False)
    bk = SGLangBackend()
    with pytest.raises(BackendCapabilityError) as exc:
        bk.run_kld(
            model=Path("/tmp/x"),
            corpus=Path("/tmp/x"),
            ref_kv_str="ctk=f16,ctv=f16",
            cand_kv_str="ctk=q8_0,ctv=q8_0",
        )
    assert "REFRACT_SGLANG_REF_URL" in str(exc.value)


def test_sglang_methods_present():
    bk = SGLangBackend()
    for meth in (
        "run_completion",
        "run_completion_trajectory",
        "run_kld",
        "tokenize_to_ids",
        "model_metadata",
    ):
        assert callable(getattr(bk, meth)), f"SGLangBackend missing {meth}"


def test_sglang_completion_and_trajectory_share_prompt_ids(monkeypatch):
    from refract.backends import sglang as module

    monkeypatch.setattr(module, "_prompt_token_ids", lambda *a, **kw: [4, 5, 6])
    calls = []

    def fake_post(url, path, body, *, timeout_s):
        calls.append((path, body))
        if body.get("return_logprob"):
            return {
                "text": "ok",
                "meta_info": {"output_token_logprobs": [[-0.1, 9, "x"]]},
            }
        return {"text": "ok", "meta_info": {"completion_tokens": 1}}

    monkeypatch.setattr(module, "_post", fake_post)
    backend = SGLangBackend()
    completion = backend.run_completion(
        model=Path("org/model"),
        prompt="hello",
        kv_config_str="ctk=f16,ctv=f16",
    )
    trajectory = backend.run_completion_trajectory(
        model=Path("org/model"),
        prompt="hello",
        kv_config_str="ctk=f16,ctv=f16",
    )
    assert completion.text == "ok"
    assert trajectory.token_ids == [9]
    assert calls[0][0] == calls[1][0] == "/generate"
    assert calls[0][1]["input_ids"] == calls[1][1]["input_ids"] == [4, 5, 6]


# --- base.Backend default detect_thinking_mode ----------------------------


class _ConcreteBackend(Backend):
    """Minimal concrete backend that lets the default detect_thinking_mode +
    model_metadata exercise their default code paths."""

    name = "test"

    def __init__(self, completion_text: str = "no thinking here"):
        self._text = completion_text

    def run_completion(self, **_kw) -> CompletionResult:
        return CompletionResult(text=self._text, n_tokens=0, metadata={})

    def run_completion_trajectory(self, **_kw) -> TrajectoryResult:
        return TrajectoryResult(token_ids=[], metadata={})

    def run_kld(self, **_kw) -> KLDResult:
        return KLDResult(mean_kld=0.0)

    def tokenize_to_ids(self, **_kw):
        return []


def test_base_detect_thinking_mode_no_markers():
    bk = _ConcreteBackend(completion_text="4")
    detected, markers = bk.detect_thinking_mode(model=Path("m"))
    assert detected is False
    assert markers == []


def test_base_detect_thinking_mode_finds_marker():
    bk = _ConcreteBackend(completion_text="<think>hmm</think> 4")
    detected, markers = bk.detect_thinking_mode(model=Path("m"))
    assert detected is True
    assert "<think>" in markers


def test_base_detect_thinking_mode_swallows_exception():
    class _Boom(_ConcreteBackend):
        def run_completion(self, **_kw):
            raise RuntimeError("boom")

    bk = _Boom()
    detected, markers = bk.detect_thinking_mode(model=Path("m"))
    assert detected is False
    assert markers == []


def test_base_model_metadata_default_shape():
    bk = _ConcreteBackend()
    info = bk.model_metadata(model=Path("/x/y.gguf"))
    assert info["backend"] == "test"
    assert info["model"] == "/x/y.gguf"


# --- backend dataclasses --------------------------------------------------


def test_completion_result_defaults():
    r = CompletionResult(text="hi", n_tokens=2)
    assert r.metadata == {}


def test_kld_result_defaults():
    r = KLDResult(mean_kld=0.5)
    assert r.ppl is None
    assert r.metadata == {}


def test_approximate_topk_kl_is_normalized_and_zero_for_equal_inputs():
    logp = {1: -0.2, 2: -2.0}
    assert approximate_topk_kl(logp, logp) == pytest.approx(0.0, abs=1e-12)


def test_approximate_topk_kl_detects_distribution_shift():
    ref = {1: -0.1, 2: -3.0}
    cand = {1: -3.0, 2: -0.1}
    assert approximate_topk_kl(ref, cand) > 1.0


def test_trajectory_result_defaults():
    r = TrajectoryResult(token_ids=[1, 2])
    assert r.metadata == {}


def test_backend_capability_error_is_runtime_error():
    assert issubclass(BackendCapabilityError, RuntimeError)


# --- llamacpp adapter (delegates to runner; mock the runner) --------------


def test_llamacpp_run_completion_delegates(monkeypatch):
    bk = LlamaCppBackend()
    captured = {}

    def fake_rc(*, model, prompt, kv, **kw):
        captured["kv_label"] = kv.label()
        captured["prompt"] = prompt
        return ("hello world", {"returncode": 0})

    monkeypatch.setattr("refract.runner.run_completion", fake_rc)
    res = bk.run_completion(
        model=Path("/m.gguf"),
        prompt="hi",
        kv_config_str="ctk=q8_0,ctv=q8_0",
    )
    assert res.text == "hello world"
    assert "ctk=q8_0" in captured["kv_label"]


def test_llamacpp_run_completion_trajectory_delegates(monkeypatch):
    bk = LlamaCppBackend()

    def fake_rct(*, model, prompt, kv, **kw):
        return ([1, 2, 3], {"returncode": 0, "n_tokens": 3})

    monkeypatch.setattr("refract.runner.run_completion_trajectory", fake_rct)
    res = bk.run_completion_trajectory(
        model=Path("/m.gguf"),
        prompt="hi",
        kv_config_str="ctk=f16,ctv=f16",
    )
    assert res.token_ids == [1, 2, 3]


def test_llamacpp_tokenize_delegates(monkeypatch):
    bk = LlamaCppBackend()
    monkeypatch.setattr(
        "refract.runner.tokenize_to_ids", lambda model, text, timeout=120.0: [4, 5, 6]
    )
    assert bk.tokenize_to_ids(model=Path("/m"), text="x") == [4, 5, 6]


def test_llamacpp_model_metadata_includes_backend_name():
    bk = LlamaCppBackend()
    info = bk.model_metadata(model=Path("/m.gguf"))
    assert info["backend"] == "llamacpp"
    assert "llama_cpp_bin_dir" in info


# --- mlx backend (translate KV; import-gate) ------------------------------


def test_mlx_translate_kv_symmetric_q8():
    from refract.backends.mlx import _translate_kv_to_mlx

    out = _translate_kv_to_mlx("ctk=q8_0,ctv=q8_0")
    assert out["kv_bits"] == 8
    assert out["kv_group_size"] == 64


def test_mlx_translate_kv_f16_no_quant():
    from refract.backends.mlx import _translate_kv_to_mlx

    out = _translate_kv_to_mlx("ctk=f16,ctv=f16")
    assert out["kv_bits"] is None


def test_mlx_translate_kv_asymmetric_raises():
    from refract.backends.mlx import _translate_kv_to_mlx

    with pytest.raises(BackendCapabilityError):
        _translate_kv_to_mlx("ctk=q8_0,ctv=turbo4")


def test_mlx_translate_kv_turbo_raises():
    from refract.backends.mlx import _translate_kv_to_mlx

    with pytest.raises(BackendCapabilityError):
        _translate_kv_to_mlx("ctk=turbo4,ctv=turbo4")


def test_mlx_translate_kv_unknown_type_raises():
    from refract.backends.mlx import _translate_kv_to_mlx

    with pytest.raises(BackendCapabilityError):
        _translate_kv_to_mlx("ctk=q9_99,ctv=q9_99")


def test_mlx_require_mlx_raises_if_missing(monkeypatch):
    """When mlx-lm isn't installed, _require_mlx must raise BackendCapabilityError."""
    import sys

    from refract.backends import mlx as mlx_mod

    # Hide mlx + mlx_lm from sys.modules + meta_path so import fails.
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    with pytest.raises(BackendCapabilityError):
        mlx_mod._require_mlx()
