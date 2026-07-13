"""Tests for MLXBackend methods using mocked mlx + mlx_lm modules.

mlx-lm doesn't have to be installed for these to run; we inject fake
modules so the backend's adapter logic gets exercised. Tests focus on
the dispatch glue (chat template, KV translation, token streaming,
softmax-based KLD) — the real ML primitives are out of scope.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

import refract.backends.mlx as mlx_mod
from refract.backends.base import (
    BackendCapabilityError,
    CompletionResult,
    KLDResult,
    TrajectoryResult,
)
from refract.backends.mlx import (
    MLXBackend,
    _apply_chat_template,
    _load_model,
    _translate_kv_to_mlx,
)


# --- _apply_chat_template (no mlx required) ------------------------------


class _FakeTokenizer:
    def __init__(self, ids=None):
        self._ids = ids or [1, 2, 3]
        self.last_messages = None
        self.last_kw = None

    def apply_chat_template(self, messages, **kw):
        self.last_messages = messages
        self.last_kw = kw
        return "RENDERED:" + " | ".join(m["content"] for m in messages)

    def encode(self, text, **kw):
        return list(self._ids)


def test_apply_chat_template_user_only():
    tok = _FakeTokenizer()
    out = _apply_chat_template(tok, "hi", None)
    assert out == "RENDERED:hi"
    assert len(tok.last_messages) == 1
    assert tok.last_messages[0]["role"] == "user"


def test_apply_chat_template_with_system():
    tok = _FakeTokenizer()
    out = _apply_chat_template(tok, "question", "you are helpful")
    assert "you are helpful" in out
    assert "question" in out
    assert tok.last_messages[0]["role"] == "system"
    assert tok.last_messages[1]["role"] == "user"


# --- mock the mlx + mlx_lm imports the backend uses ---------------------


@pytest.fixture
def fake_mlx(monkeypatch):
    """Inject fake mlx + mlx_lm modules so MLXBackend methods can run.

    Returns (mx_module, mlx_lm_module, cache_module) so tests can wire up
    per-test behaviour.
    """
    # mx.random.seed + mx.array + mx.softmax + mx.log + mx.sum + mx.mean
    mx = types.SimpleNamespace()
    mx.random = types.SimpleNamespace(seed=mock.MagicMock())

    class _MxArray:
        def __init__(self, data):
            self.data = data
        def __getitem__(self, idx):
            return self
        def item(self):
            # Used to coerce mx scalars to python floats. Return a small
            # non-zero value so the v0.3.2.3 KLD validity-counting path
            # (n_valid = mx.sum(finite_mask).item()) treats the chunk as
            # contributing rather than skipping every chunk as non-finite.
            return 1.0

    mx.array = lambda data: _MxArray(data)
    mx.softmax = lambda a, axis=-1: a
    mx.log = lambda a: a
    mx.sum = lambda *a, **kw: _MxArray("sum")
    mx.mean = lambda a: _MxArray("mean")
    # Numerically-stable KL stubs added in v0.3.2.3.
    mx.logsumexp = lambda a, axis=-1, keepdims=False: a
    mx.exp = lambda a: a
    mx.isfinite = lambda a: a
    mx.where = lambda mask, a, b: a
    mx.zeros_like = lambda a: a

    # mlx_lm.generate / stream_generate / load
    mlx_lm = types.SimpleNamespace()

    def _generate(m, tok, **kw):
        return "fake completion text<|im_end|>"
    mlx_lm.generate = _generate

    def _stream_generate(m, tok, **kw):
        # Yield 3 records each carrying a token ID.
        for tid in (10, 20, 30):
            yield types.SimpleNamespace(token=tid)
    mlx_lm.stream_generate = _stream_generate

    fake_tok = _FakeTokenizer()
    fake_model = mock.MagicMock(name="model")
    fake_model.return_value = _MxArray("logits")  # m(inp, cache=...) → logits

    def _load(path):
        return fake_model, fake_tok
    mlx_lm.load = _load

    # cache module
    cache_mod = types.SimpleNamespace()
    cache_mod.make_prompt_cache = lambda m: ["fake_cache_obj"]
    cache_mod.maybe_quantize_kv_cache = mock.MagicMock()
    mlx_lm.models = types.SimpleNamespace(cache=cache_mod)

    # _require_mlx returns this triple — patch the function directly so
    # tests don't have to wrestle with sys.modules + lazy imports.
    monkeypatch.setattr(mlx_mod, "_require_mlx",
                        lambda: (mx, mlx_lm, cache_mod))
    # Also expose mlx_lm via the import inside _load_model.
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    # Reset the model cache between tests.
    mlx_mod._MODEL_CACHE.clear()

    return mx, mlx_lm, cache_mod


# --- _load_model caches across calls ------------------------------------


def test_load_model_caches(fake_mlx):
    p = Path("/m.gguf")
    a = _load_model(p)
    b = _load_model(p)
    assert a is b  # same cached tuple


# --- MLXBackend.run_completion ------------------------------------------


def test_mlx_run_completion_strips_im_end_marker(fake_mlx):
    bk = MLXBackend()
    res = bk.run_completion(
        model=Path("/m"), prompt="hi",
        kv_config_str="ctk=q8_0,ctv=q8_0",
        n_predict=10,
    )
    assert isinstance(res, CompletionResult)
    assert "fake completion text" in res.text
    assert "<|im_end|>" not in res.text
    # KV bits propagated through metadata.
    assert res.metadata["kv_kwargs"]["kv_bits"] == 8


def test_mlx_run_completion_without_chat_template(fake_mlx):
    bk = MLXBackend()
    res = bk.run_completion(
        model=Path("/m"), prompt="raw prompt",
        kv_config_str="ctk=f16,ctv=f16",
        apply_chat_template=False,
    )
    assert "fake completion" in res.text
    # kv_bits=None for f16
    assert res.metadata["kv_kwargs"]["kv_bits"] is None


def test_mlx_run_completion_asymmetric_raises(fake_mlx):
    bk = MLXBackend()
    with pytest.raises(BackendCapabilityError):
        bk.run_completion(
            model=Path("/m"), prompt="x",
            kv_config_str="ctk=q8_0,ctv=q4_0",
        )


# --- MLXBackend.run_completion_trajectory --------------------------------


def test_mlx_trajectory_collects_token_ids(fake_mlx):
    bk = MLXBackend()
    res = bk.run_completion_trajectory(
        model=Path("/m"), prompt="x",
        kv_config_str="ctk=q8_0,ctv=q8_0",
    )
    assert isinstance(res, TrajectoryResult)
    assert res.token_ids == [10, 20, 30]
    assert res.metadata["n_tokens"] == 3


def test_mlx_trajectory_handles_mx_array_token_scalar(fake_mlx, monkeypatch):
    """`record.token` is sometimes an mx.array scalar — int(tid) raises
    TypeError, so the code falls back to tid.item()."""
    class _MxScalar:
        def __init__(self, v):
            self._v = v
        def __int__(self):
            raise TypeError("not directly convertible")
        def item(self):
            return self._v

    def _stream_with_scalars(m, tok, **kw):
        for v in (7, 8):
            yield types.SimpleNamespace(token=_MxScalar(v))

    monkeypatch.setattr(fake_mlx[1], "stream_generate", _stream_with_scalars)
    bk = MLXBackend()
    res = bk.run_completion_trajectory(
        model=Path("/m"), prompt="x",
        kv_config_str="ctk=f16,ctv=f16",
    )
    assert res.token_ids == [7, 8]


def test_mlx_trajectory_skips_records_without_token(fake_mlx, monkeypatch):
    def _stream_with_none(m, tok, **kw):
        yield types.SimpleNamespace(token=None)
        yield types.SimpleNamespace(token=99)

    monkeypatch.setattr(fake_mlx[1], "stream_generate", _stream_with_none)
    bk = MLXBackend()
    res = bk.run_completion_trajectory(
        model=Path("/m"), prompt="x",
        kv_config_str="ctk=f16,ctv=f16",
    )
    assert res.token_ids == [99]


# --- MLXBackend.run_kld --------------------------------------------------


def test_mlx_run_kld_basic(fake_mlx, tmp_path, monkeypatch):
    """Native KLD path: tokenize corpus, two forwards per chunk, accumulate."""
    corpus = tmp_path / "c.txt"
    corpus.write_text("text " * 200)  # need ≥ ctx tokens
    # Make the fake tokenizer return enough tokens for at least 1 ctx-sized chunk.
    fake_tok = _FakeTokenizer(ids=list(range(1024)))

    def _load(path):
        m = mock.MagicMock()
        m.return_value = _FakeMxLogits()
        return m, fake_tok

    monkeypatch.setattr(fake_mlx[1], "load", _load)
    mlx_mod._MODEL_CACHE.clear()

    bk = MLXBackend()
    res = bk.run_kld(
        model=tmp_path / "m.gguf", corpus=corpus,
        ref_kv_str="ctk=f16,ctv=f16",
        cand_kv_str="ctk=q8_0,ctv=q8_0",
        chunks=2, ctx=64,
    )
    assert isinstance(res, KLDResult)
    # mean_kld is 0.0 from our fake softmax (returns logits unchanged).
    assert res.chunks <= 2
    assert res.ctx == 64


class _FakeMxLogits:
    """Stand-in for mx.array logits — supports indexing + arithmetic enough
    for the KLD math in MLXBackend.run_kld."""
    def __getitem__(self, idx):
        return self
    def __add__(self, other):
        return self
    def __sub__(self, other):
        return self
    def __mul__(self, other):
        return self
    def __rmul__(self, other):
        return self
    def item(self):
        return 0.0


def test_mlx_run_kld_corpus_too_short_raises(fake_mlx, tmp_path, monkeypatch):
    corpus = tmp_path / "c.txt"
    corpus.write_text("tiny")
    fake_tok = _FakeTokenizer(ids=[1, 2, 3])  # only 3 tokens

    def _load(path):
        return mock.MagicMock(), fake_tok

    monkeypatch.setattr(fake_mlx[1], "load", _load)
    mlx_mod._MODEL_CACHE.clear()

    bk = MLXBackend()
    with pytest.raises(BackendCapabilityError, match="too short"):
        bk.run_kld(
            model=tmp_path / "m.gguf", corpus=corpus,
            ref_kv_str="ctk=f16,ctv=f16",
            cand_kv_str="ctk=f16,ctv=f16",
            chunks=4, ctx=128,
        )


def test_mlx_run_kld_zero_chunks_raises(fake_mlx, tmp_path, monkeypatch):
    """If chunks calculation yields 0, raise rather than divide by zero."""
    corpus = tmp_path / "c.txt"
    corpus.write_text("x")
    # Tokenizer returns exactly ctx tokens — len // ctx = 1, but if chunks=0
    # parameter is passed, min(0, 1) = 0 → should raise.
    fake_tok = _FakeTokenizer(ids=list(range(64)))

    def _load(path):
        return mock.MagicMock(), fake_tok

    monkeypatch.setattr(fake_mlx[1], "load", _load)
    mlx_mod._MODEL_CACHE.clear()

    bk = MLXBackend()
    with pytest.raises(BackendCapabilityError, match="zero chunks"):
        bk.run_kld(
            model=tmp_path / "m.gguf", corpus=corpus,
            ref_kv_str="ctk=f16,ctv=f16",
            cand_kv_str="ctk=f16,ctv=f16",
            chunks=0, ctx=64,
        )


# --- MLXBackend.tokenize_to_ids -----------------------------------------


def test_mlx_tokenize_empty_text_returns_empty(fake_mlx):
    bk = MLXBackend()
    assert bk.tokenize_to_ids(model=Path("/m"), text="") == []


def test_mlx_tokenize_text_returns_list(fake_mlx, monkeypatch):
    fake_tok = _FakeTokenizer(ids=[5, 6, 7])

    def _load(path):
        return mock.MagicMock(), fake_tok

    monkeypatch.setattr(fake_mlx[1], "load", _load)
    mlx_mod._MODEL_CACHE.clear()

    bk = MLXBackend()
    out = bk.tokenize_to_ids(model=Path("/m"), text="hello")
    assert out == [5, 6, 7]


# --- MLXBackend.model_metadata ------------------------------------------


def test_mlx_model_metadata_includes_versions(fake_mlx, monkeypatch):
    """When importlib.metadata can resolve mlx + mlx-lm versions, they're
    included in the metadata dict."""
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: "9.9.9")

    bk = MLXBackend()
    info = bk.model_metadata(model=Path("/m"))
    assert info["backend"] == "mlx"
    assert info["mlx_lm_version"] == "9.9.9"
    assert info["mlx_version"] == "9.9.9"


def test_mlx_model_metadata_swallows_version_error(fake_mlx, monkeypatch):
    """If the version probe raises, metadata still returns backend + path."""
    import importlib.metadata as md

    def boom(name):
        raise RuntimeError("no metadata")

    monkeypatch.setattr(md, "version", boom)

    bk = MLXBackend()
    info = bk.model_metadata(model=Path("/m"))
    assert info["backend"] == "mlx"
    assert "mlx_lm_version" not in info
