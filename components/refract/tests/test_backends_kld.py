"""Cover the LlamaCppBackend.run_kld + axes.kld backend-dispatch paths."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from refract.axes.kld import run_kld
from refract.backends.base import BackendCapabilityError, KLDResult as BKLDResult
from refract.backends.llamacpp import LlamaCppBackend
from refract.runner import KVConfig, set_active_backend


def test_llamacpp_run_kld_delegates_and_cleans_up(tmp_path, monkeypatch):
    """LlamaCppBackend.run_kld: builds base, scores cand, returns KLDResult."""
    bk = LlamaCppBackend()
    corpus = tmp_path / "c.txt"
    corpus.write_bytes(b"text content")

    captured = {}

    def fake_base(**kw):
        # Simulate llama-perplexity creating the base file
        Path(kw["base_path"]).write_bytes(b"base data")
        captured["base_built"] = True
        return {}

    def fake_score(**kw):
        captured["scored_kv"] = kw["kv"].label()
        return {"mean_kld": 0.05, "ppl": 7.2,
                "rms_dp_pct": 0.5, "same_topp_pct": 99.5}

    monkeypatch.setattr("refract.runner.run_perplexity_kld_base", fake_base)
    monkeypatch.setattr("refract.runner.run_perplexity_kld", fake_score)
    res = bk.run_kld(
        model=tmp_path / "m.gguf", corpus=corpus,
        ref_kv_str="ctk=f16,ctv=f16",
        cand_kv_str="ctk=q8_0,ctv=q8_0",
    )
    assert isinstance(res, BKLDResult)
    assert res.mean_kld == 0.05
    assert res.ppl == 7.2
    assert captured["base_built"]
    assert "ctk=q8_0" in captured["scored_kv"]
    # Base file gets cleaned up in finally block.


def test_axes_kld_uses_active_backend_when_non_llamacpp(tmp_path):
    """run_kld on axes/kld delegates to active backend when name != llamacpp."""
    corpus = tmp_path / "c.txt"
    corpus.write_bytes(b"text")

    class _FakeBackend:
        name = "mlx"
        def run_kld(self, **kw):
            return BKLDResult(
                mean_kld=0.0, ppl=None,
                rms_dp_pct=None, same_topp_pct=None,
                chunks=8, ctx=64, metadata={"base_path": "via-fake"},
            )

    set_active_backend(_FakeBackend())
    try:
        res = run_kld(
            model=tmp_path / "m.gguf", corpus=corpus,
            reference_kv=KVConfig(), candidate_kv=KVConfig(),
            chunks=8, ctx=64, progress=False,
        )
        assert res.score == pytest.approx(100.0)
        assert res.mean_kld == 0.0
        assert res.is_self_reference is True
        assert res.base_path == "via-fake"
    finally:
        set_active_backend(None)


def test_axes_kld_user_supplied_base_validates_corpus(tmp_path, monkeypatch):
    """When base_path is supplied, sidecar mismatch raises BEFORE scoring."""
    set_active_backend(None)
    corpus_a = tmp_path / "a.txt"
    corpus_a.write_bytes(b"corpus A")
    corpus_b = tmp_path / "b.txt"
    corpus_b.write_bytes(b"a totally different corpus")
    base = tmp_path / "base.bin"
    base.write_bytes(b"x")

    # Build sidecar from corpus_a; then call run_kld with corpus_b.
    from refract.runner import write_corpus_sidecar
    write_corpus_sidecar(base, corpus_a)

    with pytest.raises(RuntimeError, match="corpus identity mismatch"):
        run_kld(
            model=tmp_path / "m.gguf", corpus=corpus_b,
            reference_kv=KVConfig(), candidate_kv=KVConfig(),
            base_path=base, progress=False,
        )
