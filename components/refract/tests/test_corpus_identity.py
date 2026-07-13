"""v0.1.3 unit tests for corpus identity / sidecar machinery in runner.py.

These guard the "wrong corpus passed to KLD candidate" foot-gun without
needing llama.cpp.
"""

from __future__ import annotations

import pytest

from refract.runner import (
    assert_corpus_matches,
    corpus_identity,
    read_corpus_sidecar,
    write_corpus_sidecar,
)


def test_corpus_identity_reports_size_and_hash(tmp_path):
    corpus = tmp_path / "wiki.raw"
    corpus.write_bytes(b"hello world\n")
    ident = corpus_identity(corpus)
    assert ident["path"] == str(corpus)
    assert ident["size_bytes"] == 12
    assert len(ident["sha256_head"]) == 64
    assert ident["sha256_head_bytes"] == 12


def test_sidecar_roundtrip(tmp_path):
    corpus = tmp_path / "wiki.raw"
    corpus.write_text("alpha beta gamma\n")
    base = tmp_path / "kld.bin"
    sidecar = write_corpus_sidecar(base, corpus)
    assert sidecar.exists()
    parsed = read_corpus_sidecar(base)
    assert parsed["path"] == str(corpus)
    assert parsed["size_bytes"] == corpus.stat().st_size


def test_assert_corpus_matches_passes_on_same_file(tmp_path):
    corpus = tmp_path / "wiki.raw"
    corpus.write_text("identical bytes")
    base = tmp_path / "kld.bin"
    write_corpus_sidecar(base, corpus)
    # No exception
    assert_corpus_matches(base, corpus)


def test_assert_corpus_matches_raises_on_mismatch(tmp_path):
    corpus_a = tmp_path / "wiki-a.raw"
    corpus_b = tmp_path / "wiki-b.raw"
    corpus_a.write_text("aaaa")
    corpus_b.write_text("bbbb")
    base = tmp_path / "kld.bin"
    write_corpus_sidecar(base, corpus_a)
    with pytest.raises(RuntimeError, match="corpus identity mismatch"):
        assert_corpus_matches(base, corpus_b)


def test_assert_corpus_matches_no_op_when_no_sidecar(tmp_path):
    """If no sidecar exists (user-supplied base built outside REFRACT),
    we treat it as 'user knows best' rather than blocking the run."""
    corpus = tmp_path / "wiki.raw"
    corpus.write_text("anything")
    base = tmp_path / "kld.bin"
    # Don't write a sidecar — this should NOT raise.
    assert_corpus_matches(base, corpus)
