"""Component fetch/default resolution uses the shared integrity/provenance boundary."""

from __future__ import annotations

import argparse
import json

import pytest

from kv_fidelity import cli, corpora
from metria.artifacts import ArtifactIntegrityError


def test_fetch_reuses_verified_cache_without_network(tmp_path, pinned_wikitext):
    first = corpora.ensure_wikitext_2(tmp_path)
    second = corpora.ensure_wikitext_2(tmp_path)
    assert first == second
    assert len(pinned_wikitext.calls) == 1
    for name, data in pinned_wikitext.data.items():
        assert (first / name).read_bytes() == data


def test_corrupt_cache_is_rejected_even_offline(tmp_path, monkeypatch, pinned_wikitext):
    target = corpora.ensure_wikitext_2(tmp_path)
    (target / "wiki.test.raw").write_bytes(b"wrong")
    args = argparse.Namespace(corpus=None, rniah_haystack=None, no_auto_fetch=True)
    monkeypatch.setattr(cli, "_KV_FIDELITY_CACHE", tmp_path)
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        cli._resolve_default_paths(args, need_corpus=True, need_haystack=False)
    assert len(pinned_wikitext.calls) == 1


def test_resolved_manifest_retains_revision_license_and_complete_hash(
    tmp_path, monkeypatch, pinned_wikitext
):
    monkeypatch.setattr(cli, "_KV_FIDELITY_CACHE", tmp_path)
    args = argparse.Namespace(corpus=None, rniah_haystack=None, no_auto_fetch=False)
    cli._resolve_default_paths(args, need_corpus=True, need_haystack=True)
    retained = args._input_artifacts
    assert (
        retained["corpus"]["sha256"] == pinned_wikitext.members["wiki.test.raw"].sha256
    )
    assert (
        retained["corpus"]["metadata"]["archive_sha256"]
        == pinned_wikitext.archive.sha256
    )
    assert retained["corpus"]["revision"] == pinned_wikitext.archive.revision
    assert retained["corpus"]["source"]["license_identifiers"] == [
        "cc-by-sa-3.0",
        "gfdl",
    ]
    assert json.loads(json.dumps(retained)) == retained
    # Repeated scoring must not keep an old verified claim after the input changes.
    args.corpus.write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError):
        cli._resolve_default_paths(args, need_corpus=True, need_haystack=True)


def test_partial_cache_is_completed_atomically(tmp_path, pinned_wikitext):
    target = tmp_path / "wikitext-2-raw"
    target.mkdir()
    (target / "wiki.test.raw").write_bytes(pinned_wikitext.data["wiki.test.raw"])
    assert corpora.ensure_wikitext_2(tmp_path) == target
    assert {p.name for p in target.iterdir()} == set(pinned_wikitext.data)
    assert not list(tmp_path.glob(".extract-*"))
