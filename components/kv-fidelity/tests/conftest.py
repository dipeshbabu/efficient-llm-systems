"""Pytest configuration for KV Fidelity tests.

Registers the `integration` marker locally so the validation test does not
emit a PytestUnknownMarkWarning, without modifying the project-wide
pyproject.toml.
"""

from __future__ import annotations

import pytest


def pytest_configure(config):  # noqa: D401
    config.addinivalue_line(
        "markers",
        "integration: requires llama.cpp + a real GGUF; takes minutes",
    )


@pytest.fixture
def pinned_wikitext(monkeypatch):
    """A tiny pinned corpus exercising the real shared artifact implementation."""
    import hashlib
    import io
    import zipfile
    from dataclasses import replace
    from types import SimpleNamespace

    import metria.artifacts as artifacts
    from kv_fidelity import corpora

    data = {
        "wiki.test.raw": b"test",
        "wiki.train.raw": b"train",
        "wiki.valid.raw": b"valid",
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in data.items():
            archive.writestr(f"wikitext-2-raw/{name}", content)
    payload = buffer.getvalue()
    archive_manifest = replace(
        corpora.WIKITEXT_ARCHIVE,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    members = {
        name: replace(
            corpora.WIKITEXT_MEMBERS[name],
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            metadata={"archive_sha256": archive_manifest.sha256},
        )
        for name, content in data.items()
    }
    calls = []

    def open_url(uri, timeout):
        calls.append(uri)
        return io.BytesIO(payload)

    monkeypatch.setattr(corpora, "WIKITEXT_ARCHIVE", archive_manifest)
    monkeypatch.setattr(corpora, "WIKITEXT_MEMBERS", members)
    monkeypatch.setattr(artifacts, "_open_url", open_url)
    return SimpleNamespace(
        data=data, archive=archive_manifest, members=members, calls=calls
    )
