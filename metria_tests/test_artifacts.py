"""Adversarial artifact boundaries and transaction/identity regression tests."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

import metria.artifacts as artifacts
from metria import ArtifactManifest, RunRecord, RunSpec, RunStatus, run_record_to_json


def _manifest(data=b"trusted bytes", **kwargs):
    values = dict(
        name="fixture",
        kind="dataset",
        uri="https://example.org/fixed-revision/data",
        revision="abc123",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        source={"license_identifiers": ["fixture-license"]},
    )
    values.update(kwargs)
    return ArtifactManifest(**values)


def _zip(tmp_path, entries, *, expected=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    data = buffer.getvalue()
    path = tmp_path / "input.zip"
    path.write_bytes(data)
    manifest = _manifest(data, path=path, name="input.zip", kind="dataset_archive")
    members = expected or {"root/data": _manifest(b"hello", name="data")}
    return manifest, members


def _extract(archive, target, members, **kwargs):
    return artifacts.extract_verified_zip(
        archive,
        target,
        members,
        max_archive_bytes=100_000,
        max_expanded_bytes=kwargs.get("max_expanded_bytes", 1000),
    )


def test_fetch_is_verified_atomic_and_rechecks_the_cache(tmp_path, monkeypatch):
    data = b"trusted bytes"
    source = _manifest(data)
    calls = []

    def open_url(uri, timeout):
        calls.append((uri, timeout))
        assert not (tmp_path / source.sha256).exists()
        return io.BytesIO(data)

    monkeypatch.setattr(artifacts, "_open_url", open_url)
    first = artifacts.fetch_artifact(source, tmp_path, max_bytes=100)
    second = artifacts.fetch_artifact(source, tmp_path, max_bytes=100)
    assert first == second
    assert len(calls) == 1
    assert first.sha256 == hashlib.sha256(data).hexdigest()
    assert first.metadata["integrity"]["verified"] is True
    assert first.revision == source.revision
    assert first.source == source.source
    assert source.path is None
    Path(first.path).write_bytes(b"corrupt")
    with pytest.raises(artifacts.ArtifactIntegrityError, match="SHA-256"):
        artifacts.fetch_artifact(source, tmp_path, max_bytes=100)
    assert not list(tmp_path.glob(".download-*"))


@pytest.mark.parametrize(
    "failure", ["checksum", "size", "limit", "interrupted", "read_error"]
)
def test_failed_downloads_leave_no_usable_or_partial_file(
    tmp_path, monkeypatch, failure
):
    data = b"trusted bytes"
    source = _manifest(data)

    class Stream(io.BytesIO):
        calls = 0

        def read1(self, count=-1):
            self.calls += 1
            if self.calls > 1 and failure == "interrupted":
                raise KeyboardInterrupt
            if self.calls > 1 and failure == "read_error":
                raise OSError("connection interrupted")
            return super().read1(count)

    monkeypatch.setattr(
        artifacts,
        "_open_url",
        lambda *a: Stream(b"wrong" if failure == "checksum" else data),
    )
    if failure == "size":
        source = replace(source, size_bytes=len(data) + 1)
    expected = (
        KeyboardInterrupt
        if failure == "interrupted"
        else (OSError if failure == "read_error" else artifacts.ArtifactIntegrityError)
    )
    with pytest.raises(expected):
        artifacts.fetch_artifact(
            source, tmp_path, max_bytes=2 if failure == "limit" else 100
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "uri",
    [
        "http://example.org/file",
        "file:///etc/passwd",
        "https://user:secret@example.org/file",
        "https://example.org/file?token=secret",
        "https://example.org/file#fragment",
    ],
)
def test_fetch_rejects_noncanonical_or_authenticated_urls(tmp_path, uri):
    with pytest.raises(ValueError, match="canonical HTTPS"):
        artifacts.fetch_artifact(_manifest(uri=uri), tmp_path, max_bytes=100)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limits_are_rejected(tmp_path, limit):
    with pytest.raises(ValueError, match="positive integer"):
        artifacts.fetch_artifact(_manifest(), tmp_path, max_bytes=limit)


def test_redirect_cannot_downgrade_https():
    with pytest.raises(artifacts.ArtifactIntegrityError, match="remain HTTPS"):
        artifacts._HTTPSRedirect().redirect_request(
            None, None, 302, "", {}, "http://example.org/file"
        )


def test_unpinned_artifact_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        artifacts.fetch_artifact(
            replace(_manifest(), sha256=None), tmp_path, max_bytes=100
        )


def test_verified_archive_cache_and_manifest_roundtrip(tmp_path):
    archive, members = _zip(tmp_path, [("root/", b""), ("root/data", b"hello")])
    target = tmp_path / "cache"
    first = _extract(archive, target, members)
    second = _extract(archive, target, members)
    assert first == second
    assert (target / "data").read_bytes() == b"hello"
    record = RunRecord(
        study_name="artifacts",
        run_id="one",
        requested=RunSpec(
            model={"id": "fixture"},
            runtime={"name": "fixture"},
            scenario={},
            measurements=("fixture",),
        ),
        resolved={},
        observed={},
        status=RunStatus.COMPLETED,
        artifacts=first,
    )
    retained = json.loads(run_record_to_json(record))["record"]["artifacts"][0]
    assert retained == artifacts.artifact_to_data(first[0])
    assert retained["revision"] == "abc123"
    assert retained["source"]["license_identifiers"] == ["fixture-license"]
    assert not list(tmp_path.glob(".extract-*"))


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "root/../../escape",
        "C:/escape",
        "root\\escape",
        "root/./data",
        "root//data",
        "root/data:stream",
        "root/data.",
        "root/CON",
        "root/COM¹",
        "root/data?stream",
        "root/data\nextra",
    ],
)
def test_unsafe_zip_paths_are_rejected_before_promotion(tmp_path, name):
    archive, members = _zip(tmp_path, [(name, b"hello")])
    target = tmp_path / "cache"
    with pytest.raises(artifacts.ArtifactIntegrityError):
        _extract(archive, target, members)
    assert not target.exists()
    assert not list(tmp_path.glob(".extract-*"))


@pytest.mark.parametrize("mode", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFSOCK])
def test_archive_links_and_special_files_are_rejected(tmp_path, mode):
    info = zipfile.ZipInfo("root/data")
    info.create_system = 3
    info.external_attr = (mode | 0o600) << 16
    archive, members = _zip(tmp_path, [(info, b"hello")])
    with pytest.raises(artifacts.ArtifactIntegrityError, match="linked"):
        _extract(archive, tmp_path / "cache", members)


@pytest.mark.parametrize(
    "entries",
    [
        [("root/data", b"hello"), ("unexpected", b"payload")],
        [("unexpected/", b""), ("root/data", b"hello")],
        [],
        [("root/data", b"hello"), ("ROOT/DATA", b"hello")],
    ],
)
def test_missing_duplicate_or_unexpected_members_are_rejected(tmp_path, entries):
    archive, members = _zip(tmp_path, entries)
    with pytest.raises(artifacts.ArtifactIntegrityError):
        _extract(archive, tmp_path / "cache", members)
    assert not (tmp_path / "cache").exists()


def test_member_hash_and_expanded_limit_are_enforced(tmp_path):
    archive, members = _zip(tmp_path, [("root/data", b"hello")])
    with pytest.raises(artifacts.ArtifactIntegrityError, match="expanded byte limit"):
        _extract(archive, tmp_path / "limited", members, max_expanded_bytes=4)
    wrong = {"root/data": replace(members["root/data"], sha256="a" * 64)}
    with pytest.raises(artifacts.ArtifactIntegrityError, match="SHA-256"):
        _extract(archive, tmp_path / "wrong", wrong)
    assert not (tmp_path / "wrong").exists()
    assert not list(tmp_path.glob(".extract-*"))


def test_archive_digest_is_rechecked_before_extraction(tmp_path):
    archive, members = _zip(tmp_path, [("root/data", b"hello")])
    Path(archive.path).write_bytes(b"corrupt")
    with pytest.raises(artifacts.ArtifactIntegrityError, match="SHA-256"):
        _extract(archive, tmp_path / "cache", members)


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_failed_promotion_restores_existing_cache(tmp_path, monkeypatch, failure):
    archive, members = _zip(tmp_path, [("root/data", b"hello")])
    target = tmp_path / "cache"
    target.mkdir()
    (target / "data").write_bytes(b"old")
    original_replace = os.replace

    def replace_path(source, destination):
        if Path(source).name == "ready":
            raise failure("interrupted promotion")
        original_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", replace_path)
    with pytest.raises(failure):
        _extract(archive, target, members)
    assert (target / "data").read_bytes() == b"old"
    assert not list(tmp_path.glob(".extract-*"))


def test_unexpected_existing_files_are_preserved(tmp_path):
    archive, members = _zip(tmp_path, [("root/data", b"hello")])
    target = tmp_path / "cache"
    target.mkdir()
    (target / "user-notes").write_bytes(b"keep")
    with pytest.raises(artifacts.ArtifactIntegrityError, match="unexpected entries"):
        _extract(archive, target, members)
    assert (target / "user-notes").read_bytes() == b"keep"


def test_cached_hardlinks_are_rejected(tmp_path):
    source = tmp_path / "original"
    source.write_bytes(b"hello")
    linked = tmp_path / "link"
    os.link(source, linked)
    with pytest.raises(artifacts.ArtifactIntegrityError, match="unlinked"):
        artifacts.verify_artifact(_manifest(b"hello"), linked, max_bytes=100)


def test_member_order_and_total_limit_are_stable_on_cache_reuse(tmp_path):
    members = {
        "root/b": _manifest(b"bb", name="b"),
        "root/a": _manifest(b"aa", name="a"),
    }
    archive, members = _zip(
        tmp_path, [("root/b", b"bb"), ("root/a", b"aa")], expected=members
    )
    target = tmp_path / "cache"
    first = _extract(archive, target, members)
    second = _extract(archive, target, members)
    assert first == second
    assert [m.name for m in first] == ["a", "b"]
    with pytest.raises(artifacts.ArtifactIntegrityError, match="expanded byte limit"):
        _extract(archive, target, members, max_expanded_bytes=3)


def test_truncated_or_invalid_zip_is_not_promoted(tmp_path):
    archive = tmp_path / "input.zip"
    archive.write_bytes(b"not a zip")
    with pytest.raises(artifacts.ArtifactIntegrityError, match="ZIP archive"):
        _extract(
            _manifest(b"not a zip", path=archive),
            tmp_path / "cache",
            {"root/data": _manifest(b"hello", name="data")},
        )
    assert not (tmp_path / "cache").exists()
    assert not list(tmp_path.glob(".extract-*"))


def test_file_mutation_during_verification_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "input"
    path.write_bytes(b"hello")
    original = artifacts._copy_hash

    def mutate_after_read(stream, output, limit):
        result = original(stream, output, limit)
        path.write_bytes(b"changed contents")
        return result

    monkeypatch.setattr(artifacts, "_copy_hash", mutate_after_read)
    with pytest.raises(artifacts.ArtifactIntegrityError, match="changed while hashing"):
        artifacts.verify_artifact(_manifest(b"hello"), path, max_bytes=100)


def test_path_and_descriptor_timestamp_precision_need_not_match(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path = tmp_path / "input"
    path.write_bytes(b"hello")
    original = os.fstat

    def handle_stat(descriptor):
        info = original(descriptor)
        return SimpleNamespace(
            st_dev=info.st_dev,
            st_ino=info.st_ino,
            st_size=info.st_size,
            st_mode=info.st_mode,
            st_nlink=info.st_nlink,
            st_mtime_ns=info.st_mtime_ns + 1,
            st_ctime_ns=info.st_ctime_ns + 1,
        )

    monkeypatch.setattr(artifacts.os, "fstat", handle_stat)
    verified = artifacts.verify_artifact(_manifest(b"hello"), path, max_bytes=100)
    assert verified.sha256 == hashlib.sha256(b"hello").hexdigest()


@pytest.mark.skipif(
    os.name == "nt" or getattr(os, "geteuid", lambda: 0)() == 0,
    reason="requires POSIX directory permission enforcement",
)
def test_unreadable_existing_subtree_is_not_silently_accepted(tmp_path):
    members = {"root/data": _manifest(b"hello", name="nested/data")}
    archive, members = _zip(tmp_path, [("root/data", b"hello")], expected=members)
    target = tmp_path / "cache"
    nested = target / "nested"
    nested.mkdir(parents=True)
    (nested / "data").write_bytes(b"hello")
    (nested / "user-notes").write_bytes(b"preserve")
    nested.chmod(0o111)
    try:
        with pytest.raises(PermissionError):
            _extract(archive, target, members)
    finally:
        nested.chmod(0o700)
    assert (nested / "user-notes").read_bytes() == b"preserve"
