"""Bounded, digest-verified artifact resolution with transactional ZIP extraction."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import IO, Any
from urllib.parse import urlsplit

from .identity import ArtifactManifest
from .recipes import _json_value

_CHUNK = 64 * 1024


class ArtifactIntegrityError(RuntimeError):
    """Artifact contents, limits, or extraction policy could not be verified."""


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ArtifactIntegrityError("artifact redirect must remain HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(uri: str, timeout_s: float):
    return urllib.request.build_opener(_HTTPSRedirect()).open(uri, timeout=timeout_s)


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("artifact byte limit must be a positive integer")
    return value


def _pinned(manifest: ArtifactManifest) -> None:
    if not isinstance(manifest, ArtifactManifest) or manifest.sha256 is None:
        raise ValueError(
            "artifact resolution requires an ArtifactManifest with SHA-256"
        )


def _regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or _reparse_point(info):
        raise ArtifactIntegrityError(
            f"artifact must be a regular, unlinked file: {path}"
        )
    return info


def _reparse_point(info: os.stat_result) -> bool:
    return bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _copy_hash(
    source: IO[bytes], target: IO[bytes] | None, limit: int
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    # read1 avoids waiting for an entire chunk on a trickling HTTP stream.
    read = getattr(source, "read1", source.read)
    while data := read(min(_CHUNK, limit - size + 1)):
        size += len(data)
        if size > limit:
            raise ArtifactIntegrityError("artifact exceeds its byte limit")
        digest.update(data)
        if target is not None:
            target.write(data)
    return digest.hexdigest(), size


def _checked(
    manifest: ArtifactManifest, path: Path, digest: str, size: int
) -> ArtifactManifest:
    if digest != manifest.sha256:
        raise ArtifactIntegrityError(f"SHA-256 mismatch for artifact {manifest.name!r}")
    if manifest.size_bytes is not None and size != manifest.size_bytes:
        raise ArtifactIntegrityError(f"size mismatch for artifact {manifest.name!r}")
    return replace(
        manifest,
        path=str(path.absolute()),
        size_bytes=size,
        metadata={
            **manifest.metadata,
            "integrity": {"algorithm": "sha256", "verified": True},
        },
    )


def verify_artifact(
    manifest: ArtifactManifest, path: str | Path, *, max_bytes: int
) -> ArtifactManifest:
    """Hash a complete local file and retain its verified identity and provenance."""

    _pinned(manifest)
    limit = _limit(max_bytes)
    candidate = Path(path)
    before = _regular(candidate)
    if before.st_size > limit:
        raise ArtifactIntegrityError("artifact exceeds its byte limit")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(candidate, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ArtifactIntegrityError("artifact changed before hashing")
        digest, size = _copy_hash(stream, None, limit)
        after = os.fstat(stream.fileno())
    current = _regular(candidate)
    # Compare like-for-like: path and descriptor timestamp precision can differ.
    if _file_identity(opened) != _file_identity(after) or _file_identity(
        before
    ) != _file_identity(current):
        raise ArtifactIntegrityError("artifact changed while hashing")
    return _checked(manifest, candidate, digest, size)


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def fetch_artifact(
    manifest: ArtifactManifest,
    cache_dir: str | Path,
    *,
    max_bytes: int,
    timeout_s: float = 30.0,
) -> ArtifactManifest:
    """Resolve an HTTPS artifact into a verified content-addressed cache file.

    SHA-256 is authoritative even when a server redirects to temporary storage.
    Only the supplied canonical source URI is retained, never a signed redirect.
    The timeout bounds individual network operations, not the full transfer.
    """

    _pinned(manifest)
    limit = _limit(max_bytes)
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("artifact timeout must be positive and finite")
    uri = urlsplit(manifest.uri or "")
    if (
        uri.scheme != "https"
        or not uri.hostname
        or uri.username
        or uri.password
        or uri.query
        or uri.fragment
    ):
        raise ValueError(
            "artifact URI must be canonical HTTPS without credentials, query, or fragment"
        )
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = cache / str(manifest.sha256)
    if target.exists() or target.is_symlink():
        return verify_artifact(manifest, target, max_bytes=limit)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".download-", dir=cache)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            with _open_url(str(manifest.uri), timeout_s) as response:
                digest, size = _copy_hash(response, output, limit)
            output.flush()
            os.fsync(output.fileno())
        result = _checked(manifest, target, digest, size)
        os.replace(temporary, target)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def _relative_name(name: str) -> PurePosixPath:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or "\0" in name
        or any(ord(character) < 32 or character in '<>"|?*' for character in name)
    ):
        raise ArtifactIntegrityError("unsafe archive member name")
    parts = name.split("/")
    if any(part in ("", ".", "..") or part.endswith((" ", ".")) for part in parts):
        raise ArtifactIntegrityError("unsafe archive member path")
    for part in parts:
        stem = part.split(".")[0].upper()
        if stem in {"CON", "PRN", "AUX", "NUL"} or (
            len(stem) == 4 and stem[:3] in {"COM", "LPT"} and stem[-1] in "123456789¹²³"
        ):
            raise ArtifactIntegrityError("non-portable archive member path")
    return PurePosixPath(name)


def _check_existing_tree(destination: Path, filenames: set[str]) -> None:
    info = destination.lstat()
    if not stat.S_ISDIR(info.st_mode) or _reparse_point(info):
        raise ArtifactIntegrityError(
            "artifact extraction destination must be a directory"
        )
    allowed_dirs = {
        str(parent)
        for filename in filenames
        for parent in PurePosixPath(filename).parents
        if str(parent) != "."
    }
    pending = [destination]
    while pending:
        # Unlike globbing, explicit enumeration propagates unreadable-directory
        # errors; an uninspected subtree must not be treated as an empty cache.
        for path in pending.pop().iterdir():
            relative = path.relative_to(destination).as_posix()
            info = path.lstat()
            if _reparse_point(info):
                raise ArtifactIntegrityError("artifact cache contains a reparse point")
            if stat.S_ISDIR(info.st_mode) and relative in allowed_dirs:
                pending.append(path)
                continue
            if relative not in filenames:
                raise ArtifactIntegrityError(
                    "existing artifact directory contains unexpected entries"
                )
            _regular(path)


def extract_verified_zip(
    archive: ArtifactManifest,
    destination: str | Path,
    members: Mapping[str, ArtifactManifest],
    *,
    max_archive_bytes: int,
    max_expanded_bytes: int,
) -> tuple[ArtifactManifest, ...]:
    """Extract only pinned regular members, then atomically install the directory.

    Mapping keys name ZIP members; each manifest's name is its destination path.
    An existing cache may contain only these files and their parent directories.
    Failed replacement restores that previous cache. Unexpected user files and
    links are never removed to make room for a download.
    """

    limit = _limit(max_expanded_bytes)
    if archive.path is None:
        raise ValueError("archive must resolve to a local file before extraction")
    verified = verify_artifact(archive, Path(archive.path), max_bytes=max_archive_bytes)
    expected = {name: members[name] for name in sorted(members)}
    if not expected:
        raise ValueError("archive extraction requires a non-empty member allowlist")
    destinations: set[str] = set()
    folded_names: set[str] = set()
    allowed_dirs: set[str] = set()
    for name, manifest in expected.items():
        source_name = _relative_name(name)
        _pinned(manifest)
        destination_name = str(_relative_name(manifest.name))
        if destination_name.casefold() in folded_names:
            raise ArtifactIntegrityError("duplicate archive destination")
        folded_names.add(destination_name.casefold())
        destinations.add(destination_name)
        allowed_dirs.update(str(p) for p in source_name.parents if str(p) != ".")
    if any(
        str(parent).casefold() in folded_names
        for name in destinations
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    ):
        raise ArtifactIntegrityError("overlapping archive destinations")
    target = Path(destination).expanduser().absolute()
    if target.exists() or target.is_symlink():
        _check_existing_tree(target, destinations)
        try:
            cached = []
            total = 0
            for manifest in expected.values():
                member = verify_artifact(
                    manifest, target / manifest.name, max_bytes=limit
                )
                total += member.size_bytes or 0
                if total > limit:
                    raise ArtifactIntegrityError("archive exceeds expanded byte limit")
                cached.append(member)
            return tuple(cached)
        except (FileNotFoundError, ArtifactIntegrityError):
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=".extract-", dir=target.parent))
    ready, previous = workspace / "ready", workspace / "previous"
    ready.mkdir(mode=0o700)
    try:
        with zipfile.ZipFile(str(verified.path)) as source:
            seen: set[str] = set()
            planned: list[tuple[zipfile.ZipInfo, ArtifactManifest]] = []
            declared_size = 0
            for info in source.infolist():
                name = info.filename[:-1] if info.is_dir() else info.filename
                _relative_name(name)
                if name.casefold() in seen:
                    raise ArtifactIntegrityError("duplicate archive member")
                seen.add(name.casefold())
                mode = stat.S_IFMT(info.external_attr >> 16)
                if info.is_dir():
                    if name not in allowed_dirs or mode not in (0, stat.S_IFDIR):
                        raise ArtifactIntegrityError(
                            "unexpected archive directory or link"
                        )
                    continue
                if (
                    name not in expected
                    or mode not in (0, stat.S_IFREG)
                    or info.flag_bits & 1
                ):
                    raise ArtifactIntegrityError(
                        "unexpected, linked, or encrypted archive member"
                    )
                declared_size += info.file_size
                if declared_size > limit:
                    raise ArtifactIntegrityError("archive exceeds expanded byte limit")
                manifest = expected[name]
                if (
                    manifest.size_bytes is not None
                    and info.file_size != manifest.size_bytes
                ):
                    raise ArtifactIntegrityError("archive member size mismatch")
                planned.append((info, manifest))
            if {info.filename for info, _ in planned} != set(expected):
                raise ArtifactIntegrityError("archive is missing required members")
            total = 0
            results = {}
            for info, manifest in planned:
                output_path = ready / manifest.name
                output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with source.open(info) as stream, output_path.open("xb") as output:
                    digest, size = _copy_hash(stream, output, limit - total)
                total += size
                result = _checked(manifest, target / manifest.name, digest, size)
                results[info.filename] = result
        if target.exists() or target.is_symlink():
            _check_existing_tree(target, destinations)
        try:
            if target.exists():
                os.replace(target, previous)
            os.replace(ready, target)
        except BaseException:
            if previous.exists() and not target.exists():
                os.replace(previous, target)
            raise
        return tuple(results[name] for name in expected)
    except (zipfile.BadZipFile, NotImplementedError) as exc:
        raise ArtifactIntegrityError("cannot validate artifact ZIP archive") from exc
    finally:
        # If restoration itself fails, retain the only copy of the old cache.
        if not previous.exists() or target.exists():
            shutil.rmtree(workspace)


def artifact_to_data(manifest: ArtifactManifest) -> dict[str, Any]:
    """Return the standard JSON-compatible artifact identity for run provenance."""

    return dict(_json_value(manifest.to_mapping(), path="artifact"))
