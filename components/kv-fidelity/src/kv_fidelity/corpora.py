"""Pinned corpus catalog using Metria's shared artifact resolver."""

from __future__ import annotations

from pathlib import Path

from metria.artifacts import extract_verified_zip, fetch_artifact, verify_artifact
from metria.identity import ArtifactManifest

WIKITEXT_REVISION = "927b3642933080f1b0e811e2f916e14c292992f9"
_SOURCE = {
    "provider": "huggingface",
    "repository": "ggml-org/ci",
    "dataset": "Salesforce/wikitext",
    "dataset_url": "https://huggingface.co/datasets/Salesforce/wikitext",
    "license_identifiers": ("cc-by-sa-3.0", "gfdl"),
    "license_source": "https://huggingface.co/datasets/Salesforce/wikitext",
}
WIKITEXT_ARCHIVE = ArtifactManifest(
    name="wikitext-2-raw-v1.zip",
    kind="dataset_archive",
    uri=f"https://huggingface.co/datasets/ggml-org/ci/resolve/{WIKITEXT_REVISION}/wikitext-2-raw-v1.zip",
    revision=WIKITEXT_REVISION,
    sha256="ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11",
    size_bytes=4_721_645,
    source=_SOURCE,
)
WIKITEXT_MEMBERS = {
    name: ArtifactManifest(
        name=name,
        kind="dataset",
        uri=WIKITEXT_ARCHIVE.uri,
        revision=WIKITEXT_REVISION,
        sha256=digest,
        size_bytes=size,
        source={**_SOURCE, "archive_member": f"wikitext-2-raw/{name}"},
        metadata={"archive_sha256": WIKITEXT_ARCHIVE.sha256},
    )
    for name, digest, size in (
        (
            "wiki.test.raw",
            "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08",
            1_290_590,
        ),
        (
            "wiki.valid.raw",
            "4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7",
            1_146_846,
        ),
        (
            "wiki.train.raw",
            "6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57",
            10_940_747,
        ),
    )
}
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_EXPANDED_BYTES = 16 * 1024 * 1024


def resolve_cached_corpus(cache_dir: Path, name: str) -> ArtifactManifest:
    """Verify the exact requested split; offline callers need no other splits."""

    return verify_artifact(
        WIKITEXT_MEMBERS[name],
        cache_dir / "wikitext-2-raw" / name,
        max_bytes=MAX_EXPANDED_BYTES,
    )


def ensure_wikitext_2(cache_dir: Path) -> Path:
    """Verify/reuse the cache, or install the complete pinned archive transactionally."""

    target = cache_dir / "wikitext-2-raw"
    try:
        for name in WIKITEXT_MEMBERS:
            resolve_cached_corpus(cache_dir, name)
        return target
    except FileNotFoundError:
        pass
    archive = fetch_artifact(
        WIKITEXT_ARCHIVE, cache_dir / "objects", max_bytes=MAX_ARCHIVE_BYTES
    )
    extract_verified_zip(
        archive,
        target,
        {
            f"wikitext-2-raw/{name}": manifest
            for name, manifest in WIKITEXT_MEMBERS.items()
        },
        max_archive_bytes=MAX_ARCHIVE_BYTES,
        max_expanded_bytes=MAX_EXPANDED_BYTES,
    )
    return target
