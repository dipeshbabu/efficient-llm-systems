"""Check published file hashes and publisher identity; print attestation URLs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_json(url: str) -> Any:
    request = urllib.request.Request(
        url, headers={"Accept": "application/vnd.pypi.integrity.v1+json"}
    )
    for attempt in range(12):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 404 or attempt == 11:
                raise
            time.sleep(10)
    raise AssertionError("unreachable")


def validate_files(files: list[dict[str, Any]], expected: dict[str, str]) -> None:
    if len(files) != 2 or {f["filename"] for f in files} != set(expected):
        raise ValueError(
            "PyPI release must contain exactly the reviewed wheel and sdist"
        )
    for item in files:
        if item["digests"]["sha256"] != expected[item["filename"]]:
            raise ValueError(f"published hash mismatch: {item['filename']}")
        if item.get("yanked"):
            raise ValueError(f"published file is yanked: {item['filename']}")


def validate_publisher(provenance: dict[str, Any]) -> None:
    publishers = {
        (
            b["publisher"]["kind"],
            b["publisher"]["repository"],
            b["publisher"]["workflow"],
            b["publisher"]["environment"],
        )
        for b in provenance["attestation_bundles"]
    }
    if publishers != {
        ("GitHub", "dipeshbabu/metria", "publish-metria.yml", "pypi-metria")
    }:
        raise ValueError(f"unexpected Trusted Publisher identity: {publishers}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist", required=True, type=Path)
    args = parser.parse_args()
    expected = {
        name: hashlib.sha256((args.dist / name).read_bytes()).hexdigest()
        for name in (
            f"metria-{args.version}-py3-none-any.whl",
            f"metria-{args.version}.tar.gz",
        )
    }
    release = load_json(f"https://pypi.org/pypi/metria/{args.version}/json")
    validate_files(release["urls"], expected)
    for item in release["urls"]:
        provenance = load_json(
            f"https://pypi.org/integrity/metria/{args.version}/"
            f"{item['filename']}/provenance"
        )
        validate_publisher(provenance)
        print(item["url"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
