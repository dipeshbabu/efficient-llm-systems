#!/usr/bin/env python3
"""Verify required root files in the built Metria source distribution."""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path, PurePosixPath

_REQUIRED_ROOT_FILES = frozenset({"pyproject.toml", "LICENSE", "NOTICE"})
_METRIA_RELEASE_FILES = frozenset(
    {
        "CHANGELOG.md",
        "README.md",
        "src/metria/__init__.py",
        "src/metria/cli.py",
        "src/metria/verification.py",
        "docs/guides/metria-verify.md",
        "tools/qualification/build_llamacpp_cpu.sh",
        "tools/qualification/prepare_cpu_verification.py",
        "tools/qualification/llamacpp-capture.patch",
        "tools/qualification/LICENSE.llama.cpp",
        "tools/qualification/llamacpp-cpu-model.json",
    }
)


def check_sdist(path: Path, *, metria_release: bool = False) -> list[str]:
    """Return validation errors for one Metria ``.tar.gz`` source distribution."""

    errors: list[str] = []
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = tuple(member.name for member in archive.getmembers())
    except (OSError, tarfile.TarError) as exc:
        return [f"{path}: cannot read source distribution: {exc}"]

    roots = {
        PurePosixPath(name).parts[0] for name in members if PurePosixPath(name).parts
    }
    if len(roots) != 1:
        errors.append(
            f"{path}: expected exactly one top-level directory; found {sorted(roots)!r}"
        )
        return errors

    root = next(iter(roots))
    member_set = set(members)
    required = _REQUIRED_ROOT_FILES
    if metria_release:
        required = required | _METRIA_RELEASE_FILES
    for filename in sorted(required):
        expected = f"{root}/{filename}"
        if expected not in member_set:
            errors.append(
                f"{path}: missing required source-distribution file {expected}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdists", nargs="+", type=Path)
    parser.add_argument("--metria-release", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    for sdist in args.sdists:
        errors.extend(check_sdist(sdist, metria_release=args.metria_release))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Validated {len(args.sdists)} source distribution(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
