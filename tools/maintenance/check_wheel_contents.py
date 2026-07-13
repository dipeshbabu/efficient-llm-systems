#!/usr/bin/env python3
"""Verify package boundaries and required data in built component wheels."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def _has_suffix(names: set[str], suffix: str) -> bool:
    return any(name.lower().endswith(suffix.lower()) for name in names)


def check_wheel(path: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())

    if "refract/__init__.py" in names:
        required = {
            "refract/prompts/v0.1.jsonl",
            "refract/prompts/README.md",
        }
        for name in sorted(required - names):
            errors.append(f"{path}: missing {name}")
        if not any(name.startswith("refract/examples/") and name.endswith(".json") for name in names):
            errors.append(f"{path}: missing packaged REFRACT JSON examples")
        if any(name.startswith("turboquant/") for name in names):
            errors.append(f"{path}: unexpectedly contains turboquant")
        if any(name.startswith("refract/tests/") for name in names):
            errors.append(f"{path}: unexpectedly contains REFRACT tests")
    elif "turboquant/__init__.py" in names:
        if any(name.startswith("refract/") for name in names):
            errors.append(f"{path}: unexpectedly contains refract")
        if any("/tests/" in name or name.startswith("tests/") for name in names):
            errors.append(f"{path}: unexpectedly contains tests")
    else:
        errors.append(f"{path}: wheel contains neither component package")

    if not _has_suffix(names, "/LICENSE"):
        errors.append(f"{path}: missing LICENSE")
    if not _has_suffix(names, "/NOTICE"):
        errors.append(f"{path}: missing NOTICE")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    for wheel in args.wheels:
        errors.extend(check_wheel(wheel))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Validated {len(args.wheels)} component wheel(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
