#!/usr/bin/env python3
"""Reject non-standard numeric constants in JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value!r} is not valid JSON")


def check(paths: list[Path]) -> int:
    failed = False
    for path in paths:
        try:
            json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_constant,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            failed = True
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="+", type=Path)
    args = parser.parse_args()
    return check(args.filenames)


if __name__ == "__main__":
    raise SystemExit(main())
