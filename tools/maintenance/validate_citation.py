# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "cffconvert==2.0.0",
# ]
# ///
"""Validate the repository's Citation File Format metadata."""

from __future__ import annotations

import sys
from pathlib import Path

from cffconvert import Citation


def main() -> int:
    """Validate the root CITATION.cff against cffconvert's CFF schema."""
    repository_root = Path(__file__).resolve().parents[2]
    citation_path = repository_root / "CITATION.cff"
    try:
        Citation(
            citation_path.read_text(encoding="utf-8"),
            src=str(citation_path),
        )
    except Exception as error:
        print(f"Invalid citation metadata: {error}", file=sys.stderr)
        return 1

    print("Citation metadata are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
