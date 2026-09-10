#!/usr/bin/env python3
"""Verify package boundaries and required data in built Metria workspace wheels."""

from __future__ import annotations

import argparse
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

AUTHOR = "Dipesh Tharu Mahato (dipeshbabu)"
METRIA_PROJECT_URLS = {
    "Homepage, https://github.com/dipeshbabu/metria",
    "Repository, https://github.com/dipeshbabu/metria",
    "Issues, https://github.com/dipeshbabu/metria/issues",
    "Documentation, https://github.com/dipeshbabu/metria/blob/main/docs/guides/metria-verify.md",
    "Changelog, https://github.com/dipeshbabu/metria/blob/main/CHANGELOG.md",
}
KV_FIDELITY_PROJECT_URLS = {
    "Homepage, https://github.com/dipeshbabu/metria/tree/main/components/kv-fidelity",
    "Repository, https://github.com/dipeshbabu/metria",
    "Issues, https://github.com/dipeshbabu/metria/issues",
    "Changelog, https://github.com/dipeshbabu/metria/blob/main/components/kv-fidelity/CHANGELOG.md",
}


def _has_suffix(names: set[str], suffix: str) -> bool:
    return any(name.lower().endswith(suffix.lower()) for name in names)


def _read_single_member(path: Path, names: set[str], suffix: str) -> str | None:
    matches = [name for name in names if name.lower().endswith(suffix.lower())]
    if len(matches) != 1:
        return None
    with zipfile.ZipFile(path) as archive:
        return archive.read(matches[0]).decode("utf-8")


def _metadata(
    path: Path,
    names: set[str],
    *,
    expected_name: str,
    expected_urls: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    metadata_text = _read_single_member(path, names, ".dist-info/METADATA")
    if metadata_text is None:
        return [f"{path}: expected exactly one METADATA file"]

    metadata = Parser().parsestr(metadata_text)
    expected_fields = {
        "Name": expected_name,
        "License-Expression": "Apache-2.0",
        "Author": AUTHOR,
    }
    for field, expected in expected_fields.items():
        actual = metadata.get(field)
        if actual != expected:
            errors.append(f"{path}: {field} is {actual!r}; expected {expected!r}")
    if expected_urls is not None:
        project_urls = set(metadata.get_all("Project-URL") or ())
        for project_url in sorted(expected_urls - project_urls):
            errors.append(f"{path}: missing Project-URL {project_url!r}")
    return errors


def _entry_points(path: Path, names: set[str]) -> str | None:
    return _read_single_member(path, names, ".dist-info/entry_points.txt")


def check_wheel(path: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())

    if "metria/__init__.py" in names:
        errors.extend(
            _metadata(
                path,
                names,
                expected_name="metria",
                expected_urls=METRIA_PROJECT_URLS,
            )
        )
        entry_points = _entry_points(path, names)
        if entry_points is None:
            errors.append(f"{path}: expected exactly one entry_points.txt file")
        elif "metria = metria.cli:main" not in entry_points:
            errors.append(f"{path}: missing metria console script")
        if any(name.startswith("kv_fidelity/") for name in names):
            errors.append(
                f"{path}: root Metria wheel unexpectedly contains kv_fidelity"
            )
        if any(name.startswith("turboquant/") for name in names):
            errors.append(f"{path}: root Metria wheel unexpectedly contains turboquant")
        if any(name.startswith("metria_tests/") for name in names):
            errors.append(f"{path}: root Metria wheel unexpectedly contains tests")
    elif "kv_fidelity/__init__.py" in names:
        errors.extend(
            _metadata(
                path,
                names,
                expected_name="kv-fidelity",
                expected_urls=KV_FIDELITY_PROJECT_URLS,
            )
        )
        entry_points = _entry_points(path, names)
        if entry_points is None:
            errors.append(f"{path}: expected exactly one entry_points.txt file")
        else:
            if "kv-fidelity = kv_fidelity.cli:main" not in entry_points:
                errors.append(f"{path}: missing kv-fidelity console script")
            if any(
                line.partition("=")[0].strip() == "refract"
                for line in entry_points.splitlines()
            ):
                errors.append(f"{path}: unexpectedly contains refract console script")

        required = {
            "kv_fidelity/prompts/v0.1.jsonl",
            "kv_fidelity/prompts/README.md",
        }
        for name in sorted(required - names):
            errors.append(f"{path}: missing {name}")
        if not any(
            name.startswith("kv_fidelity/examples/") and name.endswith(".json")
            for name in names
        ):
            errors.append(f"{path}: missing packaged KV Fidelity JSON examples")
        if any(name.startswith("metria/") for name in names):
            errors.append(f"{path}: unexpectedly contains metria")
        if any(name.startswith("turboquant/") for name in names):
            errors.append(f"{path}: unexpectedly contains turboquant")
        if any(name.startswith("kv_fidelity/tests/") for name in names):
            errors.append(f"{path}: unexpectedly contains KV Fidelity tests")
    elif "turboquant/__init__.py" in names:
        errors.extend(_metadata(path, names, expected_name="turboquant-reference"))
        if any(name.startswith("metria/") for name in names):
            errors.append(f"{path}: unexpectedly contains metria")
        if any(name.startswith("kv_fidelity/") for name in names):
            errors.append(f"{path}: unexpectedly contains kv_fidelity")
        if any("/tests/" in name or name.startswith("tests/") for name in names):
            errors.append(f"{path}: unexpectedly contains tests")
    else:
        errors.append(f"{path}: wheel contains no recognized Metria workspace package")

    if any(name.startswith("refract/") for name in names):
        errors.append(f"{path}: unexpectedly contains refract")
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
    print(f"Validated {len(args.wheels)} workspace wheel(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
