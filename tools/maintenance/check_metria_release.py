"""Validate a root Metria release version and, for uploads, its tested tag."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

tomllib = importlib.import_module("tomllib" if sys.version_info >= (3, 11) else "tomli")


# Keep aligned with the main-branch baseline. Reading branch protection itself
# requires administration permission, which a release workflow must not receive.
REQUIRED_CHECKS = [
    {"context": name, "app_id": 15368}
    for name in (
        "CI required",
        "Core / ubuntu-24.04 / Python 3.10",
        "Core / windows-2025 / Python 3.13",
        "Core / macos-15 / Python 3.13",
        "Root Metria distribution",
        "Analyze (actions)",
        "Analyze (python)",
    )
] + [{"context": "CodeQL", "app_id": 57789}]


def validate_source(
    root: Path, version: str, *, allow_development: bool = False
) -> None:
    pattern = r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    if allow_development:
        pattern += r"(?:\.dev(0|[1-9]\d*))?"
    if re.fullmatch(pattern, version) is None:
        raise ValueError("release version must have three numeric components")
    project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]
    if project["name"] != "metria" or project["version"] != version:
        raise ValueError("root distribution name/version does not match the release")
    tree = ast.parse((root / "src/metria/__init__.py").read_text("utf-8"))
    runtime_versions = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    ]
    if runtime_versions != [version]:
        raise ValueError("runtime version does not match the release")
    changelog = (root / "CHANGELOG.md").read_text("utf-8")
    heading = (
        "## Unreleased" if allow_development and ".dev" in version else f"## {version}"
    )
    if heading not in changelog.splitlines():
        raise ValueError("changelog needs an exact release heading")


def validate_checks(
    required: list[dict[str, Any]], checks: list[dict[str, Any]]
) -> None:
    if not required:
        raise ValueError("main has no required status checks")
    for requirement in required:
        matching = [
            check
            for check in checks
            if check["name"] == requirement["context"]
            and (
                requirement.get("app_id") in (None, -1)
                or check["app"]["id"] == requirement["app_id"]
            )
        ]
        if not matching:
            raise ValueError(f"missing required check: {requirement['context']}")
        latest = max(matching, key=lambda check: check["id"])
        if latest["status"] != "completed" or latest["conclusion"] != "success":
            raise ValueError(f"required check did not pass: {requirement['context']}")


def _json_api(endpoint: str) -> Any:
    return json.loads(
        subprocess.check_output(
            ["gh", "api", endpoint, "--paginate", "--slurp"], text=True
        )
    )


def _commit_checks(repository: str, sha: str) -> list[dict[str, Any]]:
    pages = _json_api(
        f"repos/{repository}/commits/{sha}/check-runs?per_page=100&filter=latest"
    )
    return [check for page in pages for check in page["check_runs"]]


def validate_commit_checks(repository: str, sha: str) -> None:
    # GitHub's aggregate CodeQL check is attached to the PR head, not the main
    # commit. Its analysis jobs still run on main and must succeed there.
    main_checks = [r for r in REQUIRED_CHECKS if r["context"] != "CodeQL"]
    validate_checks(main_checks, _commit_checks(repository, sha))
    pages = _json_api(f"repos/{repository}/commits/{sha}/pulls")
    merged = [
        pr
        for page in pages
        for pr in page
        if pr.get("merged_at") is not None
        and pr["base"]["ref"] == "main"
        and pr["merge_commit_sha"] == sha
    ]
    if len(merged) != 1:
        raise ValueError("release commit must be the exact merge of one PR into main")
    validate_checks(
        REQUIRED_CHECKS, _commit_checks(repository, merged[0]["head"]["sha"])
    )


def validate_publish_ref(version: str) -> None:
    if os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise ValueError("publication requires an explicit workflow dispatch")
    if (
        os.environ.get("GITHUB_REF_TYPE") != "tag"
        or os.environ.get("GITHUB_REF_NAME") != f"metria-v{version}"
    ):
        raise ValueError(f"publication requires tag metria-v{version}")
    repository = os.environ["GITHUB_REPOSITORY"]
    sha = os.environ["GITHUB_SHA"]
    subprocess.run(["git", "fetch", "--no-tags", "origin", "main"], check=True)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "origin/main"], check=True
    )
    validate_commit_checks(repository, sha)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--allow-development", action="store_true")
    parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    version = (
        args.version
        or tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"][
            "version"
        ]
    )
    validate_source(
        root, version, allow_development=args.allow_development and not args.publish
    )
    if args.publish:
        validate_publish_ref(version)
    if args.github_output:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            output.write(f"version={version}\n")
        with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as environment:
            environment.write(f"RELEASE_VERSION={version}\n")
    print(f"Metria {version}: release validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
