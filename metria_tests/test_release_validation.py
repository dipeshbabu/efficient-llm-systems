"""Publishing must reject mismatched sources and incomplete/spoofed CI evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_path = (
    Path(__file__).resolve().parents[1] / "tools/maintenance/check_metria_release.py"
)
_spec = importlib.util.spec_from_file_location("check_metria_release", _path)
assert _spec is not None and _spec.loader is not None
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)

_pypi_spec = importlib.util.spec_from_file_location(
    "verify_metria_pypi", _path.with_name("verify_metria_pypi.py")
)
assert _pypi_spec is not None and _pypi_spec.loader is not None
pypi = importlib.util.module_from_spec(_pypi_spec)
_pypi_spec.loader.exec_module(pypi)


@pytest.mark.parametrize(
    "version", ["0.1", "0.1.0.dev0", "0.1.0rc1", "01.1.0", "../0.1.0"]
)
def test_reject_non_release_versions(tmp_path, version):
    with pytest.raises(ValueError, match="three numeric"):
        release.validate_source(tmp_path, version)


def test_source_versions_and_changelog_must_match(tmp_path):
    (tmp_path / "src/metria").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "metria"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    init = tmp_path / "src/metria/__init__.py"
    init.write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("## 0.1.0\n", encoding="utf-8")
    release.validate_source(tmp_path, "0.1.0")
    with pytest.raises(ValueError, match="distribution"):
        release.validate_source(tmp_path, "0.1.1")
    init.write_text('__version__ = "0.1.0.dev0"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="runtime"):
        release.validate_source(tmp_path, "0.1.0")
    init.write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    changelog.write_text("## 0.1.0.dev0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changelog"):
        release.validate_source(tmp_path, "0.1.0")


def _development_source(tmp_path, monkeypatch):
    (tmp_path / "src/metria").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "metria"\nversion = "0.1.1.dev0"\n', encoding="utf-8"
    )
    (tmp_path / "src/metria/__init__.py").write_text(
        '__version__ = "0.1.1.dev0"\n', encoding="utf-8"
    )
    (tmp_path / "CHANGELOG.md").write_text("## Unreleased\n", encoding="utf-8")
    monkeypatch.setattr(
        release, "__file__", str(tmp_path / "tools/maintenance/check.py")
    )


def test_development_validation_exports_source_version(tmp_path, monkeypatch):
    _development_source(tmp_path, monkeypatch)
    output, environment = tmp_path / "output", tmp_path / "environment"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_ENV", str(environment))
    monkeypatch.setattr("sys.argv", ["check", "--allow-development", "--github-output"])
    assert release.main() == 0
    assert output.read_text() == "version=0.1.1.dev0\n"
    assert environment.read_text() == "RELEASE_VERSION=0.1.1.dev0\n"


def test_development_version_cannot_enter_publish_validation(tmp_path, monkeypatch):
    _development_source(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", ["check", "--allow-development", "--publish"])
    monkeypatch.setattr(
        release,
        "validate_publish_ref",
        lambda _: pytest.fail("must reject before publishing checks"),
    )
    with pytest.raises(ValueError, match="three numeric"):
        release.main()


@pytest.mark.parametrize(
    "version", ["0.1.1.dev0\nENV=bad", "0.1.1.dev-1", "0.1.1rc1", "0.1.1.dev01"]
)
def test_development_version_export_cannot_inject_data(tmp_path, version):
    with pytest.raises(ValueError, match="three numeric"):
        release.validate_source(tmp_path, version, allow_development=True)


def _check(identifier=1, **updates):
    return {
        "id": identifier,
        "name": "CI required",
        "app": {"id": 15368},
        "status": "completed",
        "conclusion": "success",
        **updates,
    }


def test_published_files_must_match_reviewed_artifacts():
    expected = {"metria.whl": "wheel-hash", "metria.tar.gz": "sdist-hash"}
    files = [
        {"filename": name, "digests": {"sha256": digest}, "yanked": False}
        for name, digest in expected.items()
    ]
    pypi.validate_files(files, expected)
    for invalid in [
        files[:1],
        files + files[:1],
        [dict(files[0], yanked=True), files[1]],
        [dict(files[0], digests={"sha256": "other-build"}), files[1]],
    ]:
        with pytest.raises(ValueError):
            pypi.validate_files(invalid, expected)


def test_published_attestation_identity_requires_the_release_environment():
    publisher = {
        "kind": "GitHub",
        "repository": "dipeshbabu/metria",
        "workflow": "publish-metria.yml",
        "environment": "pypi-metria",
    }
    pypi.validate_publisher({"attestation_bundles": [{"publisher": publisher}]})
    for invalid in [
        [],
        [{"publisher": dict(publisher, environment="unprotected")}],
        [{"publisher": dict(publisher, repository="someone/metria")}],
    ]:
        with pytest.raises(ValueError):
            pypi.validate_publisher({"attestation_bundles": invalid})


def test_required_checks_need_latest_success_from_expected_app():
    required = [{"context": "CI required", "app_id": 15368}]
    release.validate_checks(required, [_check()])
    for checks in [
        [],
        [_check(app={"id": 999})],
        [_check(conclusion="neutral")],
        [_check(status="in_progress", conclusion=None)],
        [_check(), _check(2, conclusion="failure")],
    ]:
        with pytest.raises(ValueError):
            release.validate_checks(required, checks)
    release.validate_checks(required, [_check(conclusion="failure"), _check(2)])
    with pytest.raises(ValueError, match="no required"):
        release.validate_checks([], [_check()])


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "unmerged",
        "different_merge",
        "different_base",
        "missing_pr",
        "codeql_failed",
        "codeql_wrong_app",
        "main_failed",
    ],
)
def test_tag_checks_require_main_ci_and_the_exact_merged_pr(monkeypatch, problem):
    main_checks = [
        _check(name=r["context"], app={"id": r["app_id"]})
        for r in release.REQUIRED_CHECKS
        if r["context"] != "CodeQL"
    ]
    pr_checks = [
        _check(name=r["context"], app={"id": r["app_id"]})
        for r in release.REQUIRED_CHECKS
    ]
    pr = {
        "merged_at": "2026-09-10T00:00:00Z",
        "merge_commit_sha": "release-commit",
        "base": {"ref": "main"},
        "head": {"sha": "pr-head"},
    }
    if problem == "unmerged":
        pr["merged_at"] = None
    elif problem == "different_merge":
        pr["merge_commit_sha"] = "other-commit"
    elif problem == "different_base":
        pr["base"] = {"ref": "other-branch"}
    elif problem == "codeql_failed":
        pr_checks[-1]["conclusion"] = "failure"
    elif problem == "codeql_wrong_app":
        pr_checks[-1]["app"] = {"id": 15368}
    elif problem == "main_failed":
        main_checks[0]["conclusion"] = "failure"
    responses = {
        "repos/owner/repo/commits/release-commit/check-runs?per_page=100&filter=latest": [
            {"check_runs": main_checks}
        ],
        "repos/owner/repo/commits/release-commit/pulls": [
            [] if problem == "missing_pr" else [pr]
        ],
        "repos/owner/repo/commits/pr-head/check-runs?per_page=100&filter=latest": [
            {"check_runs": pr_checks}
        ],
    }
    monkeypatch.setattr(release, "_json_api", responses.__getitem__)
    if problem is None:
        release.validate_commit_checks("owner/repo", "release-commit")
    else:
        with pytest.raises(ValueError):
            release.validate_commit_checks("owner/repo", "release-commit")


@pytest.mark.parametrize(
    ("event", "ref_type", "ref_name"),
    [
        ("pull_request", "tag", "metria-v0.1.0"),
        ("workflow_dispatch", "branch", "main"),
        ("workflow_dispatch", "tag", "metria-v0.1.1"),
        ("workflow_dispatch", "tag", "kv-fidelity-v0.1.0"),
    ],
)
def test_publish_rejects_unapproved_event_or_wrong_tag(
    monkeypatch, event, ref_type, ref_name
):
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("GITHUB_REF_TYPE", ref_type)
    monkeypatch.setenv("GITHUB_REF_NAME", ref_name)
    with pytest.raises(ValueError):
        release.validate_publish_ref("0.1.0")
