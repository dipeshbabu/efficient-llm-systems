# Publish Metria 0.1.0

The first root release is `metria==0.1.0`, classified as Alpha. Its qualified
end-to-end CLI scope is local llama.cpp CPU thread-count verification.
The broader framework roadmap and component release gates are independent.
The release decision is recorded in
[issue #109](https://github.com/dipeshbabu/metria/issues/109).

Preparation does not upload anything to PyPI. A maintainer explicitly starts
publication after configuring the account-side publisher.

## One-time PyPI setup

In your PyPI account's [Publishing settings](https://pypi.org/manage/account/publishing/),
add a pending GitHub Trusted Publisher with these exact values:

| Field | Value |
|---|---|
| PyPI project name | `metria` |
| Owner | `dipeshbabu` |
| Repository | `metria` |
| Workflow filename | `publish-metria.yml` |
| Environment | `pypi-metria` |

Use the existing project's publishing settings instead if you already own
`metria`. A missing project page does not reserve a name or prove ownership.
The first successful upload creates the project for a pending publisher.
See [PyPI's pending publisher instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

GitHub's `pypi-metria` environment requires `@dipeshbabu` review, allows
only `metria-v*` tags, and disables administrator bypass. Self-review remains
allowed for the single maintainer. No PyPI API token is stored in GitHub.
Only the protected publishing job receives OIDC permission.

## Validate and publish

1. Update the root version in `pyproject.toml`, `src/metria/__init__.py`,
   and `uv.lock`; update `CHANGELOG.md` and `docs/releases/<VERSION>.md`.
   Merge by squash after all CI and security checks pass.
2. Wait for checks on the resulting `main` commit, then create
   `metria-v<VERSION>` at that commit. The publication guard checks the
   required main jobs and their GitHub app identities on the tagged commit.
   It also finds the exact PR merged into that commit and verifies all eight
   required checks on its head. GitHub attaches the aggregate `CodeQL` result
   to the PR, while its analysis jobs run on both the PR and `main`.
3. Run a validation-only release build:

   ```bash
   gh workflow run publish-metria.yml --repo dipeshbabu/metria \
     --ref metria-v0.1.0 -f version=0.1.0 -F publish=false
   ```

   Download `metria-0.1.0` from the completed run. It contains the wheel,
   source archive, and `SHA256SUMS`. The workflow tests the root package,
   builds its wheel from the source archive, checks metadata and setup assets,
   and clean-installs both formats on Linux (Python 3.10 and 3.14),
   Windows, and macOS (Python 3.13). It never uploads when `publish=false`.
4. After PyPI setup, start the publishing run yourself:

   ```bash
   gh workflow run publish-metria.yml --repo dipeshbabu/metria \
     --ref metria-v0.1.0 -f version=0.1.0 -F publish=true
   ```

5. Review this run's checks, artifacts, and hashes, then approve the
   `pypi-metria` deployment in GitHub Actions. This run builds fresh artifacts,
   so its hashes are authoritative for the upload. The protected job verifies
   those hashes and publishes only the wheel and source archive using
   [Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
   with attestations.
6. The remaining jobs verify PyPI file hashes and publisher identity, validate
   signatures, install the published package in a clean environment, and
   publish the GitHub release. An existing draft is finalized with the exact
   published files. Confirm `python -m pip install metria==0.1.0` works.

Do not rerun the upload if PyPI succeeded but a later verification or GitHub
release step failed. Rerun only failed downstream jobs or complete their
verification manually using the successful run's artifacts. Do not move the
release tag or replace a published version.

The local llama.cpp qualification is an additional integration check using an
installed wheel; it does not claim general hardware, quality, or speedup coverage.
