# Releasing Python packages

Package publication is a maintainer-only operation. Build artifacts are created
in a job without publishing credentials, then a separate protected job uploads
the verified artifacts with PyPI Trusted Publishing. No PyPI token is stored in
GitHub.

For the root `metria` distribution, follow the separate
[Metria release procedure](releasing-metria.md). The component identities and
release blockers below apply to their respective packages.

## KV Fidelity package identity

This repository publishes KV Fidelity as `kv-fidelity`, with Python import
`kv_fidelity` and CLI command `kv-fidelity`. PyPI's legacy `refract-llm`
project is controlled by a legacy maintainer and its releases were not
published from this repository. See
[the package-identity record](../../components/kv-fidelity/PACKAGE-IDENTITY.md)
before changing release or ownership settings.

No `kv-fidelity` release exists yet. Until the first release is verified,
user documentation must direct users to a source checkout instead of claiming
that a package-index version is available. The first planned stable version is
0.3.5.

### One-time setup for `kv-fidelity`

1. In PyPI, create a pending Trusted Publisher for project `kv-fidelity` with:
   - GitHub owner: `dipeshbabu`
   - repository: `metria`
   - workflow: `publish-kv-fidelity.yml`
   - environment: `pypi-kv-fidelity`
2. Create the GitHub environment `pypi-kv-fidelity`. Require a reviewer,
   disallow administrator bypass, and restrict deployments to tags matching
   `kv-fidelity-v*`.
3. Establish two independent PyPI recovery paths before the first release.
   Prefer a PyPI organization account; otherwise keep at least two trusted
   people as project Owners, with two-factor authentication and separate
   recovery codes.
4. Confirm that release-blocking issues #8, #11, #12, and #40 are closed. The
   workflow also checks them and refuses to build while any is open.
5. Keep the GitHub environment and Trusted Publisher values exact. The
   workflow has no password or long-lived token fallback.

### `kv-fidelity` release procedure

1. In a pull request, change the development version in
   `components/kv-fidelity/pyproject.toml` and
   `components/kv-fidelity/src/kv_fidelity/__init__.py` to the stable release version.
   Finalize the matching dated section in
   `components/kv-fidelity/CHANGELOG.md`. The first planned release is 0.3.5.
2. Merge only after the required CI and security checks pass on the exact
   release commit.
3. Tag the merge commit as `kv-fidelity-v<VERSION>` and push the tag. The tag
   must point to a commit reachable from `main`.
4. Dispatch the publishing workflow from that exact tag:

   ```bash
   gh workflow run publish-kv-fidelity.yml \
     --ref kv-fidelity-v0.3.5 \
     -f version=0.3.5
   ```

5. Review the clean-wheel smoke results and artifact hashes, then approve the
   protected `pypi-kv-fidelity` deployment.
6. Let the workflow finish. After upload, it installs the exact version from
   PyPI in a clean environment; verifies distribution metadata, Apache-2.0
   license files, project links, `kv-fidelity --version`, and bundled prompts;
   checks the exact Trusted Publisher identity through PyPI's Integrity API;
   cryptographically verifies the wheel and source-distribution attestations;
   and creates the matching GitHub release.

PyPI does not permit replacing files for an existing version. If publication
succeeds but a later verification or GitHub release step fails, do not rerun
the publish job. Repair the post-publish step against the existing PyPI files
and tag.

## TurboQuant reference package

`turboquant-reference` is an independently versioned alpha package. It uses
semantic versioning and supports the Python versions declared in its package
manifest. During the `0.x` series, incompatible API changes may ship in a minor
release; patch releases preserve documented public APIs. The responsible
maintainer owns its changelog and release notes according to
[MAINTAINERS.md](../../MAINTAINERS.md).

### One-time setup for `turboquant-reference`

1. Verify that the exact `turboquant-reference` PyPI project is controlled by
   the project maintainer or is still available. Do not publish under a name
   owned by an unrelated project.
2. Create a GitHub environment named `pypi-turboquant-reference`. Require a
   reviewer, prevent administrator bypass, and restrict deployments to tags
   matching `turboquant-reference-v*`.
3. Add a PyPI Trusted Publisher, or a pending publisher for the first release,
   with:
   - owner: `dipeshbabu`
   - repository: `metria`
   - workflow: `publish-turboquant-reference.yml`
   - environment: `pypi-turboquant-reference`
4. Keep the environment and publisher configuration synchronized. Do not add a
   password or long-lived PyPI token fallback.

### `turboquant-reference` release procedure

1. Update `components/turboquant-reference/pyproject.toml` and convert the
   relevant `Unreleased` changelog entries into a dated
   `[VERSION] - YYYY-MM-DD` section in a pull request.
2. Merge only after all required checks pass.
3. Tag the merge commit as `turboquant-reference-v<VERSION>` and push the tag.
   The tag must point to a commit reachable from `main`.
4. Dispatch the package-specific workflow from that exact tag:

   ```bash
   gh workflow run publish-turboquant-reference.yml \
     --ref turboquant-reference-v0.1.0 \
     -f version=0.1.0
   ```

5. Review the build logs, SHA-256 hashes, clean-wheel smoke test, and demo
   output before approving the protected deployment. The workflow publishes
   with attestations and then creates a matching GitHub release containing the
   verified wheel and source distribution.

If PyPI publication succeeds but GitHub release creation fails, do not rerun
the publish job against the existing version. Download the retained workflow
artifact, verify its hashes against the build log, and create the release for
the existing tag manually.
