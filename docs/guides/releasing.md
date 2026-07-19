# Releasing Python packages

Package publication is a maintainer-only operation. Build artifacts are created
in a job without publishing credentials, then a separate protected job uploads
the verified artifacts with PyPI Trusted Publishing. No PyPI token is stored in
GitHub.

## One-time setup for `refract-llm`

1. Create a GitHub environment named `pypi`. Require a reviewer, prevent
   administrators from bypassing the review, and restrict deployments to tags
   matching `refract-v*`.
2. In the `refract-llm` PyPI project, add a GitHub Trusted Publisher with:
   - owner: `dipeshbabu`
   - repository: `efficient-llm-systems`
   - workflow: `publish-refract.yml`
   - environment: `pypi`
3. Keep the GitHub environment name and Trusted Publisher configuration in
   sync. The workflow intentionally has no password fallback.

## `refract-llm` release procedure

1. Update `components/refract/pyproject.toml` and
   `components/refract/CHANGELOG.md` in a pull request.
2. Merge only after all required checks pass.
3. Tag the merge commit as `refract-v<VERSION>` and push the tag. The tag must
   point to a commit reachable from `main`.
4. Dispatch the publishing workflow from that exact tag:

   ```bash
   gh workflow run publish-refract.yml \
     --ref refract-v0.3.4 \
     -f version=0.3.4
   ```

5. Approve the protected `pypi` deployment after reviewing the build job and
   artifact hashes. Confirm the new files and provenance on PyPI, then create
   the matching GitHub release.

PyPI does not permit replacing files for an existing version. If a version is
already present, increment the package version and create a new tag instead of
trying to overwrite it.

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
   - repository: `efficient-llm-systems`
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
