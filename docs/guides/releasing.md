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

## Release procedure

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

`turboquant-reference` is buildable but is not published by this workflow. Do
not reuse the `refract-llm` Trusted Publisher for it. Before publishing it,
verify the PyPI project name and ownership, decide its release policy, and add a
separate protected environment and package-specific workflow.
