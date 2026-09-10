# Repository settings baseline

This page records contributor-facing GitHub settings that are not represented
fully by tracked files. Maintainers should compare it with the live repository
after policy or ownership changes.

Release controls last reviewed: 2026-09-10.

## About and discovery

| Setting | Expected value |
|---|---|
| Visibility | Public |
| Default branch | `main` |
| Description | Research, reference implementations, and evaluation tools for efficient LLM inference, KV-cache compression, quantization, and behavioral fidelity. |
| Homepage | `https://github.com/dipeshbabu/efficient-llm-systems/blob/main/docs/index.md` |
| Topics | `llm`, `inference`, `quantization`, `kv-cache`, `compression`, `evaluation`, `reproducible-research`, `machine-learning`, `llama-cpp`, `mlx`, `vllm`, `sglang` |

The project uses GitHub's default repository card rather than a custom social
preview. A custom image is deferred until the project has a stable visual
identity; maintainers should not add a temporary or component-specific image
that misrepresents the umbrella repository.

## Contributor features

| Feature | Expected state | Rationale |
|---|---|---|
| Issues | Enabled | Structured forms route bugs, questions, proposals, and research evidence. |
| Blank issues | Disabled | `.github/ISSUE_TEMPLATE/config.yml` provides explicit private and documentation routes. |
| Discussions | Disabled | The structured question and evidence forms capture the environment and reproducibility details currently needed for useful support. Reconsider when conversational traffic needs a separate forum. |
| Wiki | Disabled | Maintained documentation belongs in reviewed, versioned files under `docs/`. |
| Projects | Enabled | Maintainers may use GitHub Projects for planning without making it a documentation source. |

The contributor journey is:

1. start at the [documentation index](../index.md) or component README;
2. use [SUPPORT.md](../../SUPPORT.md) to choose a public or private route;
3. submit a structured issue when documentation does not resolve the need; and
4. follow [CONTRIBUTING.md](../../CONTRIBUTING.md) and
   [GOVERNANCE.md](../../GOVERNANCE.md) for proposed changes.

## Protection and automation

The `main` branch requires pull requests, strict required checks, linear history,
and resolved conversations. Protections apply to administrators, and force-pushes
and deletion are disabled. PR branches must be up to date with `main` before
merging. The expected required checks are:

- `CI required`;
- `Core / ubuntu-24.04 / Python 3.10`;
- `Core / windows-2025 / Python 3.13`;
- `Core / macos-15 / Python 3.13`;
- `Root Metria distribution`;
- `Analyze (actions)`;
- `Analyze (python)`; and
- `CodeQL`.

The CI and analysis jobs are bound to the GitHub Actions app; the `CodeQL`
result is bound to GitHub Advanced Security. `CI required` also gates component
tests, coverage, lint, and component distribution validation.

Only squash merges are enabled. Merge commits and rebase merges are disabled.
The approving-review count remains zero under the single-maintainer policy in
[GOVERNANCE.md](../../GOVERNANCE.md); a PR and the required checks are still
mandatory.

Automatic deletion of merged pull request branches is enabled
(`delete_branch_on_merge: true`). GitHub deletes the head branch after a pull
request is merged, including merges into `main`. Protected branches retain
their deletion protection.

The repository permits GitHub-owned Actions plus the explicitly allowed
`astral-sh/setup-uv` and `pypa/gh-action-pypi-publish` actions. Full commit-SHA
pinning is required. New third-party Actions require a deliberate allowlist and
supply-chain review; prefer locked in-repository tooling when practical.

Dependabot security updates, secret scanning, and secret-scanning push
protection are enabled. Release environment and publishing controls are
documented in [the release guide](../guides/releasing.md).

The root publishing environment `pypi-metria` was configured on 2026-09-10:
required reviewer `@dipeshbabu`, `can_admins_bypass: false`, and a selected
**tag** policy matching `metria-v*`. Branches cannot deploy. Self-review is
allowed so the single maintainer can approve an explicitly requested upload.
The workflow defaults to validation only; see the
[root release procedure](../guides/releasing-metria.md).

The expected component publishing environments are `pypi-kv-fidelity` for
tags matching `kv-fidelity-v*` and `pypi-turboquant-reference` for tags
matching `turboquant-reference-v*`. Both require review, prevent administrator
bypass, and map to separate PyPI Trusted Publishers. The obsolete `pypi`
environment must not be referenced by a workflow.

The `pypi-kv-fidelity` environment was created on 2026-07-25 with
`@dipeshbabu` as required reviewer and the expected tag policy. Its live
`can_admins_bypass` setting remains `true`; change it to `false` in the GitHub
environment settings before its first release. The root environment's bypass
control was set through the REST API and verified by reading it back.

## Audit procedure

Maintainers can inspect the live baseline with read-only commands:

```bash
gh repo view dipeshbabu/efficient-llm-systems \
  --json description,homepageUrl,repositoryTopics,hasDiscussionsEnabled,hasWikiEnabled
gh api repos/dipeshbabu/efficient-llm-systems/community/profile
gh api repos/dipeshbabu/efficient-llm-systems/branches/main/protection
gh api repos/dipeshbabu/efficient-llm-systems/actions/permissions
```

When a non-file setting changes, update this page in the same issue or pull
request. Settings that affect security, releases, contributor access, or the
support path also require the review described in
[GOVERNANCE.md](../../GOVERNANCE.md).
