# Repository settings baseline

This page records contributor-facing GitHub settings that are not represented
fully by tracked files. Maintainers should compare it with the live repository
after policy or ownership changes.

Last reviewed: 2026-07-19.

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

The `main` branch is protected with strict required checks, linear history,
resolved conversations, administrator enforcement, and disabled force-pushes
and deletion. The expected required checks are:

- `CI required`;
- `Analyze (actions)`; and
- `Analyze (python)`.

The repository permits GitHub-owned Actions plus the explicitly allowed
`astral-sh/setup-uv` and `pypa/gh-action-pypi-publish` actions. Full commit-SHA
pinning is required. New third-party Actions require a deliberate allowlist and
supply-chain review; prefer locked in-repository tooling when practical.

Dependabot security updates, secret scanning, and secret-scanning push
protection are enabled. Release environment and publishing controls are
documented in [the release guide](../guides/releasing.md).

The expected protected publishing environments are `pypi` for tags matching
`refract-v*` and `pypi-turboquant-reference` for tags matching
`turboquant-reference-v*`. Both require review, prevent administrator bypass,
and map to separate PyPI Trusted Publishers.

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
