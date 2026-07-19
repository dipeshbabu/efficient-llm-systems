# Maintainers and ownership

This file records active maintainers and their areas of responsibility. It
describes project ownership; it is not a guarantee of response time or an
invitation to contact maintainers privately for general support.

Use [SUPPORT.md](SUPPORT.md) for public support routes,
[SECURITY.md](SECURITY.md) for vulnerability reports, and the
[Code of Conduct](CODE_OF_CONDUCT.md#enforcement) for conduct concerns.

## Active maintainers

| Maintainer | Role | Scope |
|---|---|---|
| [@dipeshbabu](https://github.com/dipeshbabu) | Lead maintainer and release manager | Repository administration, REFRACT, TurboQuant reference, tools, research and artifacts, documentation, security coordination, and releases |

## Component ownership

| Area | Paths | Responsible maintainer | Stability |
|---|---|---|---|
| REFRACT | `components/refract/` | `@dipeshbabu` | Published beta package |
| TurboQuant reference | `components/turboquant-reference/` | `@dipeshbabu` | Research alpha package |
| Repository tools | `tools/` | `@dipeshbabu` | Mixed experimental and operational tooling |
| Maintained documentation | `docs/`, root community files | `@dipeshbabu` | Current guidance and policy |
| Research record | `research/` | `@dipeshbabu` | Dated evidence and interpretation |
| Generated evidence | `artifacts/` | `@dipeshbabu` | Retained benchmark and validation output |
| CI and releases | `.github/workflows/`, package manifests | `@dipeshbabu` | Protected automation |

## Permission model

The lead maintainer currently holds repository administration, merge, release,
security-coordination, and package-publishing responsibility. Permissions are
granted by scope and should follow least privilege. PyPI publication remains
behind the protected `pypi` environment and Trusted Publishing workflow.

There is currently one active maintainer, so the repository does not use a
`CODEOWNERS` file or require an approval the author cannot independently give.
If ownership becomes shared, add `CODEOWNERS`, require independent review for
protected paths, and keep its entries synchronized with this table.

## Updating this file

Maintainer nominations, inactivity, removal, and restoration follow
[GOVERNANCE.md](GOVERNANCE.md#maintainer-lifecycle). Any change to active
ownership requires a pull request that updates this file and, when applicable,
GitHub permissions, branch protection, release environments, package publisher
access, and security contacts in the same transition.
