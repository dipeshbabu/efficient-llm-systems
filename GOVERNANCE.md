# Project governance

Efficient LLM Systems is a maintainer-led open-source research monorepo. This
document explains how decisions are made, how responsibilities are assigned,
and how contributors can propose changes or challenge a decision.

## Project scope

The project maintains four kinds of material with different stability needs:

- published software, currently REFRACT;
- portable research reference implementations, currently TurboQuant;
- diagnostics, validation, benchmark, and conversion tools; and
- research reports and retained experimental evidence.

The repository favors reproducible evidence, explicit compatibility boundaries,
and preservation of negative or superseded results. Production engine
integrations remain owned by their upstream projects.

## Roles

### Contributors

Anyone who participates constructively under the
[Code of Conduct](CODE_OF_CONDUCT.md) is a contributor. Contributors may open
issues, propose designs, submit pull requests, review changes, reproduce
results, and help other users. Contribution does not require commit access.

### Maintainers

Maintainers are trusted contributors with responsibility for one or more
project areas. Depending on their assigned scope, they may triage issues,
review and merge pull requests, manage releases, maintain repository settings,
or coordinate private security and conduct reports.

The current maintainers and component ownership are listed in
[MAINTAINERS.md](MAINTAINERS.md). That file is the source of truth while the
project has a single maintainer; `CODEOWNERS` is not currently used.

## Decision-making

The project seeks rough consensus through public issues and pull requests.
Evidence and the documented compatibility contract carry more weight than
vote counts. The maintainer responsible for an area makes the final decision
after considering technical risk, contributor feedback, maintenance cost, and
the quality of supporting evidence.

Decisions that affect only implementation details can be made in a pull
request. Open a design issue before implementation when a proposal would:

- add or break a public API, CLI option, report schema, or package contract;
- change scoring methodology, benchmark interpretation, or a published claim;
- add a runtime dependency, network download, service, or supported backend;
- materially change security, privacy, licensing, release, or repository
  policy;
- remove retained research evidence or rewrite historical conclusions; or
- amend governance or maintainer responsibilities.

The issue should state the problem, intended outcome, alternatives, migration
or reproducibility impact, and how the result will be validated. Routine fixes,
tests, refactors that preserve behavior, and documentation corrections may go
directly to a pull request.

## Review and merge expectations

All changes to `main` go through a pull request. A merge requires the protected
branch checks, including the aggregate CI check and configured security
analysis, to pass. Conversations must be resolved, the branch must be current,
and history remains linear. Force-pushes and branch deletion are disabled on
`main`.

Because the project currently has one maintainer, branch protection does not
require an approving review that the author cannot independently provide. The
maintainer still performs a final diff and validation review before merge.
External review is strongly preferred for high-risk changes. If a second
active maintainer is added, the project should require at least one independent
approval and introduce `CODEOWNERS` aligned with [MAINTAINERS.md](MAINTAINERS.md).

Additional expectations depend on the change:

- **Published software:** preserve public behavior or document the migration,
  update tests and changelog entries, and verify built distributions.
- **Releases:** follow the protected, trusted-publishing procedure and verify
  artifacts from the exact tagged commit.
- **Research claims:** identify the model, engine revision, data, hardware,
  configuration, command, date, baselines, raw output, and limitations.
- **Retained artifacts:** do not mechanically rewrite historical evidence;
  correct interpretation in a new dated record and link the sources.
- **Security-sensitive changes:** avoid public exploit details and follow
  [SECURITY.md](SECURITY.md) until coordinated disclosure is complete.

## Compatibility and deprecation

Supported release lines are defined by [SECURITY.md](SECURITY.md). Component
stability labels in the root README set expectations, but alpha or beta status
does not remove the obligation to document user-visible changes.

For published packages and stable command/report contracts:

- announce a deprecation in the changelog and relevant documentation;
- keep the old behavior for at least one minor release when practical;
- provide a migration path and a planned removal version or condition; and
- version report-schema changes and reject unsafe silent interpretation.

An immediate breaking correction is permitted when continuing the old behavior
would create a security vulnerability, corrupt evidence, or produce materially
false results. The pull request and release notes must explain that exception.
Experimental scripts may change faster, but their supported inputs and known
limitations should remain explicit.

## Maintainer lifecycle

### Nomination and access

A prospective maintainer should demonstrate sustained, constructive
contributions; sound review judgment; respect for reproducibility, security,
and compatibility; and consistent adherence to the Code of Conduct. A public
governance issue nominates the contributor, describes the proposed scope, and
allows community feedback. Existing maintainers grant the minimum GitHub and
release permissions needed for that scope after the decision is recorded.

### Inactivity and stepping down

Maintainers may step down or narrow their scope at any time through a pull
request updating [MAINTAINERS.md](MAINTAINERS.md). After roughly six months
without project activity, another maintainer may propose inactive or emeritus
status after attempting private contact. Access should be removed promptly
when it is no longer needed, while prior contributions continue to be credited.
Returning maintainers can be restored through the normal nomination process.

### Removal

Maintainer access may be suspended immediately for a credible security risk or
removed for repeated policy or Code of Conduct violations. Except where safety,
privacy, or legal obligations prevent disclosure, the project records the
scope change and non-sensitive rationale publicly.

## Conflicts and appeals

A maintainer with a personal or financial conflict should disclose it and
recuse from the final decision when another maintainer can decide. Technical
disagreements should first be resolved in the relevant issue using reproducible
evidence and clearly stated tradeoffs.

If no disinterested maintainer exists, the lead maintainer documents the
decision and reasoning publicly, except for confidential security or conduct
matters, and invites review from established contributors or an appropriate
upstream community. Conduct complaints follow the private process in
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md#enforcement).

## Amending governance

Governance changes require a public issue and pull request. The pull request
must explain the problem being solved, transition effects, and any required
GitHub permission or branch-protection changes. Update `GOVERNANCE.md`,
`MAINTAINERS.md`, and the audited repository settings together when their
claims would otherwise diverge.
