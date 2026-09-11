# Verification acceptance policies

The development version of Metria supports optional, user-defined acceptance
policies. Published Metria 0.1.0 reports `VERIFIED` when comparison and analysis
complete; it does not evaluate these policies. Use the current workspace for this
feature until the next root release is published.

A policy answers whether a valid reference/candidate comparison meets your
criteria. Metria provides no universal behavior or safety threshold. `PASS`
means only that the criteria you supplied were met.

Add an optional top-level `policy` to an existing verification recipe:

```json
{
  "schema": "metria.verification_policy.v1",
  "criteria": [
    {
      "target": "behavior.trajectory_agreement",
      "version": "0.3.4",
      "unit": "fraction",
      "min": 0.98
    },
    {
      "target": "behavior.full_match_rate",
      "version": "0.3.4",
      "unit": "fraction",
      "min": 0.90
    }
  ]
}
```

The numbers above are illustrative user choices, not recommended deployment
thresholds. The fragment is the value of `recipe.policy`; keep the recipe's
existing `metria.study_recipe.v1` schema and reference/candidate configuration.

From the repository root, run:

```bash
uv sync --locked --all-packages
uv run --locked --all-packages metria recipe validate study.json
uv run --locked --all-packages metria verify study.json --output policy-check
```

The output directory must be new. See the [local verifier guide](metria-verify.md)
for preparing the qualified CPU runtime, model, and recipe.

## Typed targets

The initial catalog refers to the `kv_fidelity.trajectory_match` analysis method
version `0.3.4`. This version identifies the analysis methodology, not the Metria
package version. Targets are registered identifiers, not arbitrary JSON paths.

| Target | Unit | Criteria |
|---|---|---|
| `behavior.trajectory_agreement` | `fraction` | Numeric `min` and/or `max`, within 0–1 |
| `behavior.full_match_rate` | `fraction` | Numeric `min` and/or `max`, within 0–1 |
| `behavior.all_trajectories_match` | `boolean` | Exact `equals: true` or `equals: false` |
| `analysis.status` | `status` | Exact `equals: "completed"`, `"failed"`, or `"skipped"` |

The verifier still requires completed analysis before evaluating a policy.
Requesting `analysis.status == "failed"` cannot make a failed execution pass.

Trajectory agreement is the existing 0–100 score divided by 100. The report
retains both the source metric identity/unit and this explicit normalization.
Exact matches use the analysis's boolean result, without rounding a numeric
score into a boolean.

Each criterion must specify a target and version. The unit may be omitted and
is resolved from the typed catalog; an explicitly wrong unit is rejected.
Numeric criteria cannot use `equals`, and boolean/status criteria cannot use
numeric bounds. Empty policies, duplicate targets, contradictory min/max bounds,
unknown fields, unsupported versions, and non-finite thresholds are rejected
before execution. A target's analysis must be declared in the recipe.

Performance targets will accompany the verifier-native measurement work. They
are not guessed from the current process timing display.

## Evaluation order and CI exits

Policy evaluation follows these gates:

```text
execution completed
  -> required identity and runtime evidence present
  -> comparison valid
  -> required analyses completed
  -> typed policy evidence available
  -> PASS or FAIL
```

`NOT_COMPARABLE`, `INSUFFICIENT_EVIDENCE`, and `EXECUTION_FAILED` never become
`PASS` through a policy. If an earlier gate fails, criteria are retained as
`NOT_EVALUATED`. If a required policy value is absent, incomplete, or has a
different unit/direction/method/version/aggregation, the outcome is
`INSUFFICIENT_EVIDENCE`. Missing values are not numeric zero. Every criterion is
retained, including observed values, source identity, and its result/reason.

| Exit | Meaning |
|---|---|
| `0` | `PASS`, or `VERIFIED` when no policy was supplied |
| `1` | `FAIL` or another unsuccessful verification outcome |
| `2` | Invalid input/configuration or a filesystem error |
| `130` | Interrupted verification |

The manifest adds a `policy` evaluation with schema `metria.policy_evaluation.v1`
when a policy is present. The policy is also part of the canonical recipe digest.
Recipes without a policy retain their previous digest shape and `VERIFIED`
completion semantics. Python callers can construct `PolicyCriterion` and
`VerificationPolicy` objects, or use `policy_from_data`.
