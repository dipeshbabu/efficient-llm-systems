# Comparing KV Fidelity reports

`kv-fidelity compare` uses Metria's `RunRecord`, `ComparisonPlan`, and
`compare_runs` APIs. It checks retained evidence before treating report scores as
directly comparable. This is available in the development workspace; the first
KV Fidelity PyPI release is still pending.

```bash
uv sync --locked --all-packages
uv run --locked kv-fidelity compare run-a.json run-b.json --json-out comparison.json
```

The command requires at least two valid reports and checks every pair. It exits
with `0` for comparable reports and `2` for incompatible or invalid input.
Malformed JSON, duplicate keys, non-finite numbers, and reports larger than
64 MiB are rejected. A bad report is never silently dropped from the comparison.

The default plan allows the candidate KV configuration to change. It controls
the reference configuration, model/tokenizer content, selected runtime files,
prompt/corpus content, generation settings, seed, suite/schema versions, active
axes, coverage context, and metric methods. Host/software and reported backend
identity define comparison blocks. Model file locations may differ when content
identities match; this waiver and its reason are retained in JSON.

Every controlled, blocking, and method mismatch is reported. Missing required
evidence is a failure even when both reports omit it. Full-vocabulary
llama-perplexity KLD and normalized top-k estimates have distinct metric
identities, as do retokenized GTM and native token-trajectory measurements.
Composite identity includes the methods of its contributing axes.

## Evidence in new reports

The existing `kv_fidelity.report.v0.3.3` fields remain readable. New reports add
`method` and `method_version` to each axis, and an optional
`extras.comparison_evidence` object with schema
`kv_fidelity.comparison_evidence.v1`:

| Field | Meaning |
|---|---|
| `requested` | Backend, KV-related settings, generation settings, and seed |
| `resolved` | Full SHA-256 identities of available local inputs and selected runtime executables |
| `observed` | Captured host/software identity; backend metadata remains in the report's existing `environment` field |
| `changed_inputs` | Input/runtime files whose identity changed during scoring |

For local llama.cpp reports, the collector hashes the complete GGUF and records
its embedded tokenizer's container identity. It hashes the prompt, corpus, and
haystack files used by the selected axes, plus the required executables. It
checks file identity again after scoring and marks changed inputs as partial
evidence. Runtime environment overrides are retained as per-variable digests,
without retaining their values. Hashing is streamed once per selected file and
is only performed when JSON or HTML output is requested.
The scoring package's Python source manifest is also retained, so two different
development builds sharing a version string cannot silently share metric identity.
Arbitrary `KV_FIDELITY_LLAMA_EXTRA_FLAGS` can replace model, prompt, or placement
arguments. Runs using them are marked incomplete for direct comparison until
effective settings can be read back; their values are not retained in this metadata.
Extra command arguments embedded in reference/candidate KV specifications are
likewise rejected for direct comparison; only the named KV fields may vary.

These are selected-file identities, not native runtime attestation. Requested
settings are not promoted into observed configuration. `COMPARABLE` means the
retained report evidence satisfies this plan; it does not certify a runtime,
prove deployment quality, or establish an inference speedup. The post-run file
check detects ordinary changes, not a hostile local filesystem or unrecorded
runtime dependencies.

Remote model names and version strings alone do not establish immutable model,
tokenizer, and runtime identity. GPU placement requests also do not identify an
active accelerator. Those missing facts cause `NOT_COMPARABLE`; GPU/vLLM
qualification remains deferred. A local CPU comparison should request
`--n-gpu-layers 0` and retain the generated report evidence.

## Historical reports and inspection overrides

Older reports and bundled examples often lack required identity or method
fields. Keep their original contents. Rerun with the current scorer to collect
new evidence when a direct comparison is needed; do not infer missing observed
facts from an old command line or filename.

For explicitly limited inspection of historical or methodologically different
reports, supply a reason:

```bash
kv-fidelity compare old-a.json old-b.json \
  --allow-incompatible "Inspect historical estimates separately" \
  --json-out inspection.json
```

This permits an inspection exit status of `0`, while text still says
`NOT_COMPARABLE` and JSON keeps `compatible: false`, all mismatches, the override
reason, and the original metric-method compatibility. It never turns
full-vocabulary and top-k KLD into the same raw metric. Use the default command
without an override for CI gating.

The comparison JSON uses `kv_fidelity.comparison.v1` and identifies input reports
by their content digests. Those digests identify saved contents; they are not
universal comparability fingerprints. Python callers can use
`kv_fidelity.comparison.report_to_run_record` and `compare_reports` directly.
