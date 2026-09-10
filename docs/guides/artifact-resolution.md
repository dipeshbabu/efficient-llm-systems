# Verified model and data artifacts

The development API in `metria.artifacts` uses `ArtifactManifest` for requested
and resolved file identity. It is part of the upcoming root release; the already
published `metria==0.1.0` is unchanged. Use `uv sync --all-packages` from a
checkout while developing against this API.

## Shared API

- `fetch_artifact(manifest, cache_dir, max_bytes=...)` streams a canonical
  HTTPS source into a temporary file, hashes while copying, and promotes the
  file into a content-addressed cache only when its SHA-256 and optional byte
  size match. SHA-256 is mandatory. Network timeouts apply to individual I/O
  operations, not a complete-transfer deadline.
- `verify_artifact(manifest, path, max_bytes=...)` rechecks the complete local
  file, rejects links/non-regular files, and detects changes during hashing.
- `extract_verified_zip(archive, destination, members, ...)` requires pinned
  member manifests and compressed/expanded byte limits. It validates every
  archive entry before writing any output and verifies each extracted member.
  Path traversal, absolute paths, links, encrypted entries, duplicates,
  unexpected entries, and oversized expansion fail closed. A failed directory
  replacement restores the prior cache.
- `artifact_to_data(manifest)` produces JSON-compatible identity for requested
  configuration, resolved input metadata, or `RunRecord.artifacts`.

Sources must supply the expected digest independently; a hash computed from an
untrusted first download is not a trust anchor. Pin an upstream revision in the
URI where the provider supports it. HTTPS redirects may reach a different
HTTPS storage host, but signed redirect URLs are never retained as provenance.
Canonical sources containing credentials, query parameters, or fragments are
rejected by the downloader.

Resolved manifests preserve the source URI, revision, name/kind, source/license
metadata, complete digest, byte size, local path, and the verification method.
They can be attached directly to `RunRecord.artifacts`. A digest identifies
contents; it does not itself prove two experiments are scientifically comparable.
Local paths are retained for reproducibility and should be reviewed before sharing.

## KV Fidelity corpus cache

`kv-fidelity fetch` uses the shared API and a small catalog in
`kv_fidelity.corpora`. It pins the WikiText-2 archive in `ggml-org/ci` to commit
`927b3642933080f1b0e811e2f916e14c292992f9`, with archive SHA-256
`ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.
The 4,721,645-byte archive contains three raw splits whose individual digests
and sizes are pinned as well. Compressed and expanded limits are 8 MiB and
16 MiB respectively.

The original [dataset card](https://huggingface.co/datasets/Salesforce/wikitext)
lists `cc-by-sa-3.0` and `gfdl` license identifiers. Those upstream declarations
are retained with the input identity. The downloaded corpus is separate from
Metria's Apache-2.0 code and is not bundled into either package.

The cache paths remain `~/.cache/kv-fidelity/wikitext-2-raw/wiki.test.raw` and
`wiki.train.raw`. Offline use of a verified test split does not require the
training split. Cache reuse hashes the requested bytes rather than trusting
file existence or a writable metadata sidecar. A corrupt cache is rejected;
remove the named corrupt cache file and rerun `kv-fidelity fetch`, or pass an
explicit custom corpus path if the different data was intentional.

Scoring and repeatability JSON reports retain verified defaults in
`extras.input_artifacts`, keyed by `corpus` and `rniah_haystack`. Repeatability
runs recheck those retained identities before each score invocation. Explicit
custom corpus paths retain their existing behavior; full comparison migration
is tracked separately in issue #11.
