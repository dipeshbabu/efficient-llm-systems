# Local llama.cpp capture provider

The first local verifier uses a pinned CPU build of llama.cpp with a small
downstream capture patch. The patch writes sampled token IDs to the existing
`KV_FIDELITY_TRAJECTORY` JSONL stream, plus a `.runtime.json` sidecar containing
context size, active thread counts, vocabulary size, and whether chat formatting
was applied. It does not change sampling or write prompt text to those files.

Upstream source: `https://github.com/ggml-org/llama.cpp`

Revision: `434ddbbc0e30522e897670681e503b797c12b7c1`

Apply `llamacpp-capture.patch` to that revision and build `llama-completion` with
CMake. Metria accepts a completion-only binary directory. The downstream patch
is maintained here; it is not an upstream llama.cpp feature or support claim.

The token provider must be qualified and its SHA-256 supplied through
`environment.llama_cpp_token_ids_capture_sha256`. An unpatched provider remains
insufficient for the local verification workflow. Qualification is specific to
the recorded build, model, host, and workload, not a claim about other engines
or hardware.

The patch is derived from MIT-licensed llama.cpp; see `LICENSE.llama.cpp`.
