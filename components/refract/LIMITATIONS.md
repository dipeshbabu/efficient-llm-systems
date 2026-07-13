# REFRACT v0.3.3 — Known limitations

REFRACT compares KV-cache configurations against the same model's fp16/bf16
behavior. It is a fidelity evaluator, not a general model-quality benchmark or
a deployment certification.

## 1. Backend metrics are not interchangeable

llama.cpp and MLX can compute full-distribution corpus KLD on their supported
paths. vLLM and SGLang expose top-k prompt log probabilities, so REFRACT uses a
normalized union-of-top-k estimate with an omitted-mass bucket. Reports label
the estimator in `axes.kld.metadata`. Compare absolute KLD values only when the
backend, top-k, corpus, and framework version match.

## 2. Corpus dependence

Axis B is measured on a corpus. Absolute magnitudes depend on corpus content
and context length. Reports include byte size and a first-MiB hash, and
llama.cpp sidecars reject accidental corpus mismatches, but the hash is an
identity guard rather than a cryptographic integrity guarantee for the entire
file.

## 3. Provisional score bands

The 90/80/60 thresholds were calibrated on a limited model and hardware
matrix. Treat EXCELLENT/PASS as a compact reading of the measured surfaces,
not proof that an application is safe. Run application-specific evaluations
before deployment.

## 4. R-NIAH needs a working reference

If the fp16/bf16 reference retrieves fewer than 20% of R-NIAH cells, the axis
is low confidence. v0.3.3 retains the raw cells in reports but excludes the
axis from the composite, preventing a candidate that merely matches a broken
reference from receiving an inflated score.

## 5. PLAD coverage can be partial

Some perturbations cannot be applied to every prompt. Such perturbations are
recorded as skipped and the axis confidence becomes `partial`; they are not
silently converted into failures or perfect scores.

## 6. Engine capabilities vary

- llama.cpp trajectory capture needs compatible TurboQuant/REFRACT binaries;
  no version-sensitive patch is bundled in the Python wheel.
- MLX is Apple Silicon-only.
- vLLM depends on the installed build's KV dtype support.
- SGLang uses a separately managed server. KLD requires reference and
  candidate endpoints, and local access to the matching tokenizer is required
  so completion and trajectory paths use identical chat-template IDs.

Run `refract selftest --backend ... --model ...` before a long score.

## 7. Determinism is engine-dependent

Greedy settings reduce variance but GPU kernels, scheduling, and engine builds
can still differ. Use `refract repeatability` and `--measure-floor` on the
actual deployment stack. A failed reference floor invalidates the comparison.

## 8. Axis coverage is intentionally finite

The current axes cover token trajectory, decoder distribution, long-context
retrieval, and local prompt perturbations. They do not directly evaluate tool
calling, structured-output validity, multilingual breadth, safety behavior,
or domain-specific task accuracy.
