# REFRACT INTERPRETATION GUIDE

How to read a REFRACT report card. Targets a reader who knows quants
generally but is new to REFRACT specifically.

## Composite score

A single 0–100 number. **Higher is better.** Computed as the harmonic
mean of every axis that ran (so any single broken axis tanks the
composite — that's intentional, the framework is fail-loud).

| Band      | Range  | Meaning                                          |
|-----------|--------|--------------------------------------------------|
| EXCELLENT | 90–100 | Within reference noise. Indistinguishable.       |
| PASS      | 80–90  | Minor drift; safe to deploy in most uses.        |
| DEGRADED  | 60–80  | Visible drift; audit on your workload first.    |
| FAIL      | 0–60   | Material quality loss; treat as broken.          |

## Per-axis interpretation

Each axis has its own 0–100 score and its own band. The **Diagnosis**
block in the report card translates the per-axis pattern into plain
English. Below is the deeper "what to do about it" guide per axis.

### Axis A: Trajectory (formerly GTM)

Measures: **Token-level agreement with fp16**. Greedy-decodes N tokens
under both reference and candidate KV configs, captures token IDs at
decode time, computes prefix agreement length / candidate length.

- **EXCELLENT (90–100):** the model decodes essentially the same tokens
  under both configs. KV quantization is doing nothing visible in the
  short-context greedy regime.
- **PASS (80–90):** tokens diverge after 5–10 steps but the leading
  context matches. Common for fine-grained quants. Safe.
- **DEGRADED (60–80):** divergence within first few tokens on most
  prompts. Suspect a softmax-shift class of error: e.g. K rotation
  collisions, V rotation breaking with insufficient bits.
- **FAIL (0–60):** model decodes essentially different text under
  cand. The KV scheme is wrecking next-token distributions.

**Remediation when low:** try a higher-bit V cache; switch from
symmetric to asymmetric turbo; try non-rotating V. If cand and ref
diverge at step 1 (median_first_divergence near 0), the problem is
purely at the per-token distribution level.

### Axis B: KLD@D (corpus KL divergence)

Measures: **Distribution-level divergence from fp16** on natural-text
inputs, averaged per token across a corpus. Bit-exact zero on Metal
when cand == ref, so any non-zero value is real signal not float
jitter.

- **EXCELLENT (90–100, KLD < 0.05 nats):** distributions essentially
  match. The candidate doesn't measurably move the model.
- **PASS (80–90, KLD ~0.05–0.2 nats):** small but measurable drift.
  Most quants land here.
- **DEGRADED (60–80, KLD ~0.2–0.5 nats):** material distribution
  shift. Argmax may still match (so trajectory could be high) but
  the second/third choices have moved.
- **FAIL (< 60, KLD > 0.5 nats):** large divergence; the per-token
  predictive distributions are fundamentally different.

**Remediation when low:** measure the noise floor first
(`--measure-floor`) to confirm the reference itself is deterministic
on your build. If floor passes, the candidate is genuinely lossy at
the distribution level. Try a higher-bit baseline (q8 instead of q4)
or a non-rotating V scheme.

**Note:** KLD@D is the best-disciplined axis. If KLD says broken,
trust it. If KLD says clean while Trajectory says broken, the bug is
likely in chat-template handling or in the test workload not the
quant itself.

### Axis C: R-NIAH (Retrieval Needle-In-A-Haystack)

Measures: **Long-context retrieval quality vs fp16**. Inserts a
sentinel fact into a long context at fractional positions; checks
whether the candidate retrieves it as well as the reference does.
Score per (length, position) cell, aggregated.

- **EXCELLENT (100):** candidate retrieves wherever the reference
  does. Either the model is robust at long context OR the model can't
  retrieve at all under either config (check `confidence` field).
- **PASS / DEGRADED / FAIL (< 100):** candidate loses retrieval at
  cells where the reference still finds the needle. Inspect the
  per-cell breakdown for the (length, position) where it breaks.

**Remediation when low:** revert to a higher-bit KV at long contexts;
or add a runtime guard that switches KV schemes by sequence length.

**Confidence:** the JSON includes `confidence: low` when
`base_acc_avg < 0.2` — the model isn't engaging retrieval at all and
R-NIAH = 100 is then a noise-floor reading. Re-check on a stronger
model or with a different needle.

### Axis D: PLAD (Perturbation-Locality Aware Drift)

Measures: **Robustness to small prompt changes vs fp16**. Generates
anchor completions, perturbs each prompt minimally (typo, casing,
punctuation, paraphrase), measures how much the candidate's output
drifts vs how much the reference's drifts.

Per-perturbation breakdown is the actionable signal:

- **typo low** → candidate is brittle to typos. Consider input
  sanitization or higher-bit fallback when typos are detected.
- **case low** → candidate is sensitive to capitalization changes.
- **punct low** → candidate's output flips on terminal punctuation
  (sometimes acts as an EOS-like signal).
- **paraphrase low** → candidate is sensitive to lexical substitution.
  Less common; usually the most worrying signal.

**Confidence:** if a perturbation can't apply to your prompts (e.g.,
no ≥4-char words → typo skipped, no synonyms in table → paraphrase
skipped), it's listed in `skipped_perturbations` and excluded from
the band.

## Pattern recognition (Diagnosis block)

The Diagnosis block fires plain-English sentences for the most common
patterns. Some examples and what they imply:

  - **"Per-token distribution is broken but high-level surfaces are
    intact"** → Trajectory and KLD low, R-NIAH and PLAD intact. The
    quant is producing different tokens at decode time but reasoning
    and retrieval still work. Common for aggressive turbo schemes.
    Likely safe if you don't depend on exact text reproduction.

  - **"Long-context retrieval degraded"** → R-NIAH low, others high.
    Candidate fails specifically at long context. Affects RAG,
    summarization, code understanding. Unsafe for those workloads.

  - **"Brittleness to small input changes"** → PLAD low, others high.
    Candidate flips its output under input variations. Affects
    real-user deployments where prompts have typos or slight
    paraphrasing. Stress-test before shipping.

  - **"Catastrophic"** → every measured axis < 60. Treat as
    non-functional; revert to higher-bit config.

## Composite vs per-axis: which to trust

If composite says PASS but one axis is DEGRADED, **read the per-axis
band first**. Composite is a summary; per-axis tells you *what kind
of failure* you have. A 95 composite hides a 99/99/99/85 reality
where the model is actually brittle to perturbations.

For shipping decisions:
  - All 4 axes ≥ 80 → safe
  - One axis 60–80 → audit the specific surface that's degraded;
    decide based on whether that surface matters to your workload
  - One axis < 60 → ship only if you understand the failure and have
    mitigations
  - Multiple axes < 60 → don't ship

## Cross-version comparisons

Reports embed `framework_version`. Comparing scores across versions
is fine for the same model+candidate, but a "+1.5 composite" between
v0.2.0 and v0.3.0 may be partly attributable to methodology changes
(e.g., chat-template handling). Apples-to-apples requires same
version on both sides.
