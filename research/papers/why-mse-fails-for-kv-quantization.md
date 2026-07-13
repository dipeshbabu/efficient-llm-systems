# When Better Reconstruction Breaks Models: Why MSE Fails for KV Cache Quantization

**Dipesh Tharu Mahato**
Independent Researcher
GitHub: [@dipeshbabu](https://github.com/dipeshbabu)

---

## TL;DR

> **Methods that are strictly better under MSE are systematically worse under KL@D in deployment. This is not noise. It is a sign inversion, reproducible across five model families and two hardware backends.**
>
> **K cache is sub-Gaussian after WHT, not Gaussian. Optimizing quantizers for MSE on this distribution does not improve model quality, it can invert it.**
>
> **The quantizer is not the problem. The objective is. MSE is a broken proxy for KV cache compression quality.**

> ## ⚠ Core Empirical Result
>
> **A centroid table that improves K-cache reconstruction MSE by 1–13% across five model families (Qwen3-0.6B, Llama-3.2-1B, Mistral-7B, Phi-3-mini, Gemma-2-2b) causes 70–90% mean KL@D regressions and 50–60% catastrophic-rate failures at the model output (F3, §7.2).**

> **This is a sign inversion: improvements in MSE systematically correspond to worse model behavior in this regime. MSE is not just noisy here, it is adversarial. Improving it can systematically move the model in the wrong direction.**

The rest of this paper is the evidence.

---

## 1. Motivation

Triggered by Mitzenmacher and Portnoy's EDEN-line critique of TurboQuant
(historical issues `#87` and `#89`;
[public endpoints unavailable](../../docs/reference/historical-forks.md#archived-identifiers)).
The companion paper [eden-optimal-s-revisit.md](./eden-optimal-s-revisit.md)
refutes the algorithmic claim at production setup.

This paper is independent. While running the EDEN tests, hypothesis H8 (distribution analysis) returned an unexpected observation: **K cache is sub-Gaussian post-WHT.** The 27 follow-ups produce no shipping change and one negative result that subsumes everything else.

EDEN is the trigger. From here forward, EDEN is context. The subject is the evaluation metric the field uses.

---

## 2. Methodology

| dimension | spec |
|---|---|
| Primary model | Qwen3-0.6B (28 layers, head_dim=128) |
| Cross-family (KL test) | Llama-3.2-1B-Instruct (16 layers, head_dim=64) |
| Five-model expansion | Qwen3-0.6B, Llama-3.2-1B, Mistral-7B, Phi-3-mini, Gemma-2-2b |
| Corpus | wikitext-2-raw, 256–1024 token prefills |
| Hardware | M5 Max (Apple Silicon, MLX); AMD MI300X (DigitalOcean ROCm 7.2) |
| Quantization stack | WHT rotation (fixed sign-flip seed) → b-bit scalar quantizer → matched-norm dequantization |
| Bit budgets | b ∈ {2, 3, 4} |
| Quality metrics | per-coord MSE; **KL@D** (next-token logit KL vs fp32 baseline) |
| Deployment bar | zero catastrophic regressions (>10% KL increase vs default) |

**Sign convention.** Throughout this paper, Δ KL is reported relative to the default Lloyd-on-Gaussian baseline: positive means worse/higher KL, negative means better/lower KL.

> **Catastrophic regression (operational definition).** A prompt where next-token KL@D under the candidate quantization scheme is more than **10% larger** than next-token KL@D under the default Lloyd-on-Gaussian baseline on the same prompt with the same fp32 reference logits. This is the threshold reported in every "X/N catastrophic" rate in this paper. The threshold is conservative: in practice, even 1% next-token KL spikes degrade end-task scores on long-form generation; 10% guarantees user-visible failure on at least the next token.

> **When MSE works, and when it doesn't.** MSE only works when the downstream operator is approximately linear in the representation. Attention is not. Distributed-mean-estimation aggregation (EDEN's original federated-learning use), top-k inner-product search (RaBitQ), and weight-quantization perplexity in the small-perturbation regime (HIGGS' linearity theorem) are the linear-operator cases where MSE proxies quality. Attention's `softmax(QKᵀ/√d)·V` is non-linear and concentrates sensitivity on a sparse subset of coordinates. KV cache quantization is in the second regime; the rest of this paper measures the failure.

The full 35-hypothesis ledger is in [Appendix B](#appendix-b-full-hypothesis-ledger). The body of the paper organizes findings into three acts. Each table includes a one-line takeaway and a one-line implication.

### 2.1 Why this can even happen (intuition before the data)

Three structural facts; no formal proof needed at this stage. The mechanism is unpacked in §8 and made concrete in §8.1.

- **MSE distributes error uniformly across coordinates.** Every coordinate counts equally in the L2 sum.
- **Attention concentrates importance sparsely.** The softmax weight on a given KV pair is dominated by its inner product with the query; small perturbations to high-attention coordinates dominate the output. Because attention scores are inner products ⟨Q,K⟩, an L2-small error in K can be large in the Q direction that matters for the next token.
- **Softmax is discontinuous near ties.** A perturbation that flips the max-pre-softmax bucket from token A to token B is a discrete output change, not a smooth one. Linear MSE additivity fails at this step.

Result: a centroid placement that lowers per-coord MSE can simultaneously flip the dominant attention weight on some prompts. The MSE sees uniform-error reduction; the attention sees a bucket-flip; KL@D sees a catastrophe. This is the failure mode the rest of the paper measures.

**An MSE-improving change is not a safe optimization. It is an uncontrolled intervention.**

---

## Act 1, The Observation

### 3. K Is Sub-Gaussian Post-WHT (H8)

For each layer of Qwen3-0.6B, dump K and V from a 512-token prefill, apply the production WHT, compare per-coord empirical statistics to the expected post-WHT N(0, 1/d).

| layer | K/V | KS-stat vs N(0, 1/d) | kurtosis | tail-99 |
|---|---|---:|---:|---:|
| 0 | K | 0.155 | **−1.56** | 0.00% |
| 4 | K | 0.121 | **−1.42** | 0.00% |
| 8 | K | 0.097 | **−1.27** | 0.00% |
| 12 | K | 0.084 | **−1.16** | 0.00% |
| 16 | K | 0.043 | −0.62 | 0.05% |
| 20 | K | 0.020 | −0.33 | 0.10% |
| 23 | K | 0.010 | −0.25 | 0.13% |
| 0 | V | 0.009 | −0.19 | 0.19% |
| 12 | V | 0.015 | −0.03 | 0.30% |
| 23 | V | 0.011 | +0.02 | 0.27% |

Gaussian baseline at this sample size: kurt ≈ 0, tail-99 ≈ 0.27%.

> **Takeaway:** K layers 0–15 are strongly sub-Gaussian. V cache is approximately Gaussian at every layer.
> **Implication:** Lloyd-Max centroids built for N(0, 1/d) over-resolve nonexistent K tails. They are systematically miscalibrated for K.

This observation reproduces across five model families ([§7](#7-the-universal-k-table)) and is independent of the quantization framework that consumes it.

---

## Act 2, Failed Paths

### 4. The Calibration Win (H9, H10)

Refit Lloyd-Max centroids on actual post-WHT K samples. Test held-out MSE, then test KL@D on the model output.

**MSE result (selected, full table 42 rows):**

| layer | b | Lloyd-Gauss MSE | data-fit MSE | Δ% |
|---|---|---:|---:|---:|
| 0 K | 2 | 2.93e+01 | 7.47e+00 | **+74.5%** |
| 0 K | 4 | 2.63e+00 | 1.18e+00 | **+55.2%** |
| 12 K | 4 | 9.45e−02 | 6.66e−02 | **+29.5%** |
| 23 K | 4 | 6.79e−02 | 6.80e−02 | −0.2% |
| 0 V | 4 | 1.91e−04 | 1.94e−04 | −1.2% |

**KL@D on the calibration prompt (b=4, all 28 layers):**

| KV strategy | KL@D | Δ vs default |
|---|---:|---:|
| default-K + default-V | 9.17e−03 | (baseline) |
| **data-driven K + default-V** | **3.29e−03** | **+64.1%** |
| identity-K + default-V (V-only floor) | 1.48e−04 | (V is 60× cheaper than K) |

> **Takeaway:** Data-driven K centroids cut MSE by 25–75% on layers 0–12 and reduce next-token KL by 64% on the calibration prompt.
> **Implication:** The sub-Gaussian observation has algorithmic teeth. K dominates KV-quantization-induced KL by ~60×; this is where wins live, *if they generalize*.

This was the high-water mark of the investigation.

### 5. Out-of-Distribution Collapse (H11–H22)

The fitted centroids do not transfer. **All single-method per-prompt fitting strategies converge to a 22–30% catastrophic rate at scale, hardware-agnostic.**

| algorithm | mean Δ KL | median Δ KL | catastrophic | sample, hardware |
|---|---:|---:|---:|---|
| tail_aware (M5) | **−3.7%** | +51% | **23.3%** | N=30, M5 |
| tail_aware (AMD MI300X) | −1.2% | +43% | 26% | N=50, MI300X |
| robust (median Lloyd, AMD) | −28.8% | +56% | 22% | N=50, MI300X |

> **Takeaway:** Single-method per-prompt fitting is net-negative on mean KL at scale.
> **Implication:** Zero-catastrophic is unreachable with single-method fitting. Same rate across M5 and MI300X excludes a hardware-specific artifact.

#### 5.1 Prefill–Decode Distribution Shift (PDS)

For 30 prompts, record per-prompt prefill K statistics: mean, std, kurtosis, KS-stat vs Gaussian, max-abs, and tail mass. Compare distributions across catastrophic vs successful prompts. **Welch's t-test p = 0.4–0.9 across all six summary statistics. The two groups are statistically indistinguishable at calibration time.**

I name the failure mode here for the rest of the paper:

> **Prefill–Decode Distribution Shift (PDS).** The calibration distribution (prefill K samples) does not match the inference distribution (decode-time K vectors). The fitted centroids over-resolve the prefill mass; the next-token K lands outside the prefill envelope and is reconstructed in the wrong direction. The signal that predicts which prompts will catastrophically regress does not exist at fit time.

> **Takeaway:** Catastrophic regressions are not a calibration-data quantity issue. They are PDS, an intrinsic property of the prefill→decode transition.
> **Implication:** No prefill-side gating, fallback heuristic, or larger calibration sample eliminates them. Quadrupling prefill context to 1024 tokens (H32) leaves the rate at 10%. Confirmed.

### 6. Variance Reduction via Ensemble (H23–H29)

Average four independently-fitted reconstructions. Each method has a different failure mode; failure modes are uncorrelated across methods on the same prompt.

```
ens_4way per layer per prompt:
  c_default = LLOYD_GAUSSIAN              # static, no fit
  c_tail    = lloyd_with_tail_synthesis(K_post_WHT)
  c_robust  = lloyd_with_median_update(K_post_WHT)
  c_bounded = lloyd_with_30pct_constraint(K_post_WHT)
  recon = mean([
      quantize_dequantize(K, c_default),
      quantize_dequantize(K, c_tail),
      quantize_dequantize(K, c_robust),
      quantize_dequantize(K, c_bounded),
  ])
```

**Scaling result (Qwen3-0.6B, b=4):**

| sample | mean Δ KL | median Δ KL | catastrophic |
|---:|---:|---:|---:|
| N=30 (M5) | +75.05% | +81.53% | **0/30 (0%)** |
| N=100 (M5) | +69.25% | +84.05% | **8/100 (8%)** |
| N=20, 1024-tok prefill | +48% | +77% | 2/20 (10%) |

The N=30 zero-catastrophic was sample variance. **True rate at scale is 8%, Wilson 95% CI [2.7%, 13.3%].**

> **Takeaway:** Ensemble reduces catastrophic rate from 22–30% (single-method) to 8% (four-method) but does not eliminate it. It trades deterministic failure for probabilistic failure.
> **Implication:** Catastrophic regressions are variance-driven, not bias-driven. Ensemble dampens the variance; nothing in the fit pipeline corrects the bias because the bias is PDS, undetectable at fit time.

ens_4way is the closest the investigation comes to a deployable algorithm. It is research material, not a shipping change. Cost: ~1.2s prefill overhead per prompt on Qwen3-0.6B; free at decode.

#### 6.1 Why 8% is unacceptable in production

A 2.7–13.3% catastrophic rate at next-token KL is unacceptable for a default-on shipping change because the failures are **nondeterministic and silent**. They cannot be caught by standard evaluation (PPL on a corpus averages them out) or monitoring (no observable signal at fit time, no obvious symptom at decode time other than the next token being wrong). Users would experience occasional degraded responses with no log entry, no error, and no reproduction path.

This is the deployment-side reason a research-grade algorithm with a strong median win (+84% KL reduction at b=4) does not ship.

---

## Act 3, The Fundamental Limitation

By this point, every reasonable variant of MSE-driven per-prompt centroid optimization has failed under deployment metrics: single-method fitting (22–30% catastrophic), held-out gating (no signal), prediction gating (misclassifies the catastrophic group), per-head fitting (worse), alpha-blending (1/8 catastrophic), best_of fallback (MSE-on-prefill is not predictive), four-method ensemble (8% floor at scale, hardware-agnostic, context-length-agnostic). The remaining family of fixes is to drop per-prompt entirely.

If per-prompt fitting has an irreducible PDS floor, fit centroids **once** on a representative pool of models and ship a static table. No per-prompt overhead, no per-prompt failure mode.

### 7. The Universal K Table

#### 7.1 Five-model cross-validation on MSE

Fit per-layer K centroids on each of {Qwen3-0.6B, Llama-3.2-1B, Mistral-7B, Phi-3-mini, Gemma-2-2b}. Compute pairwise L2 distances on per-layer-averaged centroids.

| comparison | L2 distance |
|---|---:|
| default Lloyd-Gauss vs Qwen K | 0.816 |
| default Lloyd-Gauss vs Llama K | 0.598 |
| Qwen K vs Llama K | 0.275 |
| Qwen V vs Llama V | 0.102 |

The strict 5-model average fails leave-one-out cross-validation on Mistral and Gemma at the MSE level. A Cluster A subset (Llama + Mistral + Gemma, excluding the Qwen + Phi-3 sub-Gaussian-extreme outliers) is MSE-positive on 8/10 sample cells, worst case −2.1%, no MSE-level catastrophic. Averaged centroids:

```
[−2.426, −1.870, −1.498, −1.196, −0.927, −0.681, −0.449, −0.218,
 +0.023, +0.268, +0.512, +0.766, +1.045, +1.358, +1.734, +2.281]
```

Compared to default Lloyd-on-Gaussian, this table compresses inner centroids and slightly extends outer ones, exactly the shape predicted by the sub-Gaussian observation. Applied cross-model it gives 1–13% K-MSE wins on Cluster A members.

> **Takeaway:** Per-model fitted K centroids cluster 3× closer to each other than to the default. The Cluster A average is MSE-positive across 5 model families.
> **Implication:** By any K-reconstruction metric, this is the deployable artifact.

#### 7.2 The KL test (F3), the assumption that fails

Apply the Cluster A universal K table to all K-cache encode/decode in Qwen3-0.6B and Llama-3.2-1B. Run 30 prompt chunks per model. Measure next-token KL@D vs fp32 baseline.

> ## ⚠ Key Result
>
> | model | mean Δ KL | median Δ KL | catastrophic (>10% KL increase) |
> |---|---:|---:|---:|
> | Qwen3-0.6B | **+71.09%** | **+26.82%** | **18/30 (60%)** |
> | Llama-3.2-1B | **+89.95%** | **+18.39%** | **16/30 (53%)** |
>
> **A centroid table that improves K reconstruction MSE by 1–13% across five model families causes 70–90% mean KL regressions and 50–60% catastrophic-rate failures at the model output.**

**The inversion is the point.** Cross-model generalization is normally a strong robustness signal: if a fitted artifact works on five different model families, the standard inference is "this captures a real underlying distributional pattern, ship it." Here that signal is real *and* it correlates with worse downstream behavior. **This is not a corner case. It is the expected outcome once the metric is misaligned with the deployment objective.** It is a structural property of the metric pair, not an artifact of any single model.

> **Generalization across models is not a safety signal when the metric is wrong.** Most ML practice treats cross-model consistency as a validation heuristic: if a method works on three model families, it probably captures something real and is safer to ship. F3 breaks that heuristic. The Cluster A universal K table generalizes cleanly across five model families on MSE *and* breaks 50–60% of prompts on KL@D. Cross-model robustness on the wrong metric provides no protection against deployment failure on the right metric.

The Cluster A universal K table is best understood as a **clean, reproducible, multi-model counterexample to MSE-based KV centroid optimization**, not as a deployable artifact. Its existence is what makes the methodology critique impossible to dismiss as a single-model fluke.

The universal K table is not doing something obviously wrong. The MSE wins are real, reproducible, and cross-validate cleanly across 5 models. They translate to catastrophic KL regressions at the model output.

This violates the implicit assumption behind every published MSE-based KV quantization result, including the EDEN-line note paper that triggered this investigation:

> **The broken assumption:** Improvements in per-vector reconstruction MSE translate monotonically, even approximately monotonically, to improvements in model-output quality.

That assumption is false in the regime measured here. F3 demonstrates a centroid change that is **strictly better** on per-coord MSE across 5 model families and **catastrophically worse** on KL@D in 50–60% of test prompts on the same models, with mean KL degradation of 70–90%.

#### 7.3 MSE–KL correlation across the investigation

The skeptic question, "does MSE correlate with KL *at all*?", has to be answered before this gets dismissed as a corner case.

Across the 35 hypotheses, the relationship between per-coord K-MSE delta and next-token KL@D delta:

| regime | MSE→KL relationship |
|---|---|
| Same prompt, MSE on calibration data (H9, H10) | strongly positive (the high-water mark, 25–75% MSE win → 64% KL reduction) |
| Same prompt, MSE on prefill, KL on decode (H11–H22) | **uncorrelated within the catastrophic group** (Welch's p = 0.4–0.9; H17) |
| Cross-prompt, single static table (H11) | weak, occasionally inverts |
| Cross-model, static universal table (F3) | **inverted** (1–13% MSE win → 70–90% mean KL regression) |
| Per-prompt ensemble (H29) | positive on median, uncorrelated on the catastrophic 8% |

> **Takeaway:** MSE and KL@D correlate only when measured on the same data the centroids were fit to. Off-distribution, the relationship weakens, then inverts.
> **Implication:** Any centroid-fitting work that reports K-MSE wins without held-out KL@D validation publishes on a metric uncorrelated with deployment outcome. The discovered MSE win may be deployable, neutral, or catastrophic; the metric does not distinguish.

**Verdict.** There is no regime in this investigation in which MSE reliably predicts KL@D outside the calibration distribution. The "MSE works on calibration data" cell is the only positive case in the table, and it is the trivial one (fitting on the same data you evaluate on).

The strongest direct correlation evidence in this investigation is H17, which is a forensic test of whether *any* prefill-K summary statistic predicts the catastrophic class. Across six features (mean, std, kurtosis, KS-stat vs Gaussian, max-abs, tail mass), Welch's t-test between catastrophic and non-catastrophic prompts gives p = 0.4–0.9. None significant at any reasonable threshold. The catastrophic class is statistically indistinguishable from the safe class on every prefill-side feature tested.

A formal per-prompt scatter (prefill-K MSE delta vs decode KL@D delta on the N=100 ens_4way run) would tighten this further. It has not been computed; the paper does not claim a Pearson/Spearman value. The H17 null result is sufficient evidence to refute the monotone-MSE-implies-monotone-KL assumption in the regime measured, but a scatter is the cleanest visual proof and is the highest-priority follow-up for this paper.

### 8. Mechanism: Why MSE Fails

The brutal simplification:

> **MSE spreads error evenly. Attention does not. Attention concentrates error on exactly the coordinates that matter.**

Three mechanisms compose:

1. **Direction vs magnitude error.** Matched-norm preserves vector magnitude exactly (cf. Appendix A of the companion paper). A centroid table that lowers per-coord MSE while distorting direction reduces the magnitude penalty in MSE but increases attention-relevant error. Default Lloyd-on-Gaussian places centroids in directions attention relies on; a fitted-on-K-marginals table does not preserve that.
2. **Per-coordinate sensitivity asymmetry.** Attention computes `softmax(QKᵀ/√d)·V`. A reconstruction error on a coordinate in a high-attention direction (one the next-token query has high inner product with) flips the rounding bucket of the largest term in the softmax pre-image. The same MSE budget concentrated on a low-attention coordinate is invisible. Per-coord MSE weights every coordinate uniformly; KL@D weights them by attention magnitude, by orders of magnitude.
3. **Softmax non-linearity.** A perturbation to the largest pre-softmax score flips the dominant attention weight from one token to another, propagating a discrete change through the rest of the forward pass. Linear MSE additivity breaks at this step. A 1% MSE win that pushes one prompt's max-pre-softmax score from "barely token A" to "barely token B" is a catastrophic KL regression by definition.

**This failure mode is invisible to any metric that aggregates error across coordinates.** Per-vector L2, per-coord MSE, mean reconstruction error, and any rate-distortion bound that integrates over the coordinate axis all average exactly the signal that determines the bucket-flip event. The only metrics that surface it are downstream: KL@D, end-task scores, output divergence on long-form generation. Reconstruction-side metrics structurally cannot.

```
True K dist (sub-Gaussian, kurt −1.6 to −0.2)
       │
       │ default Lloyd-on-Gaussian over-resolves tails K does not have
       ▼
Refit Lloyd on real K  ─────►  +1–13% per-coord MSE
       │
       │ static, applied uniformly across prompts
       ▼
Some prompts: rounding decisions flip in high-attention directions
       │
       ▼
softmax non-linearly amplifies coordinate-specific errors
       │
       ▼
KL@D: +71% to +90% mean regression, 50–60% catastrophic
```

The variance reduction in ens_4way works because averaging four reconstructions smears the rounding-bucket-flip events across uncorrelated centroid sets. Any single fit might flip the wrong bucket on prompt P; the ensemble interpolates between flipped and unflipped, dampening the softmax discontinuity. Static universal tables cannot do this, they always flip the same buckets in the same direction across all prompts.

#### 8.1 A 2-D illustration

To make the softmax-flip mechanism concrete, take a degenerate 2-token attention with `Q = [1, 0]`, two K vectors `K₁ = [1.00, 0.10]` and `K₂ = [0.95, 0.10]`. Pre-softmax scores: `⟨Q, K₁⟩ = 1.00`, `⟨Q, K₂⟩ = 0.95`. Softmax weights: `[0.5125, 0.4875]`, attention to token 1.

Quantize K with two candidate centroid placements. **Candidate A** rounds K₁ to `[1.00, 0.10]` and K₂ to `[0.96, 0.10]`. Per-coord squared error on K: `0 + 0.0001 = 0.0001`. **Candidate B** rounds K₁ to `[0.94, 0.10]` and K₂ to `[0.96, 0.10]`. Per-coord squared error on K: `0.0036 + 0.0001 = 0.0037`. By MSE, **A is 37× better than B**.

Post-quantization pre-softmax scores. Under A: `⟨Q, K̂₁⟩ = 1.00`, `⟨Q, K̂₂⟩ = 0.96`. Softmax: `[0.5100, 0.4900]`. Attention still to token 1, KL ≈ 6e-5. Under B: `⟨Q, K̂₁⟩ = 0.94`, `⟨Q, K̂₂⟩ = 0.96`. Softmax: `[0.4950, 0.5050]`. **Attention flipped to token 2**, KL ≈ 1.4e-3 (23× worse than A).

Now reverse the magnitudes. Suppose K₁ and K₂ have nearly equal pre-softmax scores (e.g. 1.000 vs 0.9999). A 0.001 error in either direction can flip the dominant attention head. The MSE-better candidate may produce the flip; the MSE-worse candidate may avoid it. In the regime where prompt-by-prompt the dominant attention margin is narrow (real KV at decode), MSE rank no longer determines KL rank. F3's universal-K table is the macroscopic version of B: 1–13% MSE-better on K reconstruction, but the rounding directions it picks happen to flip the dominant attention bucket on 50–60% of prompts.

This is why the failure mode is not detectable from K-statistics alone. It depends on **which centroid the rounding picks** and **whether that pick happens to land on the high-attention coordinate of the next-token query** — a property of the prompt's downstream consumer, not the K vector being quantized.

---

## 9. What Matters, What Doesn't (Empirical Summary)

Compressed read of the entire investigation:

| factor | first-order or second-order | direction |
|---|---|---|
| **Rotation choice** (WHT vs dense Haar) | **first-order** | WHT eliminates the catastrophic-tail (companion paper §7) |
| **Prefill-decode distribution mismatch (PDS)** | **first-order** | Bound on per-prompt fitting; ~8–30% catastrophic floor |
| **Operator non-linearity (attention softmax)** | **first-order** | Breaks the MSE→KL link; this paper §8 |
| Per-vector scale formula (matched-norm vs EDEN-S) | second-order | ±1% on synthetic, 0.5–9% on real KV (companion paper §6, §9) |
| Centroid optimization on MSE | **unreliable, frequently negative under deployment metrics** | Calibration-positive, deployment-negative (this paper §4–§7) |
| Bit budget (b=2 vs b=3 vs b=4) | second-order at b≥3, problematic at b=2 | ens_4way 0/30 catastrophic at b=3 (AMD); 17% at b=2 |
| Hardware (M5 vs MI300X) | none | Same catastrophic rates within sample variance |

The screenshot-friendly version: **rotation matters, PDS matters, the operator's non-linearity matters; per-vector scale formulas and MSE-optimal centroid tables don't.**

---

## 10. Implications for KV Quantization

> **If you only take one thing from this paper: stop evaluating KV quantization with MSE alone.**

This is not an EDEN-vs-TurboQuant question. The companion paper handles that. This is an evaluation-methodology question, and the implications affect the field broadly:

1. **MSE-only evaluation is insufficient.** Any centroid-fitting work that reports K-MSE wins without held-out KL@D validation publishes on a metric that does not predict deployment. F3 demonstrates 13% MSE improvements with 90% mean KL regressions on the same data.
2. **Static K-centroid tables are fragile.** Uniform centroid placement that diverges from default Lloyd-on-Gaussian risks catastrophic KL even when MSE-positive. The default does not win on K-MSE, it wins on KL@D for reasons not captured by reconstruction error.
3. **Per-prompt adaptation has an irreducible variance floor.** Single-method fitting fails catastrophically at 22–30% rates. Four-method ensemble brings it to 8%. Same hardware-agnostic floor at long context. PDS is the limiting factor, and PDS is undetectable at fit time.
4. **K is the only cache worth re-fitting.** V is approximately Gaussian; default Lloyd is correct for V. K dominates KV-induced KL by ~60×.
5. **A KL-aware centroid objective is the next obvious step.** Replace the Lloyd MSE update with a forward-pass-aware objective: a centroid move that increases MSE by ε but decreases KL@D by 10ε is a strict improvement on the metric that ships. Requires gradient access through one model forward pass per Lloyd iteration. Not run in this investigation.
6. **The deployment bar requires multi-token quality validation.** Next-token KL@D catches the 8% catastrophic; multi-token PPL averaging may obscure it. Either metric is necessary; neither alone is sufficient.

### 10.1 What actually works (operational guidance)

For practitioners shipping KV cache quantization today:

- **Default to Lloyd-on-Gaussian centroids for both K and V.** Empirically the strongest baseline on KL@D; F3 confirms divergence from this default risks catastrophic regression even when MSE-positive.
- **Spend the bit budget on rotation conditioning, not centroid optimization.** WHT with random sign flips is first-order; switching from EDEN-S to matched-norm to optimal-S is second-order on real KV.
- **If you must adapt, adapt per-prompt with an ensemble.** ens_4way is the only candidate in this investigation that produced positive median KL gains without 22%+ catastrophic. Variance reduction across uncorrelated fitting strategies is the only known mechanism.
- **Restrict adaptation to K cache.** V is approximately Gaussian, default is already correct, refitting only adds risk.
- **Validate every change on KL@D *and* an end-task metric.** Per-coord MSE is not sufficient. Next-token KL@D catches PDS-driven catastrophes that multi-token PPL averages out.
- **Report catastrophic-rate confidence intervals, not just means.** A method with mean +69% but 8% catastrophic at Wilson-95% CI [2.7%, 13.3%] is not a deployable shipping change.

### 10.2 Why this wasn't caught earlier

Prior MSE-based KV quantization work is not careless. Three structural blind spots make the failure mode hard to surface during normal development:

1. **Calibration sets are short and prefill-only.** Standard practice: dump K from a 256–512 token prefill, fit centroids, validate on the same prefill. The decode-time distribution shift (PDS) does not appear in this loop. The fitted centroids look excellent on every visible metric and the catastrophic regressions are invisible until the model actually generates tokens.
2. **MSE is the natural objective at fit time.** Lloyd-Max is an MSE optimizer; the RD curve language is MSE; the published bounds (TurboQuant Theorems 1–3, EDEN-S derivation, HIGGS' linearity theorem) are all MSE. There is no widely-deployed KL-aware centroid-fit primitive, so the field defaults to MSE because nothing else is available.
3. **Standard PPL benchmarks average across thousands of tokens.** The 8% catastrophic rate at next-token KL@D translates to a small mean-PPL nudge that lives well within the ±0.05 noise floor of wikitext-103 PPL evaluations. A method can have a 50% catastrophic rate per-prompt and still post a "neutral" PPL number on the standard benchmarks. Multi-token averaging is itself a proxy that hides the failure.

The blind spots compound: short prefill calibration → MSE-on-prefill objective → averaged PPL validation. At every stage the right metric to surface the failure is one step further out than the methodology actually goes. This paper's contribution is to push each stage one step further: longer-context decode evaluation, KL@D instead of MSE, per-prompt catastrophic rates instead of corpus averages.

The uncomfortable summary: **the field optimized what was easy to measure, not what mattered, and the two diverge in this regime.** MSE is differentiable, has closed-form Lloyd-Max optimizers, fits in a rate-distortion framework, and produces clean theorems. KL@D requires a model forward pass, has no clean closed-form optimizer, and produces messy per-prompt numbers. Methodology gravitates toward the metric that supports the clean math even when the clean math is on the wrong metric.

---

## 11. Final Status

| candidate | strongest empirical result | blocker |
|---|---|---|
| Sub-Gaussian K observation | reproducible across 5 model families, KS-stat 0.04–0.16, p~0 | observation, not algorithm |
| **ens_4way ensemble** | +84% median KL on Qwen3-0.6B b=4; AMD b=3 N=30: 0/30 catastrophic | 8% catastrophic ceiling at M5 N=100 |
| Universal K table (5-model fit) | 1–13% K-MSE wins, MSE-cross-validates | F3: +71% to +90% mean KL regression, 50–60% catastrophic |
| KL-aware Lloyd objective | not run | future work |

Nothing deployed. The companion paper [eden-optimal-s-revisit.md](./eden-optimal-s-revisit.md) closed issues #87 and #89.

The most durable result is the negative one: **per-coord MSE on K reconstruction is not a valid quality proxy for KV cache quantization.** The sub-Gaussian K observation, the ens_4way variance-reduction principle, and the PDS diagnosis are byproducts of the path that arrived at this conclusion.

---

## 12. Limitations and Future Work

1. **No PPL evaluation.** All KL claims are next-token KL@D. Multi-token PPL (REFRACT, wikitext-103) was not measured. ens_4way's 8% next-token rate may not survive multi-token averaging; F3's 50–60% likely does, but this is not proven.
2. **Single primary model for the deployable-algorithm arc.** Qwen3-0.6B drives the H8–H29 arc. F1–F3 cover five models for the universal-K story.
3. **No external comparison.** This investigation lives entirely inside the TurboQuant+ design space. KIVI, SmoothQuant, GPTQ K-cache variants were out of scope.
4. **Catastrophic-rate confidence intervals are wide.** N=30 and N=100 give Wilson-95% CIs that span 5–10 percentage points. The "8% ceiling" is empirical, not a tight bound.
5. **No formal MSE-vs-KL scatter.** The H17 null result is a categorical correlation refutation; a per-prompt scatter at N=100 would be a continuous one. Highest-priority follow-up.

**Direction for the field.** Future KV quantization work should treat downstream behavior (KL@D, end-task metrics) as first-class objectives during centroid optimization, rather than relying on per-vector reconstruction proxies. The Lloyd-Max algorithm is MSE-by-construction; replacing its update step with a KL-aware proxy is the cleanest path forward.

---

## Appendix A: Key Hypotheses Cited in the Body

| # | Hypothesis | Outcome |
|---|---|---|
| **8** | Are post-WHT KV vectors Gaussian? | **K is sub-Gaussian, V is Gaussian** |
| 9 | Data-driven Lloyd centroids on real KV | 25–75% MSE improvement on K |
| 10 | Does H9's MSE win transfer to logit KL? | Yes: 64% KL reduction on calibration prompt |
| **17** | Forensics: what predicts catastrophic chunks | **No prefill statistic predicts catastrophic failure (PDS)** |
| 22 | Validate single-method at scale (N=30) | 23.3% catastrophic, mean Δ KL negative |
| 23 | Ensemble of default + tail | +55% mean, 13% catastrophic |
| 27 | AMD MI300X cross-validation (N=50) | 26% catastrophic, hardware-agnostic |
| **28** | ens_4way at scale (N=30) | +75% mean, +81% median, 0/30 catastrophic |
| **29** | ens_4way scale validation (N=100) | **+69% mean, +84% median, 8/100 catastrophic** |
| 32 | ens_4way long context (1024-tok, N=20) | 10% catastrophic, same rate, longer prefill |
| H_univ | Static "universal K" table from 2 models | L2 3× closer cross-model than to default |
| F2 | 5-model universal K cross-validation (MSE) | Cluster A: +1–13% MSE on 8/10 cells |
| **F3** | **Cluster A universal K → model KL test** | **+71% to +90% mean KL regression, 50–60% catastrophic** |

Full ledger of all 35 hypotheses (including the EDEN-arc H1–H7, secondary fitting variants, and not-yet-run items) in [Appendix B](#appendix-b-full-hypothesis-ledger).

---

## Appendix B: Full Hypothesis Ledger

| # | Hypothesis | Outcome |
|---|---|---|
| 1 | EDEN's d=128 b=4 → 2.25% MSE replicates | EDEN ref 19% better; gap is dense-rotation artifact |
| 2 | Isolate which EDEN component drives gap | Hadamard rotation = entire gap (~71%); S and centroids each ~1% |
| 3 | Catastrophic failure mode in dense rotation | 1% of vectors with cos(c,x)=0.86 vs median 0.996 |
| 4 | WHT eliminates the failure mode | Confirmed: p99 0.144 → 0.018 |
| 5 | TQ_prod chain vs direct b-bit on real KV | Direct b-bit dominates by 5–6× MSE |
| 6 | EDEN-unbiased vs biased vs matched-norm | Matched-norm wins on real KV (0.5–9%) |
| 7 | Per-layer global S calibration | Marginal: −2% to +2% across layers |
| 8 | Are post-WHT KV vectors Gaussian? | K is sub-Gaussian, V is Gaussian |
| 9 | Data-driven Lloyd centroids on real KV | 25–75% MSE improvement on K (early/mid layers) |
| 10 | Does H9's MSE win transfer to logit KL? | Yes: 64% KL reduction on calibration prompt |
| 11 | Held-out generalization (calib ≠ test) | 1/5 catastrophic regression, too brittle |
| 12 | Per-prompt online calibration | 5/8 win, 3/8 lose; tail-clipping at decode is failure mode |
| 13 | Tail_aware: synthetic tail samples | 80% mean KL reduction, 0/6 on small sample (sample noise) |
| 14 | Validate across bit widths (b=2, 3, 4) | Median wins at all bits; ~12% catastrophic rate |
| 15 | best_of fallback via prefill MSE | MSE-on-prefill is not a useful gate |
| 16 | Per-head centroid fitting | Worse than per-layer (less data per fit hurts) |
| 17 | Forensics: what predicts catastrophic chunks | No prefill statistic predicts catastrophic failure |
| 18 | Held-out CV gate | MSE-on-prefill not predictive |
| 19 | Pred-gate via held-out token KL | Misclassifies catastrophic case |
| 20 | Centroid-values-only (keep default boundaries) | 4/8 catastrophic |
| 21 | Alpha-blend centroids | α=0.9 best; 1/8 catastrophic |
| 22 | Validate at scale (N=30) | Catastrophic 23.3%; mean Δ negative (−3.7%) |
| 23 | Ensemble of default + tail | +55% mean, +71% median, 4/30 catastrophic (13%) |
| 24 | MSE screening of 8 fitting variants | S5 robust −52% MSE best |
| 25 | bounded at scale (N=30) | +12% mean, +51% median, 6/30 catastrophic |
| 26 | Paper expansion (HL5–HL8 + MH1–MH4) | Used for eden-optimal-s-revisit.md |
| 27 | AMD MI300X cross-validation (N=50) | tail_aware: 13/50 (26%) catastrophic |
| 28 | ens_4way at scale (N=30) | +75% mean, +81% median, 0/30 catastrophic |
| 29 | ens_4way scale validation (N=100) | +69% mean, +84% median, 8/100 catastrophic |
| 30 | ens_4way at b=2, b=3 (AMD) | b=3: 0/30, +75%. b=2: 17%, +30% |
| 31 | ens_4way on Llama-3.2-1B | Pending |
| 32 | ens_4way long context (1024-tok, N=20) | Same 10% catastrophic rate |
| 33–35 | REFRACT PPL eval, low-bit cross-family, hybrid | Not run |
| H_univ | Static "universal K" table from 2 models | L2 3× closer cross-model than to default |
| F1 | Dump K from 3 new families | 5-model dataset assembled |
| F2 | 5-model universal K cross-validation (MSE) | Strict FAIL; Cluster A subset +1–13% MSE |
| F3 | Cluster A universal K → model KL test | +71% to +90% mean KL regression, 50–60% catastrophic |

---

## References

- TurboQuant paper (Google Research / NYU, ICLR 2026): [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- DRIVE paper (NeurIPS 2021): [arXiv:2105.08339](https://arxiv.org/abs/2105.08339)
- EDEN paper (ICML 2022): [proceedings.mlr.press/v162/vargaftik22a.html](https://proceedings.mlr.press/v162/vargaftik22a.html)
- Note on TurboQuant and DRIVE/EDEN: [arXiv:2604.18555](https://arxiv.org/abs/2604.18555)
- Companion paper (algorithmic claim): [eden-optimal-s-revisit.md](./eden-optimal-s-revisit.md)
- Issue #87 (Mitzenmacher): historical identifier ([public endpoint unavailable](../../docs/reference/historical-forks.md#archived-identifiers))
- Issue #89 (Portnoy): historical identifier ([public endpoint unavailable](../../docs/reference/historical-forks.md#archived-identifiers))
- Related work (cited if developed further): KIVI (Liu et al., 2024), HIGGS, SmoothQuant (Xiao et al.), OmniQuant, GPTQ
