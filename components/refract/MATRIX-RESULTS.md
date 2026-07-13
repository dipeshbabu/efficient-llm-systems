# REFRACT matrix results — 2026-04-30

Permanent record of every matrix cell run today. Reverse-chronological:
v0.3.0 chat-template results at top, then v0.2.x results, then negative
control. JSON outputs live under `/tmp/refract-matrix-*` with the
timestamp in the directory name.

Reference KV: `ctk=f16,ctv=f16` (the model's own fp16-KV baseline).

---

## v0.3.0 chat-template matrix — COMPLETE

Output: `/tmp/refract-matrix-v0.3-20260430-120257/`
Candidate: `ctk=q8_0,ctv=turbo4`
Changes: `--jinja` chat template + `-rea off` + R-NIAH `-sys` for haystack.
Total runtime: 7336s (2h 2min) on M5 Max 128GB.

| Model                  | Composite | Band      | Trajectory | KLD   | KLD nats | R-NIAH | PLAD  |
|------------------------|-----------|-----------|------------|-------|----------|--------|-------|
| qwen3.5-2B-Q8          |  81.48    | PASS      |   60.07    | 98.35 | 0.0167   | 100.00 | 81.47 |
| gemma-4-E2B-Q4         |  78.51    | DEGRADED  |   52.73    | 93.50 | 0.0672   | 100.00 | 88.57 |
| phi-4-Q8               |  90.25    | EXCELLENT |   77.95    | 99.55 | 0.0046   | 100.00 | 87.35 |
| qwen2.5-7B-Q8          |  77.98    | DEGRADED  |   55.13    | 98.75 | 0.0126   | 100.00 | 76.73 |
| Mistral-24B-Q4         |  90.86    | EXCELLENT |   76.65    | 99.71 | 0.0029   | 100.00 | 91.34 |
| **gemma-4-26B-A4B-Q8** | **29.12** | **FAIL**  |   17.32    | 17.59 | 1.7381   | 100.00 | 78.40 |
| **gemma-4-31B-Q8**     | **50.78** | **FAIL**  |   26.41    | 49.23 | 0.7086   | 100.00 | 94.45 |

### v0.2.0 → v0.3.0 comparison

| Model              | v0.2.0 → v0.3.0 Composite | R-NIAH (raw → templated)              | PLAD Δ           |
|--------------------|---------------------------|---------------------------------------|------------------|
| qwen3.5-2B-Q8      | 83.19 → 81.48 (−1.71)     | 100 (refusal) → 100 (real, base=0.67) | 88.76 → 81.47 (−7.29) |
| gemma-4-E2B-Q4     | 77.78 → 78.51 (+0.74)     | 100 (refusal) → 100 (real, base=0.83) | 84.95 → 88.57 (+3.62) |
| phi-4-Q8           | 90.25 → 90.25 (no change) | 100 → 100                             | 87.35 → 87.35    |
| qwen2.5-7B-Q8      | 76.12 → 77.98 (+1.86)     | 88.89 → 100 (1-cell loss → clean)     | 76.73 → 76.73    |
| Mistral-24B-Q4    | 90.86 → 90.86 (no change) | 100 → 100                             | 91.34 → 91.34    |
| gemma-4-26B-A4B-Q8 | 29.14 → 29.12 (−0.02)     | 100 (refusal) → 100 (real, base=0.67) | 79.01 → 78.40 (−0.61) |
| gemma-4-31B-Q8     | 50.48 → 50.78 (+0.30)     | 100 (refusal) → 100 (real)            | 90.49 → 94.45 (+3.96) |

### Conclusions

1. **Composite ranking is unchanged** between v0.2.0 and v0.3.0. The
   v0.2.0 ranking on this candidate (q8_0/turbo4) was correct even with
   the methodology bug.
2. **Per-cell measurement validity is fixed.** R-NIAH's "refusal artifact
   100s" became "real-engagement 100s" on gemmas + small qwens.
3. **PLAD shifts most** under chat-template handling. Perturbations
   propagate through templated prompts differently than raw prompts.
   Mostly small (±4) but qwen3.5-2B dropped 7.3 points.
4. **Two models had zero change** (phi-4, Mistral-24B). They engaged in
   raw mode already.
5. **Negative control (sym turbo @ 11 FAIL) and positive controls hold**
   across methodology versions. Framework credibility is rooted in
   per-axis bit-exactness on Metal, not the chat-template fix.

---

## v0.2.0 main matrix — 2026-04-30

Output: `/tmp/refract-matrix-v0.2-20260430-084540/`
Candidate: `ctk=q8_0,ctv=turbo4`
Total runtime: 7370s (2h 2min) on M5 Max 128GB.

| Model                  | Composite | Band     | Trajectory | KLD   | KLD nats | R-NIAH | PLAD  |
|------------------------|-----------|----------|------------|-------|----------|--------|-------|
| qwen3.5-2B-Q8          | ~89       | PASS     | 60.07      | 98.35 | 0.0167   | 100.00*| 88.76 |
| gemma-4-E2B-Q4         | ~80       | PASS     | 52.73      | 93.50 | 0.0672   | 100.00*| 84.95 |
| phi-4-Q8               | ~90       | PASS     | 77.95      | 99.55 | 0.0046   | 100.00 | 87.35 |
| qwen2.5-7B-Q8          | ~76       | DEGRADED | 55.13      | 98.75 | 0.0126   |  88.89 | 76.73 |
| Mistral-24B-Q4         | ~89       | PASS     | 76.65      | 99.71 | 0.0029   | 100.00 | 91.34 |
| **gemma-4-26B-A4B-Q8** | **~29**   | **FAIL** |   17.32    | 17.59 | 1.7381   | 100.00*| 79.01 |
| **gemma-4-31B-Q8**     | **~50**   | **FAIL** |   26.41    | 49.23 | 0.7086   | 100.00*| 90.49 |

`*` = R-NIAH inflated by chat-template / refusal artifacts. See v0.2.1
re-run below for re-measured numbers on the affected models.

### PLAD per-perturbation breakdown

| Model              | typo   | case   | punct  | paraphrase |
|--------------------|--------|--------|--------|------------|
| qwen3.5-2B-Q8      | 88.76* | 88.76* | 88.76* | nan        |
| gemma-4-E2B-Q4     | 80.20  | 86.00  | 89.00  | nan        |
| phi-4-Q8           | 86.30  | 89.10  | 88.41  | nan        |
| qwen2.5-7B-Q8      | 70.65  | 78.88  | 80.55  | nan        |
| Mistral-24B-Q4     | 91.20  | 90.10  | 92.50  | nan        |
| gemma-4-26B-A4B-Q8 | 72.43  | 86.32  | 78.82  | nan        |
| gemma-4-31B-Q8     | 88.33  | 87.76  | 94.88  | nan        |

`paraphrase = nan` across all models — the v0.1 prompt set has no words in
the small built-in synonym table. Either expand the table or skip
paraphrase entirely; tracked for v0.2.2.

---

## v0.2.0 extras — Llama-4 Scout & symmetric-turbo negative control

Output: `/tmp/refract-matrix-v0.2-extras-20260430-104924/`

### Llama-4 Scout 17B-16E Q4_K_M

Candidate: `ctk=q8_0,ctv=turbo4` (same as main matrix)

| Axis       | Score  | Band      |
|------------|--------|-----------|
| Trajectory |  73.58 | DEGRADED  |
| KLD@D      |  97.32 | EXCELLENT (0.0272 nats) |
| R-NIAH     | 100.00 | EXCELLENT (base engaged at 4K/8K, both fail at 16K) |
| PLAD       |  93.54 | EXCELLENT |
| **Composite** | **~89.77** | **PASS** (just below EXCELLENT) |

Runtime: 1884s (31 min). 60 GB GGUF (split into 2 parts).

Caveat for posting: REFRACT measures KV-cache quantization fidelity vs the
model's *own* fp16 baseline. This score does NOT say Llama-4 Scout is a
good model — only that the q8/turbo4 quant doesn't degrade it from its
own fp16 reference.

### Negative control: gemma-4-26B-A4B-Q8 + symmetric turbo

Candidate: `ctk=turbo4,ctv=turbo4` (the "known catastrophic" config from
the paper; rotation collisions on both K and V)

| Axis       | Score  | Band      |
|------------|--------|-----------|
| Trajectory |   3.93 | FAIL      |
| KLD@D      |  11.84 | FAIL  (2.1334 nats) |
| R-NIAH     | 100.00 | EXCELLENT (real signal post v0.2.1 needle fix) |
| PLAD       |  72.21 | DEGRADED (paraphrase NaN -> partial) |
| **Composite** | **~11** | **FAIL** |

Runtime: 1161s (19 min).

### Comparison on the same model

| Candidate                     | Composite | KLD nats | Trajectory |
|-------------------------------|-----------|----------|------------|
| ctk=q8_0,ctv=turbo4 (asym)    | ~29 FAIL  | 1.738    | 17.32      |
| ctk=turbo4,ctv=turbo4 (sym)   | ~11 FAIL  | 2.133    | 3.93       |

Symmetric is empirically worse than asymmetric on every distribution-level
surface, matching the paper's analytical prediction (rotation collisions
on K and V destroy per-head distribution geometry). Framework correctly
distinguishes the two.

---

## v0.2.1 R-NIAH re-run with neutral needle + n_predict=256

Output: `/tmp/refract-rniah-rerun-20260430-114912/`

Re-ran R-NIAH only (skip A/B/D) on the 4 models that scored
`base_acc = 0` in every cell with the v0.2.0 password-themed needle.

| Model              | v0.2.0 R-NIAH | v0.2.1 R-NIAH | base_avg | cand_avg | engagement |
|--------------------|---------------|---------------|----------|----------|------------|
| qwen3.5-2B-Q8      | 100 (refusal) | **88.89**     | 0.44     | 0.44     | 1 cell deg |
| gemma-4-E2B-Q4     | 100 (refusal) |     100.00    | 0.83     | 0.83     | clean      |
| gemma-4-26B-A4B-Q8 | 100 (refusal) |     100.00    | 0.67     | 0.67     | clean (<16K) |
| gemma-4-31B-Q8     | 100 (refusal) | **88.89**     | 0.67     | 0.56     | 1 cell deg |

The other 3 models in the main matrix (phi-4, qwen2.5-7B, Mistral-24B)
plus Llama-4 Scout had `base_acc > 0` on the original run and don't need
re-runs; their v0.2.0 R-NIAH numbers are valid.

Why the v0.2.0 R-NIAH inflation happened: the "secret password" needle
phrasing + raw `Q: ... A:` prompt format combined badly with instruct-
tuned models (especially gemmas + small qwens). The needle wording
triggered refusal on safety-trained models, and the raw prompt format
didn't engage chat mode anyway. Both fixed in v0.2.1 / v0.3.0.

---

## Open follow-ups

- **PLAD paraphrase NaN across all models** — synonym table is too small
  for the v0.1 prompt set. Expand or skip the perturbation. v0.2.2.
- **PLAD band display** for `paraphrase = NaN` shows as `FAIL` — should
  show `n/a` or `skipped`. Cosmetic; v0.2.2.
- **R-NIAH base-confidence guard** — when `base_avg < ~0.2` flag the
  R-NIAH score as low confidence. Avoids 100 readings on models that
  legitimately can't retrieve. v0.3.1.
- **T-Call axis** for tool-call fidelity. Builds on v0.3 chat-template
  machinery. Tracked for v0.4.
