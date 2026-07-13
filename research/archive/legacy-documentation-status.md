# Documentation status guide

This directory contains both current guidance and dated research records.
They serve different purposes:

- Start with `getting-started.md`, `turboquant-recommendations.md`, and
  `benchmarks.md` for the current repository guidance.
- Files under `papers/` are dated experiment reports. Their conclusions are
  evidence snapshots, not automatically the latest recommendation.
- Investigation notes and raw logs preserve failed hypotheses and superseded
  implementations intentionally. Check their date and the later changelog or
  recommendation guide before treating a result as current.
- Production inference implementations live in the external engine projects
  linked from the root README; this repository contains the Python reference,
  evaluation framework, scripts, and evidence.

When two reports conflict, prefer the newer controlled experiment and retain
the older document as historical evidence rather than rewriting its result.
