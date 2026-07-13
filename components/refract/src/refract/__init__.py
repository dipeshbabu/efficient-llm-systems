"""REFRACT — REFerence-anchored Robust Acid-test for Compressed Transformers.

A benchmaxx-resistant alternative to corpus PPL for evaluating KV-cache
quantization quality. Replaces "lower PPL = better" with four reference-
anchored fidelity surfaces:

  - Axis A (Trajectory): decode-time token agreement.
  - Axis B (KLD@D): next-token distribution divergence.
  - Axis C (R-NIAH): long-context retrieval fidelity.
  - Axis D (PLAD): robustness under small prompt perturbations.

See:
  - components/refract/README.md for usage.
  - research/papers/attn-rotation-and-ppl-artifact.md for the motivating paper.
"""

from __future__ import annotations

__version__ = "0.3.4"
__report_schema__ = "refract.report.v0.3.2"
__all__ = ["__version__", "__report_schema__"]
