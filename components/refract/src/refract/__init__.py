"""REFRACT v0.1 — REFerence-anchored Robust Acid-test for Compressed Transformers.

A benchmaxx-resistant alternative to corpus PPL for evaluating KV-cache
quantization quality. Replaces "lower PPL = better" with a composite of:

  - Axis A (GTM): Greedy Trajectory Match — do quantized + reference produce
    the same generated tokens?
  - Axis B (KLD@D): KL Divergence vs the fp16-KV reference distribution
    (v0.1 uses corpus KLD via llama-perplexity as a proxy for trajectory KLD).

See:
  - components/refract/README.md for usage.
  - research/papers/attn-rotation-and-ppl-artifact.md for the motivating paper.
"""

from __future__ import annotations

__version__ = "0.3.3"
__report_schema__ = "refract.report.v0.3.2"
__all__ = ["__version__", "__report_schema__"]
