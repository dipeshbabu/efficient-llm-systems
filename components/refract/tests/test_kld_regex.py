"""v0.1.3 regression tests for the KLD output parser regexes.

The KLD axis greps llama-perplexity's stdout/stderr for:
  - "Final estimate: PPL = ..."
  - "Mean    KLD: ..."
  - "RMS Δp: ... %"
  - "Same top-p: ... %"  (sometimes absent on gemma; must not crash)

We exercise the regex layer directly (no subprocess) so future llama.cpp
output changes that drift the format are caught before the matrix run.
"""

from __future__ import annotations

from refract.runner import _KLD_MEAN_RE, _PPL_RE, _RMS_DP_RE, _TOPP_RE, _first_float


CANNED_FULL = """\
... loading model ...
perplexity: calculating perplexity over 32 chunks
[1]4.21,[2]4.13,[3]4.55
====== Perplexity statistics ======
Mean PPL(Q)                   :   6.146 ± 0.034
Mean PPL(base)                :   6.139 ± 0.033
====== KL divergence statistics ======
Mean    KLD:   1.738070 ±   0.035748
Maximum KLD:  18.221
99.9%   KLD:   8.421
Mean   Δp:   1.45 ± 0.02 %
RMS Δp:   3.13 %
Same top p: 95.59 %
Final estimate: PPL = 6.1460 +/- 0.03402
"""


CANNED_NO_TOPP = """\
... loading model ...
====== KL divergence statistics ======
Mean    KLD:   0.067200 ±   0.001234
RMS Δp:   1.10 %
Final estimate: PPL = 5.4321 +/- 0.02100
"""


def test_parses_mean_kld():
    assert _first_float(_KLD_MEAN_RE, CANNED_FULL) == 1.738070


def test_parses_ppl():
    assert _first_float(_PPL_RE, CANNED_FULL) == 6.1460


def test_parses_rms_dp():
    assert _first_float(_RMS_DP_RE, CANNED_FULL) == 3.13


def test_parses_same_topp_with_dash_or_space():
    """The regex tolerates 'Same top p' (space) and 'Same top-p' (dash)."""
    assert _first_float(_TOPP_RE, CANNED_FULL) == 95.59
    dashed = CANNED_FULL.replace("Same top p", "Same top-p")
    assert _first_float(_TOPP_RE, dashed) == 95.59


def test_missing_topp_returns_none():
    """Gemma matrix runs sometimes omit the Same top-p line; the parser
    must return None instead of crashing — KLDResult.same_topp_pct is
    Optional precisely for this case."""
    assert _first_float(_TOPP_RE, CANNED_NO_TOPP) is None
    # Other fields still parse fine
    assert _first_float(_KLD_MEAN_RE, CANNED_NO_TOPP) == 0.067200


def test_missing_kld_returns_none():
    text = "no perplexity output, just an error"
    assert _first_float(_KLD_MEAN_RE, text) is None
