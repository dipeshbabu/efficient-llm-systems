"""Tests for axes' pure logic: PLAD perturbations, R-NIAH helpers, GTM/KLD math."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from refract.axes.gtm import _diff as gtm_diff, _tokenize_words, _load_prompts
from refract.axes.kld import _kld_to_score
from refract.axes.plad import (
    _apply_case,
    _apply_paraphrase,
    _apply_punct,
    _apply_typo,
    _eligible_words,
    _levenshtein,
    _normalized_drift,
    _PERTURBATION_FUNCS,
)
from refract.axes.rniah import (
    DEFAULT_NEEDLE,
    _build_prompt,
    _extract_password_keyword,
    _nearest_sentence_boundary,
    _scored,
)
from refract.axes.trajectory import _diff as traj_diff


# --- KLD axis math --------------------------------------------------------


def test_kld_to_score_zero():
    assert _kld_to_score(0.0) == pytest.approx(100.0)


def test_kld_to_score_negative_clamped():
    # llama-perplexity should never report negative; defensive clamp.
    assert _kld_to_score(-1.0) == pytest.approx(100.0)


def test_kld_to_score_monotonic_decreasing():
    a = _kld_to_score(0.5)
    b = _kld_to_score(1.0)
    c = _kld_to_score(2.0)
    assert a > b > c


def test_kld_to_score_paper_headline_matches_motivation():
    # 1.738 nats → ~17.6 (paper headline number)
    s = _kld_to_score(1.738)
    assert 17.0 < s < 18.5


# --- GTM diff -------------------------------------------------------------


def test_gtm_diff_identical():
    fd, prefix = gtm_diff([1, 2, 3], [1, 2, 3])
    assert fd is None
    assert prefix == 3


def test_gtm_diff_divergence():
    fd, prefix = gtm_diff([1, 2, 3], [1, 2, 4])
    assert fd == 2
    assert prefix == 2


def test_gtm_diff_one_is_prefix():
    fd, prefix = gtm_diff([1, 2, 3], [1, 2])
    # Boundary divergence reported at the shorter length
    assert fd == 2
    assert prefix == 2


def test_gtm_diff_empty_inputs():
    fd, prefix = gtm_diff([], [])
    assert fd is None  # both empty == identical
    assert prefix == 0


def test_traj_diff_matches_gtm_diff():
    # Trajectory and GTM use independent _diff impls; they must agree.
    for a, b in [
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 4]),
        ([], []),
    ]:
        assert traj_diff(a, b) == gtm_diff(a, b)


# --- GTM legacy whitespace tokenizer (kept for tests) ---------------------


def test_tokenize_words_drops_blanks():
    assert _tokenize_words("  hello   world  ") == ["hello", "world"]


def test_load_prompts_skips_comments_and_blanks(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text(
        '{"id": "p1", "prompt": "a"}\n'
        "\n"
        "# comment\n"
        '{"id": "p2", "prompt": "b"}\n'
    )
    prompts = _load_prompts(p)
    assert [pp["id"] for pp in prompts] == ["p1", "p2"]


# --- PLAD perturbations ---------------------------------------------------


def test_eligible_words_skips_stopwords():
    out = _eligible_words("The cat is happy")
    words = [w for (_, _, w) in out]
    assert "cat" in words
    assert "happy" in words
    assert "The" not in words  # stopword


def test_apply_typo_swaps_two_chars():
    rng = random.Random(0)
    out = _apply_typo("hello banana", rng)
    assert out is not None
    # The output differs from input in exactly the swap region
    assert out != "hello banana"
    # Length unchanged
    assert len(out) == len("hello banana")


def test_apply_typo_returns_none_when_no_long_words():
    rng = random.Random(0)
    assert _apply_typo("a b c", rng) is None  # all words < 4 chars


def test_apply_case_lowers_capitalized_words():
    out = _apply_case("Hello World")
    assert out == "hello world"


def test_apply_case_returns_none_when_no_capitals():
    assert _apply_case("hello world") is None


def test_apply_punct_removes_question_mark():
    assert _apply_punct("Are you here?") == "Are you here"


def test_apply_punct_removes_period():
    assert _apply_punct("Yes.") == "Yes"


def test_apply_punct_appends_question_mark_when_missing():
    assert _apply_punct("Hello") == "Hello?"


def test_apply_paraphrase_substitutes_synonym():
    rng = random.Random(0)
    out = _apply_paraphrase("This is a big house.", rng)
    # "big" → "large" (in synonym table)
    assert out is not None
    assert "large" in out.lower()


def test_apply_paraphrase_preserves_first_letter_case():
    rng = random.Random(0)
    out = _apply_paraphrase("Big things happen.", rng)
    assert out is not None
    # Surface case preservation: "Big" → "Large"
    assert "Large" in out


def test_apply_paraphrase_returns_none_for_no_known_synonyms():
    rng = random.Random(0)
    assert _apply_paraphrase("xyzzy plugh", rng) is None


def test_perturbation_funcs_complete():
    for name in ("typo", "case", "punct", "paraphrase"):
        assert name in _PERTURBATION_FUNCS


# --- PLAD edit distance ---------------------------------------------------


def test_levenshtein_identical():
    assert _levenshtein([1, 2, 3], [1, 2, 3]) == 0


def test_levenshtein_one_substitution():
    assert _levenshtein([1, 2, 3], [1, 9, 3]) == 1


def test_levenshtein_one_insertion():
    assert _levenshtein([1, 2], [1, 2, 3]) == 1


def test_levenshtein_one_deletion():
    assert _levenshtein([1, 2, 3], [1, 2]) == 1


def test_levenshtein_empty_a():
    assert _levenshtein([], [1, 2, 3]) == 3


def test_levenshtein_empty_b():
    assert _levenshtein([1, 2, 3], []) == 3


def test_normalized_drift_both_empty(monkeypatch):
    monkeypatch.setattr("refract.axes.plad.tokenize_to_ids", lambda *a, **kw: [])
    assert _normalized_drift(Path("m"), "", "") == 0.0


def test_normalized_drift_anchor_empty(monkeypatch):
    def fake_tok(model, text):
        return [] if not text else [1, 2]
    monkeypatch.setattr("refract.axes.plad.tokenize_to_ids", fake_tok)
    assert _normalized_drift(Path("m"), "", "x") == 1.0


def test_normalized_drift_capped_at_1(monkeypatch):
    def fake_tok(model, text):
        if text == "a":
            return [1]
        return [9, 9, 9, 9, 9]  # 5-token, very different
    monkeypatch.setattr("refract.axes.plad.tokenize_to_ids", fake_tok)
    d = _normalized_drift(Path("m"), "a", "perturbed")
    assert d == 1.0  # capped


# --- R-NIAH helpers -------------------------------------------------------


def test_extract_password_keyword_default_needle():
    kw = _extract_password_keyword(DEFAULT_NEEDLE)
    assert kw == "APRICOT-7-BLUE"


def test_extract_password_keyword_picks_longest():
    kw = _extract_password_keyword("Code A-B and CODE-NAME-FOO")
    assert kw == "CODE-NAME-FOO"


def test_extract_password_keyword_falls_back_to_last_word():
    kw = _extract_password_keyword("plain words here")
    assert kw == "here"


def test_extract_password_keyword_empty_returns_input():
    assert _extract_password_keyword("") == ""


def test_nearest_sentence_boundary_at_zero():
    assert _nearest_sentence_boundary("anything", 0) == 0


def test_nearest_sentence_boundary_past_end():
    text = "short"
    assert _nearest_sentence_boundary(text, 999) == len(text)


def test_nearest_sentence_boundary_finds_period_space():
    text = "Sentence one. Sentence two. Sentence three."
    out = _nearest_sentence_boundary(text, 14)
    # Should snap to the boundary right after "one. "
    assert out == 14


def test_nearest_sentence_boundary_falls_back_to_target():
    # No "." anywhere within window → falls back to target
    text = "a" * 1000
    assert _nearest_sentence_boundary(text, 500) == 500


def test_build_prompt_inserts_needle_at_position():
    haystack = "First sentence. Second sentence. Third sentence."
    sys_msg, user_msg = _build_prompt(haystack, "NEEDLE-X.", "Q?", 0.5)
    assert "NEEDLE-X." in sys_msg
    assert user_msg == "Q?"


def test_scored_substring_match_case_insensitive():
    assert _scored("the answer is APRICOT", "apricot") == 1


def test_scored_no_match_returns_zero():
    assert _scored("the sky is blue", "APRICOT") == 0
