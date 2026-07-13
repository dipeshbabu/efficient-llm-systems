"""Tests for axes.trajectory (v0.1.4) and v0.2 skeletons.

Pure logic + skeleton smoke tests. The real subprocess-driven behaviour
of ``run_trajectory`` is exercised by the integration-tier tests under
``test_validation.py`` (skipped unless ``REFRACT_RUN_INTEGRATION=1``).
"""

from __future__ import annotations

import json

import pytest

from refract.axes import plad as plad_mod
from refract.axes import rniah as rniah_mod
from refract.axes import trajectory as traj_mod
from refract.axes.trajectory import (
    TrajectoryResult,
    _diff,
    _load_prompts,
)


# ---------------------------------------------------------------------------
# _diff: same shape as gtm._diff but operates on int token-ID lists
# ---------------------------------------------------------------------------


def test_diff_identical():
    assert _diff([1, 2, 3], [1, 2, 3]) == (None, 3)


def test_diff_first_position():
    # Diverge at step 0
    assert _diff([1, 2, 3], [9, 2, 3]) == (0, 0)


def test_diff_middle():
    assert _diff([1, 2, 3, 4], [1, 2, 9, 4]) == (2, 2)


def test_diff_cand_is_prefix_of_ref():
    # cand stops short — divergence is at the boundary
    assert _diff([1, 2, 3, 4], [1, 2, 3]) == (3, 3)


def test_diff_ref_is_prefix_of_cand():
    # ref stops short (e.g. EOS)
    assert _diff([1, 2, 3], [1, 2, 3, 4]) == (3, 3)


def test_diff_both_empty():
    # No tokens decoded by either side. Treated as full match (zero-length).
    assert _diff([], []) == (None, 0)


def test_diff_one_empty():
    # Boundary divergence at position 0 when one side decoded nothing.
    assert _diff([1], []) == (0, 0)
    assert _diff([], [1]) == (0, 0)


# ---------------------------------------------------------------------------
# _load_prompts
# ---------------------------------------------------------------------------


def test_load_prompts(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        '{"id": "p1", "prompt": "hello"}\n'
        "# comment line, ignored\n"
        "\n"
        '{"id": "p2", "category": "smoke", "prompt": "world"}\n'
    )
    out = _load_prompts(p)
    assert len(out) == 2
    assert out[0]["id"] == "p1"
    assert out[1]["category"] == "smoke"


# ---------------------------------------------------------------------------
# TrajectoryResult shape
# ---------------------------------------------------------------------------


def test_trajectory_result_has_gtm_compatible_shape():
    """Drop-in compat with GTMResult: same fields, same types.

    The composite scorer (score.composite_score) only consumes ``score`` so
    technically only that field matters, but reporting code accesses the
    diagnostics; pin the shape so a swap-in doesn't surprise downstream
    consumers.
    """
    r = TrajectoryResult(
        score=87.5,
        full_match_rate=0.6,
        median_first_divergence=12,
        mean_prefix_agreement_length=42.0,
        mean_cand_length=48.0,
        mean_ref_length=48.0,
        n_prompts=10,
        n_tokens_each=50,
        per_prompt=[],
    )
    # Same field names as GTMResult so the report layer is indifferent.
    from refract.axes.gtm import GTMResult

    gtm_fields = {f for f in GTMResult.__dataclass_fields__}
    traj_fields = {f for f in TrajectoryResult.__dataclass_fields__}
    # Trajectory must include every diagnostic field GTM exposed.
    assert gtm_fields <= traj_fields, (
        f"Missing fields vs GTMResult: {gtm_fields - traj_fields}"
    )


# ---------------------------------------------------------------------------
# run_trajectory: subprocess path is mocked. We verify the empty-trajectory
# protective error fires when the patched binary isn't present.
# ---------------------------------------------------------------------------


def test_run_trajectory_raises_on_empty_ref_and_cand(monkeypatch, tmp_path):
    """Both sides empty → patched binary is missing; raise loudly so the
    user knows to rebuild llama-completion. Silent fallback would produce
    a 0-tokens score that looks like data."""
    p = tmp_path / "prompts.jsonl"
    p.write_text('{"id": "p1", "prompt": "hello"}\n')

    def fake_run(*args, **kwargs):
        return ([], {"n_tokens": 0})

    monkeypatch.setattr(traj_mod, "run_completion_trajectory", fake_run)

    from refract.runner import KVConfig

    with pytest.raises(RuntimeError, match="patched llama-completion"):
        traj_mod.run_trajectory(
            model=tmp_path / "fake.gguf",
            reference_kv=KVConfig.parse("ctk=f16,ctv=f16"),
            candidate_kv=KVConfig.parse("ctk=q8_0,ctv=q8_0"),
            prompts_path=p,
            n_predict=8,
            progress=False,
        )


def test_run_trajectory_happy_path(monkeypatch, tmp_path):
    """Two sequences differ at step 5; verify score, divergence, and
    per-prompt diagnostics."""
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        '{"id": "p1", "prompt": "hello"}\n'
        '{"id": "p2", "category": "smoke", "prompt": "world"}\n'
    )

    # ref: 10 tokens; cand: matches first 5 then diverges at 5; 8 tokens total
    ref_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cand_tokens = [1, 2, 3, 4, 5, 99, 7, 8]

    call_order = []

    def fake_run(model, prompt, kv, **_kw):
        call_order.append((prompt, kv.label()))
        # Reference call first per prompt, then candidate. Trajectory v0.1.4
        # alternates ref/cand in run_trajectory().
        is_ref = "f16" in kv.label()
        return (ref_tokens if is_ref else cand_tokens, {"n_tokens": 0})

    monkeypatch.setattr(traj_mod, "run_completion_trajectory", fake_run)

    from refract.runner import KVConfig

    res = traj_mod.run_trajectory(
        model=tmp_path / "fake.gguf",
        reference_kv=KVConfig.parse("ctk=f16,ctv=f16"),
        candidate_kv=KVConfig.parse("ctk=q8_0,ctv=q8_0"),
        prompts_path=p,
        n_predict=10,
        progress=False,
    )

    # Both prompts diverge at step 5 ⇒ prefix_agreement_length = 5
    # mean_cand_length = 8, score = 100 * 5 / 8 = 62.5
    assert res.n_prompts == 2
    assert res.full_match_rate == 0.0
    assert res.mean_prefix_agreement_length == 5.0
    assert res.mean_cand_length == 8.0
    assert res.median_first_divergence == 5
    assert res.score == pytest.approx(62.5)
    # Per-prompt diagnostics carry token-ID lists, not text — the v0.1.4 fix.
    assert res.per_prompt[0]["ref_token_ids"] == ref_tokens
    assert res.per_prompt[0]["cand_token_ids"] == cand_tokens
    assert res.per_prompt[0]["matched"] is False
    assert res.per_prompt[0]["first_divergence"] == 5


# ---------------------------------------------------------------------------
# v0.2 skeletons: importable, fields stable, run_* raises NotImplementedError
# ---------------------------------------------------------------------------


def test_rniah_module_constants():
    """v0.2 implementer must keep DEFAULT_* stable so reporting code can rely
    on them without hardcoding strings."""
    assert rniah_mod.DEFAULT_LENGTHS == (4096, 8192, 16384, 32768, 65536)
    assert rniah_mod.DEFAULT_POSITIONS == (0.10, 0.50, 0.90)
    assert "APRICOT" in rniah_mod.DEFAULT_NEEDLE


def test_rniah_dataclass_shapes_stable():
    cell_fields = {f for f in rniah_mod.RNIAHCell.__dataclass_fields__}
    assert {"length", "position", "n_trials", "base_acc",
            "cand_acc", "degradation"} <= cell_fields

    res_fields = {f for f in rniah_mod.RNIAHResult.__dataclass_fields__}
    assert {"score", "n_cells", "cells", "skipped_cells",
            "needle", "notes", "password_keyword"} <= res_fields


def test_rniah_extract_password_keyword_default():
    assert rniah_mod._extract_password_keyword(
        "The secret password is APRICOT-7-BLUE."
    ) == "APRICOT-7-BLUE"


def test_rniah_extract_password_keyword_fallback():
    # No all-caps token → fall back to the last whitespace-token.
    assert rniah_mod._extract_password_keyword(
        "the password is hunter2."
    ) == "hunter2"


def test_rniah_nearest_sentence_boundary_finds_period():
    text = "Foo. Bar baz qux. Quux."
    # target=10 → there's a period at 16; nearest going forward is 16+2=18,
    # nearest going backward is 4 (first period+space). Outward scan gives 4
    # before 18 because abs(10-4)=6 < abs(18-10)=8.
    out = rniah_mod._nearest_sentence_boundary(text, 10)
    assert out in (4, 18)


def test_rniah_run_skips_when_all_lengths_exceed_ctx_max(tmp_path, monkeypatch):
    """If every cell is skipped because length > ctx_max, the run returns
    score=0.0 with all cells in skipped_cells and a clear note."""
    haystack = tmp_path / "haystack.txt"
    haystack.write_text("Hello world. " * 200)

    # Stub out the costly subprocess paths that we don't reach in this case.
    monkeypatch.setattr(rniah_mod, "tokenize_to_ids",
                        lambda model, text: [1] * (len(text) // 4))

    from refract.runner import KVConfig

    res = rniah_mod.run_rniah(
        model=tmp_path / "fake.gguf",
        haystack_corpus=haystack,
        reference_kv=KVConfig.parse("ctk=f16,ctv=f16"),
        candidate_kv=KVConfig.parse("ctk=q8_0,ctv=q8_0"),
        ctx_max=128,  # smaller than every default length
        n_trials=1,
        progress=False,
    )
    assert res.score == 0.0
    assert res.n_cells == 0
    assert len(res.skipped_cells) == len(rniah_mod.DEFAULT_LENGTHS) * len(
        rniah_mod.DEFAULT_POSITIONS
    )


def test_rniah_run_full_match(tmp_path, monkeypatch):
    """Both base and candidate find the password in every cell → score 100."""
    haystack = tmp_path / "haystack.txt"
    haystack.write_text("Lorem ipsum dolor sit amet. " * 1000)

    monkeypatch.setattr(rniah_mod, "tokenize_to_ids",
                        lambda model, text: [1] * max(1, len(text) // 4))

    def fake_run_completion(model, prompt, kv, **_kw):
        # Both KV configs respond with the password every time.
        return ("APRICOT-7-BLUE is the password.", {})

    monkeypatch.setattr(rniah_mod, "run_completion", fake_run_completion)

    from refract.runner import KVConfig

    res = rniah_mod.run_rniah(
        model=tmp_path / "fake.gguf",
        haystack_corpus=haystack,
        reference_kv=KVConfig.parse("ctk=f16,ctv=f16"),
        candidate_kv=KVConfig.parse("ctk=q8_0,ctv=q8_0"),
        ctx_max=8192,
        lengths=(1024, 2048),
        positions=(0.5,),
        n_trials=1,
        progress=False,
    )
    assert res.n_cells == 2
    assert res.score == pytest.approx(100.0)
    for cell in res.cells:
        assert cell.base_acc == 1.0
        assert cell.cand_acc == 1.0
        assert cell.degradation == 0.0


def test_rniah_run_candidate_loses_at_long_ctx(tmp_path, monkeypatch):
    """Realistic shape: cand finds password at short context but loses at
    long context. R-NIAH score reflects the per-cell degradation."""
    haystack = tmp_path / "haystack.txt"
    haystack.write_text("Lorem ipsum dolor sit amet. " * 4000)

    monkeypatch.setattr(rniah_mod, "tokenize_to_ids",
                        lambda model, text: [1] * max(1, len(text) // 4))

    def fake_run_completion(model, prompt, kv, **kw):
        # ctx tells us the cell's length here (we set ctx = length + n_predict + 32).
        # f16 always finds it. q8_0 fails when ctx > ~2200.
        is_ref = "f16" in kv.label()
        ctx = kw.get("ctx", 0)
        if is_ref or ctx <= 2200:
            return ("Found it: APRICOT-7-BLUE.", {})
        return ("Sorry, I do not recall.", {})

    monkeypatch.setattr(rniah_mod, "run_completion", fake_run_completion)

    from refract.runner import KVConfig

    res = rniah_mod.run_rniah(
        model=tmp_path / "fake.gguf",
        haystack_corpus=haystack,
        reference_kv=KVConfig.parse("ctk=f16,ctv=f16"),
        candidate_kv=KVConfig.parse("ctk=q8_0,ctv=q8_0"),
        ctx_max=8192,
        lengths=(1024, 4096),  # 1024 → short (ok), 4096 → long (cand fails)
        positions=(0.5,),
        n_trials=1,
        progress=False,
    )
    # Two cells: short cell base=1, cand=1, deg=0; long cell base=1, cand=0, deg=1.
    # mean_deg = 0.5, score = 50.
    assert res.n_cells == 2
    short = next(c for c in res.cells if c.length == 1024)
    long_ = next(c for c in res.cells if c.length == 4096)
    assert short.degradation == 0.0
    assert long_.degradation == 1.0
    assert res.score == pytest.approx(50.0)


# ---- PLAD ------------------------------------------------------------------


def test_plad_module_constants():
    assert plad_mod.DEFAULT_PERTURBATIONS == ("typo", "case", "punct", "paraphrase")
    assert plad_mod.DEFAULT_ALPHA == 5.0


def test_plad_dataclass_shapes_stable():
    pp_fields = {f for f in plad_mod.PLADPerPrompt.__dataclass_fields__}
    assert {"prompt_id", "perturbation", "perturbed_prompt",
            "ref_drift", "cand_drift", "excess_drift", "plad_pp"} <= pp_fields

    res_fields = {f for f in plad_mod.PLADResult.__dataclass_fields__}
    assert {"score", "per_perturbation_score", "per_prompt",
            "n_prompts", "n_perturbations"} <= res_fields


def test_plad_levenshtein_sanity():
    assert plad_mod._levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert plad_mod._levenshtein([1, 2, 3], [1, 9, 3]) == 1
    assert plad_mod._levenshtein([1, 2, 3], []) == 3
    assert plad_mod._levenshtein([], [1, 2, 3]) == 3


def test_plad_apply_typo_swaps_two_chars():
    import random as _r
    out = plad_mod._apply_typo("Please describe Paris", _r.Random(0))
    assert out is not None
    assert out != "Please describe Paris"
    # Should still be roughly the same length (single swap)
    assert abs(len(out) - len("Please describe Paris")) <= 1


def test_plad_apply_typo_no_eligible_word():
    import random as _r
    # All words ≤3 chars or stopwords; should return None.
    assert plad_mod._apply_typo("a b cd ef", _r.Random(0)) is None


def test_plad_apply_case_lowers_first_letters():
    out = plad_mod._apply_case("Hello World")
    assert out == "hello world"


def test_plad_apply_case_no_change_returns_none():
    assert plad_mod._apply_case("hello world") is None


def test_plad_apply_punct_adds_question_mark():
    assert plad_mod._apply_punct("hello") == "hello?"


def test_plad_apply_punct_removes_question_mark():
    assert plad_mod._apply_punct("hello?") == "hello"


def test_plad_apply_paraphrase_uses_synonym():
    import random as _r
    out = plad_mod._apply_paraphrase("Make a big house", _r.Random(0))
    assert out is not None
    # 'make' → 'create' or 'big' → 'large'; either way the prompt changed.
    assert out != "Make a big house"
    assert any(w in out.lower() for w in ("create", "large"))


def test_plad_run_happy_path(tmp_path, monkeypatch):
    """Realistic shape: cand drifts a lot more than ref under perturbation."""
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        '{"id": "p1", "prompt": "Please describe Paris briefly"}\n'
    )

    # Stub tokenize_to_ids to return character codes — gives a deterministic
    # token sequence with a clear edit-distance signal.
    monkeypatch.setattr(plad_mod, "tokenize_to_ids",
                        lambda model, text: [ord(c) for c in text or ""])

    # Reference: stable answer regardless of perturbation.
    # Candidate: gets perturbed prompts wrong (different surface form).
    def fake_run_completion(model, prompt, kv, **_kw):
        is_ref = "f16" in kv.label()
        if is_ref:
            return ("Paris is in France.", {})
        # Candidate drifts on perturbed prompts but stays put on the anchor.
        if prompt == "Please describe Paris briefly":
            return ("Paris is in France.", {})
        return ("xx" * 30, {})  # very different from anchor

    monkeypatch.setattr(plad_mod, "run_completion", fake_run_completion)

    from refract.runner import KVConfig

    res = plad_mod.run_plad(
        model=tmp_path / "fake.gguf",
        prompts_path=p,
        reference_kv=KVConfig.parse("ctk=f16,ctv=f16"),
        candidate_kv=KVConfig.parse("ctk=q8_0,ctv=q8_0"),
        progress=False,
    )

    # At least some perturbations applied (typo, case, punct guaranteed for
    # this prompt; paraphrase may or may not).
    assert len(res.per_prompt) >= 3
    # Reference drift is ~0; candidate drift is >0; excess > 0; plad_pp < 100.
    for r in res.per_prompt:
        assert r.ref_drift == pytest.approx(0.0)
        assert r.cand_drift > 0.0
        assert r.excess_drift > 0.0
        assert r.plad_pp < 100.0
    # Overall score is below 100.
    assert 0.0 < res.score < 100.0


def test_plad_run_unknown_perturbation_raises(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text('{"id": "p1", "prompt": "x"}\n')
    from refract.runner import KVConfig
    with pytest.raises(ValueError, match="Unknown perturbations"):
        plad_mod.run_plad(
            model=tmp_path / "fake.gguf",
            prompts_path=p,
            reference_kv=KVConfig.parse("ctk=f16,ctv=f16"),
            candidate_kv=KVConfig.parse("ctk=q8_0,ctv=q8_0"),
            perturbations=("nonsense",),
        )


# ---- 4-axis composite ------------------------------------------------------


def test_composite_score_4_axes():
    from refract.score import composite_score

    c = composite_score(
        gtm_score=80.0, kld_score=99.0,
        rniah_score=70.0, plad_score=90.0,
    )
    # Harmonic mean of [80, 99, 70, 90] = 4 / (1/80 + 1/99 + 1/70 + 1/90)
    expected = 4 / (1 / 80 + 1 / 99 + 1 / 70 + 1 / 90)
    assert c.composite == pytest.approx(expected, rel=1e-6)
    assert c.rniah_score == 70.0
    assert c.plad_score == 90.0


def test_composite_score_2_axes_unchanged():
    """v0.1 callers passing only gtm + kld must get the same number they
    used to (no rniah / no plad in the harmonic mean)."""
    from refract.score import composite_score

    c = composite_score(gtm_score=92.0, kld_score=99.5)
    expected = 2 / (1 / 92.0 + 1 / 99.5)
    assert c.composite == pytest.approx(expected, rel=1e-6)
    assert c.rniah_score is None
    assert c.plad_score is None
