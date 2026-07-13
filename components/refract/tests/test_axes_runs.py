"""End-to-end axis run tests: mock the runner subprocess wrappers and
exercise run_gtm / run_trajectory / run_kld / run_rniah / run_plad."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from refract.axes.gtm import GTMResult, run_gtm
from refract.axes.kld import KLDResult, run_kld
from refract.axes.plad import run_plad
from refract.axes.rniah import RNIAHResult, run_rniah
from refract.axes.trajectory import TrajectoryResult, run_trajectory
from refract.runner import KVConfig


def _write_prompts(path: Path, prompts: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(p) for p in prompts))


# --- run_gtm --------------------------------------------------------------


def test_run_gtm_perfect_match(tmp_path, monkeypatch):
    pp = tmp_path / "p.jsonl"
    _write_prompts(pp, [
        {"id": "p1", "category": "cat", "prompt": "say hi"},
        {"id": "p2", "category": "cat", "prompt": "say bye"},
    ])

    monkeypatch.setattr("refract.axes.gtm.run_completion",
                        lambda **kw: ("hello world", {}))
    monkeypatch.setattr("refract.axes.gtm.tokenize_to_ids",
                        lambda model, text: [1, 2, 3])

    res = run_gtm(
        model=Path("m.gguf"),
        reference_kv=KVConfig(), candidate_kv=KVConfig(),
        prompts_path=pp, n_predict=10, progress=False,
    )
    assert isinstance(res, GTMResult)
    assert res.score == pytest.approx(100.0)
    assert res.full_match_rate == 1.0
    assert res.median_first_divergence is None


def test_run_gtm_partial_divergence(tmp_path, monkeypatch):
    pp = tmp_path / "p.jsonl"
    _write_prompts(pp, [{"id": "p1", "prompt": "x"}])

    # Different completions per call → ref vs cand
    calls = {"n": 0}

    def fake_completion(**kw):
        calls["n"] += 1
        return ("ref" if calls["n"] % 2 == 1 else "cand", {})

    def fake_tok(model, text):
        if text == "ref":
            return [1, 2, 3, 4]
        return [1, 2, 9, 9]

    monkeypatch.setattr("refract.axes.gtm.run_completion", fake_completion)
    monkeypatch.setattr("refract.axes.gtm.tokenize_to_ids", fake_tok)
    res = run_gtm(
        model=Path("m"), reference_kv=KVConfig(), candidate_kv=KVConfig(),
        prompts_path=pp, n_predict=10, progress=False,
    )
    # prefix=2, mean_cand=4 → score 50
    assert res.score == pytest.approx(50.0)
    assert res.full_match_rate == 0.0
    assert res.median_first_divergence == 2


def test_run_gtm_empty_prompts_raises(tmp_path):
    pp = tmp_path / "empty.jsonl"
    pp.write_text("")
    with pytest.raises(ValueError):
        run_gtm(
            model=Path("m"), reference_kv=KVConfig(), candidate_kv=KVConfig(),
            prompts_path=pp, progress=False,
        )


def test_run_gtm_emits_inflation_note(tmp_path, monkeypatch):
    pp = tmp_path / "p.jsonl"
    _write_prompts(pp, [{"id": "p1", "prompt": "x"}])
    monkeypatch.setattr("refract.axes.gtm.run_completion",
                        lambda **kw: ("text", {}))
    # tokenize_to_ids returns >> n_predict → should emit "inflation" note.
    monkeypatch.setattr("refract.axes.gtm.tokenize_to_ids",
                        lambda model, text: list(range(200)))
    res = run_gtm(
        model=Path("m"), reference_kv=KVConfig(), candidate_kv=KVConfig(),
        prompts_path=pp, n_predict=10, progress=False,
    )
    assert any("inflated" in n.lower() for n in res.notes)


def test_run_gtm_tokenizer_failure_propagates(tmp_path, monkeypatch):
    pp = tmp_path / "p.jsonl"
    _write_prompts(pp, [{"id": "p1", "prompt": "x"}])
    monkeypatch.setattr("refract.axes.gtm.run_completion",
                        lambda **kw: ("text", {}))

    def boom(*a, **kw):
        raise OSError("tokenizer broken")

    monkeypatch.setattr("refract.axes.gtm.tokenize_to_ids", boom)
    with pytest.raises(RuntimeError, match="tokenize_to_ids failed"):
        run_gtm(
            model=Path("m"), reference_kv=KVConfig(), candidate_kv=KVConfig(),
            prompts_path=pp, progress=False,
        )


# --- run_trajectory -------------------------------------------------------


def test_run_trajectory_perfect_match(tmp_path, monkeypatch):
    pp = tmp_path / "p.jsonl"
    _write_prompts(pp, [{"id": "p1", "prompt": "x"}])
    monkeypatch.setattr(
        "refract.axes.trajectory.run_completion_trajectory",
        lambda **kw: ([1, 2, 3, 4], {}),
    )
    res = run_trajectory(
        model=Path("m"), reference_kv=KVConfig(), candidate_kv=KVConfig(),
        prompts_path=pp, n_predict=10, progress=False,
    )
    assert isinstance(res, TrajectoryResult)
    assert res.score == pytest.approx(100.0)
    assert res.full_match_rate == 1.0


def test_run_trajectory_both_empty_raises(tmp_path, monkeypatch):
    pp = tmp_path / "p.jsonl"
    _write_prompts(pp, [{"id": "p1", "prompt": "x"}])
    monkeypatch.setattr(
        "refract.axes.trajectory.run_completion_trajectory",
        lambda **kw: ([], {}),
    )
    with pytest.raises(RuntimeError, match="empty"):
        run_trajectory(
            model=Path("m"), reference_kv=KVConfig(), candidate_kv=KVConfig(),
            prompts_path=pp, progress=False,
        )


def test_run_trajectory_emits_short_cand_note(tmp_path, monkeypatch):
    pp = tmp_path / "p.jsonl"
    _write_prompts(pp, [{"id": "p1", "prompt": "x"}])
    monkeypatch.setattr(
        "refract.axes.trajectory.run_completion_trajectory",
        lambda **kw: ([1, 2], {}),  # 2 tokens, n_predict=10 → short
    )
    res = run_trajectory(
        model=Path("m"), reference_kv=KVConfig(), candidate_kv=KVConfig(),
        prompts_path=pp, n_predict=10, progress=False,
    )
    assert any("stopped before" in n for n in res.notes)


def test_run_trajectory_empty_prompts_raises(tmp_path):
    pp = tmp_path / "p.jsonl"
    pp.write_text("")
    with pytest.raises(ValueError):
        run_trajectory(
            model=Path("m"), reference_kv=KVConfig(), candidate_kv=KVConfig(),
            prompts_path=pp, progress=False,
        )


# --- run_kld --------------------------------------------------------------


def test_run_kld_zero_kld_gives_100(tmp_path, monkeypatch):
    corpus = tmp_path / "c.txt"
    corpus.write_text("some text")

    def fake_base(**kw):
        Path(kw["base_path"]).write_bytes(b"x")  # simulate base file
        return {"base_path": str(kw["base_path"])}

    def fake_score(**kw):
        return {"mean_kld": 0.0, "ppl": 5.0,
                "rms_dp_pct": 0.1, "same_topp_pct": 99.9}

    monkeypatch.setattr("refract.axes.kld.run_perplexity_kld_base", fake_base)
    monkeypatch.setattr("refract.axes.kld.run_perplexity_kld", fake_score)
    res = run_kld(
        model=Path("m"), corpus=corpus,
        reference_kv=KVConfig(), candidate_kv=KVConfig(),
        progress=False,
    )
    assert isinstance(res, KLDResult)
    assert res.score == pytest.approx(100.0)
    assert res.mean_kld == 0.0
    assert res.is_self_reference is True


def test_run_kld_self_reference_flag(tmp_path, monkeypatch):
    corpus = tmp_path / "c.txt"
    corpus.write_text("text")

    def fake_base(**kw):
        Path(kw["base_path"]).write_bytes(b"x")
        return {}

    monkeypatch.setattr("refract.axes.kld.run_perplexity_kld_base", fake_base)
    monkeypatch.setattr(
        "refract.axes.kld.run_perplexity_kld",
        lambda **kw: {"mean_kld": 0.0, "ppl": None,
                      "rms_dp_pct": None, "same_topp_pct": None},
    )
    ref = KVConfig.parse("ctk=q8_0,ctv=q8_0")
    cand_same = KVConfig.parse("ctk=q8_0,ctv=q8_0")
    cand_diff = KVConfig.parse("ctk=f16,ctv=f16")
    r1 = run_kld(model=Path("m"), corpus=corpus,
                 reference_kv=ref, candidate_kv=cand_same, progress=False)
    r2 = run_kld(model=Path("m"), corpus=corpus,
                 reference_kv=ref, candidate_kv=cand_diff, progress=False)
    assert r1.is_self_reference is True
    assert r2.is_self_reference is False


# --- run_rniah ------------------------------------------------------------


def test_run_rniah_skips_lengths_above_ctx_max(tmp_path, monkeypatch):
    haystack = tmp_path / "h.txt"
    # Need enough text for char slicing.
    haystack.write_text(("Sentence one. " * 5000))

    monkeypatch.setattr("refract.axes.rniah.run_completion",
                        lambda **kw: ("APRICOT-7-BLUE is here.", {}))
    monkeypatch.setattr("refract.axes.rniah.tokenize_to_ids",
                        lambda model, text: list(range(len(text) // 4)))

    res = run_rniah(
        model=Path("m"), haystack_corpus=haystack,
        reference_kv=KVConfig(), candidate_kv=KVConfig(),
        ctx_max=4096, lengths=(4096, 32768),
        positions=(0.5,), n_trials=1, progress=False,
    )
    assert isinstance(res, RNIAHResult)
    # 32768 cell skipped because > ctx_max
    assert (32768, 0.5) in res.skipped_cells
    # 4096 ran → cells include length=4096
    assert any(c.length == 4096 for c in res.cells)


def test_run_rniah_no_cells_returns_zero_score(tmp_path, monkeypatch):
    haystack = tmp_path / "h.txt"
    haystack.write_text("x")
    monkeypatch.setattr("refract.axes.rniah.run_completion",
                        lambda **kw: ("any", {}))
    monkeypatch.setattr("refract.axes.rniah.tokenize_to_ids",
                        lambda model, text: [1])
    res = run_rniah(
        model=Path("m"), haystack_corpus=haystack,
        reference_kv=KVConfig(), candidate_kv=KVConfig(),
        ctx_max=100, lengths=(4096,),  # 4096 > ctx_max → skipped
        positions=(0.5,), n_trials=1, progress=False,
    )
    assert res.score == 0.0
    assert any("No R-NIAH cells" in n for n in res.notes)


def test_run_rniah_keyword_substring_match(tmp_path, monkeypatch):
    haystack = tmp_path / "h.txt"
    haystack.write_text("Sentence one. " * 2000)
    # Reference always matches; candidate never matches.
    seq = iter([
        "yes APRICOT-7-BLUE here",  # ref
        "no answer",                # cand
    ] * 100)
    monkeypatch.setattr("refract.axes.rniah.run_completion",
                        lambda **kw: (next(seq), {}))
    monkeypatch.setattr("refract.axes.rniah.tokenize_to_ids",
                        lambda model, text: list(range(len(text) // 4)))
    res = run_rniah(
        model=Path("m"), haystack_corpus=haystack,
        reference_kv=KVConfig(), candidate_kv=KVConfig(),
        ctx_max=4096, lengths=(4096,), positions=(0.5,),
        n_trials=1, progress=False,
    )
    assert res.cells[0].base_acc == 1.0
    assert res.cells[0].cand_acc == 0.0
    assert res.cells[0].degradation == 1.0
    # 100 * (1 - 1) = 0
    assert res.score == pytest.approx(0.0)


# --- run_plad -------------------------------------------------------------


def test_run_plad_basic(tmp_path, monkeypatch):
    pp = tmp_path / "p.jsonl"
    _write_prompts(pp, [
        {"id": "p1", "prompt": "Show me a big building."},
        {"id": "p2", "prompt": "Find the small house."},
    ])
    monkeypatch.setattr("refract.axes.plad.run_completion",
                        lambda **kw: ("answer", {}))
    monkeypatch.setattr("refract.axes.plad.tokenize_to_ids",
                        lambda model, text: [1, 2, 3])
    res = run_plad(
        model=Path("m"), prompts_path=pp,
        reference_kv=KVConfig(), candidate_kv=KVConfig(),
        progress=False,
    )
    # Score = 100 (no excess drift; both ref+cand return same tokens).
    assert res.score == pytest.approx(100.0)
    assert res.n_perturbations == 4


def test_run_plad_unknown_perturbation_raises(tmp_path):
    pp = tmp_path / "p.jsonl"
    _write_prompts(pp, [{"id": "p1", "prompt": "x"}])
    with pytest.raises(ValueError, match="Unknown perturbations"):
        run_plad(
            model=Path("m"), prompts_path=pp,
            reference_kv=KVConfig(), candidate_kv=KVConfig(),
            perturbations=("nonsense",), progress=False,
        )


def test_run_plad_empty_prompts_raises(tmp_path):
    pp = tmp_path / "p.jsonl"
    pp.write_text("")
    with pytest.raises(ValueError):
        run_plad(
            model=Path("m"), prompts_path=pp,
            reference_kv=KVConfig(), candidate_kv=KVConfig(),
            progress=False,
        )


def test_run_plad_zero_eligible_perturbations_raises(tmp_path, monkeypatch):
    pp = tmp_path / "p.jsonl"
    # Prompt with no eligible word for any perturbation:
    # - typo needs ≥4-char non-stopword (here every word is stopword/short)
    # - case needs a capitalized word (none)
    # - punct ALWAYS applies (toggles trailing punctuation), so we need to
    #   force perturbations=("typo",) here to trigger the empty path.
    _write_prompts(pp, [{"id": "p1", "prompt": "a is on the at to of"}])
    monkeypatch.setattr("refract.axes.plad.run_completion",
                        lambda **kw: ("x", {}))
    monkeypatch.setattr("refract.axes.plad.tokenize_to_ids",
                        lambda model, text: [1])
    with pytest.raises(RuntimeError, match="zero"):
        run_plad(
            model=Path("m"), prompts_path=pp,
            reference_kv=KVConfig(), candidate_kv=KVConfig(),
            perturbations=("typo",), progress=False,
        )
