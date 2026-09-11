"""Shared fixture builders for KV Fidelity tests.

Constructs synthetic GTM / KLD / R-NIAH / PLAD results that match the
real dataclass shapes — enough to exercise reporting + composite paths
without spinning up llama.cpp or MLX.
"""

from __future__ import annotations

import math
from typing import Optional

from kv_fidelity.axes.gtm import GTMResult
from kv_fidelity.axes.kld import KLDResult
from kv_fidelity.axes.plad import PLADPerPrompt, PLADResult
from kv_fidelity.axes.rniah import RNIAHCell, RNIAHResult
from kv_fidelity.axes.trajectory import TrajectoryResult


def make_comparable_report() -> dict:
    """Complete synthetic evidence for comparison tests, with no real runtime."""
    from kv_fidelity.comparison import EVIDENCE_SCHEMA
    from kv_fidelity.report import json_report
    from kv_fidelity.score import composite_score

    gtm = make_gtm()
    kld = make_kld()
    kld.metadata = {"kld_estimator": "llama_perplexity", "full_vocabulary": True}

    def identity(char):
        return {"sha256": char * 64, "size_bytes": 100}

    report = json_report(
        model="synthetic.gguf",
        reference_label="ctk=f16,ctv=f16",
        candidate_label="ctk=q8_0,ctv=q8_0",
        composite=composite_score(gtm.score, kld.score),
        gtm=gtm,
        kld=kld,
        extras={
            "comparison_evidence": {
                "schema": EVIDENCE_SCHEMA,
                "requested": {
                    "runtime": {
                        "backend": "llamacpp",
                        "settings": {"n_gpu_layers": 0},
                        "environment_overrides": {},
                    },
                    "scenario": {
                        "generation": {
                            "ctx": 512,
                            "n_predict": 128,
                            "chunks": 32,
                            "temperature": 0.0,
                        }
                    },
                    "trial_policy": {"seed": 42},
                },
                "resolved": {
                    "model": identity("a"),
                    "tokenizer": identity("a"),
                    "suite": identity("f"),
                    "runtime": {
                        "artifacts": {
                            name: identity("b")
                            for name in (
                                "llama-cli",
                                "llama-tokenize",
                                "llama-perplexity",
                            )
                        }
                    },
                    "inputs": {"prompts": identity("c"), "corpus": identity("d")},
                },
                "observed": {
                    "hardware": {
                        "host": {"hostname_sha256": "e" * 64, "cpu_count": 4},
                        "platform": {"system": "test", "machine": "test-cpu"},
                        "software": {"python_version": "test"},
                    }
                },
                "changed_inputs": [],
            }
        },
    )
    report["environment"] = {
        "backend": "llamacpp",
        "llama_cpp_commit": "synthetic-test-build",
    }
    report["timestamp"] = "2026-01-01T00:00:00"
    return report


def make_gtm(score: float = 95.0, n: int = 30) -> GTMResult:
    return GTMResult(
        score=score,
        full_match_rate=0.9 if score > 80 else 0.2,
        median_first_divergence=10 if score < 100 else None,
        mean_prefix_agreement_length=score,
        mean_cand_length=100.0,
        mean_ref_length=100.0,
        n_prompts=n,
        n_tokens_each=128,
        per_prompt=[],
        notes=[],
    )


def make_trajectory(score: float = 95.0, n: int = 30) -> TrajectoryResult:
    return TrajectoryResult(
        score=score,
        full_match_rate=0.9 if score > 80 else 0.2,
        median_first_divergence=10 if score < 100 else None,
        mean_prefix_agreement_length=score,
        mean_cand_length=100.0,
        mean_ref_length=100.0,
        n_prompts=n,
        n_tokens_each=128,
        per_prompt=[],
        notes=[],
    )


def make_kld(score: float = 99.0, mean_kld: Optional[float] = None) -> KLDResult:
    if mean_kld is None:
        mean_kld = -math.log(max(score, 1e-9) / 100.0)
    return KLDResult(
        score=score,
        mean_kld=mean_kld,
        ppl=8.5,
        rms_dp_pct=1.2,
        same_topp_pct=99.5,
        base_path="/tmp/base.bin",
        chunks=32,
        ctx=512,
        is_self_reference=False,
        corpus={
            "path": "wiki.test.raw",
            "size_bytes": 1234,
            "sha256_head": "abc" * 21 + "ab",
            "sha256_head_bytes": 1024 * 1024,
        },
    )


def make_rniah_high_base(score: float = 100.0) -> RNIAHResult:
    """R-NIAH with base_acc that engages the task (avg >= 0.2)."""
    cells = [
        RNIAHCell(
            length=4096,
            position=0.10,
            n_trials=1,
            base_acc=1.0,
            cand_acc=1.0,
            degradation=0.0,
            base_hits=1,
            cand_hits=1,
        ),
        RNIAHCell(
            length=4096,
            position=0.50,
            n_trials=1,
            base_acc=1.0,
            cand_acc=1.0,
            degradation=0.0,
            base_hits=1,
            cand_hits=1,
        ),
        RNIAHCell(
            length=8192,
            position=0.10,
            n_trials=1,
            base_acc=1.0,
            cand_acc=1.0,
            degradation=0.0,
            base_hits=1,
            cand_hits=1,
        ),
    ]
    return RNIAHResult(
        score=score,
        n_cells=len(cells),
        cells=cells,
        skipped_cells=[],
        needle="Note: APRICOT-7-BLUE rare paint.",
        needle_keyword="APRICOT-7-BLUE",
        notes=[],
    )


def make_rniah_low_base() -> RNIAHResult:
    """R-NIAH where base_acc averages below 0.2 (low confidence)."""
    cells = [
        RNIAHCell(
            length=4096,
            position=0.10,
            n_trials=1,
            base_acc=0.0,
            cand_acc=0.0,
            degradation=0.0,
            base_hits=0,
            cand_hits=0,
        ),
        RNIAHCell(
            length=4096,
            position=0.50,
            n_trials=1,
            base_acc=0.0,
            cand_acc=0.0,
            degradation=0.0,
            base_hits=0,
            cand_hits=0,
        ),
    ]
    return RNIAHResult(
        score=100.0,
        n_cells=len(cells),
        cells=cells,
        skipped_cells=[(16384, 0.10)],
        needle="Note: X.",
        needle_keyword="X",
        notes=[],
    )


def make_plad(score: float = 88.0, with_nan: bool = False) -> PLADResult:
    per_pert = {
        "typo": 90.0,
        "case": 88.0,
        "punct": 86.0,
        "paraphrase": float("nan") if with_nan else 88.0,
    }
    notes = ["1 (prompt, perturbation) pair was skipped"] if with_nan else []
    return PLADResult(
        score=score,
        per_perturbation_score=per_pert,
        per_prompt=[
            PLADPerPrompt(
                prompt_id="p1",
                perturbation="typo",
                perturbed_prompt="hi",
                ref_drift=0.05,
                cand_drift=0.06,
                excess_drift=0.01,
                plad_pp=95.0,
            ),
        ],
        n_prompts=30,
        n_perturbations=4,
        notes=notes,
    )
