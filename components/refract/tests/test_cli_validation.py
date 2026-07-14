"""Parser-level validation for score inputs and model identifiers."""

from __future__ import annotations

import argparse

import pytest

import refract.cli as cli


def _score_argv(*extra: str) -> list[str]:
    return [
        "score",
        "--model",
        "Qwen/Qwen3",
        "--candidate",
        "ctk=q8_0,ctv=q8_0",
        *extra,
    ]


def _capture_score_args(monkeypatch, *extra: str) -> argparse.Namespace:
    captured: dict[str, argparse.Namespace] = {}

    def fake_run_score(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_score", fake_run_score)
    assert cli.main(_score_argv(*extra)) == 0
    return captured["args"]


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--chunks", "0"),
        ("--chunks", "-1"),
        ("--ctx", "0"),
        ("--ctx", "-1"),
        ("--n-predict", "0"),
        ("--n-predict", "-1"),
        ("--rniah-up-to", "0"),
        ("--rniah-up-to", "-1"),
        ("--rniah-ctx-max", "0"),
        ("--rniah-ctx-max", "-1"),
        ("--rniah-trials", "0"),
        ("--rniah-trials", "-1"),
    ],
)
def test_score_parser_rejects_non_positive_counts(monkeypatch, flag, value):
    monkeypatch.setattr(
        cli,
        "_run_score",
        lambda args: pytest.fail("invalid input reached score execution"),
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(_score_argv(flag, value))
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "value",
    ["", ",", "4096,,8192", "not-an-int", "0", "4096,-1"],
)
def test_score_parser_rejects_empty_malformed_or_non_positive_lengths(
    monkeypatch, value
):
    monkeypatch.setattr(
        cli,
        "_run_score",
        lambda args: pytest.fail("invalid lengths reached score execution"),
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(_score_argv("--rniah-lengths", value))
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "value",
    [
        "",
        ",",
        "0.1,,0.9",
        "not-a-float",
        "nan",
        "inf",
        "-0.1",
        "1.1",
    ],
)
def test_score_parser_rejects_invalid_rniah_positions(monkeypatch, value):
    monkeypatch.setattr(
        cli,
        "_run_score",
        lambda args: pytest.fail("invalid positions reached score execution"),
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(_score_argv("--rniah-positions", value))
    assert exc.value.code == 2


def test_score_parser_converts_valid_rniah_lists(monkeypatch):
    args = _capture_score_args(
        monkeypatch,
        "--rniah-lengths",
        "4096, 8192",
        "--rniah-positions",
        "0, 0.5, 1",
    )
    assert args.rniah_lengths == (4096, 8192)
    assert args.rniah_positions == (0.0, 0.5, 1.0)


def test_score_parser_preserves_hugging_face_model_id(monkeypatch):
    args = _capture_score_args(monkeypatch)
    assert args.model == "Qwen/Qwen3"
    assert isinstance(args.model, str)


def test_score_parser_preserves_local_model_path(monkeypatch, tmp_path):
    model = str(tmp_path / "MODEL.GGUF")
    captured: dict[str, argparse.Namespace] = {}

    def fake_run_score(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_score", fake_run_score)
    assert (
        cli.main(
            [
                "score",
                "--model",
                model,
                "--candidate",
                "ctk=q8_0,ctv=q8_0",
            ]
        )
        == 0
    )
    assert captured["args"].model == model


def test_selftest_parser_preserves_hugging_face_model_id(monkeypatch):
    captured: dict[str, argparse.Namespace] = {}

    def fake_run_selftest(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_selftest", fake_run_selftest)
    assert cli.main(["selftest", "--model", "Qwen/Qwen3"]) == 0
    assert captured["args"].model == "Qwen/Qwen3"


def test_repeatability_parser_preserves_hugging_face_model_id(monkeypatch):
    captured: dict[str, argparse.Namespace] = {}

    def fake_run_repeatability(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_repeatability", fake_run_repeatability)
    assert (
        cli.main(
            [
                "repeatability",
                "--model",
                "Qwen/Qwen3",
                "--candidate",
                "ctk=q8_0,ctv=q8_0",
            ]
        )
        == 0
    )
    assert captured["args"].model == "Qwen/Qwen3"
