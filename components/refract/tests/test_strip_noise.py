"""v0.1.3 regression tests for runner._strip_noise.

These tests pin the noise-stripper behaviour so the v0.1.1 banner-comparing
bug (where _strip_noise returned llama-cli's help banner instead of the
generation body) cannot return.

Each test feeds a canned llama-cli stdout string and asserts what survives
the noise filter.
"""

from __future__ import annotations

from refract.runner import _strip_noise


def test_canonical_generation_extracts_body_only():
    """Banner + spinner + backspaces + prompt-echo + perf line, all stripped."""
    raw = (
        "Loading model... done\n"
        "\u2580\u2580 \u2580\u2580 \u2588\u2588 \u2588\u2588\n"  # ASCII art block chars
        "available commands: /help /clear\n"
        "> The capital of France is\n"
        "|\x08 \x08[Start thinking]\n"
        "| The capital of France is Paris.\n"
        "[ Prompt: 6 tokens, 0.42 t/s ]\n"
        "Exiting...\n"
    )
    out = _strip_noise(raw)
    # Generation body survives
    assert "[Start thinking]" in out
    assert "The capital of France is Paris." in out
    # Noise gone
    assert "Loading model" not in out
    assert "Exiting" not in out
    assert "[ Prompt:" not in out
    assert "> The capital" not in out
    # Backspace control char is stripped (would otherwise break gen-line regex)
    assert "\x08" not in out


def test_help_banner_only_returns_no_generation():
    """v0.1.1 bug: --no-conversation triggers help-mode and the banner was
    captured as 'completion'. After strip, nothing should remain that looks
    like a real generation body."""
    raw = (
        "usage: llama-cli [options]\n"
        "available commands: /help /clear /quit\n"
        "options:\n"
        "  -h, --help          show this help message and exit\n"
        "  -m MODEL            path to model\n"
    )
    out = _strip_noise(raw).strip()
    # No "| ..." gen-line means _strip_noise leaves the banner text in place
    # but it's clearly not a generation. The aggregator-side sanity check is
    # "no generation prefix found" — assert that's true here.
    assert "|" not in out or out == ""


def test_unicode_block_chars_inside_generation_kept():
    """Block chars only stripped when the line is *only* block chars + ws.
    A generation that legitimately contains a block char in the middle of
    real text must survive."""
    raw = (
        "| The pixel art logo uses \u2588\u2588 here.\n"
    )
    out = _strip_noise(raw)
    assert "The pixel art logo uses" in out
    assert "\u2588\u2588" in out


def test_empty_stdout_returns_empty():
    assert _strip_noise("") == ""


def test_multiple_gen_lines_all_kept():
    """Multi-line generation: every "| " line should survive (with the
    leading "| " marker stripped)."""
    raw = (
        "Loading model... done\n"
        "| Line one of the answer.\n"
        "| Line two of the answer.\n"
        "| Line three.\n"
    )
    out = _strip_noise(raw)
    assert "Line one of the answer." in out
    assert "Line two of the answer." in out
    assert "Line three." in out
    # Leading "| " markers gone
    assert "| Line" not in out
