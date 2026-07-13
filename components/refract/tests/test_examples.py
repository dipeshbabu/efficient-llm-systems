"""Integrity checks for the shipped JSON and HTML report examples."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).parents[1] / "src" / "refract" / "examples"
_JSON_EXAMPLES = sorted(_EXAMPLES_DIR.glob("*.json"))
_HTML_EXAMPLES = sorted(_EXAMPLES_DIR.glob("*.html"))


def _strict_json_loads(value: str):
    def reject_nonstandard_constant(constant: str):
        raise ValueError(f"non-standard JSON constant: {constant}")

    return json.loads(value, parse_constant=reject_nonstandard_constant)


def test_example_json_and_html_files_have_matching_stems():
    json_stems = {path.stem for path in _JSON_EXAMPLES}
    html_stems = {path.stem for path in _HTML_EXAMPLES}

    assert json_stems
    assert html_stems == json_stems


@pytest.mark.parametrize("json_path", _JSON_EXAMPLES, ids=lambda path: path.stem)
def test_example_json_is_strict(json_path: Path):
    report = _strict_json_loads(json_path.read_text(encoding="utf-8"))

    assert isinstance(report, dict)


@pytest.mark.parametrize("html_path", _HTML_EXAMPLES, ids=lambda path: path.stem)
def test_example_html_embeds_strict_json(html_path: Path):
    rendered = html_path.read_text(encoding="utf-8")
    match = re.search(
        r'<details class="json-toggle">.*?<pre>(.*?)</pre>', rendered, re.DOTALL
    )

    assert match is not None
    embedded = _strict_json_loads(html.unescape(match.group(1)))
    assert isinstance(embedded, dict)
