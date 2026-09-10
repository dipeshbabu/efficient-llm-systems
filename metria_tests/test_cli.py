from __future__ import annotations

import io
import json
from pathlib import Path

from metria import (
    ComparisonPlan,
    RunSpec,
    StudyRecipe,
    StudySpec,
    dump_study_recipe,
    study_recipe_digest,
)
from metria.cli import main
from metria.recipes import STUDY_RECIPE_SCHEMA


def _recipe() -> StudyRecipe:
    measurement = "kv_fidelity.decode_time_trajectory"
    return StudyRecipe(
        study=StudySpec(
            name="cli-study",
            runs=(
                RunSpec(
                    model={"id": "example/model"},
                    runtime={"name": "vllm"},
                    scenario={"context": 512, "max_tokens": 16},
                    measurements=(measurement,),
                ),
            ),
            comparison=ComparisonPlan(
                analyses=("kv_fidelity.trajectory_match",),
            ),
        ),
        measurement_configs={
            measurement: {"prompts": ({"id": "p1", "prompt": "private cli prompt"},)}
        },
        environment={"hardware_class": "test-gpu"},
    )


def _recipe_file(tmp_path: Path) -> tuple[Path, StudyRecipe]:
    recipe = _recipe()
    path = tmp_path / "study.json"
    dump_study_recipe(path, recipe)
    return path, recipe


def test_cli_version() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = main(["--version"], stdout=stdout, stderr=stderr)

    assert status == 0
    from metria import __version__

    assert stdout.getvalue().strip() == f"metria {__version__}"
    assert stderr.getvalue() == ""


def test_recipe_validate_human_output(tmp_path: Path) -> None:
    path, recipe = _recipe_file(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = main(
        ["recipe", "validate", str(path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0
    assert STUDY_RECIPE_SCHEMA in stdout.getvalue()
    assert recipe.study.name in stdout.getvalue()
    assert study_recipe_digest(recipe) in stdout.getvalue()
    assert "private cli prompt" not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_recipe_validate_json_output_is_structural_only(tmp_path: Path) -> None:
    path, recipe = _recipe_file(tmp_path)
    stdout = io.StringIO()

    status = main(
        ["recipe", "validate", str(path), "--json"],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    payload = json.loads(stdout.getvalue())

    assert status == 0
    assert payload["valid"] is True
    assert payload["schema"] == STUDY_RECIPE_SCHEMA
    assert payload["study"] == "cli-study"
    assert payload["runs"] == 1
    assert payload["runtimes"] == ["vllm"]
    assert payload["measurements"] == ["kv_fidelity.decode_time_trajectory"]
    assert payload["analyses"] == ["kv_fidelity.trajectory_match"]
    assert payload["digest"] == study_recipe_digest(recipe)
    assert "private cli prompt" not in stdout.getvalue()


def test_recipe_digest_outputs_only_digest(tmp_path: Path) -> None:
    path, recipe = _recipe_file(tmp_path)
    stdout = io.StringIO()

    status = main(
        ["recipe", "digest", str(path)],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert status == 0
    assert stdout.getvalue() == study_recipe_digest(recipe) + "\n"


def test_recipe_normalize_to_stdout_can_emit_sensitive_recipe_input(
    tmp_path: Path,
) -> None:
    path, _ = _recipe_file(tmp_path)
    stdout = io.StringIO()

    status = main(
        ["recipe", "normalize", str(path)],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert status == 0
    payload = json.loads(stdout.getvalue())
    assert payload["schema"] == STUDY_RECIPE_SCHEMA
    assert "private cli prompt" in stdout.getvalue()


def test_recipe_normalize_can_write_file(tmp_path: Path) -> None:
    path, _ = _recipe_file(tmp_path)
    output = tmp_path / "normalized.json"
    stdout = io.StringIO()

    status = main(
        ["recipe", "normalize", str(path), "--output", str(output)],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert status == 0
    assert stdout.getvalue() == ""
    assert (
        json.loads(output.read_text(encoding="utf-8"))["schema"] == STUDY_RECIPE_SCHEMA
    )


def test_invalid_recipe_returns_error_without_traceback(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = main(
        ["recipe", "validate", str(path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 2
    assert stdout.getvalue() == ""
    assert "metria: error:" in stderr.getvalue()
    assert str(path) in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()


def test_missing_recipe_file_returns_concise_error(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    stderr = io.StringIO()

    status = main(
        ["recipe", "digest", str(path)],
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert status == 2
    assert "metria: error:" in stderr.getvalue()
    assert str(path) in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()


def test_cli_without_command_returns_help_on_stderr() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = main([], stdout=stdout, stderr=stderr)

    assert status == 2
    assert stdout.getvalue() == ""
    assert "usage: metria" in stderr.getvalue()
