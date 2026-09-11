from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from metria import (
    ComparisonPlan,
    PolicyCriterion,
    RunSpec,
    StudyRecipe,
    StudySpec,
    TreatmentSpec,
    TreatmentType,
    VerificationPolicy,
    dump_study_recipe,
    load_study_recipe,
    study_recipe_digest,
    study_recipe_from_data,
    study_recipe_to_data,
    study_recipe_to_json,
)
from metria.recipes import STUDY_RECIPE_SCHEMA


def _recipe(
    *,
    environment: dict[str, Any] | None = None,
    waivers: dict[str, str] | None = None,
) -> StudyRecipe:
    measurement = "kv_fidelity.decode_time_trajectory"
    study = StudySpec(
        name="recipe-study",
        runs=(
            RunSpec(
                model={"id": "example/model", "revision": "abc123"},
                runtime={"name": "vllm", "dtype": "bfloat16"},
                scenario={"context": 1024, "max_tokens": 32},
                measurements=(measurement,),
                treatments=(
                    TreatmentSpec(
                        name="vllm.kv_cache",
                        kind=TreatmentType.RUNTIME_FEATURE,
                        config={"dtype": "fp8"},
                    ),
                ),
                trial_policy={"warmup": 1, "repetitions": 3},
                environment_selector={"hardware_class": "h100"},
            ),
        ),
        comparison=ComparisonPlan(
            vary=frozenset({"runtime", "treatments"}),
            control=frozenset({"model", "scenario", "measurements"}),
            block_by=frozenset({"observed.hardware_class"}),
            waivers=waivers or {},
            analyses=("kv_fidelity.trajectory_match",),
        ),
        constants={"suite": "trajectory-v1"},
        metadata={"owner": "example"},
    )
    return StudyRecipe(
        study=study,
        measurement_configs={
            measurement: {
                "prompts": (
                    {"id": "p1", "prompt": "private prompt one"},
                    {"id": "p2", "prompt": "private prompt two"},
                ),
                "generation": {"temperature": 0.0},
            }
        },
        environment=environment or {"hardware_class": "h100"},
    )


def test_recipe_round_trip_preserves_requested_study_semantics() -> None:
    recipe = _recipe(
        waivers={
            "observed.environment.zone": "cross-zone qualification",
        }
    )

    data = study_recipe_to_data(recipe)
    parsed = study_recipe_from_data(data)

    assert parsed == recipe
    assert data["schema"] == STUDY_RECIPE_SCHEMA
    assert data["study"]["comparison"]["vary"] == ["runtime", "treatments"]
    assert data["study"]["comparison"]["analyses"] == ["kv_fidelity.trajectory_match"]
    assert data["study"]["comparison"]["waivers"] == {
        "observed.environment.zone": "cross-zone qualification",
    }
    assert data["study"]["runs"][0]["treatments"][0]["kind"] == "runtime_feature"


def test_recipe_without_waivers_keeps_existing_canonical_shape() -> None:
    data = study_recipe_to_data(_recipe())

    assert "waivers" not in data["study"]["comparison"]


def test_verification_policy_round_trips_and_changes_recipe_identity():
    original = _recipe()
    assert "policy" not in study_recipe_to_data(original)
    policy = VerificationPolicy(
        (PolicyCriterion("behavior.trajectory_agreement", "0.3.4", minimum=0.98),)
    )
    recipe = replace(original, policy=policy)
    assert study_recipe_from_data(study_recipe_to_data(recipe)) == recipe
    assert study_recipe_digest(recipe) != study_recipe_digest(original)
    with pytest.raises(ValueError, match="undeclared analyses"):
        replace(recipe, study=replace(recipe.study, comparison=ComparisonPlan()))


def test_recipe_serializes_waivers_deterministically() -> None:
    first = _recipe(
        waivers={
            "observed.environment.zone": "cross-zone qualification",
            "resolved.runtime.version": "mixed runtime qualification",
        }
    )
    second = _recipe(
        waivers={
            "resolved.runtime.version": "mixed runtime qualification",
            "observed.environment.zone": "cross-zone qualification",
        }
    )

    assert study_recipe_to_json(first) == study_recipe_to_json(second)
    assert study_recipe_digest(first) == study_recipe_digest(second)


def test_recipe_rejects_invalid_waiver_mapping() -> None:
    data = study_recipe_to_data(_recipe())
    data["study"]["comparison"]["waivers"] = {
        "observed.environment.zone": "",
    }

    with pytest.raises(TypeError, match="values must be non-empty strings"):
        study_recipe_from_data(data)

    data = study_recipe_to_data(_recipe())
    data["study"]["comparison"]["waivers"] = {
        "": "reason",
    }

    with pytest.raises(TypeError, match="keys must be non-empty strings"):
        study_recipe_from_data(data)


def test_recipe_json_is_deterministic_and_digest_ignores_mapping_insertion_order() -> (
    None
):
    first = _recipe(environment={"hardware_class": "h100", "zone": "a"})
    second = _recipe(environment={"zone": "a", "hardware_class": "h100"})

    assert study_recipe_digest(first) == study_recipe_digest(second)
    assert study_recipe_to_json(first) == study_recipe_to_json(second)
    assert len(study_recipe_digest(first)) == 64


def test_dump_and_load_recipe_round_trip(tmp_path: Path) -> None:
    recipe = _recipe()
    path = tmp_path / "study.json"

    dump_study_recipe(path, recipe)
    loaded = load_study_recipe(path)

    assert loaded == recipe
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text)["schema"] == STUDY_RECIPE_SCHEMA


def test_recipe_rejects_unknown_schema_and_unknown_fields() -> None:
    data = study_recipe_to_data(_recipe())
    data["schema"] = "metria.study_recipe.v999"
    with pytest.raises(ValueError, match="unsupported recipe schema"):
        study_recipe_from_data(data)

    data = study_recipe_to_data(_recipe())
    data["mystery"] = True
    with pytest.raises(ValueError, match="recipe contains unknown fields: mystery"):
        study_recipe_from_data(data)

    data = study_recipe_to_data(_recipe())
    data["study"]["runs"][0]["mystery"] = True
    with pytest.raises(ValueError, match=r"run\[0\] contains unknown fields: mystery"):
        study_recipe_from_data(data)


def test_recipe_rejects_invalid_treatment_kind() -> None:
    data = study_recipe_to_data(_recipe())
    data["study"]["runs"][0]["treatments"][0]["kind"] = "not_a_treatment"

    with pytest.raises(ValueError, match="must be one of"):
        study_recipe_from_data(data)


def test_recipe_rejects_measurement_config_not_requested_by_study() -> None:
    recipe = _recipe()

    with pytest.raises(
        ValueError, match="not requested by the study: typo.measurement"
    ):
        StudyRecipe(
            study=recipe.study,
            measurement_configs={"typo.measurement": {}},
        )


def test_recipe_requires_each_measurement_config_to_be_a_mapping() -> None:
    recipe = _recipe()
    measurement = recipe.study.runs[0].measurements[0]

    with pytest.raises(TypeError, match="must be a mapping"):
        StudyRecipe(
            study=recipe.study,
            measurement_configs={measurement: ["bad"]},  # type: ignore[dict-item]
        )


def test_recipe_rejects_non_json_programmatic_values() -> None:
    recipe = _recipe()
    bad_study = StudySpec(
        name=recipe.study.name,
        runs=(
            RunSpec(
                model={"id": "example/model", "bad": {1, 2}},
                runtime={"name": "vllm"},
                scenario={"context": 512},
                measurements=("kv_fidelity.decode_time_trajectory",),
            ),
        ),
        comparison=ComparisonPlan(),
    )
    bad_recipe = StudyRecipe(study=bad_study)

    with pytest.raises(TypeError, match="contains non-JSON value"):
        study_recipe_to_data(bad_recipe)


def test_recipe_rejects_non_finite_floats() -> None:
    recipe = _recipe(environment={"temperature": float("nan")})

    with pytest.raises(ValueError, match="non-finite float"):
        study_recipe_to_json(recipe)


def test_invalid_json_file_reports_recipe_path(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON recipe") as exc_info:
        load_study_recipe(path)

    assert str(path) in str(exc_info.value)


def test_serialized_recipe_contains_requested_prompt_text() -> None:
    recipe = _recipe()

    text = study_recipe_to_json(recipe)

    assert "private prompt one" in text
    assert "private prompt two" in text


@pytest.mark.parametrize(
    ("original", "duplicate"),
    [
        ('"schema":', '"schema": "invalid.schema", "schema":'),
        ('"context":', '"context": 1, "context":'),
        ('"temperature":', '"temperature": 1.0, "temperature":'),
    ],
)
def test_recipe_loader_rejects_duplicate_object_keys(
    tmp_path: Path, original: str, duplicate: str
) -> None:
    path = tmp_path / "ambiguous.json"
    text = study_recipe_to_json(_recipe()).replace(original, duplicate, 1)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_study_recipe(path)
