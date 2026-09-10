"""Versioned JSON recipes for portable Metria study intent."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ._freeze import freeze_mapping
from .models import ComparisonPlan, RunSpec, StudySpec, TreatmentSpec, TreatmentType

STUDY_RECIPE_SCHEMA = "metria.study_recipe.v1"


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys before a decoder can overwrite earlier values."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class StudyRecipe:
    """Serializable study intent plus execution inputs outside ``StudySpec``.

    ``measurement_configs`` contains method-specific requested inputs such as a
    prompt set. ``environment`` contains shared environment input passed to the
    current study executor. Both are requested configuration, not observed run
    evidence.
    """

    study: StudySpec
    measurement_configs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate recipe routing and detach mutable caller-owned state."""

        requested_measurements = {
            measurement for run in self.study.runs for measurement in run.measurements
        }
        configs: dict[str, Mapping[str, Any]] = {}
        for name, config in self.measurement_configs.items():
            if not isinstance(name, str) or not name:
                raise TypeError("measurement config names must be non-empty strings")
            if not isinstance(config, Mapping):
                raise TypeError(f"measurement config {name!r} must be a mapping")
            configs[name] = config
        unknown = sorted(set(configs) - requested_measurements)
        if unknown:
            raise ValueError(
                "measurement configs are not requested by the study: "
                + ", ".join(unknown)
            )
        object.__setattr__(self, "measurement_configs", freeze_mapping(configs))
        object.__setattr__(self, "environment", freeze_mapping(self.environment))


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Require a string-keyed mapping at a recipe schema boundary."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
    return value


def _keys(
    value: Mapping[str, Any],
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    """Reject missing and unknown fields instead of silently changing intent."""

    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")
    unknown = sorted(set(value) - required - optional)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(unknown)}")


def _string_sequence(value: Any, *, name: str) -> tuple[str, ...]:
    """Parse a JSON array of strings into an immutable tuple."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an array of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise TypeError(f"{name} must contain only non-empty strings")
    return result


def _string_mapping(value: Any, *, name: str) -> dict[str, str]:
    """Parse a JSON object whose keys and values are non-empty strings."""

    mapping = _mapping(value, name=name)
    result: dict[str, str] = {}
    for key, item in mapping.items():
        if not key.strip():
            raise TypeError(f"{name} keys must be non-empty strings")
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"{name} values must be non-empty strings")
        result[key] = item
    return result


def _treatment_from_data(value: Any, *, index: int) -> TreatmentSpec:
    """Parse one versioned treatment object."""

    mapping = _mapping(value, name=f"treatment[{index}]")
    _keys(
        mapping,
        name=f"treatment[{index}]",
        required=frozenset({"name", "kind"}),
        optional=frozenset({"config"}),
    )
    name = mapping["name"]
    kind = mapping["kind"]
    if not isinstance(name, str) or not name:
        raise TypeError(f"treatment[{index}].name must be a non-empty string")
    if not isinstance(kind, str):
        raise TypeError(f"treatment[{index}].kind must be a string")
    try:
        treatment_type = TreatmentType(kind)
    except ValueError as exc:
        supported = ", ".join(item.value for item in TreatmentType)
        raise ValueError(
            f"treatment[{index}].kind must be one of: {supported}"
        ) from exc
    config = _mapping(mapping.get("config", {}), name=f"treatment[{index}].config")
    return TreatmentSpec(name=name, kind=treatment_type, config=config)


def run_spec_from_data(value: Any, *, index: int = 0) -> RunSpec:
    """Parse one requested run without resolving runtime-specific state."""

    mapping = _mapping(value, name=f"run[{index}]")
    _keys(
        mapping,
        name=f"run[{index}]",
        required=frozenset({"model", "runtime", "scenario", "measurements"}),
        optional=frozenset({"treatments", "trial_policy", "environment_selector"}),
    )
    raw_treatments = mapping.get("treatments", ())
    if isinstance(raw_treatments, (str, bytes)) or not isinstance(
        raw_treatments, Sequence
    ):
        raise TypeError(f"run[{index}].treatments must be an array")
    treatments = tuple(
        _treatment_from_data(item, index=treatment_index)
        for treatment_index, item in enumerate(raw_treatments)
    )
    return RunSpec(
        model=_mapping(mapping["model"], name=f"run[{index}].model"),
        runtime=_mapping(mapping["runtime"], name=f"run[{index}].runtime"),
        scenario=_mapping(mapping["scenario"], name=f"run[{index}].scenario"),
        measurements=_string_sequence(
            mapping["measurements"],
            name=f"run[{index}].measurements",
        ),
        treatments=treatments,
        trial_policy=_mapping(
            mapping.get("trial_policy", {}),
            name=f"run[{index}].trial_policy",
        ),
        environment_selector=_mapping(
            mapping.get("environment_selector", {}),
            name=f"run[{index}].environment_selector",
        ),
    )


def _comparison_from_data(value: Any) -> ComparisonPlan:
    """Parse study comparison roles, waivers, and ordered pairwise analyses."""

    mapping = _mapping(value, name="study.comparison")
    _keys(
        mapping,
        name="study.comparison",
        required=frozenset(),
        optional=frozenset({"vary", "control", "block_by", "waivers", "analyses"}),
    )
    return ComparisonPlan(
        vary=frozenset(
            _string_sequence(mapping.get("vary", ()), name="study.comparison.vary")
        ),
        control=frozenset(
            _string_sequence(
                mapping.get("control", ()),
                name="study.comparison.control",
            )
        ),
        block_by=frozenset(
            _string_sequence(
                mapping.get("block_by", ()),
                name="study.comparison.block_by",
            )
        ),
        waivers=_string_mapping(
            mapping.get("waivers", {}),
            name="study.comparison.waivers",
        ),
        analyses=_string_sequence(
            mapping.get("analyses", ()),
            name="study.comparison.analyses",
        ),
    )


def _study_from_data(value: Any) -> StudySpec:
    """Parse a complete study specification from JSON-compatible data."""

    mapping = _mapping(value, name="study")
    _keys(
        mapping,
        name="study",
        required=frozenset({"name", "runs", "comparison"}),
        optional=frozenset({"constants", "metadata"}),
    )
    name = mapping["name"]
    if not isinstance(name, str):
        raise TypeError("study.name must be a string")
    raw_runs = mapping["runs"]
    if isinstance(raw_runs, (str, bytes)) or not isinstance(raw_runs, Sequence):
        raise TypeError("study.runs must be an array")
    return StudySpec(
        name=name,
        runs=tuple(
            run_spec_from_data(run, index=index) for index, run in enumerate(raw_runs)
        ),
        comparison=_comparison_from_data(mapping["comparison"]),
        constants=_mapping(mapping.get("constants", {}), name="study.constants"),
        metadata=_mapping(mapping.get("metadata", {}), name="study.metadata"),
    )


def study_recipe_from_data(value: Any) -> StudyRecipe:
    """Validate and parse a ``metria.study_recipe.v1`` mapping."""

    mapping = _mapping(value, name="recipe")
    _keys(
        mapping,
        name="recipe",
        required=frozenset({"schema", "study"}),
        optional=frozenset({"measurement_configs", "environment"}),
    )
    schema = mapping["schema"]
    if schema != STUDY_RECIPE_SCHEMA:
        raise ValueError(
            f"unsupported recipe schema {schema!r}; expected {STUDY_RECIPE_SCHEMA!r}"
        )
    return StudyRecipe(
        study=_study_from_data(mapping["study"]),
        measurement_configs=_mapping(
            mapping.get("measurement_configs", {}),
            name="recipe.measurement_configs",
        ),
        environment=_mapping(
            mapping.get("environment", {}),
            name="recipe.environment",
        ),
    )


def _json_value(value: Any, *, path: str) -> Any:
    """Convert immutable Metria request values into strict JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Enum):
        return _json_value(value.value, path=path)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains non-JSON value "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def run_spec_to_data(run: RunSpec, *, path: str = "run") -> dict[str, Any]:
    """Serialize one requested run using the canonical recipe RunSpec shape."""

    return {
        "model": _json_value(run.model, path=f"{path}.model"),
        "runtime": _json_value(run.runtime, path=f"{path}.runtime"),
        "scenario": _json_value(run.scenario, path=f"{path}.scenario"),
        "measurements": list(run.measurements),
        "treatments": [
            {
                "name": treatment.name,
                "kind": treatment.kind.value,
                "config": _json_value(
                    treatment.config,
                    path=f"{path}.treatment.config",
                ),
            }
            for treatment in run.treatments
        ],
        "trial_policy": _json_value(
            run.trial_policy,
            path=f"{path}.trial_policy",
        ),
        "environment_selector": _json_value(
            run.environment_selector,
            path=f"{path}.environment_selector",
        ),
    }


def study_recipe_to_data(recipe: StudyRecipe) -> dict[str, Any]:
    """Return the canonical schema shape as JSON-compatible Python values."""

    study = recipe.study
    runs = [
        run_spec_to_data(run, path=f"study.runs[{index}]")
        for index, run in enumerate(study.runs)
    ]
    comparison: dict[str, Any] = {
        "vary": sorted(study.comparison.vary),
        "control": sorted(study.comparison.control),
        "block_by": sorted(study.comparison.block_by),
        "analyses": list(study.comparison.analyses),
    }
    if study.comparison.waivers:
        comparison["waivers"] = {
            dimension: study.comparison.waivers[dimension]
            for dimension in sorted(study.comparison.waivers)
        }
    return {
        "schema": STUDY_RECIPE_SCHEMA,
        "study": {
            "name": study.name,
            "runs": runs,
            "comparison": comparison,
            "constants": _json_value(study.constants, path="study.constants"),
            "metadata": _json_value(study.metadata, path="study.metadata"),
        },
        "measurement_configs": _json_value(
            recipe.measurement_configs,
            path="measurement_configs",
        ),
        "environment": _json_value(
            recipe.environment,
            path="environment",
        ),
    }


def study_recipe_to_json(recipe: StudyRecipe, *, indent: int | None = 2) -> str:
    """Serialize a recipe deterministically while remaining human-readable."""

    return json.dumps(
        study_recipe_to_data(recipe),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
    )


def study_recipe_digest(recipe: StudyRecipe) -> str:
    """Return SHA-256 of the canonical compact JSON recipe representation."""

    canonical = json.dumps(
        study_recipe_to_data(recipe),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_study_recipe(path: str | Path) -> StudyRecipe:
    """Load and validate a UTF-8 JSON recipe from disk."""

    recipe_path = Path(path)
    try:
        raw = json.loads(
            recipe_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON recipe {recipe_path}: {exc.msg}") from exc
    return study_recipe_from_data(raw)


def dump_study_recipe(path: str | Path, recipe: StudyRecipe) -> None:
    """Write a deterministic UTF-8 JSON recipe with a trailing newline."""

    Path(path).write_text(
        study_recipe_to_json(recipe) + "\n",
        encoding="utf-8",
    )
