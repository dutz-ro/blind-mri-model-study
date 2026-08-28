"""Look up clinical schemas by name instead of hard-coding one file.

The pipeline previously loaded exactly one schema through
``load_knee_meniscus_schema()``, whose filename was a literal. Every prompt,
plan, validator and container inherited that single choice, so a second body
region could not be expressed at all.

Schemas are registered by their ``schema_version``, which is also the value a
model must echo, so the identifier that travels through plans, manifests and
provenance is the same one used to resolve the file.
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator

from .errors import MRIStudyError

# Registered schema version -> packaged filename.
SCHEMA_FILES = {
    "knee-meniscus-finding-v2": "knee-meniscus-finding-v2.schema.json",
    "mri-finding-core-v1": "mri-finding-core-v1.schema.json",
}

DEFAULT_SCHEMA_VERSION = "knee-meniscus-finding-v2"


def available_schemas() -> list[str]:
    return sorted(SCHEMA_FILES)


@cache
def _load(schema_version: str) -> dict[str, Any]:
    filename = SCHEMA_FILES.get(schema_version)
    if filename is None:
        raise MRIStudyError(
            f"Unknown clinical schema: {schema_version}. "
            f"Registered: {', '.join(available_schemas())}"
        )
    path = files("mristudy.schemas").joinpath(filename)
    with path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    declared = (schema.get("properties") or {}).get("schema_version", {}).get("const")
    if declared != schema_version:
        raise MRIStudyError(
            f"Schema file for {schema_version} declares schema_version {declared!r}."
        )
    return schema


def load_schema(schema_version: str = DEFAULT_SCHEMA_VERSION) -> dict[str, Any]:
    """Return a registered schema. The result is cached and must not be mutated."""
    return json.loads(json.dumps(_load(schema_version)))


def validate_result(
    result: dict[str, Any],
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> list[dict[str, str]]:
    validator = Draft202012Validator(_load(schema_version))
    return [
        {
            "path": ".".join(str(part) for part in error.absolute_path),
            "message": error.message,
        }
        for error in sorted(validator.iter_errors(result), key=lambda item: list(item.path))
    ]


def schema_version_of(result: dict[str, Any]) -> str | None:
    value = result.get("schema_version") if isinstance(result, dict) else None
    return value if isinstance(value, str) else None


def empty_result(schema_version: str = DEFAULT_SCHEMA_VERSION, **extra: Any) -> dict[str, Any]:
    """Build the canonical empty, conservative result for a schema.

    Boundary constants are always false, so an empty result can never be read as
    a treatment recommendation or a repairability determination.
    """
    boundary = {
        "treatment_recommendation_provided": False,
        "repairability_determined": False,
        "vascularity_directly_measured": False,
    }
    result: dict[str, Any] = {
        "schema_version": schema_version,
        "study_purpose": "exploratory_non_diagnostic",
        "findings": [],
        "global_limitations": [],
        "treatment_boundary": boundary,
    }
    required = set(_load(schema_version).get("required") or [])
    if "body_region" in required:
        result["body_region"] = extra.get("body_region", "unspecified")
    return result
