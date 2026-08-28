from __future__ import annotations

import json
from typing import Any

from .schema_registry import SCHEMA_FILES, empty_result, load_schema, validate_result

SCHEMA_VERSION = "knee-meniscus-finding-v2"

# Region-specific helpers are retained so existing callers and locked artifacts
# keep working. New code should use the registry directly and pass an explicit
# schema version.
__all__ = [
    "SCHEMA_FILES",
    "SCHEMA_VERSION",
    "clinical_summary",
    "empty_knee_meniscus_result",
    "load_knee_meniscus_schema",
    "parse_structured_model_result",
    "validate_knee_meniscus_result",
]


def load_knee_meniscus_schema() -> dict[str, Any]:
    return load_schema(SCHEMA_VERSION)


def validate_knee_meniscus_result(result: dict[str, Any]) -> list[dict[str, str]]:
    return validate_result(result, SCHEMA_VERSION)


def empty_knee_meniscus_result() -> dict[str, Any]:
    return empty_result(SCHEMA_VERSION)


def parse_structured_model_result(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("schema_version") == SCHEMA_VERSION:
        result = response
    else:
        choices = response.get("choices") or []
        if not choices:
            raise ValueError("Model response has no choices.")
        content = ((choices[0].get("message") or {}).get("content") or "").strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        result = json.loads(content)
    errors = validate_knee_meniscus_result(result)
    if errors:
        raise ValueError(f"Structured result failed schema validation: {errors[:3]}")
    return result


def structured_validation_error(response: dict[str, Any]) -> str:
    try:
        parse_structured_model_result(response)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return str(error)
    return ""


def clinical_summary(result: dict[str, Any]) -> dict[str, Any]:
    findings = result.get("findings") or []
    return {
        "schema_version": result.get("schema_version"),
        "finding_count": len(findings),
        "findings": [
            {
                "finding_id": item["finding_id"],
                "meniscus": item["meniscus"],
                "segment": item["segment"],
                "tear_status": item["tear_status"],
                "tear_pattern": item["tear_pattern"],
                "orientation": item["orientation"],
                "vascular_zone_estimate": item["vascular_zone_estimate"],
                "capsular_relation": item["capsular_relation"],
                "surface_extension": item["surface_extension"],
                "stability_features": item["stability_features"],
                "healing_context": item["healing_context"],
                "imaging_factors_for_specialist_review": item[
                    "imaging_factors_for_specialist_review"
                ],
                "traceable_evidence": item["traceable_evidence"],
                "counter_evidence": item["counter_evidence"],
                "confidence": item["confidence"],
            }
            for item in findings
        ],
        "treatment_boundary": result.get("treatment_boundary"),
        "global_limitations": result.get("global_limitations") or [],
    }
