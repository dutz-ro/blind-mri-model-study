from __future__ import annotations

from typing import Any

from output_contract import normalize_structural_result, validate_result_evidence

__all__ = [
    "normalize_structural_result",
    "validate_result_evidence",
    "validate_window_payload",
]

OUTPUT_SCHEMA = "knee-meniscus-finding-v2"


def _expected_schema(approved: dict[str, Any]) -> str:
    """The approved window names the schema; fall back to the knee default."""
    declared = approved.get("output_schema")
    return declared if isinstance(declared, str) and declared else OUTPUT_SCHEMA

def validate_window_payload(payload: dict[str, Any], approved: dict[str, Any]) -> list[str]:
    errors = []
    instance_uids = payload.get("instance_uids") or []
    evidence_ids = payload.get("evidence_ids") or []
    if not instance_uids:
        errors.append("empty instance list")
    if len(instance_uids) > 32:
        errors.append("more than 32 instances")
    if len(instance_uids) != len(evidence_ids):
        errors.append("instance and evidence lengths differ")
    if len(instance_uids) != len(set(instance_uids)):
        errors.append("duplicate instance UID")
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("duplicate evidence ID")
    if payload.get("output_schema") != _expected_schema(approved):
        errors.append("unsupported output schema")
    for key in (
        "study_uid",
        "series_uid",
        "role",
        "window_id",
        "instance_uids",
        "evidence_ids",
        "prompt_sha256",
        "output_schema",
        "max_new_tokens",
        "generation",
        "decoder",
    ):
        if payload.get(key) != approved.get(key):
            errors.append(f"{key} differs from approved window")
    return errors
