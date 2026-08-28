from __future__ import annotations

from typing import Any

from ..clinical_schema import validate_knee_meniscus_result
from .evidence_trace import is_evidence_id


def validate_window_request(payload: dict[str, Any], approved_window: dict[str, Any]) -> list[str]:
    errors = []
    instance_uids = payload.get("instance_uids") or []
    evidence_ids = payload.get("evidence_ids") or []
    if not instance_uids:
        errors.append("instance_uids must not be empty")
    if len(instance_uids) > 32:
        errors.append("window exceeds 32 instances")
    if len(instance_uids) != len(evidence_ids):
        errors.append("instance_uids and evidence_ids lengths differ")
    if len(instance_uids) != len(set(instance_uids)):
        errors.append("duplicate instance UID")
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("duplicate evidence ID")
    if instance_uids != approved_window.get("instance_uids"):
        errors.append("instance UIDs differ from approved order")
    if evidence_ids != approved_window.get("evidence_ids"):
        errors.append("evidence IDs differ from approved order")
    if payload.get("study_uid") != approved_window.get("study_uid"):
        errors.append("study UID differs from approved window")
    if payload.get("series_uid") != approved_window.get("series_uid"):
        errors.append("series UID differs from approved window")
    if payload.get("window_id") != approved_window.get("window_id"):
        errors.append("window ID differs from approved window")
    if payload.get("output_schema") != "knee-meniscus-finding-v2":
        errors.append("unsupported output schema")
    if payload.get("prompt_sha256") != approved_window.get("prompt_sha256"):
        errors.append("prompt SHA-256 differs from approved window")
    return errors


def evidence_traceability_warnings(
    result: dict[str, Any],
    allowed_evidence_ids: list[str],
) -> list[str]:
    """Report weak traceability without rejecting the clinical content.

    A completed run produced findings whose image citation was placed in
    ``finding_id`` while ``traceable_evidence`` stayed empty. Schema validation
    accepted that, because an empty evidence list is structurally legal, so the
    report was valid yet visually untraceable.

    These conditions are returned as warnings rather than errors on purpose:
    discarding a produced clinical result is worse than surfacing it with an
    explicit caveat. The runner records them and grades every finding, so the
    weakness is visible instead of silent.
    """
    warnings: list[str] = []
    if not isinstance(result, dict):
        return warnings
    allowed = set(allowed_evidence_ids)
    for finding in result.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        identifier = finding.get("finding_id")
        cited = [
            evidence.get("export_id")
            for key in ("traceable_evidence", "counter_evidence")
            for evidence in finding.get(key) or []
            if isinstance(evidence, dict)
        ]
        if is_evidence_id(identifier):
            warnings.append(
                "finding_id is an image evidence identifier; image citations belong in "
                f"traceable_evidence: {identifier}"
            )
        if not cited:
            warnings.append(f"finding has no image citation: {identifier}")
        elif not set(cited) & allowed:
            warnings.append(
                f"finding cites no evidence from its own window: {identifier}"
            )
    return warnings


def validate_window_result(
    result: dict[str, Any],
    allowed_evidence_ids: list[str] | None = None,
) -> list[str]:
    """Validate clinical structure and invariants, independently of scope.

    The optional evidence list is retained for compatibility and warning
    generation elsewhere. A canonically valid clinical result must not trigger
    another model generation merely because a citation is from another window.
    """
    errors = [item["message"] for item in validate_knee_meniscus_result(result)]
    if not isinstance(result, dict):
        return errors + ["result must be an object"]
    for finding in result.get("findings") or []:
        if not isinstance(finding, dict):
            errors.append("finding must be an object")
            continue
        supporting = [
            evidence.get("export_id")
            for evidence in finding.get("traceable_evidence") or []
            if isinstance(evidence, dict)
        ]
        counter = [
            evidence.get("export_id")
            for evidence in finding.get("counter_evidence") or []
            if isinstance(evidence, dict)
        ]
        for key in ("traceable_evidence", "counter_evidence"):
            for evidence in finding.get(key) or []:
                if not isinstance(evidence, dict):
                    errors.append(f"{key} entry must be an object")
                    continue
                reference = evidence.get("export_id")
                expected_role = "supporting" if key == "traceable_evidence" else "counter"
                if evidence.get("role") != expected_role:
                    errors.append(
                        f"evidence role must be {expected_role} in {key}: {reference}"
                    )
        if len(supporting) != len(set(supporting)) or len(counter) != len(set(counter)):
            errors.append(
                f"duplicate evidence reference within finding: {finding.get('finding_id')}"
            )
        if set(supporting) & set(counter):
            errors.append(
                "same evidence is both supporting and counter evidence: "
                f"{finding.get('finding_id')}"
            )
        vascular = finding.get("vascular_zone_estimate") or {}
        if not isinstance(vascular, dict):
            errors.append("vascular_zone_estimate must be an object")
        elif vascular.get("directly_observed_vascularity") is not False:
            errors.append("directly_observed_vascularity must be false")
    boundary = result.get("treatment_boundary") or {}
    if not isinstance(boundary, dict):
        return errors + ["treatment_boundary must be an object"]
    if boundary.get("treatment_recommendation_provided") is not False:
        errors.append("treatment recommendation must be false")
    if boundary.get("repairability_determined") is not False:
        errors.append("repairability determination must be false")
    if boundary.get("vascularity_directly_measured") is not False:
        errors.append("vascularity_directly_measured must be false")
    return errors
