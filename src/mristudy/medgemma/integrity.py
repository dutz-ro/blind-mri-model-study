from __future__ import annotations

from typing import Any

from ..clinical_schema import load_knee_meniscus_schema
from ..errors import MRIStudyError
from ..utils import canonical_json, sha256_bytes
from .evidence import ROLE_ORDER
from .evidence_trace import EVIDENCE_TRACE_POLICY
from .output_contract import decoder_contract, output_contract, policy_digest


def recompute_plan_sha256(plan: dict[str, Any]) -> str:
    return sha256_bytes(
        canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"})
    )


def require_valid_plan_digest(plan: dict[str, Any]) -> None:
    if plan.get("plan_sha256") != recompute_plan_sha256(plan):
        raise MRIStudyError("MedGemma run plan SHA-256 is invalid.")


def recompute_evidence_map_sha256(manifest: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(manifest.get("entries") or []))


def require_valid_evidence_map(
    manifest: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    actual = recompute_evidence_map_sha256(manifest)
    if manifest.get("map_sha256") != actual:
        raise MRIStudyError("MedGemma evidence-map SHA-256 is invalid.")
    if plan.get("evidence_map_sha256") != actual:
        raise MRIStudyError("MedGemma plan is not bound to the evidence map.")

    entries = manifest.get("entries") or []
    if manifest.get("evidence_count") != len(entries):
        raise MRIStudyError("MedGemma evidence-map count is invalid.")
    by_id = {item["evidence_id"]: item for item in entries}
    if len(by_id) != len(entries):
        raise MRIStudyError("Duplicate evidence ID in MedGemma evidence map.")
    uid_to_id = {}
    for evidence_id, item in by_id.items():
        uid = item["sop_instance_uid"]
        if uid in uid_to_id and uid_to_id[uid] != evidence_id:
            raise MRIStudyError("One SOP Instance UID maps to multiple evidence IDs.")
        uid_to_id[uid] = evidence_id
    if len(entries) != plan.get("total_source_slices"):
        raise MRIStudyError("MedGemma evidence map does not cover every source slice.")

    expected_memberships: dict[str, set[str]] = {key: set() for key in by_id}
    for window in plan.get("windows") or []:
        for uid, evidence_id in zip(
            window["instance_uids"], window["evidence_ids"], strict=True
        ):
            item = by_id.get(evidence_id)
            if item is None:
                raise MRIStudyError("Window references an unknown evidence ID.")
            if item["sop_instance_uid"] != uid:
                raise MRIStudyError("Window evidence ID maps to a different SOP Instance UID.")
            if item["series_uid"] != window["series_uid"]:
                raise MRIStudyError("Window evidence belongs to another series.")
            if item["role"] != window["role"]:
                raise MRIStudyError("Window evidence belongs to another role.")
            expected_memberships[evidence_id].add(window["window_id"])
    for evidence_id, item in by_id.items():
        if set(item.get("window_ids") or []) != expected_memberships[evidence_id]:
            raise MRIStudyError("Evidence-map window memberships do not match the plan.")
    return by_id


def recompute_coverage(plan: dict[str, Any]) -> dict[str, Any]:
    windows = plan.get("windows") or []
    memberships = [uid for window in windows for uid in window.get("instance_uids") or []]
    unique = set(memberships)
    total = int(plan.get("total_source_slices", 0))
    coverage = round(100.0 * len(unique) / total, 6) if total else 0.0
    return {
        "unique_analyzed_slices": len(unique),
        "coverage_percent": coverage,
        "total_windows": len(windows),
        "maximum_window_size": max((len(window["instance_uids"]) for window in windows), default=0),
        "duplicate_window_memberships": len(memberships) - len(unique),
    }


def require_complete_plan(plan: dict[str, Any]) -> None:
    require_valid_plan_digest(plan)
    actual = recompute_coverage(plan)
    for key, value in actual.items():
        if plan.get(key) != value:
            raise MRIStudyError(f"MedGemma plan coverage field is invalid: {key}")
    if actual["coverage_percent"] != 100.0:
        raise MRIStudyError("MedGemma run plan coverage is below 100%.")
    if plan.get("missing_instance_uids") or plan.get("unexpected_instance_uids"):
        raise MRIStudyError("MedGemma run plan has missing or unexpected instance UIDs.")
    replay_sha256 = plan.get("historical_replay_sha256")
    if not isinstance(replay_sha256, str) or len(replay_sha256) != 64:
        raise MRIStudyError("MedGemma run plan lacks the historical replay digest.")
    if any(len(window["instance_uids"]) > 32 for window in plan.get("windows") or []):
        raise MRIStudyError("MedGemma run plan contains a window larger than 32.")
    windows_by_role: dict[str, list[dict[str, Any]]] = {}
    expected_decoder = decoder_contract(load_knee_meniscus_schema())
    if plan.get("decoder") != expected_decoder:
        raise MRIStudyError("MedGemma decoder contract differs from approved contract.")
    if plan.get("schema_version", 0) >= 3:
        if plan.get("output_contract") != output_contract():
            raise MRIStudyError("MedGemma output contract differs from approved contract.")
        trace = plan.get("evidence_trace_contract") or {}
        if trace.get("policy") != EVIDENCE_TRACE_POLICY:
            raise MRIStudyError("MedGemma evidence trace contract differs from approved contract.")
        expected_trace_digest = policy_digest(
            EVIDENCE_TRACE_POLICY,
            {
                "policy": EVIDENCE_TRACE_POLICY,
                "known_global_evidence_retained": True,
                "unknown_evidence_quarantined": True,
                "linkage_requires_current_window": True,
            },
        )
        if trace.get("policy_sha256") != expected_trace_digest:
            raise MRIStudyError("MedGemma evidence trace policy digest is invalid.")
    for window in plan.get("windows") or []:
        if window.get("max_new_tokens") != 4096:
            raise MRIStudyError("MedGemma window generation limit differs from approved contract.")
        if window.get("maximum_structural_corrections") != 2:
            raise MRIStudyError("MedGemma correction policy differs from approved contract.")
        if window.get("generation") != {
            "do_sample": False,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 8,
            "json_schema_enforced": True,
            "canonical_eos_only": True,
        }:
            raise MRIStudyError("MedGemma generation policy differs from approved contract.")
        if window.get("decoder") != expected_decoder:
            raise MRIStudyError("MedGemma window decoder differs from approved contract.")
        if plan.get("schema_version", 0) >= 3 and window.get(
            "output_contract"
        ) != output_contract():
            raise MRIStudyError("MedGemma window output contract differs from approved contract.")
        windows_by_role.setdefault(window["role"], []).append(window)
    if plan.get("operation") == "medgemma-full-coverage-window-analysis":
        if set(windows_by_role) != set(ROLE_ORDER):
            raise MRIStudyError("MedGemma full-coverage plan does not contain every required role.")
    for role, windows in windows_by_role.items():
        ordered = sorted(windows, key=lambda item: item["start_index"])
        if ordered[0]["start_index"] != 0:
            raise MRIStudyError(f"MedGemma role does not start at its first slice: {role}")
        source_count = ordered[0]["source_instance_count"]
        if ordered[-1]["end_index"] != source_count - 1:
            raise MRIStudyError(f"MedGemma role does not end at its final slice: {role}")
        if any(window["source_instance_count"] != source_count for window in ordered):
            raise MRIStudyError(f"MedGemma source count is inconsistent within role: {role}")
        for left, right in zip(ordered, ordered[1:], strict=False):
            overlap = left["end_index"] - right["start_index"] + 1
            if overlap < plan["minimum_overlap"]:
                raise MRIStudyError(f"MedGemma overlap is below minimum for role: {role}")
            if left["overlap_next"] != overlap or right["overlap_previous"] != overlap:
                raise MRIStudyError(f"MedGemma overlap metadata is invalid for role: {role}")
