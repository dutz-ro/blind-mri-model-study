from __future__ import annotations

from typing import Any

from ..clinical_schema import load_knee_meniscus_schema
from ..profiles import StudyProfile
from ..prompts import (
    json_suffix_for,
    medgemma_json_suffix_v2,
    medgemma_knee_meniscus_prompt_v2,
    medgemma_prompt_for_profile,
    medgemma_window_prompt,
)
from ..schema_registry import load_schema
from ..utils import canonical_json, sha256_bytes, utc_now
from .evidence import ROLE_ORDER, assign_global_evidence, finalize_evidence_map
from .evidence_trace import EVIDENCE_TRACE_POLICY
from .integrity import recompute_plan_sha256
from .output_contract import decoder_contract, output_contract, policy_digest
from .windows import plan_windows


def build_medgemma_window_plan(
    *,
    series_by_uid: dict[str, dict[str, Any]],
    orthanc_manifest: dict[str, Any],
    model_repository: str,
    model_revision: str,
    runtime_digest: str,
    historical_replay_sha256: str,
    excluded_series: list[dict[str, str]],
    window_size: int = 32,
    minimum_overlap: int = 8,
    profile: StudyProfile | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Without a profile the knee prompt, schema and role order are used, so an
    # existing plan regenerates to the same shape.
    if profile is None:
        role_order = ROLE_ORDER
        output_schema = "knee-meniscus-finding-v2"
        base_prompt = medgemma_knee_meniscus_prompt_v2()
        output_suffix = medgemma_json_suffix_v2()
        decoder = decoder_contract(load_knee_meniscus_schema())
    else:
        role_order = tuple(
            role for role in profile.role_order if role in
            {series["role"] for series in series_by_uid.values()}
        )
        output_schema = profile.output_schema
        base_prompt = medgemma_prompt_for_profile(profile)
        output_suffix = json_suffix_for(output_schema)
        decoder = decoder_contract(load_schema(output_schema))
        window_size = profile.maximum_window_size
        minimum_overlap = profile.minimum_overlap

    evidence_map, _ = assign_global_evidence(series_by_uid, role_order)
    by_role = {series["role"]: series for series in series_by_uid.values()}
    windows: list[dict[str, Any]] = []
    prompt_sha256 = sha256_bytes(base_prompt.encode())

    for role in role_order:
        series = by_role[role]
        series_plan = plan_windows(
            series["instances"],
            window_size,
            minimum_overlap,
            window_prefix=role,
        )
        for window in series_plan.windows:
            instance_uids = list(window.instance_uids)
            evidence_ids = list(window.evidence_ids)
            for evidence_id in evidence_ids:
                evidence_map[evidence_id]["window_ids"].append(window.window_id)
            input_list_sha256 = sha256_bytes(
                canonical_json(
                    {
                        "instance_uids": instance_uids,
                        "evidence_ids": evidence_ids,
                    }
                )
            )
            window_record = {
                "window_id": window.window_id,
                "role": role,
                "study_uid": series["instances"][0]["study_uid"],
                "series_uid": series["series_uid"],
                "series_description": series["series_description"],
                "ordering_method": series["ordering_method"],
                "ordering_warning": series["ordering_warning"],
                "start_index": window.start_index,
                "end_index": window.end_index,
                "source_instance_count": len(series["instances"]),
                "instance_uids": instance_uids,
                "evidence_ids": evidence_ids,
                "input_list_sha256": input_list_sha256,
                "overlap_previous": window.overlap_previous,
                "overlap_next": window.overlap_next,
                "output_schema": output_schema,
                "max_new_tokens": 4096,
                "maximum_structural_corrections": 2,
                "generation": {
                    "do_sample": False,
                    "repetition_penalty": 1.05,
                    "no_repeat_ngram_size": 8,
                    "json_schema_enforced": True,
                    "canonical_eos_only": True,
                },
                    "decoder": decoder,
                    "output_contract": output_contract(),
                    "model_repository": model_repository,
                "model_revision": model_revision,
                "runtime_digest": runtime_digest,
            }
            # Keep the legacy suffix construction for a profile-less caller,
            # but use the shared builder for every real profile-driven plan.
            window_prompt = (
                medgemma_window_prompt(base_prompt, window_record)
                if profile is not None
                else (
                    base_prompt
                    + f"\nThis window has role {role}, ID {window.window_id}, and "
                    + f"series description {series['series_description']!r}. Report only findings "
                    + "evaluable from this contiguous window. Use indeterminate for fields "
                    + "requiring "
                    + "another plane. Cite only the supplied global evidence IDs.\n"
                    + output_suffix
                )
            )
            window_record["prompt_sha256"] = sha256_bytes(window_prompt.encode())
            windows.append(window_record)

    evidence_manifest = finalize_evidence_map(evidence_map)
    source_slices = sum(len(series["instances"]) for series in series_by_uid.values())
    covered_uids = {
        uid
        for window in windows
        for uid in window["instance_uids"]
    }
    expected_uids = {
        item["sop_instance_uid"]
        for series in series_by_uid.values()
        for item in series["instances"]
    }
    missing = sorted(expected_uids - covered_uids)
    unexpected = sorted(covered_uids - expected_uids)
    duplicate_memberships = sum(len(window["instance_uids"]) for window in windows) - len(
        covered_uids
    )
    coverage = 100.0 * len(covered_uids) / source_slices

    if missing or unexpected or coverage != 100.0:
        raise ValueError("MedGemma window plan does not provide exact complete coverage.")
    if any(len(window["instance_uids"]) > window_size for window in windows):
        raise ValueError("MedGemma window exceeds maximum size.")
    if any(
        window["overlap_previous"] < minimum_overlap
        for window in windows
        if window["overlap_previous"]
    ):
        raise ValueError("MedGemma window plan violates minimum overlap.")
    if not expected_uids.issubset(set(orthanc_manifest.get("instance_uids", expected_uids))):
        # Older Orthanc manifests store only an aggregate hash. In that case exact
        # membership is guaranteed by the prior exact upload verification.
        if "instance_uids" in orthanc_manifest:
            raise ValueError("Window plan contains a UID absent from Orthanc.")

    plan = {
        "schema_version": 3,
        "operation": "medgemma-full-coverage-window-analysis",
        "created_at": utc_now(),
        "window_size": window_size,
        "minimum_overlap": minimum_overlap,
        "total_source_slices": source_slices,
        "unique_analyzed_slices": len(covered_uids),
        "coverage_percent": round(coverage, 6),
        "total_windows": len(windows),
        "maximum_window_size": max(len(window["instance_uids"]) for window in windows),
        "duplicate_window_memberships": duplicate_memberships,
        "missing_instance_uids": missing,
        "unexpected_instance_uids": unexpected,
        "excluded_series": excluded_series,
        "radiology_report_used": False,
        "treatment_recommendation_allowed": False,
        "repairability_determination_allowed": False,
        "vascularity_directly_measured": False,
        "model_repository": model_repository,
        "model_revision": model_revision,
        "runtime_digest": runtime_digest,
        "historical_replay_sha256": historical_replay_sha256,
        "prompt_sha256": prompt_sha256,
        "output_schema": output_schema,
        "profile": profile.name if profile else "knee-meniscus-v1",
        "body_region": profile.body_region if profile else "knee",
        "role_order": list(role_order),
        "decoder": decoder,
        "output_contract": output_contract(),
        "evidence_trace_contract": {
            "policy": EVIDENCE_TRACE_POLICY,
            "policy_sha256": policy_digest(
                EVIDENCE_TRACE_POLICY,
                {
                    "policy": EVIDENCE_TRACE_POLICY,
                    "known_global_evidence_retained": True,
                    "unknown_evidence_quarantined": True,
                    "linkage_requires_current_window": True,
                },
            ),
        },
        "orthanc_manifest_sha256": orthanc_manifest["manifest_sha256"],
        "evidence_map_sha256": evidence_manifest["map_sha256"],
        "windows": windows,
    }
    plan["plan_sha256"] = recompute_plan_sha256(plan)
    return plan, evidence_manifest
