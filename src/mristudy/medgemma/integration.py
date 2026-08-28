from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from ..clinical_schema import empty_knee_meniscus_result
from ..profiles import IntegrationPolicy, resolve_path
from ..schema_registry import empty_result
from ..utils import canonical_json, sha256_bytes

# Defaults reproduce the knee behaviour exactly, so an integration performed
# without a profile is byte-identical to the locked results.
KNEE_ANATOMY_KEY = ("meniscus", "segment")
KNEE_STATUS_PATH = "tear_status"
KNEE_POSITIVE = frozenset({"suspected", "probable"})
KNEE_NEGATIVE = frozenset({"absent"})


def _anatomy_key(
    finding: dict[str, Any],
    policy: IntegrationPolicy | None,
) -> tuple[Any, ...]:
    paths = policy.anatomy_key_paths if policy else KNEE_ANATOMY_KEY
    return tuple(resolve_path(finding, path) for path in paths)


def _status_of(finding: dict[str, Any], policy: IntegrationPolicy | None) -> Any:
    return resolve_path(finding, policy.status_path if policy else KNEE_STATUS_PATH)


def _empty_for(policy_schema: str | None, body_region: str | None) -> dict[str, Any]:
    if not policy_schema:
        return empty_knee_meniscus_result()
    return empty_result(policy_schema, body_region=body_region or "unspecified")

MULTIPLANAR_CLASSES = {
    "confirmed_multiplanar",
    "supported_single_plane",
    "window_local_only",
    "contradicted_between_planes",
    "indeterminate",
    "not_comparable",
}


def _evidence_ids(finding: dict[str, Any]) -> set[str]:
    return {
        item["export_id"]
        for key in ("traceable_evidence", "counter_evidence")
        for item in finding.get(key, [])
    }


KNEE_CLAIM_KEY = (
    "meniscus",
    "segment",
    "tear_status",
    "tear_pattern",
    "orientation",
    "vascular_zone_estimate.zone",
    "vascular_zone_estimate.basis",
    "capsular_relation.distance_mm_estimate",
    "capsular_relation.measurement_method",
    "capsular_relation.peripheral_extension",
    "capsular_relation.meniscocapsular_separation",
    "surface_extension.superior",
    "surface_extension.inferior",
    "surface_extension.free_edge",
    "stability_features.displaced_fragment",
    "stability_features.flap",
    "stability_features.bucket_handle_pattern",
    "stability_features.imaging_signs_of_mechanical_obstruction",
    "healing_context.extrusion_mm_estimate",
    "healing_context.parameniscal_cyst",
    "healing_context.root_involvement",
    "healing_context.tissue_quality_imaging_estimate",
    "healing_context.chronicity_imaging_estimate",
)


def _claim_key(
    finding: dict[str, Any],
    policy: IntegrationPolicy | None = None,
) -> tuple[Any, ...]:
    """Identity of the assertion, so contradictions stay separate variants."""
    paths = policy.claim_key_paths if policy else KNEE_CLAIM_KEY
    return tuple(resolve_path(finding, path) for path in paths)


def _merge_findings(members: list[dict[str, Any]], finding_id: str) -> dict[str, Any]:
    finding = deepcopy(members[0]["finding"])
    finding["finding_id"] = finding_id
    supporting = {}
    counter = {}
    limitations = set()
    factors = set()
    for member in members:
        source = member["finding"]
        for item in source.get("traceable_evidence", []):
            supporting[item["export_id"]] = item
        for item in source.get("counter_evidence", []):
            counter[item["export_id"]] = item
        limitations.update(source.get("limitations", []))
        factors.update(source.get("imaging_factors_for_specialist_review", []))
    finding["traceable_evidence"] = [supporting[key] for key in sorted(supporting)]
    finding["counter_evidence"] = [counter[key] for key in sorted(counter)]
    finding["limitations"] = sorted(limitations)
    finding["imaging_factors_for_specialist_review"] = sorted(factors)
    return finding


def _connected_components(
    observations: list[dict[str, Any]],
    policy: IntegrationPolicy | None = None,
) -> list[list[dict[str, Any]]]:
    parents = list(range(len(observations)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(observations):
        for right_index in range(left_index + 1, len(observations)):
            right = observations[right_index]
            same_anatomy = _anatomy_key(left["finding"], policy) == _anatomy_key(
                right["finding"], policy
            )
            # Only evidence actually supplied to each source window can prove
            # that two window observations share the same visual support.
            # Known global citations remain visible but cannot manufacture a
            # cross-window link when the model never saw them in that call.
            shared = bool(
                set(left["link_eligible_evidence_ids"])
                & set(right["link_eligible_evidence_ids"])
            )
            if left["window_id"] != right["window_id"] and same_anatomy and shared:
                union(left_index, right_index)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, observation in enumerate(observations):
        grouped[root(index)].append(observation)
    return [
        sorted(group, key=lambda item: (item["window_id"], item["source_finding_id"]))
        for _, group in sorted(grouped.items())
    ]


def integrate_series(
    role: str,
    series_uid: str,
    window_results: list[dict[str, Any]],
    evidence_map: dict[str, dict[str, Any]] | None = None,
    policy: IntegrationPolicy | None = None,
    schema_version: str | None = None,
    body_region: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observations = []
    for window in sorted(window_results, key=lambda item: item["window_id"]):
        trace_by_finding = {
            str(item.get("finding_id")): item
            for item in (window.get("evidence_summary") or {}).get("findings") or []
        }
        for finding in window["result"]["findings"]:
            trace = trace_by_finding.get(str(finding.get("finding_id")), {})
            cited_ids = sorted(_evidence_ids(finding))
            # Legacy/synthetic callers predate the trace sidecar. Their cited
            # evidence is the only available linkage signal and preserves the
            # established integration behaviour. Production v2 artifacts
            # always provide the explicit eligibility list.
            eligible_ids = (
                trace.get("link_eligible_evidence_ids")
                if trace
                else cited_ids
            )
            observations.append(
                {
                    "source_finding_id": finding["finding_id"],
                    "window_id": window["window_id"],
                    "finding": finding,
                    "evidence_ids": cited_ids,
                    "link_eligible_evidence_ids": sorted(eligible_ids or []),
                    "evidence_dispositions": list(trace.get("dispositions") or []),
                }
            )

    groups = _connected_components(observations, policy)

    integrated = _empty_for(schema_version, body_region)
    provenance = []
    for group_index, group in enumerate(groups, start=1):
        evidence_occurrences: dict[str, int] = defaultdict(int)
        for observation in group:
            for evidence_id in observation["evidence_ids"]:
                evidence_occurrences[evidence_id] += 1
        shared_ids = {
            evidence_id for evidence_id, count in evidence_occurrences.items() if count > 1
        }
        variants: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for observation in group:
            finding = observation["finding"]
            supporting = {
                item["export_id"] for item in finding.get("traceable_evidence", [])
            }
            counter = {item["export_id"] for item in finding.get("counter_evidence", [])}
            shared_role_signature = tuple(
                sorted(
                    (evidence_id, "supporting" if evidence_id in supporting else "counter")
                    for evidence_id in shared_ids
                    if evidence_id in supporting or evidence_id in counter
                )
            )
            variants[(_claim_key(finding, policy), shared_role_signature)].append(observation)
        for variant_index, members in enumerate(
            sorted(
                variants.values(),
                key=lambda values: repr(_claim_key(values[0]["finding"], policy)),
            ),
            start=1,
        ):
            finding = _merge_findings(
                members,
                f"{role}-finding-{group_index:04d}-{variant_index:02d}",
            )
            integrated["findings"].append(finding)
            provenance.append(
                {
                    "integrated_finding_id": finding["finding_id"],
                    "source_finding_ids": [member["source_finding_id"] for member in members],
                    "window_ids": sorted({member["window_id"] for member in members}),
                    "evidence_ids": sorted(
                        {
                            evidence_id
                            for member in members
                            for evidence_id in member["evidence_ids"]
                        }
                    ),
                    "link_eligible_evidence_ids": sorted(
                        {
                            evidence_id
                            for member in members
                            for evidence_id in member["link_eligible_evidence_ids"]
                        }
                    ),
                    "evidence_dispositions": [
                        disposition
                        for member in members
                        for disposition in member["evidence_dispositions"]
                    ],
                    "single_window_only": len({member["window_id"] for member in members}) == 1,
                    "repeated_across_windows": len({member["window_id"] for member in members}) > 1,
                    "repeated_with_distinct_evidence": len(
                        {
                            evidence_id
                            for member in members
                            for evidence_id in member["evidence_ids"]
                        }
                    ) > 1,
                    "claim_variant_count_in_group": len(variants),
                    "contradiction_preserved": len(variants) > 1,
                    "consecutive_evidence": _consecutive_evidence(
                        {
                            evidence_id
                            for member in members
                            for evidence_id in member["evidence_ids"]
                        },
                        evidence_map or {},
                    ),
                }
            )
    integrated["global_limitations"] = sorted(
        {
            limitation
            for window in window_results
            for limitation in window["result"].get("global_limitations", [])
        }
    )
    matrix = {
        "integration_level": "series",
        "role": role,
        "series_uid": series_uid,
        "source_windows": sorted({item["window_id"] for item in window_results}),
        "source_observations": len(observations),
        "integrated_findings": len(integrated["findings"]),
        "finding_provenance": provenance,
        "overlap_is_not_independent_evidence": True,
    }
    matrix["matrix_sha256"] = sha256_bytes(canonical_json(matrix))
    return integrated, matrix


def _consecutive_evidence(
    evidence_ids: set[str],
    evidence_map: dict[str, dict[str, Any]],
) -> bool:
    indices = sorted(
        {
            int(evidence_map[evidence_id]["geometric_index"])
            for evidence_id in evidence_ids
            if evidence_id in evidence_map
        }
    )
    return any(right - left == 1 for left, right in zip(indices, indices[1:], strict=False))


def integrate_multiplanar(
    series_results: list[dict[str, Any]],
    policy: IntegrationPolicy | None = None,
    schema_version: str | None = None,
    body_region: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in series_results:
        provenance_by_finding = {
            entry["integrated_finding_id"]: entry
            for entry in (item.get("provenance") or {}).get("finding_provenance") or []
        }
        for finding in item["result"]["findings"]:
            grouped[_anatomy_key(finding, policy)].append(
                {
                    "role": item["role"],
                    "finding": finding,
                    "evidence_provenance": provenance_by_finding.get(
                        finding["finding_id"], {}
                    ),
                }
            )

    final = _empty_for(schema_version, body_region)
    matrix_groups = []
    for group_index, (anatomy, observations) in enumerate(sorted(grouped.items()), start=1):
        roles = sorted({item["role"] for item in observations})
        statuses = {_status_of(item["finding"], policy) for item in observations}
        positive = bool(statuses & (policy.positive_statuses if policy else KNEE_POSITIVE))
        negative = bool(statuses & (policy.negative_statuses if policy else KNEE_NEGATIVE))
        if positive and negative:
            classification = "contradicted_between_planes"
        elif len(roles) > 1 and positive:
            classification = "confirmed_multiplanar"
        elif len(roles) == 1 and positive:
            repeated = any(
                provenance.get("repeated_with_distinct_evidence")
                for item in series_results
                if item["role"] == roles[0]
                for provenance in item.get("provenance", {}).get("finding_provenance", [])
                if provenance["integrated_finding_id"]
                in {observation["finding"]["finding_id"] for observation in observations}
            )
            classification = "supported_single_plane" if repeated else "window_local_only"
        elif statuses <= {"indeterminate", "not_evaluable"}:
            classification = "indeterminate"
        elif len(roles) == 1:
            classification = "window_local_only"
        else:
            classification = "not_comparable"
        assert classification in MULTIPLANAR_CLASSES

        variants: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for observation in observations:
            variants[_claim_key(observation["finding"], policy)].append(observation)
        output_ids = []
        output_provenance = []
        for variant_index, members in enumerate(
            sorted(
                variants.values(),
                key=lambda values: repr(_claim_key(values[0]["finding"], policy)),
            ),
            start=1,
        ):
            normalized_members = [
                {
                    "finding": member["finding"],
                    "window_id": member["role"],
                    "source_finding_id": member["finding"]["finding_id"],
                }
                for member in members
            ]
            finding = _merge_findings(
                normalized_members,
                f"multiplanar-{group_index:04d}-{variant_index:02d}",
            )
            final["findings"].append(finding)
            output_ids.append(finding["finding_id"])
            output_provenance.append(
                {
                    "output_finding_id": finding["finding_id"],
                    "source_finding_ids": [
                        member["finding"]["finding_id"] for member in members
                    ],
                    "source_roles": sorted({member["role"] for member in members}),
                    "evidence_ids": sorted(
                        {
                            evidence_id
                            for member in members
                            for evidence_id in _evidence_ids(member["finding"])
                        }
                    ),
                    "link_eligible_evidence_ids": sorted(
                        {
                            evidence_id
                            for member in members
                            for evidence_id in member["evidence_provenance"].get(
                                "link_eligible_evidence_ids", []
                            )
                        }
                    ),
                    "evidence_dispositions": [
                        disposition
                        for member in members
                        for disposition in member["evidence_provenance"].get(
                            "evidence_dispositions", []
                        )
                    ],
                    "claim_key": list(_claim_key(members[0]["finding"], policy)),
                }
            )
        anatomy_paths = policy.anatomy_key_paths if policy else KNEE_ANATOMY_KEY
        matrix_groups.append(
            {
                "anatomy_key_paths": list(anatomy_paths),
                "anatomy_key": list(anatomy),
                # Retained so existing knee readers keep working unchanged.
                **(
                    {"meniscus": anatomy[0], "segment": anatomy[1]}
                    if anatomy_paths == KNEE_ANATOMY_KEY
                    else {}
                ),
                "classification": classification,
                "series_roles": roles,
                "source_finding_ids": [item["finding"]["finding_id"] for item in observations],
                "output_finding_ids": output_ids,
                "output_finding_provenance": output_provenance,
                "all_evidence_ids": sorted(
                    {
                        evidence_id
                        for item in observations
                        for evidence_id in _evidence_ids(item["finding"])
                    }
                ),
                "claim_variants": len(variants),
                "arguments_for": sorted(
                    {
                        evidence["description"]
                        for item in observations
                        for evidence in item["finding"].get("traceable_evidence", [])
                    }
                ),
                "arguments_against": sorted(
                    {
                        evidence["description"]
                        for item in observations
                        for evidence in item["finding"].get("counter_evidence", [])
                    }
                ),
                "contradictions_preserved": len(variants) > 1 or classification
                == "contradicted_between_planes",
            }
        )
    final["global_limitations"] = sorted(
        {
            limitation
            for item in series_results
            for limitation in item["result"].get("global_limitations", [])
        }
    )
    matrix = {
        "integration_level": "multiplanar",
        "groups": matrix_groups,
        "agreement_is_not_medical_validation": True,
        "contradictions_preserved": True,
        "cross_plane_spatial_identity_inferred": False,
    }
    matrix["matrix_sha256"] = sha256_bytes(canonical_json(matrix))
    return final, matrix


def validate_integration_evidence(
    result: dict[str, Any],
    evidence_map: dict[str, dict[str, Any]],
) -> None:
    allowed = set(evidence_map)
    for finding in result.get("findings", []):
        for evidence_id in _evidence_ids(finding):
            if evidence_id not in allowed:
                raise ValueError(f"Integrated result contains unknown evidence ID: {evidence_id}")
