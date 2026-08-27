from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from ..clinical_schema import empty_knee_meniscus_result
from ..utils import canonical_json, sha256_bytes

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


def _claim_key(finding: dict[str, Any]) -> tuple[Any, ...]:
    return (
        finding["meniscus"],
        finding["segment"],
        finding["tear_status"],
        finding["tear_pattern"],
        finding["orientation"],
        finding["vascular_zone_estimate"]["zone"],
        finding["vascular_zone_estimate"]["basis"],
        finding["capsular_relation"]["distance_mm_estimate"],
        finding["capsular_relation"]["measurement_method"],
        finding["capsular_relation"]["peripheral_extension"],
        finding["capsular_relation"]["meniscocapsular_separation"],
        finding["surface_extension"]["superior"],
        finding["surface_extension"]["inferior"],
        finding["surface_extension"]["free_edge"],
        finding["stability_features"]["displaced_fragment"],
        finding["stability_features"]["flap"],
        finding["stability_features"]["bucket_handle_pattern"],
        finding["stability_features"]["imaging_signs_of_mechanical_obstruction"],
        finding["healing_context"]["extrusion_mm_estimate"],
        finding["healing_context"]["parameniscal_cyst"],
        finding["healing_context"]["root_involvement"],
        finding["healing_context"]["tissue_quality_imaging_estimate"],
        finding["healing_context"]["chronicity_imaging_estimate"],
    )


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


def _connected_components(observations: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
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
            same_anatomy = (
                left["finding"]["meniscus"], left["finding"]["segment"]
            ) == (
                right["finding"]["meniscus"], right["finding"]["segment"]
            )
            shared = bool(set(left["evidence_ids"]) & set(right["evidence_ids"]))
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    observations = []
    for window in sorted(window_results, key=lambda item: item["window_id"]):
        for finding in window["result"]["findings"]:
            observations.append(
                {
                    "source_finding_id": finding["finding_id"],
                    "window_id": window["window_id"],
                    "finding": finding,
                    "evidence_ids": sorted(_evidence_ids(finding)),
                }
            )

    groups = _connected_components(observations)

    integrated = empty_knee_meniscus_result()
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
            variants[(_claim_key(finding), shared_role_signature)].append(observation)
        for variant_index, members in enumerate(
            sorted(variants.values(), key=lambda values: repr(_claim_key(values[0]["finding"]))),
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in series_results:
        for finding in item["result"]["findings"]:
            grouped[(finding["meniscus"], finding["segment"])].append(
                {"role": item["role"], "finding": finding}
            )

    final = empty_knee_meniscus_result()
    matrix_groups = []
    for group_index, (anatomy, observations) in enumerate(sorted(grouped.items()), start=1):
        roles = sorted({item["role"] for item in observations})
        statuses = {item["finding"]["tear_status"] for item in observations}
        positive = bool(statuses & {"suspected", "probable"})
        negative = "absent" in statuses
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
            variants[_claim_key(observation["finding"])].append(observation)
        output_ids = []
        output_provenance = []
        for variant_index, members in enumerate(
            sorted(variants.values(), key=lambda values: repr(_claim_key(values[0]["finding"]))),
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
                    "claim_key": list(_claim_key(members[0]["finding"])),
                }
            )
        matrix_groups.append(
            {
                "meniscus": anatomy[0],
                "segment": anatomy[1],
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
