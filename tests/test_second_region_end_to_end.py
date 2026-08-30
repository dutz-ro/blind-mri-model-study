"""A second body region works without changing code.

This is the modularity claim under test: planning, windowing, evidence
identity, validation, series integration and multiplanar integration all run
for a brain study with unknown pathology, driven only by a profile and the
anatomy-independent schema.
"""

from __future__ import annotations

from typing import Any

from mristudy.medgemma.integration import integrate_multiplanar, integrate_series
from mristudy.medgemma.output_contract import validate_window_finding_semantics
from mristudy.medgemma.validation import evidence_traceability_warnings
from mristudy.profiles import load_profile_by_name
from mristudy.schema_registry import empty_result, load_schema, validate_result

PROFILE = load_profile_by_name("brain-unknown-v1")
SCHEMA = PROFILE.output_schema


def _finding(
    finding_id: str,
    region: str,
    certainty: str,
    evidence: list[str],
    observation_class: str = "signal_abnormality",
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "observation_class": observation_class,
        "certainty_status": certainty,
        "localization": {
            "laterality": "left",
            "region": region,
            "subregion": None,
            "description": f"{region}, periventricular white matter",
        },
        "signal": {
            "t1": "hypointense",
            "t2": "hyperintense",
            "fluid_sensitive": "hyperintense",
            "enhancement": "no_contrast_administered",
            "restricted_diffusion": "indeterminate",
        },
        "morphology": {
            "largest_dimension_mm_estimate": None,
            "measurement_method": "not_measurable",
            "shape": "irregular",
            "margins": "ill_defined",
            "mass_effect": "absent",
        },
        "imaging_factors_for_specialist_review": ["lesion_distribution"],
        "traceable_evidence": [
            {"export_id": item, "role": "supporting", "description": "d"} for item in evidence
        ],
        "counter_evidence": [],
        "limitations": ["exploratory"],
        "confidence": "low",
    }


def _result(findings: list[dict[str, Any]]) -> dict[str, Any]:
    result = empty_result(SCHEMA, body_region="brain")
    result["findings"] = findings
    return result


def test_core_schema_accepts_a_brain_finding() -> None:
    result = _result([_finding("w1-01", "frontal_lobe", "suspected", ["image-000001"])])
    assert validate_result(result, SCHEMA) == []


def test_unexpected_pathology_can_be_reported() -> None:
    """The point of the core schema: an unanticipated finding is expressible."""
    result = _result(
        [
            _finding(
                "w1-02",
                "pineal_region",
                "indeterminate",
                ["image-000002"],
                observation_class="unexpected_finding",
            )
        ]
    )
    assert validate_result(result, SCHEMA) == []


def test_knee_focal_semantics_do_not_constrain_the_core_schema() -> None:
    result = _result(
        [
            _finding("w1-01", "frontal_lobe", "suspected", ["image-000001"]),
            _finding("w1-02", "parietal_lobe", "suspected", ["image-000002"]),
            _finding("w1-03", "temporal_lobe", "suspected", ["image-000003"]),
            _finding("w1-04", "occipital_lobe", "suspected", ["image-000004"]),
        ]
    )
    assert validate_window_finding_semantics(result) == []


def test_core_schema_keeps_the_safety_boundaries() -> None:
    schema = load_schema(SCHEMA)
    boundary = schema["properties"]["treatment_boundary"]["properties"]
    assert boundary["treatment_recommendation_provided"]["const"] is False
    assert boundary["repairability_determined"]["const"] is False
    assert boundary["vascularity_directly_measured"]["const"] is False


def test_boundary_violation_is_rejected() -> None:
    result = _result([])
    result["treatment_boundary"]["treatment_recommendation_provided"] = True
    assert validate_result(result, SCHEMA) != []


def test_evidence_identifier_shape_is_enforced() -> None:
    result = _result([_finding("w1-03", "frontal_lobe", "suspected", ["img-1"])])
    assert validate_result(result, SCHEMA) != []


def test_series_integration_groups_by_profile_anatomy_key() -> None:
    windows = [
        {
            "window_id": "primary-volume-window-0001",
            "role": "primary-volume",
            "series_uid": "1.2.3",
            "start_index": 0,
            "end_index": 1,
            "result": _result(
                [_finding("w1-01", "frontal_lobe", "suspected", ["image-000001"])]
            ),
        },
        {
            "window_id": "primary-volume-window-0002",
            "role": "primary-volume",
            "series_uid": "1.2.3",
            "start_index": 1,
            "end_index": 2,
            "result": _result(
                [_finding("w2-01", "frontal_lobe", "suspected", ["image-000001"])]
            ),
        },
    ]
    result, provenance = integrate_series(
        "primary-volume",
        "1.2.3",
        windows,
        policy=PROFILE.integration,
        schema_version=SCHEMA,
        body_region="brain",
    )
    assert validate_result(result, SCHEMA) == []
    # Shared evidence plus the same location means one claim, not two.
    assert len(result["findings"]) == 1
    assert provenance["role"] == "primary-volume"


def test_multiplanar_integration_uses_the_profile_status_partition() -> None:
    series_results = [
        {
            "role": "primary-volume",
            "result": _result(
                [_finding("s1", "frontal_lobe", "suspected", ["image-000001"])]
            ),
        },
        {
            "role": "t2-axial",
            "result": _result(
                [_finding("s2", "frontal_lobe", "probable", ["image-000050"])]
            ),
        },
    ]
    final, provenance = integrate_multiplanar(
        series_results,
        policy=PROFILE.integration,
        schema_version=SCHEMA,
        body_region="brain",
    )
    assert validate_result(final, SCHEMA) == []
    groups = provenance["groups"] if "groups" in provenance else provenance.get("matrix") or []
    assert groups
    classifications = {item["classification"] for item in groups}
    assert "confirmed_multiplanar" in classifications


def test_contradiction_between_planes_is_preserved() -> None:
    series_results = [
        {
            "role": "primary-volume",
            "result": _result(
                [_finding("s1", "occipital_lobe", "probable", ["image-000001"])]
            ),
        },
        {
            "role": "t2-axial",
            "result": _result(
                [_finding("s2", "occipital_lobe", "absent", ["image-000060"])]
            ),
        },
    ]
    _, provenance = integrate_multiplanar(
        series_results,
        policy=PROFILE.integration,
        schema_version=SCHEMA,
        body_region="brain",
    )
    groups = provenance["groups"] if "groups" in provenance else provenance.get("matrix") or []
    assert any(item["classification"] == "contradicted_between_planes" for item in groups)


def test_traceability_warnings_apply_to_any_region() -> None:
    result = _result([_finding("image-000001", "frontal_lobe", "suspected", [])])
    warnings = evidence_traceability_warnings(result, ["image-000001"])
    assert any("finding_id" in item for item in warnings)
    assert any("no image citation" in item for item in warnings)
