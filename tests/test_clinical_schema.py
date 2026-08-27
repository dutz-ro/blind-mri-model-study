import json

import pytest

from mristudy.clinical_schema import (
    clinical_summary,
    empty_knee_meniscus_result,
    parse_structured_model_result,
    validate_knee_meniscus_result,
)
from mristudy.models.runs import supplemental_schema_plan
from mristudy.prompts import knee_meniscus_prompt_v2


def finding() -> dict:
    return {
        "finding_id": "finding-1",
        "meniscus": "medial",
        "segment": "posterior_horn",
        "tear_status": "suspected",
        "tear_pattern": "vertical_longitudinal",
        "orientation": "vertical",
        "vascular_zone_estimate": {
            "zone": "red_white",
            "directly_observed_vascularity": False,
            "basis": "estimated_from_meniscocapsular_distance",
            "confidence": "medium",
            "limitations": ["Conventional MRI does not directly measure perfusion."],
        },
        "capsular_relation": {
            "distance_mm_estimate": 3.2,
            "measurement_method": "geometry_calibrated",
            "peripheral_extension": "present",
            "meniscocapsular_separation": "absent",
        },
        "surface_extension": {
            "superior": "present",
            "inferior": "indeterminate",
            "free_edge": "absent",
        },
        "stability_features": {
            "displaced_fragment": "absent",
            "flap": "absent",
            "bucket_handle_pattern": "absent",
            "imaging_signs_of_mechanical_obstruction": "absent",
        },
        "healing_context": {
            "tissue_quality_imaging_estimate": "preserved_appearing",
            "chronicity_imaging_estimate": "indeterminate",
            "extrusion_mm_estimate": 1.5,
            "parameniscal_cyst": "absent",
            "root_involvement": "absent",
        },
        "imaging_factors_for_specialist_review": [
            "Peripheral location and estimated red-white zone."
        ],
        "traceable_evidence": [
            {
                "export_id": "image-000123",
                "role": "supporting",
                "description": "Linear signal on consecutive sagittal sections.",
            }
        ],
        "counter_evidence": [],
        "limitations": ["No direct assessment of tissue healing capacity."],
        "confidence": "medium",
    }


def valid_result() -> dict:
    result = empty_knee_meniscus_result()
    result["findings"] = [finding()]
    return result


def test_extended_schema_accepts_anatomic_vascular_zone_estimate() -> None:
    result = valid_result()
    assert validate_knee_meniscus_result(result) == []
    parsed = parse_structured_model_result(
        {"choices": [{"message": {"content": json.dumps(result)}}]}
    )
    summary = clinical_summary(parsed)
    assert summary["findings"][0]["vascular_zone_estimate"]["zone"] == "red_white"
    assert not summary["treatment_boundary"]["repairability_determined"]


def test_schema_rejects_direct_vascularity_and_treatment_recommendation() -> None:
    result = valid_result()
    result["findings"][0]["vascular_zone_estimate"]["directly_observed_vascularity"] = True
    result["treatment_boundary"]["treatment_recommendation_provided"] = True
    errors = validate_knee_meniscus_result(result)
    assert len(errors) == 2


def test_prompt_forbids_direct_treatment_and_vascularity_claims() -> None:
    prompt = knee_meniscus_prompt_v2()
    assert "Do not diagnose, recommend treatment" in prompt
    assert "directly_observed_vascularity=false" in prompt
    assert "repairability" in prompt


def test_invalid_structured_result_is_not_silently_accepted() -> None:
    with pytest.raises(ValueError):
        parse_structured_model_result({"choices": [{"message": {"content": "{}"}}]})


def test_supplement_plan_binds_v2_schema_and_does_not_replace_lock() -> None:
    exports = {"exports": [{"export_id": "image-000001", "sha256": "abc"}]}
    plan = supplemental_schema_plan(
        exports,
        {"gpt-sol": "gpt-5.6-sol"},
        ["image-000001"],
    )
    assert plan["output_schema"] == "knee-meniscus-finding-v2"
    assert plan["does_not_replace_existing_lock"]
