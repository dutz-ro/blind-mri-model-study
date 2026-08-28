from copy import deepcopy

import pytest

from mristudy.medgemma.validation import validate_window_request, validate_window_result
from tests.test_clinical_schema import valid_result


def approved_window() -> dict:
    return {
        "study_uid": "1.2.3",
        "series_uid": "1.2.4",
        "role": "primary-volume",
        "window_id": "primary-volume-window-0001",
        "instance_uids": ["1.2.5", "1.2.6"],
        "evidence_ids": ["image-000001", "image-000002"],
        "output_schema": "knee-meniscus-finding-v2",
    }


def test_request_requires_exact_approved_uid_order() -> None:
    approved = approved_window()
    assert validate_window_request(dict(approved), approved) == []
    changed = dict(approved)
    changed["instance_uids"] = list(reversed(changed["instance_uids"]))
    assert "instance UIDs differ from approved order" in validate_window_request(changed, approved)


def test_result_scope_is_not_a_canonical_validation_error() -> None:
    result = valid_result()
    result["findings"][0]["traceable_evidence"][0]["export_id"] = "image-999999"
    errors = validate_window_result(result, ["image-000001", "image-000002"])
    assert not any("outside approved window" in error for error in errors)
    assert errors == []


def test_result_rejects_duplicate_evidence_reference() -> None:
    result = valid_result()
    duplicate = deepcopy(result["findings"][0]["traceable_evidence"][0])
    result["findings"][0]["counter_evidence"] = [duplicate]
    errors = validate_window_result(result, ["image-000123"])
    assert any("supporting and counter" in error for error in errors)


def test_same_evidence_can_support_distinct_findings() -> None:
    result = valid_result()
    duplicate_finding = deepcopy(result["findings"][0])
    duplicate_finding["finding_id"] = "finding-2"
    result["findings"].append(duplicate_finding)
    assert validate_window_result(result, ["image-000123"]) == []


def test_evidence_role_must_match_collection() -> None:
    result = valid_result()
    result["findings"][0]["traceable_evidence"][0]["role"] = "counter"
    errors = validate_window_result(result, ["image-000123"])
    assert any("role must be supporting" in error for error in errors)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.update(findings={}),
        lambda result: result.update(findings=[None]),
        lambda result: result.update(treatment_boundary=[]),
        lambda result: result["findings"][0].update(vascular_zone_estimate=[]),
        lambda result: result["findings"][0].update(traceable_evidence=[None]),
    ],
)
def test_semantic_validator_never_raises_for_malformed_nested_values(mutation) -> None:
    result = valid_result()
    mutation(result)
    errors = validate_window_result(result, ["image-000123"])
    assert errors
