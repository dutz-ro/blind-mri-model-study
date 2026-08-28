import importlib.util
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from tests.test_clinical_schema import valid_result

CONTRACT_PATH = (
    Path(__file__).parents[1] / "deploy/scaleway/medgemma/contracts.py"
)
sys.path.insert(0, str(CONTRACT_PATH.parent))
OUTPUT_CONTRACT_PATH = (
    Path(__file__).parents[1] / "src/mristudy/medgemma/output_contract.py"
)
OUTPUT_SPEC = importlib.util.spec_from_file_location("output_contract", OUTPUT_CONTRACT_PATH)
assert OUTPUT_SPEC and OUTPUT_SPEC.loader
OUTPUT_MODULE = importlib.util.module_from_spec(OUTPUT_SPEC)
sys.modules["output_contract"] = OUTPUT_MODULE
OUTPUT_SPEC.loader.exec_module(OUTPUT_MODULE)
SPEC = spec_from_file_location("medgemma_runtime_contracts", CONTRACT_PATH)
assert SPEC and SPEC.loader
CONTRACTS = module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACTS)


def approved() -> dict:
    return {
        "study_uid": "1.2.3",
        "series_uid": "1.2.4",
        "role": "primary-volume",
        "window_id": "primary-volume-window-0001",
        "instance_uids": ["1.2.5", "1.2.6"],
        "evidence_ids": ["image-000001", "image-000002"],
        "prompt_sha256": "prompt",
        "output_schema": "knee-meniscus-finding-v2",
        "max_new_tokens": 4096,
        "generation": {
            "do_sample": False,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 8,
            "json_schema_enforced": True,
        },
        "decoder": OUTPUT_MODULE.decoder_contract(
            __import__("mristudy.clinical_schema", fromlist=["load_knee_meniscus_schema"])
            .load_knee_meniscus_schema()
        ),
    }


def test_runtime_contract_accepts_exact_window() -> None:
    assert CONTRACTS.validate_window_payload(dict(approved()), approved()) == []


def test_runtime_contract_rejects_unapproved_uid_and_order() -> None:
    payload = dict(approved())
    payload["instance_uids"] = ["1.2.6", "1.2.999"]
    errors = CONTRACTS.validate_window_payload(payload, approved())
    assert "instance_uids differs from approved window" in errors


def test_runtime_contract_rejects_duplicate_and_oversized_lists() -> None:
    payload = dict(approved())
    payload["instance_uids"] = ["1.2.5"] * 33
    payload["evidence_ids"] = ["image-000001"] * 33
    errors = CONTRACTS.validate_window_payload(payload, approved())
    assert "more than 32 instances" in errors
    assert "duplicate instance UID" in errors
    assert "duplicate evidence ID" in errors


def test_runtime_contract_rejects_changed_generation_policy() -> None:
    payload = dict(approved())
    payload["generation"] = dict(payload["generation"])
    payload["generation"]["repetition_penalty"] = 1.0
    errors = CONTRACTS.validate_window_payload(payload, approved())
    assert "generation differs from approved window" in errors


def test_runtime_contract_rejects_changed_decoder_policy() -> None:
    payload = dict(approved())
    payload["decoder"] = dict(payload["decoder"])
    payload["decoder"]["version"] = "wrong"
    errors = CONTRACTS.validate_window_payload(payload, approved())
    assert "decoder differs from approved window" in errors


def test_runtime_contract_separates_evidence_scope_from_boundary_violation() -> None:
    result = valid_result()
    result["findings"][0]["traceable_evidence"][0]["export_id"] = "image-999999"
    result["treatment_boundary"]["repairability_determined"] = True
    errors = CONTRACTS.validate_result_evidence(result, ["image-000001"])
    assert "evidence outside approved window" not in errors
    assert "repairability determined" in errors


def test_runtime_contract_accepts_well_formed_out_of_window_evidence() -> None:
    result = valid_result()
    result["findings"][0]["traceable_evidence"][0]["export_id"] = "image-999999"
    assert CONTRACTS.validate_result_evidence(result, ["image-000001"]) == []


def test_structural_normalization_is_conservative_and_audited() -> None:
    result = valid_result()
    finding = result["findings"][0]
    finding["segment"] = "entire"
    del finding["confidence"]
    normalized, actions = CONTRACTS.normalize_structural_result(result)
    assert normalized["findings"][0]["segment"] == "multiple_segments"
    assert normalized["findings"][0]["confidence"] == "low"
    assert result["findings"][0]["segment"] == "entire"
    assert "confidence" not in result["findings"][0]
    assert actions == [
        {
            "rule_id": "segment-exact-alias-v1",
            "path": "findings.0.segment",
            "operation": "enum_alias",
            "old_value": "entire",
            "new_value": "multiple_segments",
        },
        {
            "rule_id": "missing-confidence-low-v1",
            "path": "findings.0.confidence",
            "operation": "missing_conservative_default",
            "old_value": None,
            "new_value": "low",
        },
    ]


def test_semantic_validator_returns_errors_for_malformed_nested_values() -> None:
    result = valid_result()
    result["findings"][0]["vascular_zone_estimate"] = "indeterminate"
    errors = CONTRACTS.validate_result_evidence(result, ["image-000123"])
    assert "vascular_zone_estimate must be an object" in errors


def test_shorthand_nested_objects_expand_conservatively() -> None:
    result = valid_result()
    finding = result["findings"][0]
    finding["segment"] = "posterior_horn_body"
    finding["vascular_zone_estimate"] = "indeterminate"
    finding["capsular_relation"] = "displaced"
    finding["surface_extension"] = "indeterminate"
    finding["stability_features"] = "displaced"
    finding["healing_context"] = "degenerative_appearing"
    normalized, actions = CONTRACTS.normalize_structural_result(result)
    normalized_finding = normalized["findings"][0]
    assert normalized_finding["segment"] == "multiple_segments"
    assert normalized_finding["vascular_zone_estimate"]["zone"] == "indeterminate"
    assert normalized_finding["capsular_relation"]["measurement_method"] == "not_measurable"
    assert normalized_finding["surface_extension"]["superior"] == "indeterminate"
    assert normalized_finding["stability_features"]["displaced_fragment"] == "present"
    assert (
        normalized_finding["healing_context"]["tissue_quality_imaging_estimate"]
        == "degenerative_appearing"
    )
    assert len(actions) == 6


def test_empty_treatment_boundary_restores_only_required_false_constants() -> None:
    result = valid_result()
    result["treatment_boundary"] = {}
    normalized, actions = CONTRACTS.normalize_structural_result(result)
    assert normalized["treatment_boundary"] == {
        "treatment_recommendation_provided": False,
        "repairability_determined": False,
        "vascularity_directly_measured": False,
    }
    boundary_actions = [
        item for item in actions if item["path"].startswith("treatment_boundary.")
    ]
    assert len(boundary_actions) == 3
    assert all(item["operation"] == "missing_contract_constant" for item in boundary_actions)
