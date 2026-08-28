from __future__ import annotations

import json
import random
from copy import deepcopy

import pytest

from mristudy.clinical_schema import load_knee_meniscus_schema
from mristudy.medgemma.output_contract import (
    classify_output,
    decoder_schema,
    enumerate_json_candidates,
    normalize_structural_result,
    strict_json_loads,
    structural_correction_errors,
    verify_normalization,
)
from tests.test_clinical_schema import valid_result


def serialized(value: dict) -> str:
    return json.dumps(value, separators=(",", ":"))


@pytest.mark.parametrize(
    "wrapper",
    [
        lambda value: value,
        lambda value: f"```json\n{value}\n```",
        lambda value: f"info\n```json\n{value}\n```",
        lambda value: f"prose before\n```json\n{value}\n```\nprose after",
        lambda value: f"prose before {value} prose after",
    ],
)
def test_candidate_extraction_recovers_complete_json(wrapper) -> None:
    text = wrapper(serialized(valid_result()))
    decision = classify_output(text, load_knee_meniscus_schema(), ["image-000123"])
    assert decision.status == "exact_valid"
    assert decision.result == valid_result()


def test_scanner_handles_braces_and_escapes_inside_strings() -> None:
    result = valid_result()
    result["global_limitations"] = ['braces { } and quote " and slash \\']
    candidates = enumerate_json_candidates(f"prefix {serialized(result)} suffix")
    assert len(candidates) == 1
    assert strict_json_loads(candidates[0].text) == result


@pytest.mark.parametrize(
    "text",
    [
        "prose only",
        "```json\n{\"incomplete\": true",
        "```json\n\n```",
        "[]",
        '{"ok":NaN}',
        '{"ok":true,"ok":false}',
    ],
)
def test_invalid_outputs_fail_closed(text: str) -> None:
    decision = classify_output(text, load_knee_meniscus_schema(), ["image-000123"])
    assert decision.result is None
    assert decision.status == "unrecoverable"


def test_distinct_valid_results_are_ambiguous() -> None:
    first = valid_result()
    second = deepcopy(first)
    second["global_limitations"] = ["different"]
    text = f"```json\n{serialized(first)}\n```\n```json\n{serialized(second)}\n```"
    decision = classify_output(text, load_knee_meniscus_schema(), ["image-000123"])
    assert decision.status == "ambiguous_multiple_valid_results"
    assert decision.result is None


def test_identical_valid_results_are_not_ambiguous() -> None:
    value = serialized(valid_result())
    decision = classify_output(
        f"```json\n{value}\n```\n```json\n{value}\n```",
        load_knee_meniscus_schema(),
        ["image-000123"],
    )
    assert decision.status == "exact_valid"


def test_out_of_window_evidence_is_canonically_valid() -> None:
    """Evidence scope is graded later and must not trigger a model correction."""
    result = valid_result()
    result["findings"][0]["traceable_evidence"][0]["export_id"] = "image-000070"
    decision = classify_output(
        serialized(result),
        load_knee_meniscus_schema(),
        ["image-000071"],
    )
    assert decision.status == "exact_valid"
    assert decision.result == result
    assert decision.validation_policy == "medgemma-output-validation-v2"
    assert structural_correction_errors(decision) == []


def test_unknown_well_formed_evidence_is_canonically_valid() -> None:
    result = valid_result()
    result["findings"][0]["traceable_evidence"][0]["export_id"] = "image-999999"
    decision = classify_output(
        serialized(result),
        load_knee_meniscus_schema(),
        ["image-000123"],
    )
    assert decision.status == "exact_valid"


def test_malformed_explicit_evidence_remains_canonically_invalid() -> None:
    result = valid_result()
    result["findings"][0]["traceable_evidence"][0]["export_id"] = "image-070"
    decision = classify_output(
        serialized(result),
        load_knee_meniscus_schema(),
        ["image-000123"],
    )
    assert decision.status == "unrecoverable"
    assert decision.result is None
    assert structural_correction_errors(decision)


def test_normalization_is_idempotent_audited_and_does_not_mutate_source() -> None:
    source = valid_result()
    source["findings"][0]["segment"] = "posterior_horn_body"
    source["findings"][0]["vascular_zone_estimate"] = "indeterminate"
    source["treatment_boundary"] = {}
    original = deepcopy(source)
    normalized, actions = normalize_structural_result(source)
    assert source == original
    assert not verify_normalization(source, normalized, actions)
    second, second_actions = normalize_structural_result(normalized)
    assert second == normalized
    assert second_actions == []


@pytest.mark.parametrize(
    "value",
    ["all", "diffuse", "posterior horn and body", "anterior_horn_body", "", None, 1],
)
def test_unknown_segment_aliases_are_not_normalized(value) -> None:
    source = valid_result()
    source["findings"][0]["segment"] = value
    normalized, actions = normalize_structural_result(source)
    assert normalized["findings"][0]["segment"] == value
    assert not [item for item in actions if item["path"].endswith("segment")]


def test_present_boundary_violation_is_never_masked() -> None:
    source = valid_result()
    source["treatment_boundary"]["repairability_determined"] = True
    normalized, actions = normalize_structural_result(source)
    assert normalized["treatment_boundary"]["repairability_determined"] is True
    assert not actions


def test_normalizer_never_changes_evidence_or_measurements() -> None:
    source = valid_result()
    source["findings"][0]["segment"] = "entire"
    before_evidence = deepcopy(source["findings"][0]["traceable_evidence"])
    before_distance = source["findings"][0]["capsular_relation"]["distance_mm_estimate"]
    normalized, _ = normalize_structural_result(source)
    assert normalized["findings"][0]["traceable_evidence"] == before_evidence
    assert (
        normalized["findings"][0]["capsular_relation"]["distance_mm_estimate"]
        == before_distance
    )


def test_seeded_extraction_fuzz_is_deterministic_and_never_raises() -> None:
    rng = random.Random(20260827)
    schema = load_knee_meniscus_schema()
    value = serialized(valid_result())
    wrappers = [
        lambda text: text,
        lambda text: f"```json\n{text}\n```",
        lambda text: f"prefix {text} suffix",
        lambda text: f"prefix ```json\n{text}\n``` suffix",
    ]
    for _ in range(1000):
        text = rng.choice(wrappers)(value)
        if rng.random() < 0.25:
            text = text[: rng.randrange(len(text) + 1)]
        first = classify_output(text, schema, ["image-000123"])
        second = classify_output(text, schema, ["image-000123"])
        assert first == second


def test_decoder_schema_relaxes_boolean_consts_only_for_lmfe() -> None:
    canonical = load_knee_meniscus_schema()
    constrained = decoder_schema(canonical)
    boundary = constrained["properties"]["treatment_boundary"]["properties"]
    assert boundary["repairability_determined"] == {"type": "boolean"}
    canonical_boundary = canonical["properties"]["treatment_boundary"]["properties"]
    assert canonical_boundary["repairability_determined"] == {"const": False}


@pytest.mark.parametrize(
    "mutator",
    [
        lambda finding: finding.update(meniscus="lateral"),
        lambda finding: finding.update(tear_status="absent"),
        lambda finding: finding.update(tear_pattern="radial"),
        lambda finding: finding.update(orientation="oblique"),
        lambda finding: finding["capsular_relation"].update(distance_mm_estimate=99.0),
        lambda finding: finding["healing_context"].update(extrusion_mm_estimate=99.0),
        lambda finding: finding["traceable_evidence"][0].update(export_id="image-999999"),
    ],
)
def test_unexplained_clinical_mutations_are_rejected_by_preservation(mutator) -> None:
    before = valid_result()
    after = deepcopy(before)
    mutator(after["findings"][0])
    errors = verify_normalization(before, after, [])
    assert errors


def test_normalizer_does_not_drop_extra_properties() -> None:
    source = valid_result()
    source["findings"][0]["unexpected"] = "preserve-invalid-field"
    normalized, _ = normalize_structural_result(source)
    assert normalized["findings"][0]["unexpected"] == "preserve-invalid-field"


def test_invalid_present_confidence_is_not_replaced() -> None:
    source = valid_result()
    source["findings"][0]["confidence"] = "certain"
    normalized, actions = normalize_structural_result(source)
    assert normalized["findings"][0]["confidence"] == "certain"
    assert not [item for item in actions if item["path"].endswith("confidence")]


def test_non_object_boundary_is_not_synthesized() -> None:
    source = valid_result()
    source["treatment_boundary"] = []
    normalized, actions = normalize_structural_result(source)
    assert normalized["treatment_boundary"] == []
    assert not actions
