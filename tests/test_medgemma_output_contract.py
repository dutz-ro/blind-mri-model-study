from __future__ import annotations

import json
import random
from copy import deepcopy

import pytest

from mristudy.clinical_schema import load_knee_meniscus_schema
from mristudy.medgemma.output_contract import (
    ARRAY_BOUNDS,
    CORRECTION_POLICY,
    DECODER_POLICY,
    MAX_FREE_TEXT_CHARACTERS,
    MAX_NEW_TOKENS,
    OUTPUT_BUDGET_MARGIN_TOKENS,
    PROVED_MAXIMUM_STRUCTURAL_CHARACTERS,
    SCHEMA_ARRAY_BOUNDS,
    classify_output,
    decoder_acceptance,
    decoder_output_bounds,
    decoder_schema,
    enumerate_json_candidates,
    generation_policy,
    maximum_structural_output,
    normalize_structural_result,
    strict_json_loads,
    structural_correction_errors,
    structural_correction_prompt,
    validate_result,
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
    assert decision.validation_policy == "medgemma-output-validation-v3"
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


def test_incomplete_json_is_classified_as_truncated() -> None:
    decision = classify_output(
        '{"schema_version":"knee-meniscus-finding-v2","findings":[',
        load_knee_meniscus_schema(),
    )
    assert decision.status == "truncated_json"
    assert decision.result is None
    assert structural_correction_errors(decision) == ["truncated_json"]


def test_structural_correction_prompt_is_compact_and_does_not_repeat_schema() -> None:
    previous = '{"schema_version":"knee-meniscus-finding-v2","findings":['
    prompt = structural_correction_prompt(previous, 1, ["truncated_json"])
    assert previous in prompt
    assert "truncated_json" in prompt
    assert "Required JSON Schema" not in prompt
    assert '"$defs"' not in prompt
    assert "Do not inspect or reinterpret images" in prompt
    assert "Do not add a clinical finding" in prompt
    assert CORRECTION_POLICY == "medgemma-structural-correction-v2"


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


def test_decoder_schema_enumerates_only_approved_evidence_ids() -> None:
    canonical = load_knee_meniscus_schema()
    constrained = decoder_schema(canonical, ["image-000123", "image-000124"])
    evidence = constrained["$defs"]["evidence"]["properties"]["export_id"]
    assert evidence == {"type": "string", "enum": ["image-000123", "image-000124"]}
    canonical_evidence = canonical["$defs"]["evidence"]["properties"]["export_id"]
    assert canonical_evidence["pattern"] == "^image-[0-9]{6}$"


def test_decoder_replay_accepts_valid_result_and_rejects_impossible_character() -> None:
    pytest.importorskip("lmformatenforcer")
    schema = load_knee_meniscus_schema()
    text = json.dumps(valid_result())
    accepted = decoder_acceptance(text, schema, ["image-000123"])
    assert accepted["accepted"] is True
    mutated = text.replace("image-000123", "image-999999")
    rejected = decoder_acceptance(mutated, schema, ["image-000123"])
    assert rejected["accepted"] is False
    assert rejected["reason"] == "character_outside_decoder_mask"


def test_generation_policy_neutralizes_repetition_processors() -> None:
    policy = generation_policy()
    assert policy["repetition_penalty"] == 1.0
    assert policy["no_repeat_ngram_size"] == 0
    assert policy["evidence_ids_enumerated"] is True
    assert DECODER_POLICY == "medgemma-lmfe-decoder-v4"


def test_absent_window_findings_are_rejected_as_non_focal() -> None:
    result = valid_result()
    finding = result["findings"][0]
    finding["tear_status"] = "absent"
    finding["tear_pattern"] = "none"
    finding["orientation"] = "not_applicable"
    decision = classify_output(
        serialized(result), load_knee_meniscus_schema(), ["image-000123"]
    )
    assert decision.result is None
    assert any(
        "absent finding must be omitted"
        in error
        for candidate in decision.candidates
        for error in candidate["validation_errors"]
    )


def test_one_finding_per_image_repetition_is_rejected() -> None:
    result = valid_result()
    duplicate = deepcopy(result["findings"][0])
    duplicate["finding_id"] = "finding-2"
    duplicate["traceable_evidence"][0]["export_id"] = "image-000124"
    result["findings"].append(duplicate)
    decision = classify_output(
        serialized(result),
        load_knee_meniscus_schema(),
        ["image-000123", "image-000124"],
    )
    assert decision.result is None
    assert any(
        "repeated anatomical claim"
        in error
        for candidate in decision.candidates
        for error in candidate["validation_errors"]
    )


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


def _walk(node, path=""):
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            yield from _walk(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}/{index}")


def test_decoder_schema_bounds_every_array_and_free_text_string() -> None:
    """An unbounded schema makes any token budget a guess rather than a limit.

    The canonical schema constrains no array length and no string length, so the
    decoder could legally emit an arbitrarily large object. A real window used
    that freedom to repeat one absent finding until the budget ran out.
    """
    canonical = load_knee_meniscus_schema()
    constrained = decoder_schema(canonical, ["image-000001"])
    unbounded_arrays = [
        path
        for path, node in _walk(constrained)
        if node.get("type") == "array" and "maxItems" not in node
    ]
    unbounded_strings = [
        path
        for path, node in _walk(constrained)
        if node.get("type") == "string"
        and "maxLength" not in node
        and not {"enum", "const", "pattern"} & set(node)
    ]
    assert not unbounded_arrays
    assert not unbounded_strings
    assert constrained["properties"]["findings"]["maxItems"] == (
        SCHEMA_ARRAY_BOUNDS["knee-meniscus-finding-v2"]["findings"]
    )
    assert (
        constrained["$defs"]["evidence"]["properties"]["description"]["maxLength"]
        == MAX_FREE_TEXT_CHARACTERS
    )


def test_canonical_schema_stays_unbounded_so_bounds_cannot_invalidate_output() -> None:
    canonical = load_knee_meniscus_schema()
    decoder_schema(canonical, ["image-000001"])
    assert "maxItems" not in canonical["properties"]["findings"]
    assert canonical == load_knee_meniscus_schema()


def test_maximum_structural_output_is_canonically_valid_and_finite() -> None:
    """The output the budget is proved against must itself be a legal result."""
    canonical = load_knee_meniscus_schema()
    evidence_ids = [f"image-{index + 1:06d}" for index in range(32)]
    maximum = maximum_structural_output(canonical, evidence_ids)
    assert validate_result(maximum, canonical) == []
    assert len(maximum["findings"]) == (
        SCHEMA_ARRAY_BOUNDS["knee-meniscus-finding-v2"]["findings"]
    )
    assert len(maximum["findings"][0]["traceable_evidence"]) == (
        ARRAY_BOUNDS["traceable_evidence"]
    )
    cited = [item["export_id"] for item in maximum["findings"][0]["traceable_evidence"]]
    assert len(cited) == len(set(cited))
    assert maximum == maximum_structural_output(canonical, evidence_ids)


def test_output_budget_leaves_margin_for_the_largest_decodable_result() -> None:
    """Characters are checked here; the runtime proves tokens with its tokenizer."""
    canonical = load_knee_meniscus_schema()
    maximum = maximum_structural_output(
        canonical, [f"image-{index + 1:06d}" for index in range(32)]
    )
    serialized = json.dumps(maximum, indent=2)
    assert OUTPUT_BUDGET_MARGIN_TOKENS > 0
    assert MAX_NEW_TOKENS > OUTPUT_BUDGET_MARGIN_TOKENS
    assert len(serialized) <= PROVED_MAXIMUM_STRUCTURAL_CHARACTERS
    assert decoder_output_bounds()["policy"] == "medgemma-decoder-output-bounds-v2"
