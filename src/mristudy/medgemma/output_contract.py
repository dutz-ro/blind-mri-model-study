from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from jsonschema import Draft202012Validator

EXTRACTION_POLICY = "medgemma-json-extraction-v1"
NORMALIZATION_POLICY = "medgemma-normalization-v2"
MAX_OUTPUT_CHARACTERS = 1_000_000
MAX_JSON_DEPTH = 128
DECODER_POLICY = "medgemma-lmfe-decoder-v1"
DECODER_IMPLEMENTATION = "lm-format-enforcer"
DECODER_VERSION = "0.11.3"
DECODER_CONFIG = {
    "force_json_field_order": True,
    "max_json_array_length": 128,
    "max_consecutive_whitespaces": 12,
}

SEGMENT_ALIASES = {
    "entire": "multiple_segments",
    "entire_meniscus": "multiple_segments",
    "whole": "multiple_segments",
    "posterior_horn_body": "multiple_segments",
}


@dataclass(frozen=True)
class JsonCandidate:
    start: int
    end: int
    source: str
    text: str


@dataclass(frozen=True)
class NormalizationAction:
    rule_id: str
    path: str
    operation: str
    old_value: Any
    new_value: Any


@dataclass(frozen=True)
class CandidateDecision:
    index: int
    source: str
    status: str
    candidate_sha256: str
    semantic_sha256: str | None
    validation_errors: tuple[str, ...]
    actions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ExtractionDecision:
    extraction_policy: str
    normalization_policy: str
    status: str
    candidate_count: int
    selected_candidate_index: int | None
    selected_candidate_sha256: str | None
    selected_semantic_sha256: str | None
    result: dict[str, Any] | None
    candidates: tuple[dict[str, Any], ...]


class DuplicateJsonKeyError(ValueError):
    pass


def decoder_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return an LMFE-compatible structural schema; canonical validation remains authoritative."""
    def convert(value: Any) -> Any:
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value
        converted = {key: convert(item) for key, item in value.items() if key != "$schema"}
        constant = converted.get("const")
        if isinstance(constant, bool):
            converted.pop("const")
            converted["type"] = "boolean"
        return converted

    return convert(schema)


def policy_digest(policy: str, payload: Any) -> str:
    return _digest({"policy": policy, "payload": payload})


def decoder_contract(schema: dict[str, Any]) -> dict[str, Any]:
    constrained_schema = decoder_schema(schema)
    payload = {
        "implementation": DECODER_IMPLEMENTATION,
        "version": DECODER_VERSION,
        "policy": DECODER_POLICY,
        "config": DECODER_CONFIG,
        "schema": constrained_schema,
    }
    return {
        "implementation": DECODER_IMPLEMENTATION,
        "version": DECODER_VERSION,
        "policy": DECODER_POLICY,
        "config": dict(DECODER_CONFIG),
        "policy_sha256": policy_digest(DECODER_POLICY, payload),
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def _digest(value: Any) -> str:
    data = value.encode() if isinstance(value, str) else _canonical_bytes(value)
    return sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number: {value}")

    value = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("The JSON root must be an object.")
    return value


def _scan_objects(text: str) -> list[JsonCandidate]:
    candidates = []
    start = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if start is None:
            if character == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                start = None
                depth = 0
        elif character in "]}":
            depth -= 1
            if depth == 0:
                candidates.append(
                    JsonCandidate(start, index + 1, "unfenced", text[start : index + 1])
                )
                start = None
            elif depth < 0:
                start = None
                depth = 0
    return candidates


def enumerate_json_candidates(text: str) -> list[JsonCandidate]:
    if len(text) > MAX_OUTPUT_CHARACTERS:
        raise ValueError("Model output exceeds the replay size limit.")
    candidates = _scan_objects(text)
    fenced_ranges = [
        (match.start(1), match.end(1))
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    ]
    result = []
    seen = set()
    for candidate in candidates:
        source = "fenced" if any(
            start <= candidate.start and candidate.end <= end for start, end in fenced_ranges
        ) else candidate.source
        key = (candidate.start, candidate.end)
        if key not in seen:
            result.append(JsonCandidate(candidate.start, candidate.end, source, candidate.text))
            seen.add(key)
    return result


def _action(
    actions: list[NormalizationAction],
    rule_id: str,
    path: str,
    operation: str,
    old_value: Any,
    new_value: Any,
) -> None:
    actions.append(NormalizationAction(rule_id, path, operation, old_value, deepcopy(new_value)))


def normalize_structural_result(
    result: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(result, dict):
        raise ValueError("Result root must be an object.")
    normalized = deepcopy(result)
    actions: list[NormalizationAction] = []
    findings = normalized.get("findings")
    if findings is not None and not isinstance(findings, list):
        return normalized, []
    for index, finding in enumerate(findings or []):
        if not isinstance(finding, dict):
            continue
        segment = finding.get("segment")
        if segment in SEGMENT_ALIASES:
            replacement = SEGMENT_ALIASES[segment]
            finding["segment"] = replacement
            _action(
                actions,
                "segment-exact-alias-v1",
                f"findings.{index}.segment",
                "enum_alias",
                segment,
                replacement,
            )
        if "confidence" not in finding:
            finding["confidence"] = "low"
            _action(
                actions,
                "missing-confidence-low-v1",
                f"findings.{index}.confidence",
                "missing_conservative_default",
                None,
                "low",
            )
        _expand_shorthand_objects(finding, index, actions)
    boundary = normalized.get("treatment_boundary")
    if isinstance(boundary, dict):
        for key in (
            "treatment_recommendation_provided",
            "repairability_determined",
            "vascularity_directly_measured",
        ):
            if key not in boundary:
                boundary[key] = False
                _action(
                    actions,
                    "missing-boundary-constant-v1",
                    f"treatment_boundary.{key}",
                    "missing_contract_constant",
                    None,
                    False,
                )
    return normalized, [asdict(action) for action in actions]


def _expand_shorthand_objects(
    finding: dict[str, Any],
    index: int,
    actions: list[NormalizationAction],
) -> None:
    expansions = {
        "vascular_zone_estimate": lambda source: {
            "zone": source
            if source in {"red_red", "red_white", "white_white", "junctional", "indeterminate"}
            else "indeterminate",
            "directly_observed_vascularity": False,
            "basis": "insufficient_anatomic_resolution",
            "confidence": "low",
            "limitations": ["Expanded from model shorthand; anatomical detail unavailable."],
        },
        "capsular_relation": lambda _source: {
            "distance_mm_estimate": None,
            "measurement_method": "not_measurable",
            "peripheral_extension": "indeterminate",
            "meniscocapsular_separation": "indeterminate",
        },
        "surface_extension": lambda _source: {
            "superior": "indeterminate",
            "inferior": "indeterminate",
            "free_edge": "indeterminate",
        },
        "stability_features": lambda source: {
            "displaced_fragment": "present" if source == "displaced" else "indeterminate",
            "flap": "indeterminate",
            "bucket_handle_pattern": "indeterminate",
            "imaging_signs_of_mechanical_obstruction": "indeterminate",
        },
        "healing_context": lambda source: {
            "tissue_quality_imaging_estimate": source
            if source in {"preserved_appearing", "degenerative_appearing", "indeterminate"}
            else "indeterminate",
            "chronicity_imaging_estimate": "indeterminate",
            "extrusion_mm_estimate": None,
            "parameniscal_cyst": "indeterminate",
            "root_involvement": "indeterminate",
        },
    }
    for key, expand in expansions.items():
        source = finding.get(key)
        if isinstance(source, str):
            replacement = expand(source)
            finding[key] = replacement
            _action(
                actions,
                f"{key}-shorthand-v1",
                f"findings.{index}.{key}",
                "expand_shorthand_object",
                source,
                replacement,
            )


def _diff(before: Any, after: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    if type(before) is not type(after):
        return [(path, before, after)]
    if isinstance(before, dict):
        changes = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else key
            if key not in before:
                changes.append((child, None, after[key]))
            elif key not in after:
                changes.append((child, before[key], None))
            else:
                changes.extend(_diff(before[key], after[key], child))
        return changes
    if isinstance(before, list):
        if len(before) != len(after):
            return [(path, before, after)]
        changes = []
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            changes.extend(_diff(left, right, f"{path}.{index}"))
        return changes
    return [] if before == after else [(path, before, after)]


def verify_normalization(
    before: dict[str, Any],
    after: dict[str, Any],
    actions: list[dict[str, Any]],
) -> list[str]:
    errors = []
    changes = _diff(before, after)
    action_paths = [item["path"] for item in actions]
    if len(action_paths) != len(set(action_paths)):
        errors.append("Normalization writes the same path more than once.")
    for path, old, new in changes:
        matches = [item for item in actions if item["path"] == path]
        if len(matches) != 1:
            errors.append(f"Unexplained normalization change: {path}")
            continue
        action = matches[0]
        if action["old_value"] != old or action["new_value"] != new:
            errors.append(f"Normalization action does not match the diff: {path}")
    for action in actions:
        if not any(path == action["path"] for path, _, _ in changes):
            errors.append(f"Normalization action has no matching diff: {action['path']}")
    second, second_actions = normalize_structural_result(after)
    if second != after or second_actions:
        errors.append("Normalization is not idempotent.")
    return errors


def validate_result(
    result: dict[str, Any],
    schema: dict[str, Any],
    allowed_evidence_ids: list[str],
) -> list[str]:
    errors = [
        error.message
        for error in sorted(
            Draft202012Validator(schema).iter_errors(result),
            key=lambda item: list(item.absolute_path),
        )
    ]
    errors.extend(validate_result_evidence(result, allowed_evidence_ids))
    return errors


def validate_result_evidence(
    result: dict[str, Any],
    allowed_evidence_ids: list[str],
) -> list[str]:
    if not isinstance(result, dict):
        return ["result must be an object"]
    errors = []
    allowed = set(allowed_evidence_ids)
    findings = result.get("findings") or []
    if not isinstance(findings, list):
        return ["findings must be an array"]
    for finding in findings:
        if not isinstance(finding, dict):
            errors.append("finding must be an object")
            continue
        supporting = finding.get("traceable_evidence") or []
        counter = finding.get("counter_evidence") or []
        if not isinstance(supporting, list):
            errors.append("traceable_evidence must be an array")
            supporting = []
        if not isinstance(counter, list):
            errors.append("counter_evidence must be an array")
            counter = []
        supporting_objects = [item for item in supporting if isinstance(item, dict)]
        counter_objects = [item for item in counter if isinstance(item, dict)]
        if len(supporting_objects) != len(supporting):
            errors.append("traceable_evidence entry must be an object")
        if len(counter_objects) != len(counter):
            errors.append("counter_evidence entry must be an object")
        supporting_ids = [item.get("export_id") for item in supporting_objects]
        counter_ids = [item.get("export_id") for item in counter_objects]
        if any(item not in allowed for item in supporting_ids + counter_ids):
            errors.append("evidence outside approved window")
        if len(supporting_ids) != len(set(supporting_ids)):
            errors.append("duplicate supporting evidence")
        if len(counter_ids) != len(set(counter_ids)):
            errors.append("duplicate counter evidence")
        if set(supporting_ids) & set(counter_ids):
            errors.append("evidence is both supporting and counter")
        if any(item.get("role") != "supporting" for item in supporting_objects):
            errors.append("invalid supporting evidence role")
        if any(item.get("role") != "counter" for item in counter_objects):
            errors.append("invalid counter evidence role")
        vascular = finding.get("vascular_zone_estimate")
        if not isinstance(vascular, dict):
            errors.append("vascular_zone_estimate must be an object")
        elif vascular.get("directly_observed_vascularity") is not False:
            errors.append("direct vascularity claim")
    boundary = result.get("treatment_boundary")
    if not isinstance(boundary, dict):
        return errors + ["treatment_boundary must be an object"]
    if boundary.get("treatment_recommendation_provided") is not False:
        errors.append("treatment recommendation provided")
    if boundary.get("repairability_determined") is not False:
        errors.append("repairability determined")
    if boundary.get("vascularity_directly_measured") is not False:
        errors.append("vascularity directly measured")
    return errors


def classify_output(
    text: str,
    schema: dict[str, Any],
    allowed_evidence_ids: list[str],
) -> ExtractionDecision:
    candidates = enumerate_json_candidates(text)
    decisions = []
    valid = []
    for index, candidate in enumerate(candidates):
        candidate_digest = _digest(candidate.text)
        try:
            parsed = strict_json_loads(candidate.text)
            exact_errors = validate_result(parsed, schema, allowed_evidence_ids)
            if not exact_errors:
                normalized = parsed
                actions = []
                status = "exact_valid"
                errors = []
            else:
                normalized, actions = normalize_structural_result(parsed)
                preservation_errors = verify_normalization(parsed, normalized, actions)
                errors = preservation_errors + validate_result(
                    normalized, schema, allowed_evidence_ids
                )
                status = "normalizable_valid" if not errors else "schema_invalid"
            semantic_digest = (
                _digest(normalized)
                if status in {"exact_valid", "normalizable_valid"}
                else None
            )
        except (ValueError, TypeError, KeyError) as error:
            normalized = None
            actions = []
            errors = [str(error)]
            status = "parse_invalid"
            semantic_digest = None
        decision = CandidateDecision(
            index,
            candidate.source,
            status,
            candidate_digest,
            semantic_digest,
            tuple(errors),
            tuple(actions),
        )
        decisions.append(decision)
        if status in {"exact_valid", "normalizable_valid"}:
            valid.append((index, candidate, normalized, decision))
    semantic_digests = {item[3].semantic_sha256 for item in valid}
    if not valid:
        status = "unrecoverable"
        selected = None
    elif len(semantic_digests) > 1:
        status = "ambiguous_multiple_valid_results"
        selected = None
    else:
        exact = [item for item in valid if item[3].status == "exact_valid"]
        selected = (exact or valid)[-1]
        status = selected[3].status
    return ExtractionDecision(
        EXTRACTION_POLICY,
        NORMALIZATION_POLICY,
        status,
        len(candidates),
        selected[0] if selected else None,
        selected[3].candidate_sha256 if selected else None,
        selected[3].semantic_sha256 if selected else None,
        selected[2] if selected else None,
        tuple(asdict(item) for item in decisions),
    )
