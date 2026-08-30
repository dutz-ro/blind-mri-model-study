from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from jsonschema import Draft202012Validator

EXTRACTION_POLICY = "medgemma-json-extraction-v2"
NORMALIZATION_POLICY = "medgemma-normalization-v2"
OUTPUT_VALIDATION_POLICY = "medgemma-output-validation-v2"
CORRECTION_POLICY = "medgemma-structural-correction-v2"
MAX_OUTPUT_CHARACTERS = 1_000_000
MAX_JSON_DEPTH = 128
DECODER_POLICY = "medgemma-lmfe-decoder-v3"
DECODER_IMPLEMENTATION = "lm-format-enforcer"
DECODER_VERSION = "0.11.3"
DECODER_CONFIG = {
    "force_json_field_order": True,
    "max_json_array_length": 128,
    "max_consecutive_whitespaces": 12,
}
GENERATION_POLICY = "medgemma-constrained-greedy-v3"
EVIDENCE_ID_PATTERN = r"^image-[0-9]{6}$"

OUTPUT_BOUNDS_POLICY = "medgemma-decoder-output-bounds-v1"
# The canonical schema deliberately bounds no array and no free-text string: it
# describes what a finding *is*, not how much of it one window may contain. That
# left the decodable output infinite, so no token budget could ever be a
# guarantee. A real medical window then emitted the same "absent, medial body,
# no tear" finding eight times, consumed the whole budget, and the window failed
# after three byte-identical attempts. Greedy decoding under a schema mask has no
# other stopping force, because suppressing repetition through the sampler is
# what broke constrained decoding in the first place.
#
# These bounds are applied only to the derived structural schema used to build
# the token mask. The canonical schema stays byte-identical and remains the
# authoritative validator, so a bounded output is always canonically valid; the
# bounds can only prevent output, never invalidate it. Repeated identical claims
# are merged by series integration anyway, so a per-window cap costs no clinical
# content.
# Every bound sits above the widest value observed across the 32 preserved
# structured window results, so the limits cut off runaway repetition rather
# than legitimate reporting. The observed maxima were: 65 free-text characters,
# 9 supporting citations, 2 counter citations, 5 finding limitations, 2 review
# factors and 8 global limitations. Findings peaked at 8, but that single case
# was the degenerate repetition this policy exists to stop; the 95th percentile
# is 2.
MAX_FREE_TEXT_CHARACTERS = 128
DEFAULT_ARRAY_BOUND = 4
ARRAY_BOUNDS = {
    "findings": 6,
    "global_limitations": 10,
    "traceable_evidence": 10,
    "counter_evidence": 4,
    "imaging_factors_for_specialist_review": 4,
    "limitations": 6,
}
# A JSON number has no length constraint in the schema and none in the decoder,
# so the worst case is modelled with a wide fixed-point placeholder instead of
# being claimed as enforced.
MAX_NUMERIC_PLACEHOLDER = 1234567890.123456

# Derived from the bounded schema by scripts/verify-decoder-output-budget.py and
# re-proved on the runtime itself, with the real tokenizer, before any medical
# access. It is not a guess: /decoder-bounds recomputes it and the probe gate
# fails if the maximum decodable output no longer fits.
MAX_NEW_TOKENS = 12288
OUTPUT_BUDGET_MARGIN_TOKENS = 512
# Measured with the pinned MedGemma tokenizer on the bounds above, for the
# widest approved window of 32 images: the maximum decodable output is 46_394
# characters and 10_791 tokens, so 11_303 tokens are required including the
# margin. The character figure is asserted locally without a tokenizer; widening
# any bound raises it, fails the test, and forces the token proof to be run
# again before the budget is trusted.
PROVED_MAXIMUM_STRUCTURAL_CHARACTERS = 46_394
LONG_OUTPUT_PROBE_RECORDS = 96
LONG_OUTPUT_PROBE_MIN_TOKENS = 4097


def generation_policy() -> dict[str, Any]:
    """The only approved generation policy for constrained canonical JSON.

    ``no_repeat_ngram_size`` and ``repetition_penalty`` are deliberately
    neutral. Canonical JSON for this schema must repeat structural n-grams:
    ``"export_id": "image-`` and ``"role": "supporting"`` recur once per cited
    image. A repetition constraint sets those continuations to ``-inf`` at the
    same positions where the schema parser already allows only one token, the
    combined mask becomes empty, and generation leaves the grammar and stalls
    into whitespace until it is cut off. Suppressing repeated normal statements
    is a prompt and schema concern, not a sampler concern.
    """
    return {
        "do_sample": False,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "json_schema_enforced": True,
        "canonical_eos_only": True,
        "evidence_ids_enumerated": True,
        "policy": GENERATION_POLICY,
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
    validation_policy: str
    status: str
    candidate_count: int
    selected_candidate_index: int | None
    selected_candidate_sha256: str | None
    selected_semantic_sha256: str | None
    result: dict[str, Any] | None
    candidates: tuple[dict[str, Any], ...]


class DuplicateJsonKeyError(ValueError):
    pass


def decoder_schema(
    schema: dict[str, Any],
    evidence_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return an LMFE-compatible structural schema; canonical validation remains authoritative.

    When the approved window's evidence IDs are supplied, every ``export_id``
    that the canonical schema constrains by pattern is additionally constrained
    to that exact enumeration. A citation outside the window then cannot be
    decoded at all, instead of being produced and quarantined afterwards. The
    canonical schema is never modified: this derived structural schema is used
    only to build the token mask.
    """
    allowed = list(dict.fromkeys(evidence_ids or []))
    named_children = {"properties", "$defs", "definitions"}

    def convert(value: Any, key: str = "") -> Any:
        if isinstance(value, list):
            return [convert(item, key) for item in value]
        if not isinstance(value, dict):
            return value
        converted: dict[str, Any] = {}
        for name, item in value.items():
            if name == "$schema":
                continue
            if name in named_children and isinstance(item, dict):
                converted[name] = {
                    child: convert(child_value, child) for child, child_value in item.items()
                }
            else:
                converted[name] = convert(item, key)
        constant = converted.get("const")
        if isinstance(constant, bool):
            converted.pop("const")
            converted["type"] = "boolean"
        if allowed and converted.get("pattern") == EVIDENCE_ID_PATTERN:
            converted.pop("pattern")
            converted["enum"] = allowed
        if converted.get("type") == "array" and "maxItems" not in converted:
            converted["maxItems"] = ARRAY_BOUNDS.get(key, DEFAULT_ARRAY_BOUND)
        if (
            converted.get("type") == "string"
            and "maxLength" not in converted
            and not {"enum", "const", "pattern"} & set(converted)
        ):
            converted["maxLength"] = MAX_FREE_TEXT_CHARACTERS
        return converted

    return convert(schema)


def decoder_output_bounds() -> dict[str, Any]:
    """The structural limits that make the decodable output finite."""
    return {
        "policy": OUTPUT_BOUNDS_POLICY,
        "max_free_text_characters": MAX_FREE_TEXT_CHARACTERS,
        "default_array_items": DEFAULT_ARRAY_BOUND,
        "array_items": dict(ARRAY_BOUNDS),
    }


def maximum_structural_output(
    schema: dict[str, Any],
    evidence_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the largest result the bounded decoder can emit for one window.

    This is the object the token budget must accommodate. It is deliberately
    built from the derived schema rather than from any observed output, so the
    budget is proved against what is decodable, not against what happened to be
    generated once.
    """
    bounded = decoder_schema(schema, evidence_ids)
    definitions = bounded.get("$defs") or bounded.get("definitions") or {}

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        reference = node.get("$ref")
        if not reference:
            return node
        return resolve(definitions[str(reference).rsplit("/", maxsplit=1)[-1]])

    def build(node: Any) -> Any:
        if not isinstance(node, dict):
            return None
        node = resolve(node)
        if "const" in node:
            return node["const"]
        if node.get("enum"):
            return max(node["enum"], key=lambda item: len(str(item)))
        declared = node.get("type")
        if isinstance(declared, list):
            declared = next((item for item in declared if item != "null"), "string")
        if declared == "object":
            return {
                name: build(child) for name, child in (node.get("properties") or {}).items()
            }
        if declared == "array":
            items = node.get("items") or {}
            count = int(node.get("maxItems", DEFAULT_ARRAY_BOUND))
            return [build(items) for _ in range(count)]
        if declared == "string":
            return _free_text_filler(int(node.get("maxLength", MAX_FREE_TEXT_CHARACTERS)))
        if declared in {"number", "integer"}:
            return MAX_NUMERIC_PLACEHOLDER
        if declared == "boolean":
            return False
        return None

    result = build(bounded)
    if isinstance(result, dict):
        _assign_distinct_evidence(result, list(dict.fromkeys(evidence_ids or [])))
    return result


_FREE_TEXT_SPECIMEN = (
    "Intrasubstance signal reaching the inferior articular surface of the posterior horn "
    "on adjacent sagittal slices, with volume averaging at the meniscocapsular margin. "
)


def _free_text_filler(length: int) -> str:
    """Fill a bounded string with ordinary clinical prose.

    A run of one repeated character tokenises far more densely than real text
    and understated the maximum output by roughly a third, which would have made
    the proved budget optimistic. The specimen models what the model actually
    emits, so the measured maximum is representative rather than flattering.
    """
    if length <= 0:
        return ""
    repeats = length // len(_FREE_TEXT_SPECIMEN) + 1
    return (_FREE_TEXT_SPECIMEN * repeats)[:length]


def _assign_distinct_evidence(result: dict[str, Any], evidence_ids: list[str]) -> None:
    """Give the maximum output distinct citations, as the contract requires.

    A finding may not cite the same image twice, nor cite one image as both
    supporting and counter evidence. Repeating a single identifier would build a
    maximum output that the canonical validator rejects, which would hide the
    very property this construction is meant to prove.
    """
    findings = result.get("findings")
    if not isinstance(findings, list) or not evidence_ids:
        return
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        cursor = 0
        for key, role in (
            ("traceable_evidence", "supporting"),
            ("counter_evidence", "counter"),
        ):
            entries = finding.get(key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if "export_id" in entry:
                    entry["export_id"] = evidence_ids[cursor % len(evidence_ids)]
                    cursor += 1
                if "role" in entry:
                    entry["role"] = role


def policy_digest(policy: str, payload: Any) -> str:
    return _digest({"policy": policy, "payload": payload})


def output_contract() -> dict[str, Any]:
    payload = {
        "extraction": EXTRACTION_POLICY,
        "normalization": NORMALIZATION_POLICY,
        "validation": OUTPUT_VALIDATION_POLICY,
        "correction": CORRECTION_POLICY,
    }
    return {
        **payload,
        "policy_sha256": policy_digest(OUTPUT_VALIDATION_POLICY, payload),
    }


def structural_correction_prompt(
    previous: str,
    attempt: int,
    errors: list[str],
) -> str:
    """Build a compact, text-only repair prompt for an accepted schema contract.

    The previous implementation appended the entire JSON Schema to every
    correction. With a partial model response this consumed most of the model
    context and three real corrections terminated around 1,000 tokens without
    EOS or a closing brace. The constrained decoder already enforces the schema,
    so repeating it is both redundant and harmful.
    """
    return (
        "Structural correction only. Do not inspect or reinterpret images. "
        "Return exactly one complete JSON object and no prose. Preserve every clinical "
        "claim, uncertainty, evidence reference, and counter-evidence already present. "
        "Do not add a clinical finding or change a measurement. If the previous response "
        "was truncated, complete only its JSON structure and required fields, using "
        "indeterminate, null, low, empty arrays, or false safety constants when a required "
        "value was not present. The constrained decoder enforces the approved schema. "
        f"Correction attempt {attempt}. Errors: {'; '.join(errors)[:1500]}\n"
        "Previous response to preserve semantically:\n"
        + previous[:12000]
    )


def structural_correction_errors(decision: ExtractionDecision) -> list[str]:
    """Return correction reasons only for canonical output failures.

    Evidence scope is deliberately absent here. A valid JSON object with a
    well-formed citation from another window or an unknown global ID is accepted
    on the first attempt and graded locally; it must never trigger another model
    generation.
    """
    if decision.result is not None:
        return []
    errors = []
    for candidate in decision.candidates:
        errors.extend(str(item) for item in candidate.get("validation_errors") or [])
    return list(dict.fromkeys(errors)) or [decision.status]


def decoder_acceptance(
    text: str,
    schema: dict[str, Any],
    evidence_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Replay text through the constrained decoder's own grammar.

    A constrained run may only emit characters the schema parser allowed. If the
    recorded text diverges, the token mask was overridden during generation and
    the output is a runtime defect, not a model answer. This check needs no GPU
    and no model, so every attempt can be audited after the fact.
    """
    from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser

    parser = JsonSchemaParser(
        decoder_schema(schema, evidence_ids),
        config=CharacterLevelParserConfig(**DECODER_CONFIG),
    )
    for index, character in enumerate(text):
        try:
            allowed = parser.get_allowed_characters()
        except Exception as error:  # noqa: BLE001 - reported, never raised onward
            return {
                "accepted": False,
                "divergence_index": index,
                "reason": f"parser_failed:{type(error).__name__}",
                "consumed_characters": index,
            }
        if character not in allowed:
            return {
                "accepted": False,
                "divergence_index": index,
                "reason": "character_outside_decoder_mask",
                "consumed_characters": index,
            }
        try:
            parser = parser.add_character(character)
        except Exception as error:  # noqa: BLE001 - reported, never raised onward
            return {
                "accepted": False,
                "divergence_index": index,
                "reason": f"parser_failed:{type(error).__name__}",
                "consumed_characters": index,
            }
    return {
        "accepted": True,
        "divergence_index": None,
        "reason": None,
        "consumed_characters": len(text),
    }


def decoder_contract(schema: dict[str, Any]) -> dict[str, Any]:
    # The digest stays window-independent: it binds the structural schema, the
    # decoder configuration and the fact that evidence IDs are enumerated per
    # window, never the enumeration of one window.
    constrained_schema = decoder_schema(schema)
    payload = {
        "implementation": DECODER_IMPLEMENTATION,
        "version": DECODER_VERSION,
        "policy": DECODER_POLICY,
        "config": DECODER_CONFIG,
        "schema": constrained_schema,
        "evidence_ids_enumerated": True,
        "evidence_id_pattern": EVIDENCE_ID_PATTERN,
        "output_bounds": decoder_output_bounds(),
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    return {
        "implementation": DECODER_IMPLEMENTATION,
        "version": DECODER_VERSION,
        "policy": DECODER_POLICY,
        "config": dict(DECODER_CONFIG),
        "evidence_ids_enumerated": True,
        "output_bounds": decoder_output_bounds(),
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
    allowed_evidence_ids: list[str] | None = None,
) -> list[str]:
    """Validate canonical model output, independently of evidence scope.

    ``allowed_evidence_ids`` is retained for API compatibility but deliberately
    ignored. Whether a well-formed image ID belongs to this window, another
    window, another series, or nowhere in the approved plan is a traceability
    decision made locally against the evidence map. It is not a reason to ask
    the model for another clinical interpretation.
    """
    errors = [
        error.message
        for error in sorted(
            Draft202012Validator(schema).iter_errors(result),
            key=lambda item: list(item.absolute_path),
        )
    ]
    errors.extend(validate_evidence_structure(result))
    return errors


def validate_evidence_structure(result: dict[str, Any]) -> list[str]:
    """Validate evidence objects and clinical safety invariants, not scope."""
    if not isinstance(result, dict):
        return ["result must be an object"]
    errors = []
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


def validate_result_evidence(
    result: dict[str, Any],
    allowed_evidence_ids: list[str] | None = None,
) -> list[str]:
    """Backward-compatible name for evidence structure validation.

    Scope validation was intentionally removed in output-validation-v2. The
    optional second argument is ignored so runtime and older callers cannot
    accidentally reintroduce the correction loop.
    """
    return validate_evidence_structure(result)


def classify_output(
    text: str,
    schema: dict[str, Any],
    allowed_evidence_ids: list[str] | None = None,
) -> ExtractionDecision:
    candidates = enumerate_json_candidates(text)
    decisions = []
    valid = []
    for index, candidate in enumerate(candidates):
        candidate_digest = _digest(candidate.text)
        try:
            parsed = strict_json_loads(candidate.text)
            exact_errors = validate_result(parsed, schema)
            if not exact_errors:
                normalized = parsed
                actions = []
                status = "exact_valid"
                errors = []
            else:
                normalized, actions = normalize_structural_result(parsed)
                preservation_errors = verify_normalization(parsed, normalized, actions)
                errors = preservation_errors + validate_result(normalized, schema)
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
        status = (
            "truncated_json"
            if text.lstrip().startswith("{") and not candidates
            else "unrecoverable"
        )
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
        OUTPUT_VALIDATION_POLICY,
        status,
        len(candidates),
        selected[0] if selected else None,
        selected[3].candidate_sha256 if selected else None,
        selected[3].semantic_sha256 if selected else None,
        selected[2] if selected else None,
        tuple(asdict(item) for item in decisions),
    )
