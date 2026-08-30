from __future__ import annotations

from collections.abc import Callable
from typing import Any

from output_contract import normalize_structural_result, validate_result_evidence

__all__ = [
    "normalize_structural_result",
    "canonical_eos_guard",
    "require_finite_logits",
    "generation_budget_exhausted",
    "restore_decoder_config",
    "validate_result_evidence",
    "validate_window_payload",
]

OUTPUT_SCHEMA = "knee-meniscus-finding-v2"


def canonical_eos_guard(
    prefix_allowed_tokens_fn: Callable[[int, Any], list[int]],
    canonical_eos_token_id: int,
    configured_eos_token_ids: int | list[int] | tuple[int, ...] | None,
) -> Callable[[int, Any], list[int]]:
    """Block alternate stop tokens while preserving the schema parser's EOS decision.

    Gemma configures both ``<eos>`` and ``<end_of_turn>`` as stopping tokens.
    LM Format Enforcer knows when canonical JSON may end and exposes tokenizer
    EOS only in that state, but ``<end_of_turn>`` is an ordinary vocabulary token
    to the parser and can terminate generation inside a JSON string. The wrapper
    removes every configured stop token except canonical EOS. It never adds EOS:
    the underlying schema parser remains the sole authority on when ending is
    legal.
    """
    configured = (
        [configured_eos_token_ids]
        if isinstance(configured_eos_token_ids, int)
        else list(configured_eos_token_ids or [])
    )
    blocked = {int(item) for item in configured if int(item) != canonical_eos_token_id}

    def guarded(batch_id: int, sent: Any) -> list[int]:
        allowed = prefix_allowed_tokens_fn(batch_id, sent)
        filtered = [token for token in allowed if token not in blocked]
        if not filtered:
            raise ValueError("Canonical EOS guard removed every allowed token.")
        return filtered

    return guarded


def require_finite_logits(input_ids: Any, scores: Any) -> Any:
    """Fail closed when combined generation constraints ban every token.

    Transformers applies schema, repetition and other logits processors in
    sequence. If their intersection is empty, greedy decoding otherwise takes
    an arbitrary token from an all-negative-infinity row and silently leaves
    the JSON grammar. This processor must be installed last.
    """
    del input_ids
    finite = scores.isfinite()
    if not bool(finite.any(dim=-1).all().item()):
        raise RuntimeError("Generation constraints removed every finite token.")
    return scores


def generation_budget_exhausted(
    generated_tokens: int,
    max_new_tokens: int,
    tolerance: int = 8,
) -> bool:
    """Identify a structurally incomplete output that consumed its token budget."""
    return generated_tokens >= max(0, max_new_tokens - tolerance)


def restore_decoder_config(
    prefix_allowed_tokens_fn: Any,
    parser_config: Any,
) -> Any:
    """Restore the approved parser config after LMFE constructs TokenEnforcer.

    lm-format-enforcer 0.11.3 replaces the supplied parser config with a new
    config containing only the tokenizer alphabet. That silently resets field
    ordering, whitespace and array limits to package defaults while runtime
    provenance still reports the approved values. Preserve the tokenizer
    alphabet but restore every approved behavioral setting.
    """
    token_enforcer = getattr(prefix_allowed_tokens_fn, "token_enforcer", None)
    root_parser = getattr(token_enforcer, "root_parser", None)
    if root_parser is None:
        raise RuntimeError("Constrained decoder callback exposes no root parser.")
    root_parser.config = parser_config
    return prefix_allowed_tokens_fn


def _expected_schema(approved: dict[str, Any]) -> str:
    """The approved window names the schema; fall back to the knee default."""
    declared = approved.get("output_schema")
    return declared if isinstance(declared, str) and declared else OUTPUT_SCHEMA

def validate_window_payload(payload: dict[str, Any], approved: dict[str, Any]) -> list[str]:
    errors = []
    instance_uids = payload.get("instance_uids") or []
    evidence_ids = payload.get("evidence_ids") or []
    if not instance_uids:
        errors.append("empty instance list")
    if len(instance_uids) > 32:
        errors.append("more than 32 instances")
    if len(instance_uids) != len(evidence_ids):
        errors.append("instance and evidence lengths differ")
    if len(instance_uids) != len(set(instance_uids)):
        errors.append("duplicate instance UID")
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("duplicate evidence ID")
    if payload.get("output_schema") != _expected_schema(approved):
        errors.append("unsupported output schema")
    for key in (
        "study_uid",
        "series_uid",
        "role",
        "window_id",
        "instance_uids",
        "evidence_ids",
        "prompt_sha256",
        "output_schema",
        "max_new_tokens",
        "generation",
        "decoder",
        "output_contract",
    ):
        if payload.get(key) != approved.get(key):
            errors.append(f"{key} differs from approved window")
    return errors
