#!/usr/bin/env python3
"""Privacy-safe diagnosis of truncated private model attempts.

Only hashes, token/brace counts, canonical statuses, and correction-prompt
lengths are printed. Raw model text and clinical content are never displayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mristudy.medgemma.output_contract import (  # noqa: E402
    classify_output,
    structural_correction_errors,
    structural_correction_prompt,
)
from mristudy.schema_registry import load_schema  # noqa: E402
from mristudy.utils import load_json, sha256_file  # noqa: E402


def _brace_state(text: str) -> dict[str, object]:
    depth = 0
    in_string = False
    escape = False
    for character in text.lstrip():
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
    return {
        "starts_with_object": text.lstrip().startswith("{"),
        "ends_with_object": text.rstrip().endswith("}"),
        "final_brace_depth": depth,
        "in_string_at_end": in_string,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, action="append", required=True)
    parser.add_argument("--expected-sha256", action="append", required=True)
    parser.add_argument("--schema", default="knee-meniscus-finding-v2")
    args = parser.parse_args()
    if len(args.attempt) != len(args.expected_sha256):
        raise SystemExit("Each private attempt requires one expected SHA-256.")

    records = []
    for index, (path, expected) in enumerate(
        zip(args.attempt, args.expected_sha256, strict=True)
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit("Private attempt digest differs from the approved replay input.")
        attempt = load_json(path)
        raw = str(attempt.get("raw_text") or "")
        decision = classify_output(raw, load_schema(args.schema))
        errors = structural_correction_errors(decision)
        correction = structural_correction_prompt(raw, index + 1, errors)
        records.append(
            {
                "attempt": attempt.get("attempt"),
                "artifact_sha256": actual,
                "raw_text_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "raw_character_count": len(raw),
                "canonical_status": decision.status,
                "candidate_count": decision.candidate_count,
                "correction_required": bool(errors),
                "correction_prompt_character_count": len(correction),
                "correction_repeats_schema": "Required JSON Schema" in correction
                or '"$defs"' in correction,
                **_brace_state(raw),
            }
        )
    report = {
        "attempts": records,
        "all_classified_truncated": all(
            item["canonical_status"] == "truncated_json" for item in records
        ),
        "all_corrections_compact": all(
            not item["correction_repeats_schema"] for item in records
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_classified_truncated"] and report["all_corrections_compact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
