#!/usr/bin/env python3
"""Prove locally that the largest decodable window output fits the token budget.

Why this exists
---------------
The canonical schema bounds no array and no free-text string. It says what a
finding is, not how much of it a single window may contain, so the set of
decodable outputs was infinite and ``max_new_tokens`` could never be a
guarantee. A real medical window then repeated the same "absent, medial body,
no tear" finding eight times, exhausted the budget, and failed after three
byte-identical corrections on a paid GPU.

The derived decoder schema now bounds every array and every free-text string.
This script builds the largest output those bounds admit and checks two
properties that must hold before a GPU is ever provisioned:

1. the maximum output is valid against the unchanged canonical schema, so the
   bounds can only prevent output and never invalidate it;
2. it fits ``MAX_NEW_TOKENS`` with margin.

Property 2 needs a tokenizer. Without ``--tokenizer`` the script verifies the
structure and reports the size, which is useful on a workstation without
``transformers``; with ``--tokenizer`` it is exact. The exact form runs inside
the built runtime image against the pinned model snapshot:

    docker run --rm --entrypoint python3 \
      -v <snapshot>:/model:ro -v <repo>:/repo:ro <image> \
      /repo/scripts/verify-decoder-output-budget.py --tokenizer /model

No model weights are loaded, no GPU is used, and no clinical text is printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mristudy.clinical_schema import load_knee_meniscus_schema  # noqa: E402
from mristudy.medgemma.output_contract import (  # noqa: E402
    MAX_NEW_TOKENS,
    OUTPUT_BUDGET_MARGIN_TOKENS,
    decoder_output_bounds,
    maximum_structural_output,
    validate_result,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", help="path to the pinned model snapshot")
    parser.add_argument("--window-images", type=int, default=32)
    arguments = parser.parse_args()

    schema = load_knee_meniscus_schema()
    evidence_ids = [f"image-{index + 1:06d}" for index in range(arguments.window_images)]
    maximum = maximum_structural_output(schema, evidence_ids)
    serialized = json.dumps(maximum, indent=2)
    canonical_errors = validate_result(maximum, schema)

    report = {
        "output_bounds": decoder_output_bounds(),
        "window_images": arguments.window_images,
        "maximum_structural_characters": len(serialized),
        "maximum_structural_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "canonical_validation_errors": canonical_errors,
        "max_new_tokens": MAX_NEW_TOKENS,
        "margin_tokens": OUTPUT_BUDGET_MARGIN_TOKENS,
        "maximum_structural_tokens": None,
        "required_tokens_with_margin": None,
        "token_budget_proved": False,
    }

    if arguments.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer)
        tokens = len(tokenizer(serialized)["input_ids"])
        report["maximum_structural_tokens"] = tokens
        report["required_tokens_with_margin"] = tokens + OUTPUT_BUDGET_MARGIN_TOKENS
        report["token_budget_proved"] = (
            tokens + OUTPUT_BUDGET_MARGIN_TOKENS <= MAX_NEW_TOKENS
        )

    report["passed"] = not canonical_errors and (
        report["token_budget_proved"] or not arguments.tokenizer
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
