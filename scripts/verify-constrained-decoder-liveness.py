#!/usr/bin/env python3
"""Privacy-safe replay of constrained-generation liveness failures.

The script accepts saved runtime attempt JSON files, but emits only hashes,
lengths and decoder divergence indexes. It never prints model text, prompts,
filenames, evidence identifiers or clinical content. No DICOM, model or GPU is
loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mristudy.medgemma.output_contract import (  # noqa: E402
    decoder_acceptance,
    generation_policy,
)
from mristudy.schema_registry import load_schema  # noqa: E402
from mristudy.utils import load_json, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, action="append", required=True)
    parser.add_argument("--expected-sha256", action="append", required=True)
    parser.add_argument("--schema", default="knee-meniscus-finding-v2")
    parser.add_argument("--evidence-id", action="append", default=[])
    args = parser.parse_args()
    if len(args.attempt) != len(args.expected_sha256):
        raise SystemExit("Each private attempt requires one expected SHA-256.")

    schema = load_schema(args.schema)
    records = []
    for path, expected in zip(args.attempt, args.expected_sha256, strict=True):
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit("Private attempt digest differs from the approved replay input.")
        raw = str(load_json(path).get("raw_text") or "")
        replay = decoder_acceptance(raw, schema, args.evidence_id or None)
        records.append(
            {
                "artifact_sha256": actual,
                "raw_text_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "raw_character_count": len(raw),
                **replay,
            }
        )

    report = {
        "generation_policy": generation_policy(),
        "attempts": records,
        "historical_decoder_violation_reproduced": any(
            not item["accepted"] for item in records
        ),
        "fixed_policy_neutralizes_repetition_constraints": (
            generation_policy()["repetition_penalty"] == 1.0
            and generation_policy()["no_repeat_ngram_size"] == 0
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
