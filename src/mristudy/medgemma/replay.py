from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..clinical_schema import load_knee_meniscus_schema
from ..errors import MRIStudyError
from ..utils import atomic_json, canonical_json, load_json, sha256_bytes, sha256_file, utc_now
from .output_contract import (
    EXTRACTION_POLICY,
    NORMALIZATION_POLICY,
    classify_output,
    decoder_contract,
)


def replay_historical_runs(runs_root: Path, output_path: Path) -> dict[str, Any]:
    schema = load_knee_meniscus_schema()
    records = []
    for raw_path in sorted(runs_root.glob("medgemma-schema-v2-*/windows/*.raw.json")):
        response = load_json(raw_path)
        allowed = response.get("evidence_ids") or []
        if not isinstance(allowed, list):
            raise MRIStudyError("Historical replay artifact has invalid evidence IDs.")
        attempt_records = []
        for attempt in response.get("attempts") or []:
            decision = classify_output(str(attempt.get("raw_text") or ""), schema, allowed)
            attempt_records.append(
                {
                    "attempt": attempt.get("attempt"),
                    "raw_text_sha256": sha256_bytes(str(attempt.get("raw_text") or "").encode()),
                    "status": decision.status,
                    "candidate_count": decision.candidate_count,
                    "selected_candidate_sha256": decision.selected_candidate_sha256,
                    "selected_semantic_sha256": decision.selected_semantic_sha256,
                    "normalization_rule_ids": sorted(
                        {
                            action["rule_id"]
                            for candidate in decision.candidates
                            for action in candidate.get("actions", [])
                        }
                    ),
                }
            )
        records.append(
            {
                "source_artifact_sha256": sha256_file(raw_path),
                "window_id_sha256": sha256_bytes(str(response.get("window_id") or "").encode()),
                "attempts": attempt_records,
            }
        )
    counts = Counter(
        attempt["status"] for record in records for attempt in record["attempts"]
    )
    report = {
        "schema_version": 1,
        "operation": "medgemma-historical-output-replay",
        "created_at": utc_now(),
        "source_artifacts": len(records),
        "attempts": sum(len(record["attempts"]) for record in records),
        "classifications": dict(sorted(counts.items())),
        "policies": {
            "extraction": EXTRACTION_POLICY,
            "normalization": NORMALIZATION_POLICY,
            "decoder": decoder_contract(schema),
        },
        "records": records,
    }
    report["report_sha256"] = replay_report_digest(report)
    atomic_json(output_path, report)
    return report


def replay_report_digest(report: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in report.items()
        if key not in {"created_at", "report_sha256"}
    }
    return sha256_bytes(canonical_json(stable))


def replay_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    classifications = report.get("classifications") or {}
    recoverable = int(classifications.get("exact_valid", 0)) + int(
        classifications.get("normalizable_valid", 0)
    )
    unrecoverable = int(classifications.get("unrecoverable", 0))
    ambiguous = int(classifications.get("ambiguous_multiple_valid_results", 0))
    return {
        "passed": ambiguous == 0 and recoverable > 0,
        "historical_attempts": int(report.get("attempts", 0)),
        "recoverable_attempts": recoverable,
        "unrecoverable_attempts": unrecoverable,
        "ambiguous_attempts": ambiguous,
        "unrecoverable_have_no_json_candidates": all(
            attempt["candidate_count"] == 0
            for record in report.get("records", [])
            for attempt in record.get("attempts", [])
            if attempt["status"] == "unrecoverable"
        ),
        "all_json_candidates_recoverable": all(
            attempt["status"] in {"exact_valid", "normalizable_valid", "unrecoverable"}
            and (attempt["status"] != "unrecoverable" or attempt["candidate_count"] == 0)
            for record in report.get("records", [])
            for attempt in record.get("attempts", [])
        ),
    }
