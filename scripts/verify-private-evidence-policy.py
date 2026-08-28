#!/usr/bin/env python3
"""Privacy-safe replay of one real model attempt under the evidence policy.

The script prints hashes, statuses and counts only. It never prints raw model
text, clinical findings, evidence descriptions, UIDs, or DICOM pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mristudy.medgemma.evidence_trace import (  # noqa: E402
    project_result_evidence,
    verify_evidence_projection,
)
from mristudy.medgemma.integrity import require_valid_evidence_map  # noqa: E402
from mristudy.medgemma.output_contract import classify_output  # noqa: E402
from mristudy.schema_registry import load_schema  # noqa: E402
from mristudy.utils import canonical_json, load_json, sha256_bytes, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--evidence-map", type=Path, required=True)
    args = parser.parse_args()

    actual = sha256_file(args.attempt)
    if actual != args.expected_sha256:
        raise SystemExit("Private attempt digest differs from the approved replay input.")

    plan = load_json(args.run_plan)
    evidence_manifest = load_json(args.evidence_map)
    evidence_map = require_valid_evidence_map(evidence_manifest, plan)
    windows = [item for item in plan["windows"] if item["window_id"] == args.window_id]
    if len(windows) != 1:
        raise SystemExit("Window is not uniquely present in the plan.")
    window = windows[0]

    attempt = load_json(args.attempt)
    raw_text = str(attempt.get("raw_text") or "")
    decision = classify_output(raw_text, load_schema(window["output_schema"]))

    report = {
        "artifact_sha256": actual,
        "raw_text_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
        "canonical_status": decision.status,
        "candidate_count": decision.candidate_count,
        "selected_candidate_sha256": decision.selected_candidate_sha256,
        "clinical_findings_preserved": False,
        "projection_valid": False,
        "scope_counts": {},
        "integration_safe_sha256": None,
        "link_eligible_count": 0,
        "correction_required": decision.result is None,
    }
    if decision.result is not None:
        safe, actions, summary = project_result_evidence(decision.result, window, evidence_map)
        projection_errors = verify_evidence_projection(decision.result, safe, actions)
        report.update(
            {
                "clinical_findings_preserved": len(safe.get("findings") or [])
                == len(decision.result.get("findings") or []),
                "projection_valid": not projection_errors,
                "scope_counts": summary.get("scope_counts") or {},
                "integration_safe_sha256": sha256_bytes(canonical_json(safe)),
                "link_eligible_count": sum(
                    len(item.get("link_eligible_evidence_ids") or [])
                    for item in summary.get("findings") or []
                ),
            }
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["canonical_status"] == "exact_valid" and report["projection_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
