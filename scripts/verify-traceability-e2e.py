"""End-to-end proof that a recovered citation survives into the final report.

Builds a small plan with two overlapping windows, drives the real runner with a
client that reproduces the observed defect (image id placed in finding_id, empty
evidence list), and checks that the citation is recovered, graded, propagated
through series and multiplanar integration, and reported.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from mristudy.medgemma.runner import run_window_plan  # noqa: E402
from mristudy.prompts import medgemma_knee_meniscus_prompt_v2  # noqa: E402
from tests.test_medgemma_runner import plan as base_plan  # noqa: E402

PROVISION = "e2e-provision"
SNAPSHOT = "e2e-snapshot"


def valid_result(finding_id, evidence_ids):
    return {
        "schema_version": "knee-meniscus-finding-v2",
        "study_purpose": "exploratory_non_diagnostic",
        "findings": [
            {
                "finding_id": finding_id,
                "meniscus": "medial",
                "segment": "posterior_horn",
                "tear_status": "suspected",
                "tear_pattern": "horizontal",
                "orientation": "oblique",
                "vascular_zone_estimate": {
                    "zone": "red_white",
                    "directly_observed_vascularity": False,
                    "basis": "estimated_from_peripheral_location",
                    "confidence": "low",
                    "limitations": ["exploratory"],
                },
                "capsular_relation": {
                    "distance_mm_estimate": None,
                    "measurement_method": "not_measurable",
                    "peripheral_extension": "indeterminate",
                    "meniscocapsular_separation": "indeterminate",
                },
                "surface_extension": {
                    "superior": "indeterminate",
                    "inferior": "indeterminate",
                    "free_edge": "indeterminate",
                },
                "stability_features": {
                    "displaced_fragment": "indeterminate",
                    "flap": "indeterminate",
                    "bucket_handle_pattern": "indeterminate",
                    "imaging_signs_of_mechanical_obstruction": "indeterminate",
                },
                "healing_context": {
                    "extrusion_mm_estimate": None,
                    "parameniscal_cyst": "indeterminate",
                    "root_involvement": "indeterminate",
                    "tissue_quality_imaging_estimate": "indeterminate",
                    "chronicity_imaging_estimate": "indeterminate",
                },
                "imaging_factors_for_specialist_review": ["peripheral_extension"],
                "traceable_evidence": [
                    {"export_id": item, "role": "supporting", "description": "d"}
                    for item in evidence_ids
                ],
                "counter_evidence": [],
                "limitations": ["exploratory"],
                "confidence": "low",
            }
        ],
        "global_limitations": ["exploratory"],
        "treatment_boundary": {
            "treatment_recommendation_provided": False,
            "repairability_determined": False,
            "vascularity_directly_measured": False,
        },
    }


class DefectiveClient:
    """Reproduces the observed defect: citation placed in finding_id."""

    def __init__(self, plan):
        self.plan = plan

    def analyze_window(self, window, prompt):
        # The model cites the first slice of the window, but in the wrong field.
        cited = window["evidence_ids"][0]
        result = valid_result(cited, [])
        raw = json.dumps(result)
        return {
            "model": window["model_repository"],
            "model_revision": window["model_revision"],
            "model_snapshot_manifest_sha256": SNAPSHOT,
            "runtime_digest": window["runtime_digest"],
            "run_plan_sha256": self.plan["plan_sha256"],
            "provision_plan_sha256": PROVISION,
            "window_id": window["window_id"],
            "role": window["role"],
            "instance_uids": window["instance_uids"],
            "evidence_ids": window["evidence_ids"],
            "output_schema": window["output_schema"],
            "prompt_sha256": window["prompt_sha256"],
            "generation": window["generation"],
            "decoder": window["decoder"],
            "raw_text": raw,
            "result": result,
            "schema_status": "valid",
            "validation_errors": [],
            "attempts": [
                {"attempt": 0, "raw_text": raw, "result": result, "validation_errors": []}
            ],
            "accepted_attempt": 0,
            "metrics": {"total_seconds": 0.0, "gpu_peak_memory_bytes": 0, "oom": False},
        }


def main():
    plan = base_plan()
    run_dir = Path(tempfile.mkdtemp()) / "run"
    final = run_window_plan(
        DefectiveClient(plan),
        plan,
        run_dir,
        medgemma_knee_meniscus_prompt_v2(),
        provision_plan_sha256=PROVISION,
        snapshot_manifest_sha256=SNAPSHOT,
    )

    assert (run_dir / "integration/evidence-traceability.json").exists()
    multiplanar = json.loads((run_dir / "integration/multiplanar.json").read_text())

    print("DEFECT INJECTED: finding_id carries the image id, evidence list empty")
    print()
    print("traceability overall :", final["evidence_traceability"]["counts"])
    print("coverage percent     :", final["evidence_traceability"]["evidence_coverage_percent"])
    print("warning present      :", "evidence_traceability_warning" in final)
    print()
    cited = [
        item["export_id"]
        for finding in multiplanar["findings"]
        for item in finding.get("traceable_evidence") or []
    ]
    print("multiplanar findings :", len(multiplanar["findings"]))
    print("citations propagated :", cited)
    print()
    ok = bool(cited)
    print("RESULT:", "PROPAGATED to final report" if ok else "LOST before final report")
    shutil.rmtree(run_dir.parent, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
