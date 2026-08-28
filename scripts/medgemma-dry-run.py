"""Execute the full approved window plan on CPU, without a GPU or medical images.

Purpose
-------
Every historical pre-lock failure was discovered on a paid GPU, after the model
had already produced a medical result, because nothing exercised the complete
runner path locally. This gate replays the entire approved plan end to end
using a client that returns responses built strictly from the plan itself.

What this proves
----------------
- every window prompt hash matches the plan
- the runner accepts a fully conformant runtime response
- window manifests, attempts, and digests are written and revalidated
- series integration and multiplanar integration complete
- evidence, treatment boundaries, and coverage validate at every stage

What this does not prove
------------------------
Model quality. The clinical content here is a fixed, deliberately empty and
conservative result. This gate verifies the contract and the pipeline, not the
medicine. No DICOM is read and no medical image is transmitted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "src"))

from mristudy.medgemma.integrity import require_valid_evidence_map  # noqa: E402
from mristudy.medgemma.runner import run_window_plan  # noqa: E402
from mristudy.profiles import load_profile_by_name  # noqa: E402
from mristudy.prompts import medgemma_prompt_for_profile  # noqa: E402
from mristudy.report_bundle import (  # noqa: E402
    build_viewer_bundle,
    evidence_summary_from_multiplanar_provenance,
)
from mristudy.schema_registry import empty_result  # noqa: E402
from mristudy.utils import load_json, sha256_bytes  # noqa: E402


def _empty_valid_result(window: dict[str, Any]) -> dict[str, Any]:
    """The canonical conservative v2 result: no findings, all boundaries false."""
    return {
        "schema_version": "knee-meniscus-finding-v2",
        "study_purpose": "exploratory_non_diagnostic",
        "findings": [],
        "global_limitations": [
            "Dry-run contract verification result. Not a clinical interpretation.",
        ],
        "treatment_boundary": {
            "treatment_recommendation_provided": False,
            "repairability_determined": False,
            "vascularity_directly_measured": False,
        },
    }


def _finding_result(window: dict[str, Any], evidence_id: str) -> dict[str, Any]:
    result = empty_result(window["output_schema"], body_region="knee")
    result["findings"] = [
        {
            "finding_id": f"{window['window_id']}-finding-01",
            "meniscus": "medial",
            "segment": "posterior_horn",
            "tear_status": "suspected",
            "tear_pattern": "vertical_longitudinal",
            "orientation": "vertical",
            "vascular_zone_estimate": {
                "zone": "red_white",
                "directly_observed_vascularity": False,
                "basis": "estimated_from_meniscocapsular_distance",
                "confidence": "low",
                "limitations": ["Synthetic dry-run finding."],
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
                "tissue_quality_imaging_estimate": "indeterminate",
                "chronicity_imaging_estimate": "indeterminate",
                "extrusion_mm_estimate": None,
                "parameniscal_cyst": "indeterminate",
                "root_involvement": "indeterminate",
            },
            "imaging_factors_for_specialist_review": [],
            "traceable_evidence": [
                {
                    "export_id": evidence_id,
                    "role": "supporting",
                    "description": "Synthetic dry-run evidence.",
                }
            ],
            "counter_evidence": [],
            "limitations": ["Synthetic dry-run finding."],
            "confidence": "low",
        }
    ]
    return result


class DryRunClient:
    """Return a response that satisfies every provenance field the runner checks.

    The response is assembled from the approved window, so a drift between the
    plan and the runner's expectations surfaces here rather than on a GPU.
    """

    def __init__(
        self,
        plan: dict[str, Any],
        provision_plan_sha256: str,
        snapshot_manifest_sha256: str,
        mode: str = "empty",
    ) -> None:
        self.plan = plan
        self.provision_plan_sha256 = provision_plan_sha256
        self.snapshot_manifest_sha256 = snapshot_manifest_sha256
        self.mode = mode
        self.calls = 0

    def analyze_window(self, window: dict[str, Any], prompt: str) -> dict[str, Any]:
        self.calls += 1
        result = _empty_valid_result(window)
        if self.mode != "empty":
            evidence_id = window["evidence_ids"][0]
            if self.mode == "unknown":
                evidence_id = "image-999999"
            elif self.mode == "adjacent":
                current = int(evidence_id.split("-")[1])
                candidate = f"image-{max(1, current - 1):06d}"
                if candidate in window["evidence_ids"]:
                    candidate = f"image-{current + len(window['evidence_ids']):06d}"
                evidence_id = candidate
            result = _finding_result(window, evidence_id)
        raw_text = json.dumps(result)
        attempt = {
            "attempt": 1,
            "raw_text": raw_text,
            "result": result,
            "normalization_actions": [],
            "validation_errors": [],
            "generation_seconds": 0.0,
            "output_contract": {
                "extraction_policy": "dry-run",
                "normalization_policy": "dry-run",
                "validation_policy": "medgemma-output-validation-v2",
                "status": "exact_valid",
                "candidate_count": 1,
                "selected_candidate_index": 0,
                "selected_candidate_sha256": sha256_bytes(raw_text.encode()),
                "selected_semantic_sha256": sha256_bytes(raw_text.encode()),
                "candidates": [],
            },
        }
        return {
            "model": window["model_repository"],
            "model_revision": window["model_revision"],
            "model_snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "runtime_digest": window["runtime_digest"],
            "run_plan_sha256": self.plan["plan_sha256"],
            "provision_plan_sha256": self.provision_plan_sha256,
            "window_id": window["window_id"],
            "role": window["role"],
            "instance_uids": window["instance_uids"],
            "evidence_ids": window["evidence_ids"],
            "output_schema": window["output_schema"],
            "prompt_sha256": window["prompt_sha256"],
            "generation": window["generation"],
            "decoder": window["decoder"],
            "output_contract": window["output_contract"],
            "raw_text": raw_text,
            "result": result,
            "schema_status": "valid",
            "validation_errors": [],
            "attempts": [attempt],
            "accepted_attempt": 1,
            "metrics": {
                "total_seconds": 0.0,
                "encoding_seconds": 0.0,
                "generation_seconds": 0.0,
                "gpu_peak_memory_bytes": 0,
                "input_images": len(window["instance_uids"]),
                "oom": False,
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, help="study identifier")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "mri-study-data",
        help="MRI study data root",
    )
    parser.add_argument("--keep", action="store_true", help="keep the dry-run directory")
    parser.add_argument(
        "--mode",
        choices=("empty", "adjacent", "unknown"),
        default="empty",
        help="adversarial evidence scenario",
    )
    args = parser.parse_args()

    study_root = args.data_root / "studies" / args.study
    if not study_root.is_dir():
        raise SystemExit(f"Study not found: {study_root}")

    plan = load_json(study_root / "state/medgemma-run-plan.json")
    state = load_json(study_root / "state/status.json")
    profile = load_profile_by_name(state["profile"])
    evidence_manifest = load_json(study_root / "manifests/medgemma-evidence-map.json")
    evidence_map = require_valid_evidence_map(evidence_manifest, plan)

    provision_plan_sha256 = "dry-run-provision-plan"
    snapshot_manifest_sha256 = "dry-run-snapshot-manifest"

    run_dir = study_root / "runs" / f"dry-run-contract-verification-{args.mode}"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    client = DryRunClient(
        plan, provision_plan_sha256, snapshot_manifest_sha256, mode=args.mode
    )
    final = run_window_plan(
        client,
        plan,
        run_dir,
        medgemma_prompt_for_profile(profile),
        evidence_map=evidence_map,
        provision_plan_sha256=provision_plan_sha256,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        profile=profile,
    )

    summary = {
        "mode": args.mode,
        "windows_executed": client.calls,
        "window_count": final["window_count"],
        "coverage": final["coverage"],
        "series_integrations": sorted(final["series_integrations"]),
        "multiplanar_result_written": bool(final.get("multiplanar_result")),
    }
    result = load_json(run_dir / final["multiplanar_result"])
    integrated_summary = evidence_summary_from_multiplanar_provenance(
        load_json(run_dir / final["multiplanar_provenance"])
    )
    bundle = build_viewer_bundle(
        {"medgemma-dry-run": result},
        evidence_manifest,
        study_root / "anonymized-dicom",
        run_dir / "viewer",
        run_plan=plan,
        export_images=False,
        evidence_summaries={"medgemma-dry-run": integrated_summary},
    )
    summary["viewer_findings"] = sum(
        len(item["findings"]) for item in bundle["models"]
    )
    summary["viewer_evidence_statuses"] = sorted(
        finding["evidenceStatus"]
        for model in bundle["models"]
        for finding in model["findings"]
    )
    print(json.dumps(summary, indent=2))

    if not args.keep:
        shutil.rmtree(run_dir)
        print("dry-run artifacts removed")
    else:
        print(f"dry-run artifacts kept at {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
