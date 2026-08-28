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
    ) -> None:
        self.plan = plan
        self.provision_plan_sha256 = provision_plan_sha256
        self.snapshot_manifest_sha256 = snapshot_manifest_sha256
        self.calls = 0

    def analyze_window(self, window: dict[str, Any], prompt: str) -> dict[str, Any]:
        self.calls += 1
        result = _empty_valid_result(window)
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

    run_dir = study_root / "runs" / "dry-run-contract-verification"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    client = DryRunClient(plan, provision_plan_sha256, snapshot_manifest_sha256)
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
        "windows_executed": client.calls,
        "window_count": final["window_count"],
        "coverage": final["coverage"],
        "series_integrations": sorted(final["series_integrations"]),
        "multiplanar_result_written": bool(final.get("multiplanar_result")),
    }
    print(json.dumps(summary, indent=2))

    if not args.keep:
        shutil.rmtree(run_dir)
        print("dry-run artifacts removed")
    else:
        print(f"dry-run artifacts kept at {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
