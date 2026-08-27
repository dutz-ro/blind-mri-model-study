from __future__ import annotations

from pathlib import Path

import pydicom

from ..utils import canonical_json, sha256_bytes, sha256_file, utc_now
from .dicomweb import DicomStoreRef


def discover_uploadable_images(root: Path) -> tuple[list[Path], list[dict]]:
    images = []
    skipped = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, OSError) as error:
            skipped.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "reason": type(error).__name__,
                }
            )
            continue
        required = (
            getattr(ds, "StudyInstanceUID", None),
            getattr(ds, "SeriesInstanceUID", None),
            getattr(ds, "SOPInstanceUID", None),
            getattr(ds, "SOPClassUID", None),
            getattr(ds, "Rows", None),
            getattr(ds, "Columns", None),
        )
        if not all(required):
            skipped.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "reason": "not-an-image-instance",
                }
            )
            continue
        images.append(path)
    return images, skipped


def stow_plan(paths: list[Path], store: DicomStoreRef) -> dict:
    files = []
    seen = set()
    total = 0
    for path in sorted(paths):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        sop_uid = str(ds.SOPInstanceUID)
        if sop_uid in seen:
            raise ValueError("Duplicate SOP Instance UID in upload plan")
        seen.add(sop_uid)
        size = path.stat().st_size
        total += size
        files.append(
            {
                "path": str(path),
                "size": size,
                "sha256": sha256_file(path),
                "study_uid": str(ds.StudyInstanceUID),
                "series_uid": str(ds.SeriesInstanceUID),
                "sop_instance_uid": sop_uid,
            }
        )
    plan = {
        "schema_version": 1,
        "operation": "dicomweb-stow",
        "created_at": utc_now(),
        "target": store.resource_name,
        "files": files,
        "total_bytes": total,
        "contains_phi": False,
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json(plan))
    return plan


def infrastructure_plan(
    *,
    project: str,
    region: str,
    study_id: str,
    upload_plan: dict,
    existing_budget_usd: float | None,
    approved_max_cost_usd: float,
    global_gpu_quota: float,
) -> dict:
    dataset = f"mristudy-{study_id}"
    store = "deidentified-input"
    plan = {
        "schema_version": 1,
        "operation": "google-healthcare-medgemma-infrastructure",
        "created_at": utc_now(),
        "project": project,
        "region": region,
        "resources": {
            "healthcare_dataset": dataset,
            "dicom_store": store,
            "service_account": f"mristudy-{study_id[:15]}",
            "vertex_endpoint": f"medgemma-dicom-{study_id}",
            "model": "google/medgemma-1.5-4b-it",
            "serving_variant": "dicom-aware",
            "machine_type": "g2-standard-4",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,
            "min_replicas": 0,
            "max_replicas": 1,
        },
        "labels": {
            "managed-by": "mristudy",
            "study-id": study_id,
            "purpose": "exploratory-analysis",
            "contains-phi": "false",
        },
        "upload": {
            "plan_sha256": upload_plan["plan_sha256"],
            "files": len(upload_plan["files"]),
            "total_bytes": upload_plan["total_bytes"],
        },
        "cost_control": {
            "approved_max_cost_usd": approved_max_cost_usd,
            "existing_google_budget_usd": existing_budget_usd,
            "endpoint_timeout_minutes": 180,
            "delete_after_inference": True,
        },
        "preconditions": {
            "privacy_approved": True,
            "generalist_results_locked": True,
            "cloud_upload_approval_required": True,
            "model_terms_acceptance_required": True,
            "global_gpu_quota": global_gpu_quota,
            "gpu_quota_sufficient": global_gpu_quota >= 1,
            "budget_matches_approved_max": (
                existing_budget_usd is not None
                and existing_budget_usd >= approved_max_cost_usd
            ),
        },
        "estimated_cost": {
            "status": "blocked",
            "reason": (
                "A deployable Model Garden configuration and GPU quota are required for a "
                "load-bearing estimate. No endpoint may be created above the approved maximum."
            ),
            "healthcare_storage_note": "Temporary Standard-class storage for less than 1 GiB.",
        },
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json(plan))
    return plan
