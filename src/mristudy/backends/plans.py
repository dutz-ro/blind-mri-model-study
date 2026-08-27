from __future__ import annotations

from pathlib import Path

import pydicom

from ..cloud.plans import discover_uploadable_images
from ..errors import MRIStudyError
from ..utils import canonical_json, sha256_bytes, sha256_file, utc_now


def local_orthanc_plan(
    anonymized_root: Path,
    endpoint_url: str,
    deployment_id: str,
) -> dict:
    paths, skipped = discover_uploadable_images(anonymized_root)
    files = []
    for path in paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        files.append(
            {
                "path": str(path),
                "relative_path": path.relative_to(anonymized_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "study_uid": str(ds.StudyInstanceUID),
                "series_uid": str(ds.SeriesInstanceUID),
                "sop_instance_uid": str(ds.SOPInstanceUID),
            }
        )
    plan = {
        "schema_version": 2,
        "operation": "local-orthanc-import",
        "created_at": utc_now(),
        "provider": "local-orthanc",
        "deployment_id": deployment_id,
        "dicomweb_url": endpoint_url.rstrip("/"),
        "files": files,
        "skipped": skipped,
        "total_bytes": sum(item["size"] for item in files),
        "contains_phi": False,
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json(plan))
    return plan


def scaleway_operation_plan(
    *,
    study_id: str,
    legacy_upload_plan: dict,
    legacy_infrastructure_plan: dict,
    orthanc_manifest: dict,
    source_public_ip: str,
    cost_limit_eur: float,
    model_revision: str | None = None,
    model_access_accepted: bool = False,
) -> tuple[dict, dict]:
    operation = {
        "schema_version": 2,
        "operation": "dicom-aware-backend",
        "created_at": utc_now(),
        "provider": "scaleway",
        "topology": "hybrid-local-dicom-temporary-gpu",
        "region_preference": "fr-par",
        "zone_preference": ["fr-par-2", "fr-par-1"],
        "gpu_preference": ["L4-1-24G", "L40S-1-48G"],
        "gpu_fallback_policy": "L40S only after documented L4 capacity failure",
        "source_public_ip": source_public_ip,
        "local_dicom": {
            "provider": "orthanc",
            "deployment_id": orthanc_manifest["deployment_id"],
            "manifest_sha256": orthanc_manifest["manifest_sha256"],
            "study_uids": orthanc_manifest["study_uids"],
            "series_uids": orthanc_manifest["series_uids"],
            "verified_instances": orthanc_manifest["verified_instances"],
            "transport": "reverse-ssh-tunnel",
            "remote_bind": "127.0.0.1:18042",
            "permissions": ["QIDO-RS", "WADO-RS"],
        },
        "medgemma": {
            "repository": "google/medgemma-1.5-4b-it",
            "revision": model_revision or "REQUIRED_BEFORE_PROVISION",
            "container_digest": "REQUIRED_BEFORE_PROVISION",
            "api_bind": "127.0.0.1:8000",
            "model_access_accepted": model_access_accepted,
            "radiology_report_allowed": False,
        },
        "security": {
            "inbound_default": "drop",
            "ssh_source_cidr": f"{source_public_ip}/32",
            "public_application_ports": [],
            "temporary_ssh_key": True,
            "host_key_pin_required": True,
        },
        "lifecycle": {
            "max_gpu_instances": 1,
            "soft_timeout_hours": 4,
            "hard_ttl_hours": 6,
            "snapshots": False,
            "automatic_backups": False,
            "destroy_and_verify": True,
        },
        "cost_control": {"maximum_eur": cost_limit_eur},
        "preconditions": {
            "scaleway_credentials_present": False,
            "huggingface_model_access": model_access_accepted,
            "synthetic_probe_required": True,
            "provider_egress_approval_required": True,
        },
    }
    operation["plan_sha256"] = sha256_bytes(canonical_json(operation))
    migration = {
        "schema_version": 1,
        "operation": "migrate-backend-plan",
        "created_at": utc_now(),
        "study_id": study_id,
        "stage_before": "CLOUD_PLAN_READY",
        "stage_after": "CLOUD_PLAN_READY",
        "from_provider": "google",
        "to_provider": "scaleway",
        "legacy_upload_plan_sha256": legacy_upload_plan["plan_sha256"],
        "legacy_infrastructure_plan_sha256": legacy_infrastructure_plan["plan_sha256"],
        "backend_operation_plan_sha256": operation["plan_sha256"],
        "approval_carried_forward": False,
        "egress_authorized": False,
    }
    migration["plan_sha256"] = sha256_bytes(canonical_json(migration))
    return migration, operation


def revalidate_planned_file(item: dict, anonymized_root: Path) -> Path:
    path = Path(item["path"])
    if path.is_symlink():
        raise MRIStudyError("Planned DICOM became a symlink.")
    try:
        path.resolve().relative_to(anonymized_root.resolve())
    except ValueError as error:
        raise MRIStudyError("Planned DICOM is outside anonymized-dicom.") from error
    if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
        raise MRIStudyError("Planned DICOM changed after planning.")
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    actual = (
        str(ds.StudyInstanceUID),
        str(ds.SeriesInstanceUID),
        str(ds.SOPInstanceUID),
    )
    planned = (item["study_uid"], item["series_uid"], item["sop_instance_uid"])
    if actual != planned:
        raise MRIStudyError("Planned DICOM UIDs changed after planning.")
    return path
