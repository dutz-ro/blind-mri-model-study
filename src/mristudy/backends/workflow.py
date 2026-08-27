from __future__ import annotations

from pathlib import Path

from ..errors import MRIStudyError
from ..utils import atomic_json, canonical_json, sha256_bytes
from .dicomweb import DicomWebClient, dicom_json_value
from .plans import revalidate_planned_file

STUDY_UID_TAG = "0020000D"
SERIES_UID_TAG = "0020000E"
INSTANCE_UID_TAG = "00080018"


def inventory_remote(client: DicomWebClient, study_uids: list[str]) -> dict:
    instances = set()
    series_uids = set()
    for study_uid in study_uids:
        for series in client.search_series(study_uid):
            series_uid = dicom_json_value(series, SERIES_UID_TAG)
            if not series_uid:
                continue
            series_uids.add(series_uid)
            for instance in client.search_instances(study_uid, series_uid):
                instance_uid = dicom_json_value(instance, INSTANCE_UID_TAG)
                if instance_uid:
                    instances.add(instance_uid)
    return {
        "study_uids": sorted(study_uids),
        "series_uids": sorted(series_uids),
        "instance_uids": sorted(instances),
    }


def import_local_orthanc(
    client: DicomWebClient,
    plan: dict,
    anonymized_root: Path,
    manifest_path: Path,
) -> dict:
    expected = {item["sop_instance_uid"] for item in plan["files"]}
    study_uids = sorted({item["study_uid"] for item in plan["files"]})
    before = inventory_remote(client, study_uids)
    unexpected_before = set(before["instance_uids"]) - expected
    if unexpected_before:
        raise MRIStudyError("Orthanc contains unexpected instances before import.")
    already_present = set(before["instance_uids"])
    uploaded = 0
    for item in plan["files"]:
        path = revalidate_planned_file(item, anonymized_root)
        if item["sop_instance_uid"] in already_present:
            continue
        client.store_instance(path, item["study_uid"])
        uploaded += 1
    remote = inventory_remote(client, study_uids)
    found = set(remote["instance_uids"])
    missing = expected - found
    extra = found - expected
    if missing or extra:
        raise MRIStudyError(
            f"Orthanc verification failed: {len(missing)} missing, {len(extra)} extra."
        )
    result = {
        "provider": client.endpoint.provider,
        "deployment_id": client.endpoint.deployment_id,
        "plan_sha256": plan["plan_sha256"],
        "uploaded_instances": uploaded,
        "verified_instances": len(found),
        "study_uids": remote["study_uids"],
        "series_uids": remote["series_uids"],
        "instance_uids": remote["instance_uids"],
        "instance_uids_sha256": sha256_bytes(canonical_json(remote["instance_uids"])),
    }
    result["manifest_sha256"] = sha256_bytes(canonical_json(result))
    atomic_json(manifest_path, result)
    return result
