from __future__ import annotations

from pathlib import Path

from ..approvals import require_approval
from ..errors import MRIStudyError
from ..utils import atomic_json
from .dicomweb import DicomStoreRef, DicomWebClient, dicom_json_value
from .medgemma import VertexEndpointRef, VertexMedGemmaClient, build_dicom_request, series_url

STUDY_UID_TAG = "0020000D"
SERIES_UID_TAG = "0020000E"
INSTANCE_UID_TAG = "00080018"


def upload_and_verify(
    client: DicomWebClient,
    plan: dict,
    approval_path: Path,
    manifest_path: Path,
) -> dict:
    require_approval(approval_path, "cloud-upload", plan)
    uploaded = []
    for item in plan["files"]:
        client.store_instance(Path(item["path"]), item["study_uid"])
        uploaded.append(item["sop_instance_uid"])

    expected = set(uploaded)
    found = set()
    for study_uid in sorted({item["study_uid"] for item in plan["files"]}):
        for series in client.search_series(study_uid):
            series_uid = dicom_json_value(series, SERIES_UID_TAG)
            if not series_uid:
                continue
            for instance in client.search_instances(study_uid, series_uid):
                instance_uid = dicom_json_value(instance, INSTANCE_UID_TAG)
                if instance_uid:
                    found.add(instance_uid)
    if expected != found.intersection(expected):
        missing = sorted(expected - found)
        raise MRIStudyError(f"DICOM Store verification failed; {len(missing)} instances missing.")
    result = {
        "target": plan["target"],
        "uploaded_instances": len(uploaded),
        "verified_instances": len(expected.intersection(found)),
        "study_uids": sorted({item["study_uid"] for item in plan["files"]}),
        "series_uids": sorted({item["series_uid"] for item in plan["files"]}),
    }
    atomic_json(manifest_path, result)
    return result


def run_medgemma_series(
    store: DicomStoreRef,
    endpoint: VertexEndpointRef,
    study_uid: str,
    series_uid: str,
    prompt: str,
) -> dict:
    request = build_dicom_request(series_url(store, study_uid, series_uid), prompt)
    return VertexMedGemmaClient(endpoint).infer(request)


def destroy_uploaded_instances(
    client: DicomWebClient,
    upload_plan: dict,
    destroy_plan: dict,
    approval_path: Path,
) -> dict:
    require_approval(approval_path, "cloud-destroy", destroy_plan)
    if destroy_plan.get("upload_plan_sha256") != upload_plan.get("plan_sha256"):
        raise MRIStudyError("Destroy plan is not tied to the current upload plan.")
    deleted = 0
    for item in upload_plan["files"]:
        client.delete_instance(item["study_uid"], item["series_uid"], item["sop_instance_uid"])
        deleted += 1
    return {"deleted_instances": deleted}
