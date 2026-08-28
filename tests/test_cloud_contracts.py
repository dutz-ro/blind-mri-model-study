import json

import httpx
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from mristudy.cloud.dicomweb import DicomStoreRef, DicomWebClient
from mristudy.cloud.medgemma import build_dicom_request, series_url
from mristudy.cloud.plans import discover_uploadable_images, infrastructure_plan, stow_plan
from mristudy.models.litellm import LiteLLMClient
from mristudy.utils import canonical_json, sha256_bytes


class Token:
    def header(self) -> str:
        return "Bearer test"


def test_qido_url_and_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/dicomWeb/studies")
        assert request.headers["authorization"] == "Bearer test"
        assert request.headers["accept"] == "application/dicom+json"
        return httpx.Response(200, json=[])

    store = DicomStoreRef("project", "europe-west4", "dataset", "store")
    client = DicomWebClient(store, Token(), httpx.MockTransport(handler))
    assert client.search_studies() == []


def test_medgemma_request_uses_application_default() -> None:
    store = DicomStoreRef("project", "europe-west4", "dataset", "store")
    source = series_url(store, "1.2.3", "1.2.4")
    request = build_dicom_request(source, "test")
    block = request["messages"][0]["content"][1]["image_dicom"]
    assert block["dicom_source"] == source
    assert block["access_credential"] == "application_default"
    json.dumps(request)


def test_litellm_upstream_verification_requires_routera() -> None:
    class Client(LiteLLMClient):
        def model_info(self) -> dict:
            return {
                "data": [
                    {
                        "model_name": "model-a",
                        "litellm_params": {
                            "model": "openai/model-a",
                            "api_base": "https://api.routera.one/v1",
                        },
                    }
                ]
            }

    result = Client("http://unused").verify_upstream_host(["model-a"], "api.routera.one")
    assert result["model-a"]["upstream_host"] == "api.routera.one"


def test_cloud_plan_excludes_non_image_dicom(tmp_path) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    image = FileDataset("image.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    image.SOPClassUID = MRImageStorage
    image.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    image.StudyInstanceUID = generate_uid()
    image.SeriesInstanceUID = generate_uid()
    image.Rows = 2
    image.Columns = 2
    image.SamplesPerPixel = 1
    image.PhotometricInterpretation = "MONOCHROME2"
    image.BitsAllocated = 16
    image.BitsStored = 12
    image.HighBit = 11
    image.PixelRepresentation = 0
    image.PixelData = b"\0" * 8
    image.save_as(tmp_path / "image.dcm", enforce_file_format=True)

    non_image = FileDataset("index.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    non_image.SOPClassUID = MRImageStorage
    non_image.SOPInstanceUID = generate_uid()
    non_image.StudyInstanceUID = image.StudyInstanceUID
    non_image.SeriesInstanceUID = image.SeriesInstanceUID
    non_image.save_as(tmp_path / "index.dcm", enforce_file_format=True)

    images, skipped = discover_uploadable_images(tmp_path)
    assert images == [tmp_path / "image.dcm"]
    assert skipped[0]["reason"] == "not-an-image-instance"

    store = DicomStoreRef("project", "europe-west4", "dataset", "store")
    upload = stow_plan(images, store)
    upload["skipped"] = skipped
    upload["plan_sha256"] = sha256_bytes(
        canonical_json({key: value for key, value in upload.items() if key != "plan_sha256"})
    )
    assert upload["plan_sha256"] == sha256_bytes(
        canonical_json({key: value for key, value in upload.items() if key != "plan_sha256"})
    )
    plan = infrastructure_plan(
        project="project",
        region="europe-west4",
        study_id="study-test",
        upload_plan=upload,
        existing_budget_usd=5,
        approved_max_cost_usd=50,
        global_gpu_quota=0,
    )
    assert not plan["preconditions"]["gpu_quota_sufficient"]
    assert not plan["preconditions"]["budget_matches_approved_max"]
