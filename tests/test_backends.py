from pathlib import Path

import httpx
import pytest

from mristudy.backends.dicomweb import BasicAuth, DicomWebClient, DicomWebEndpoint
from mristudy.backends.plans import revalidate_planned_file
from mristudy.backends.scaleway import final_provision_plan
from mristudy.dicom.fixtures import create_synthetic_mri
from mristudy.errors import MRIStudyError
from mristudy.utils import sha256_file


def test_generic_orthanc_dicomweb_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/dicom-web/studies"
        assert request.headers["authorization"].startswith("Basic ")
        return httpx.Response(200, json=[])

    client = DicomWebClient(
        DicomWebEndpoint("http://orthanc/dicom-web/", "local-orthanc", "test"),
        BasicAuth("user", "password"),
        httpx.MockTransport(handler),
    )
    assert client.search_studies() == []


def test_stow_uses_multipart_related(tmp_path: Path) -> None:
    create_synthetic_mri(tmp_path, slices=1)
    path = next(tmp_path.glob("*.dcm"))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.headers["content-type"].startswith("multipart/related")
        assert b"Content-Type: application/dicom" in request.content
        return httpx.Response(200, json={})

    client = DicomWebClient(
        DicomWebEndpoint("http://orthanc/dicom-web", "local-orthanc", "test"),
        BasicAuth("user", "password"),
        httpx.MockTransport(handler),
    )
    assert client.store_instance(path) == {}


def test_revalidate_rejects_changed_anonymous_dicom(tmp_path: Path) -> None:
    root = tmp_path / "anonymous"
    create_synthetic_mri(root, slices=1)
    path = next(root.glob("*.dcm"))
    import pydicom

    ds = pydicom.dcmread(path, stop_before_pixels=True)
    item = {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "study_uid": str(ds.StudyInstanceUID),
        "series_uid": str(ds.SeriesInstanceUID),
        "sop_instance_uid": str(ds.SOPInstanceUID),
    }
    assert revalidate_planned_file(item, root) == path
    path.write_bytes(path.read_bytes() + b"changed")
    try:
        revalidate_planned_file(item, root)
    except Exception as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("Modified DICOM was accepted")


def test_scaleway_final_plan_pins_cost_and_artifacts(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.tar"
    archive.write_bytes(b"runtime")
    cloud_init = tmp_path / "cloud-init.yaml"
    cloud_init.write_text("#cloud-config\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model_file = snapshot / "config.json"
    model_file.write_text("{}", encoding="utf-8")
    (snapshot / "manifest.sha256").write_text(
        f"{sha256_file(model_file)}  config.json\n", encoding="utf-8"
    )
    operation = {
        "plan_sha256": "backend-plan",
        "lifecycle": {"hard_ttl_hours": 6},
        "security": {"ssh_source_cidr": "192.0.2.1/32"},
        "medgemma": {
            "repository": "google/medgemma-1.5-4b-it",
            "revision": "revision",
        },
        "cost_control": {"maximum_eur": 50},
    }
    plan = final_provision_plan(
        operation_plan=operation,
        project_id="project",
        zone="fr-par-2",
        server_types=[
            {
                "name": "L4-1-24G",
                "availability": "available",
                "hourly_price": {
                    "currency_code": "EUR",
                    "units": 0,
                    "nanos": 787_500_000,
                },
            },
            {
                "name": "L40S-1-48G",
                "availability": "shortage",
                "hourly_price": {
                    "currency_code": "EUR",
                    "units": 1,
                    "nanos": 469_916_000,
                },
            },
        ],
        marketplace_image_id="image",
        container_digest="sha256:image",
        container_tar_path=archive,
        container_tar_sha256=sha256_file(archive),
        cloud_init_path=cloud_init,
        model_snapshot_path=snapshot,
        model_snapshot_manifest_sha256=sha256_file(snapshot / "manifest.sha256"),
        medgemma_run_plan_sha256="medgemma-plan",
        evidence_map_sha256="evidence-map",
    )
    assert plan["resources"]["server"]["type"] == "L4-1-24G"
    assert plan["cost"]["l4_hourly"] == 0.7875
    assert plan["cost"]["l4_compute_ceiling_excluding_storage_and_ip"] == 4.725
    assert plan["resources"]["fallback"]["requires_new_plan_and_approval"]
    assert plan["medgemma_run_plan_sha256"] == "medgemma-plan"


def test_scaleway_plan_requires_medgemma_and_evidence_digests(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.tar"
    archive.write_bytes(b"runtime")
    cloud_init = tmp_path / "cloud-init.yaml"
    cloud_init.write_text("#cloud-config\n", encoding="utf-8")
    operation = {
        "plan_sha256": "backend-plan",
        "lifecycle": {"hard_ttl_hours": 6},
        "security": {"ssh_source_cidr": "192.0.2.1/32"},
        "medgemma": {"repository": "model", "revision": "revision"},
        "cost_control": {"maximum_eur": 50},
    }
    with pytest.raises(MRIStudyError, match="bind MedGemma"):
        final_provision_plan(
            operation_plan=operation,
            project_id="project",
            zone="fr-par-2",
            server_types=[],
            marketplace_image_id="image",
            container_digest="sha256:image",
            container_tar_path=archive,
            container_tar_sha256=sha256_file(archive),
            cloud_init_path=cloud_init,
        )
