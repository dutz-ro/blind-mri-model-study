from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_local_dicom_network_is_configurable_and_avoids_common_docker_space() -> None:
    compose = (ROOT / "deploy/local-dicom/compose.yaml").read_text(encoding="utf-8")
    script = (ROOT / "scripts/local-dicom-stack.sh").read_text(encoding="utf-8")

    assert "MRI_STUDY_DICOM_SUBNET" in compose
    assert "10.253.50.0/24" in compose
    assert "172.30.50.0/24" not in compose
    for name in (
        "MRI_STUDY_POSTGRES_IP",
        "MRI_STUDY_ORTHANC_IP",
        "MRI_STUDY_OHIF_IP",
        "MRI_STUDY_GATEWAY_IP",
    ):
        assert name in compose
        assert name in script


def test_medical_services_are_published_only_on_loopback() -> None:
    compose = (ROOT / "deploy/local-dicom/compose.yaml").read_text(encoding="utf-8")
    assert "127.0.0.1:18042:8042" in compose
    assert "127.0.0.1:13000:80" in compose
    assert "0.0.0.0:18042" not in compose
    assert "0.0.0.0:13000" not in compose
