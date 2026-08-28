from pathlib import Path

import pytest

from mristudy.clinical_schema import load_knee_meniscus_schema
from mristudy.errors import MRIStudyError
from mristudy.medgemma.output_contract import decoder_contract, output_contract
from mristudy.medgemma.probes import run_required_probes


class FakeProbeClient:
    def __init__(self, fail_count: int | None = None):
        self.fail_count = fail_count

    def synthetic_probe(self, image_count: int):
        return {
            "passed": image_count != self.fail_count,
            "oom": image_count == self.fail_count,
            "schema_valid": True,
            "image_count": image_count,
            "evidence_ids": [f"image-{index:06d}" for index in range(1, image_count + 1)],
            "expected_marker": f"MRI_STUDY_SYNTHETIC_OK_{image_count}",
            "model_response": f"MRI_STUDY_SYNTHETIC_OK_{image_count}",
            "model": "google/medgemma-1.5-4b-it",
            "model_revision": "revision",
            "runtime_digest": "sha256:runtime",
            "decoder": decoder_contract(load_knee_meniscus_schema()),
            "output_contract": output_contract(),
            "model_snapshot_manifest_sha256": "snapshot",
            "approved_plan_sha256": "run-plan",
            "provision_plan_sha256": "provision-plan",
            "metrics": {
                "total_seconds": 1,
                "encoding_seconds": 0.2,
                "generation_seconds": 0.8,
                "gpu_peak_memory_bytes": 100,
                "tokens": 10,
            },
        }

    def synthetic_dicom_probe(self):
        return {
            "passed": True,
            "oom": False,
            "schema_valid": True,
            "image_count": 32,
            "source_type": "contiguous_synthetic_dicom",
            "evidence_ids": [f"image-{index:06d}" for index in range(1, 33)],
            "expected_marker": "MRI_STUDY_SYNTHETIC_OK_32",
            "model_response": "MRI_STUDY_SYNTHETIC_OK_32",
            "model": "google/medgemma-1.5-4b-it",
            "model_revision": "revision",
            "runtime_digest": "sha256:runtime",
            "decoder": decoder_contract(load_knee_meniscus_schema()),
            "output_contract": output_contract(),
            "model_snapshot_manifest_sha256": "snapshot",
            "approved_plan_sha256": "run-plan",
            "provision_plan_sha256": "provision-plan",
        }

    def synthetic_json_probe(self):
        return {
            "passed": True,
            "oom": False,
            "schema_valid": True,
            "status": "exact_valid",
            "candidate_count": 1,
            "decoder": decoder_contract(load_knee_meniscus_schema()),
            "output_contract": output_contract(),
            "model": "google/medgemma-1.5-4b-it",
            "model_revision": "revision",
            "runtime_digest": "sha256:runtime",
            "model_snapshot_manifest_sha256": "snapshot",
            "approved_plan_sha256": "run-plan",
            "provision_plan_sha256": "provision-plan",
        }


def test_required_probes_include_1_8_16_32_and_dicom(tmp_path: Path) -> None:
    report = run_required_probes(
        FakeProbeClient(),
        tmp_path / "probes.json",
        expected_model="google/medgemma-1.5-4b-it",
        expected_revision="revision",
        expected_runtime_digest="sha256:runtime",
        expected_snapshot_manifest_sha256="snapshot",
        expected_run_plan_sha256="run-plan",
        expected_provision_plan_sha256="provision-plan",
        expected_output_contract=output_contract(),
    )
    assert [item["image_count"] for item in report["image_probes"]] == [1, 8, 16, 32]
    assert report["dicom_probe"]["image_count"] == 32
    assert report["json_decoder_probe"]["passed"]
    assert not report["medical_data_authorized"]


def test_32_image_probe_failure_blocks_medical_data(tmp_path: Path) -> None:
    with pytest.raises(MRIStudyError, match="32-image"):
        run_required_probes(
            FakeProbeClient(fail_count=32),
            tmp_path / "probes.json",
            expected_model="google/medgemma-1.5-4b-it",
            expected_revision="revision",
            expected_runtime_digest="sha256:runtime",
            expected_snapshot_manifest_sha256="snapshot",
            expected_run_plan_sha256="run-plan",
            expected_provision_plan_sha256="provision-plan",
            expected_output_contract=output_contract(),
        )


def test_probe_rejects_wrong_model_identity(tmp_path: Path) -> None:
    client = FakeProbeClient()
    original = client.synthetic_probe

    def wrong_model(image_count: int):
        result = original(image_count)
        result["model_revision"] = "wrong"
        return result

    client.synthetic_probe = wrong_model
    with pytest.raises(MRIStudyError, match="unapproved model"):
        run_required_probes(
            client,
            tmp_path / "probes.json",
            expected_model="google/medgemma-1.5-4b-it",
            expected_revision="revision",
            expected_runtime_digest="sha256:runtime",
            expected_snapshot_manifest_sha256="snapshot",
            expected_run_plan_sha256="run-plan",
            expected_provision_plan_sha256="provision-plan",
            expected_output_contract=output_contract(),
        )


def test_probe_rejects_wrong_capacity_marker(tmp_path: Path) -> None:
    client = FakeProbeClient()
    original = client.synthetic_probe

    def wrong_marker(image_count: int):
        result = original(image_count)
        result["model_response"] = "wrong"
        return result

    client.synthetic_probe = wrong_marker
    with pytest.raises(MRIStudyError, match="marker did not match"):
        run_required_probes(
            client,
            tmp_path / "probes.json",
            expected_model="google/medgemma-1.5-4b-it",
            expected_revision="revision",
            expected_runtime_digest="sha256:runtime",
            expected_snapshot_manifest_sha256="snapshot",
            expected_run_plan_sha256="run-plan",
            expected_provision_plan_sha256="provision-plan",
            expected_output_contract=output_contract(),
        )


def test_probe_rejects_failed_constrained_json_decoder(tmp_path: Path) -> None:
    client = FakeProbeClient()
    original = client.synthetic_json_probe

    def failed_decoder():
        result = original()
        result["passed"] = False
        return result

    client.synthetic_json_probe = failed_decoder
    with pytest.raises(MRIStudyError, match="constrained JSON"):
        run_required_probes(
            client,
            tmp_path / "probes.json",
            expected_model="google/medgemma-1.5-4b-it",
            expected_revision="revision",
            expected_runtime_digest="sha256:runtime",
            expected_snapshot_manifest_sha256="snapshot",
            expected_run_plan_sha256="run-plan",
            expected_provision_plan_sha256="provision-plan",
            expected_output_contract=output_contract(),
        )


def test_image_probe_without_decoder_provenance_blocks_medical_access(
    tmp_path: Path,
) -> None:
    """A capacity probe must prove the decoder that medical generation will use."""
    client = FakeProbeClient()
    original = client.synthetic_probe

    def missing_decoder(image_count: int):
        result = original(image_count)
        result.pop("decoder", None)
        return result

    client.synthetic_probe = missing_decoder
    with pytest.raises(MRIStudyError, match="decoder provenance"):
        run_required_probes(
            client,
            tmp_path / "probes.json",
            expected_model="google/medgemma-1.5-4b-it",
            expected_revision="revision",
            expected_runtime_digest="sha256:runtime",
            expected_snapshot_manifest_sha256="snapshot",
            expected_run_plan_sha256="run-plan",
            expected_provision_plan_sha256="provision-plan",
            expected_output_contract=output_contract(),
        )
    assert not (tmp_path / "probes.json").exists()


def test_dicom_probe_without_decoder_provenance_blocks_medical_access(
    tmp_path: Path,
) -> None:
    client = FakeProbeClient()
    original = client.synthetic_dicom_probe

    def missing_decoder():
        result = original()
        result.pop("decoder", None)
        return result

    client.synthetic_dicom_probe = missing_decoder
    with pytest.raises(MRIStudyError, match="32-DICOM"):
        run_required_probes(
            client,
            tmp_path / "probes.json",
            expected_model="google/medgemma-1.5-4b-it",
            expected_revision="revision",
            expected_runtime_digest="sha256:runtime",
            expected_snapshot_manifest_sha256="snapshot",
            expected_run_plan_sha256="run-plan",
            expected_provision_plan_sha256="provision-plan",
            expected_output_contract=output_contract(),
        )
    assert not (tmp_path / "probes.json").exists()


def test_probe_with_stale_output_contract_blocks_medical_access(tmp_path: Path) -> None:
    client = FakeProbeClient()
    original = client.synthetic_probe

    def stale_contract(image_count: int):
        result = original(image_count)
        result["output_contract"] = {"validation": "old"}
        return result

    client.synthetic_probe = stale_contract
    with pytest.raises(MRIStudyError, match="unapproved output contract"):
        run_required_probes(
            client,
            tmp_path / "probes.json",
            expected_model="google/medgemma-1.5-4b-it",
            expected_revision="revision",
            expected_runtime_digest="sha256:runtime",
            expected_snapshot_manifest_sha256="snapshot",
            expected_run_plan_sha256="run-plan",
            expected_provision_plan_sha256="provision-plan",
            expected_output_contract=output_contract(),
        )
    assert not (tmp_path / "probes.json").exists()
