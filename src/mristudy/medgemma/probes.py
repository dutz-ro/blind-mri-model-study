from __future__ import annotations

from typing import Any, Protocol

from ..errors import MRIStudyError
from ..utils import atomic_json, utc_now
from .output_contract import (
    LONG_OUTPUT_PROBE_MIN_TOKENS,
    LONG_OUTPUT_PROBE_RECORDS,
    MAX_NEW_TOKENS,
    OUTPUT_BUDGET_MARGIN_TOKENS,
    decoder_output_bounds,
)


class ProbeClient(Protocol):
    def synthetic_probe(self, image_count: int) -> dict[str, Any]: ...

    def synthetic_dicom_probe(self) -> dict[str, Any]: ...

    def synthetic_json_probe(self) -> dict[str, Any]: ...

    def synthetic_liveness_probe(self) -> dict[str, Any]: ...

    def decoder_bounds_probe(self) -> dict[str, Any]: ...


def run_required_probes(
    client: ProbeClient,
    report_path,
    *,
    expected_model: str,
    expected_revision: str,
    expected_runtime_digest: str,
    expected_snapshot_manifest_sha256: str,
    expected_run_plan_sha256: str,
    expected_provision_plan_sha256: str,
    expected_output_contract: dict[str, Any],
) -> dict[str, Any]:
    image_probes = []
    for count in (1, 8, 16, 32):
        result = client.synthetic_probe(count)
        image_probes.append(result)
        if not result.get("passed") or result.get("oom") or not result.get("schema_valid"):
            raise MRIStudyError(f"Synthetic {count}-image probe failed.")
        if result.get("evidence_ids") != [
            f"image-{index:06d}" for index in range(1, count + 1)
        ]:
            raise MRIStudyError(f"Synthetic {count}-image probe changed evidence IDs.")
        marker = f"MRI_STUDY_SYNTHETIC_OK_{count}"
        if result.get("expected_marker") != marker or result.get("model_response") != marker:
            raise MRIStudyError(f"Synthetic {count}-image probe marker did not match.")
        if (
            result.get("model") != expected_model
            or result.get("model_revision") != expected_revision
        ):
            raise MRIStudyError(f"Synthetic {count}-image probe used an unapproved model.")
        expected_provenance = {
            "runtime_digest": expected_runtime_digest,
            "model_snapshot_manifest_sha256": expected_snapshot_manifest_sha256,
            "approved_plan_sha256": expected_run_plan_sha256,
            "provision_plan_sha256": expected_provision_plan_sha256,
        }
        if any(result.get(key) != value for key, value in expected_provenance.items()):
            raise MRIStudyError(f"Synthetic {count}-image probe used stale provenance.")
        if result.get("decoder") is None:
            raise MRIStudyError(
                f"Synthetic {count}-image probe did not report decoder provenance."
            )
        if result.get("output_contract") != expected_output_contract:
            raise MRIStudyError(
                f"Synthetic {count}-image probe used an unapproved output contract."
            )
    dicom_probe = client.synthetic_dicom_probe()
    expected_dicom_ids = [f"image-{index:06d}" for index in range(1, 33)]
    if (
        not dicom_probe.get("passed")
        or dicom_probe.get("oom")
        or dicom_probe.get("image_count") != 32
        or dicom_probe.get("source_type") != "contiguous_synthetic_dicom"
        or dicom_probe.get("schema_valid") is not True
        or dicom_probe.get("evidence_ids") != expected_dicom_ids
        or dicom_probe.get("expected_marker") != "MRI_STUDY_SYNTHETIC_OK_32"
        or dicom_probe.get("model_response") != "MRI_STUDY_SYNTHETIC_OK_32"
        or dicom_probe.get("model") != expected_model
        or dicom_probe.get("model_revision") != expected_revision
        or dicom_probe.get("runtime_digest") != expected_runtime_digest
        or dicom_probe.get("model_snapshot_manifest_sha256")
        != expected_snapshot_manifest_sha256
        or dicom_probe.get("approved_plan_sha256") != expected_run_plan_sha256
        or dicom_probe.get("provision_plan_sha256") != expected_provision_plan_sha256
        or dicom_probe.get("decoder") is None
        or dicom_probe.get("output_contract") != expected_output_contract
    ):
        raise MRIStudyError("Synthetic 32-DICOM probe failed.")
    json_probe = client.synthetic_json_probe()
    if (
        not json_probe.get("passed")
        or json_probe.get("oom")
        or json_probe.get("schema_valid") is not True
        or json_probe.get("status") not in {"exact_valid", "normalizable_valid"}
        or json_probe.get("decoder") is None
        or json_probe.get("model") != expected_model
        or json_probe.get("model_revision") != expected_revision
        or json_probe.get("runtime_digest") != expected_runtime_digest
        or json_probe.get("model_snapshot_manifest_sha256")
        != expected_snapshot_manifest_sha256
        or json_probe.get("approved_plan_sha256") != expected_run_plan_sha256
        or json_probe.get("provision_plan_sha256") != expected_provision_plan_sha256
        or json_probe.get("output_contract") != expected_output_contract
    ):
        raise MRIStudyError("Synthetic constrained JSON probe failed.")
    liveness_probe = client.synthetic_liveness_probe()
    if (
        not liveness_probe.get("passed")
        or liveness_probe.get("oom")
        or liveness_probe.get("schema_valid") is not True
        or liveness_probe.get("record_count") != LONG_OUTPUT_PROBE_RECORDS
        or liveness_probe.get("evidence_ids_covered") is not True
        or liveness_probe.get("distinct_evidence_ids") != 32
        or liveness_probe.get("generated_tokens", 0) < LONG_OUTPUT_PROBE_MIN_TOKENS
        or liveness_probe.get("minimum_required_tokens") != LONG_OUTPUT_PROBE_MIN_TOKENS
        or liveness_probe.get("input_images") != 32
        or (liveness_probe.get("decoder_replay") or {}).get("accepted") is not True
        or liveness_probe.get("generation", {}).get("repetition_penalty") != 1.0
        or liveness_probe.get("generation", {}).get("no_repeat_ngram_size") != 0
        or liveness_probe.get("decoder") is None
        or liveness_probe.get("model") != expected_model
        or liveness_probe.get("model_revision") != expected_revision
        or liveness_probe.get("runtime_digest") != expected_runtime_digest
        or liveness_probe.get("model_snapshot_manifest_sha256")
        != expected_snapshot_manifest_sha256
        or liveness_probe.get("approved_plan_sha256") != expected_run_plan_sha256
        or liveness_probe.get("provision_plan_sha256") != expected_provision_plan_sha256
        or liveness_probe.get("output_contract") != expected_output_contract
    ):
        raise MRIStudyError("Synthetic constrained-decoder liveness probe failed.")
    bounds_probe = client.decoder_bounds_probe()
    # A medical window failed after producing structurally valid output that did
    # not fit the token budget, because nothing bounded the decodable result. The
    # runtime now proves, with the tokenizer that will generate the windows, that
    # its largest decodable output fits with margin before any image is sent.
    if (
        not bounds_probe.get("passed")
        or bounds_probe.get("canonical_validation_errors")
        or bounds_probe.get("max_new_tokens") != MAX_NEW_TOKENS
        or bounds_probe.get("margin_tokens") != OUTPUT_BUDGET_MARGIN_TOKENS
        or bounds_probe.get("output_bounds") != decoder_output_bounds()
        or int(bounds_probe.get("required_tokens_with_margin") or MAX_NEW_TOKENS + 1)
        > MAX_NEW_TOKENS
        or bounds_probe.get("decoder") is None
        or bounds_probe.get("model") != expected_model
        or bounds_probe.get("model_revision") != expected_revision
        or bounds_probe.get("runtime_digest") != expected_runtime_digest
        or bounds_probe.get("model_snapshot_manifest_sha256")
        != expected_snapshot_manifest_sha256
        or bounds_probe.get("approved_plan_sha256") != expected_run_plan_sha256
        or bounds_probe.get("provision_plan_sha256") != expected_provision_plan_sha256
        or bounds_probe.get("output_contract") != expected_output_contract
    ):
        raise MRIStudyError("Decoder output budget probe failed.")
    report = {
        "completed_at": utc_now(),
        "image_probes": image_probes,
        "dicom_probe": dicom_probe,
        "json_decoder_probe": json_probe,
        "decoder_liveness_probe": liveness_probe,
        "decoder_bounds_probe": bounds_probe,
        "medical_data_authorized": False,
        "required_32_image_probe_passed": True,
        "required_decoder_liveness_probe_passed": True,
        "required_output_budget_probe_passed": True,
    }
    atomic_json(report_path, report)
    return report
