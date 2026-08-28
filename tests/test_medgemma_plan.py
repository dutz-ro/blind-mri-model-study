from pathlib import Path

import pydicom
from pydicom.uid import generate_uid

from mristudy.dicom.fixtures import create_synthetic_mri
from mristudy.medgemma.evidence import (
    ROLE_ORDER,
    assign_global_evidence,
    finalize_evidence_map,
    ordered_series_instances,
)
from mristudy.medgemma.plan import build_medgemma_window_plan


def make_series(root: Path, role: str, count: int, offset: int) -> tuple[str, str]:
    series_root = root / role
    create_synthetic_mri(series_root, slices=count)
    first = next(series_root.glob("*.dcm"))
    dataset = pydicom.dcmread(first)
    series_uid = generate_uid(entropy_srcs=["mristudy-window-test", role, "series"])
    for index, path in enumerate(sorted(series_root.glob("*.dcm"))):
        dataset = pydicom.dcmread(path)
        dataset.SeriesInstanceUID = series_uid
        dataset.SOPInstanceUID = generate_uid(
            entropy_srcs=["mristudy-window-test", role, str(index)]
        )
        dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        dataset.SeriesDescription = role
        dataset.ProtocolName = role
        dataset.ImagePositionPatient = [0.0, 0.0, float(offset + index)]
        dataset.InstanceNumber = index + 1
        dataset.save_as(path, enforce_file_format=True)
    return series_uid, role


def test_full_plan_has_617_unique_slices_and_26_windows(tmp_path: Path) -> None:
    counts = dict(zip(ROLE_ORDER, [430, 50, 50, 60, 27], strict=True))
    role_by_series = {}
    offset = 0
    for role, count in counts.items():
        series_uid, _ = make_series(tmp_path, role, count, offset)
        role_by_series[series_uid] = role
        offset += count
    series = ordered_series_instances(tmp_path, role_by_series)
    evidence_map, _ = assign_global_evidence(series)
    orthanc = {
        "manifest_sha256": "orthanc",
        "instance_uids": sorted(item["sop_instance_uid"] for item in evidence_map.values()),
    }
    plan, evidence = build_medgemma_window_plan(
        series_by_uid=series,
        orthanc_manifest=orthanc,
        model_repository="google/medgemma-1.5-4b-it",
        model_revision="revision",
        runtime_digest="sha256:runtime",
        historical_replay_sha256="a" * 64,
        excluded_series=[{"series_description": "MPR-SmartKnee", "reason": "excluded"}],
    )
    assert plan["total_source_slices"] == 617
    assert plan["unique_analyzed_slices"] == 617
    assert plan["coverage_percent"] == 100.0
    assert plan["total_windows"] == 26
    assert plan["maximum_window_size"] <= 32
    assert plan["minimum_overlap"] == 8
    assert plan["missing_instance_uids"] == []
    assert plan["unexpected_instance_uids"] == []
    assert evidence["evidence_count"] == 617
    ids = [item["evidence_id"] for item in evidence["entries"]]
    assert ids == [f"image-{index:06d}" for index in range(1, 618)]
    overlapping = [item for item in evidence["entries"] if len(item["window_ids"]) == 2]
    assert len(overlapping) == plan["duplicate_window_memberships"]
    assert plan["duplicate_window_memberships"] == 168
    counts_by_role = {
        role: sum(window["role"] == role for window in plan["windows"])
        for role in ROLE_ORDER
    }
    assert counts_by_role == {
        "primary-volume": 18,
        "sagittal-meniscus": 2,
        "coronal-confirmation": 2,
        "axial-confirmation": 3,
        "anatomy-correlation": 1,
    }
    assert all(window["model_revision"] == "revision" for window in plan["windows"])
    assert all(window["runtime_digest"] == "sha256:runtime" for window in plan["windows"])
    assert all(
        window["generation"]
        == {
            "do_sample": False,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 8,
            "json_schema_enforced": True,
            "canonical_eos_only": True,
        }
        for window in plan["windows"]
    )
    assert all(window["decoder"] == plan["decoder"] for window in plan["windows"])
    assert plan["decoder"]["implementation"] == "lm-format-enforcer"
    assert plan["decoder"]["version"] == "0.11.3"
    assert plan["decoder"]["config"]["max_json_array_length"] == 128
    assert plan["historical_replay_sha256"] == "a" * 64
    assert plan["schema_version"] == 3
    assert plan["output_contract"]["validation"] == "medgemma-output-validation-v2"
    assert plan["evidence_trace_contract"]["policy"] == "evidence-traceability-v2"
    assert all(window["output_contract"] == plan["output_contract"] for window in plan["windows"])


def test_geometry_fallback_is_explicit(tmp_path: Path) -> None:
    series_uid, role = make_series(tmp_path, "primary-volume", 3, 0)
    for path in (tmp_path / role).glob("*.dcm"):
        dataset = pydicom.dcmread(path)
        del dataset.ImagePositionPatient
        dataset.save_as(path, enforce_file_format=True)
    result = ordered_series_instances(tmp_path, {series_uid: role})[series_uid]
    assert result["ordering_method"] == "instance_number_fallback"
    assert result["ordering_warning"]


def test_global_evidence_id_never_points_to_two_uids() -> None:
    series = {
        "series-1": {
            "series_uid": "series-1",
            "role": "primary-volume",
            "series_description": "primary",
            "ordering_method": "dicom_geometry",
            "ordering_warning": None,
            "instances": [
                {
                    "study_uid": "study",
                    "series_uid": "series-1",
                    "sop_instance_uid": "1.2.3.1",
                    "geometric_position": 1.0,
                    "instance_number": 1,
                }
            ],
        }
    }
    for role in ROLE_ORDER[1:]:
        uid = f"series-{role}"
        series[uid] = {
            "series_uid": uid,
            "role": role,
            "series_description": role,
            "ordering_method": "dicom_geometry",
            "ordering_warning": None,
            "instances": [
                {
                    "study_uid": "study",
                    "series_uid": uid,
                    "sop_instance_uid": f"1.2.3.{len(series) + 1}",
                    "geometric_position": 1.0,
                    "instance_number": 1,
                }
            ],
        }
    evidence, _ = assign_global_evidence(series)
    manifest = finalize_evidence_map(evidence)
    assert len({item["sop_instance_uid"] for item in manifest["entries"]}) == len(
        manifest["entries"]
    )


def test_duplicate_sop_uid_across_series_is_rejected() -> None:
    series = {}
    for index, role in enumerate(ROLE_ORDER):
        series_uid = f"series-{index}"
        series[series_uid] = {
            "series_uid": series_uid,
            "role": role,
            "series_description": role,
            "ordering_method": "dicom_geometry",
            "ordering_warning": None,
            "instances": [
                {
                    "study_uid": "study",
                    "series_uid": series_uid,
                    "sop_instance_uid": "1.2.3.duplicate" if index < 2 else f"1.2.3.{index}",
                    "geometric_position": 1.0,
                    "instance_number": 1,
                }
            ],
        }
    import pytest

    with pytest.raises(ValueError, match="duplicated across selected series"):
        assign_global_evidence(series)
