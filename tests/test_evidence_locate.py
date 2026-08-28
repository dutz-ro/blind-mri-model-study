"""Resolving an evidence identifier back to a checkable image.

A finding is only actionable if a human can put the underlying slice in front of
a clinician and confirm it is the same image the model saw.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mristudy.errors import MRIStudyError
from mristudy.medgemma.evidence_locate import (
    collect_cited_evidence_ids,
    locate_evidence,
    locate_many,
)

SOP = "1.2.826.0.1.3680043.8.498.111"
SERIES = "1.2.826.0.1.3680043.8.498.222"
STUDY = "1.2.826.0.1.3680043.8.498.333"
ORIGINAL_SOP = "1.3.46.670589.11.29072.5.0.999"


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_count": 1,
        "entries": [
            {
                "evidence_id": "image-000211",
                "study_uid": STUDY,
                "series_uid": SERIES,
                "sop_instance_uid": SOP,
                "geometric_index": 210,
                "geometric_position": -12.5,
                "role": "primary-volume",
                "window_ids": [],
            }
        ],
    }


def _run_plan() -> dict[str, Any]:
    return {
        "windows": [
            {"window_id": "primary-volume-window-0009", "evidence_ids": ["image-000211"]},
            {"window_id": "primary-volume-window-0010", "evidence_ids": ["image-000212"]},
        ]
    }


def _write_instance(root: Path) -> Path:
    pydicom = pytest.importorskip("pydicom")
    root.mkdir(parents=True, exist_ok=True)
    dataset = pydicom.Dataset()
    dataset.SOPInstanceUID = SOP
    dataset.SeriesInstanceUID = SERIES
    dataset.StudyInstanceUID = STUDY
    dataset.SOPClassUID = pydicom.uid.MRImageStorage
    dataset.file_meta = pydicom.dataset.FileMetaDataset()
    dataset.file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    dataset.file_meta.MediaStorageSOPInstanceUID = SOP
    dataset.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    path = root / "IM000454"
    dataset.save_as(path, enforce_file_format=True)
    return path


def test_located_reference_identifies_the_exact_instance(tmp_path: Path) -> None:
    root = tmp_path / "anonymized-dicom"
    written = _write_instance(root / "DICOM")
    reference = locate_evidence(
        "image-000211", _manifest(), root, run_plan=_run_plan()
    )
    assert reference["sop_instance_uid"] == SOP
    assert reference["role"] == "primary-volume"
    assert reference["geometric_index"] == 210
    assert reference["anonymised_relative_path"] == "DICOM/IM000454"
    assert reference["anonymised_sha256"]
    assert written.exists()


def test_window_membership_is_reported(tmp_path: Path) -> None:
    """Overlap means a citation can belong to a neighbouring window."""
    root = tmp_path / "anonymized-dicom"
    _write_instance(root / "DICOM")
    reference = locate_evidence(
        "image-000211", _manifest(), root, run_plan=_run_plan()
    )
    assert reference["window_ids"] == ["primary-volume-window-0009"]


def test_missing_instance_is_reported_not_fatal(tmp_path: Path) -> None:
    root = tmp_path / "anonymized-dicom"
    root.mkdir(parents=True)
    reference = locate_evidence("image-000211", _manifest(), root)
    assert reference["anonymised_path"] is None
    assert "locate_warning" in reference


def test_unknown_evidence_identifier_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "anonymized-dicom"
    root.mkdir(parents=True)
    with pytest.raises(MRIStudyError):
        locate_evidence("image-999999", _manifest(), root)


def test_original_reference_requires_explicit_opt_in(tmp_path: Path) -> None:
    """Resolving the original UID re-identifies the study."""
    root = tmp_path / "anonymized-dicom"
    _write_instance(root / "DICOM")
    reference = locate_evidence("image-000211", _manifest(), root)
    assert "original_sop_instance_uid" not in reference
    with pytest.raises(MRIStudyError):
        locate_evidence(
            "image-000211", _manifest(), root, include_original_reference=True
        )


def test_original_reference_never_exposes_the_deidentification_secret(
    tmp_path: Path,
) -> None:
    root = tmp_path / "anonymized-dicom"
    _write_instance(root / "DICOM")
    uid_map = tmp_path / "uid-map.json"
    uid_map.write_text(
        json.dumps(
            {
                "secret": "MUST-NOT-LEAK",
                "subject_id": "MUST-NOT-LEAK",
                "date_shift_days": -1582,
                "uid_map": {ORIGINAL_SOP: SOP},
            }
        ),
        encoding="utf-8",
    )
    reference = locate_evidence(
        "image-000211",
        _manifest(),
        root,
        uid_map_path=uid_map,
        include_original_reference=True,
    )
    assert reference["original_sop_instance_uid"] == ORIGINAL_SOP
    serialised = json.dumps(reference)
    assert "MUST-NOT-LEAK" not in serialised
    assert "date_shift_days" not in reference
    assert "secret" not in reference


def test_unmapped_anonymous_uid_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "anonymized-dicom"
    _write_instance(root / "DICOM")
    uid_map = tmp_path / "uid-map.json"
    uid_map.write_text(json.dumps({"uid_map": {}}), encoding="utf-8")
    reference = locate_evidence(
        "image-000211",
        _manifest(),
        root,
        uid_map_path=uid_map,
        include_original_reference=True,
    )
    assert reference["original_sop_instance_uid"] is None
    assert "original_reference_warning" in reference


def test_locate_many_preserves_request_order(tmp_path: Path) -> None:
    root = tmp_path / "anonymized-dicom"
    _write_instance(root / "DICOM")
    manifest = _manifest()
    manifest["entries"].append(
        {
            "evidence_id": "image-000212",
            "study_uid": STUDY,
            "series_uid": SERIES,
            "sop_instance_uid": "1.2.826.0.1.3680043.8.498.444",
            "geometric_index": 211,
            "geometric_position": -13.5,
            "role": "primary-volume",
            "window_ids": [],
        }
    )
    references = locate_many(["image-000212", "image-000211"], manifest, root)
    assert [item["evidence_id"] for item in references] == [
        "image-000212",
        "image-000211",
    ]


def test_collect_cited_evidence_ids_is_ordered_and_deduplicated() -> None:
    result = {
        "findings": [
            {
                "traceable_evidence": [
                    {"export_id": "image-000002"},
                    {"export_id": "image-000001"},
                ],
                "counter_evidence": [{"export_id": "image-000002"}],
            },
            {
                "traceable_evidence": [{"export_id": "image-000003"}],
                "counter_evidence": [],
            },
        ]
    }
    assert collect_cited_evidence_ids(result) == [
        "image-000002",
        "image-000001",
        "image-000003",
    ]


def test_collect_handles_result_without_findings() -> None:
    assert collect_cited_evidence_ids({"findings": []}) == []
    assert collect_cited_evidence_ids({}) == []
