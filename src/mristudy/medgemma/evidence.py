from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pydicom

from ..utils import canonical_json, sha256_bytes

ROLE_ORDER = (
    "primary-volume",
    "sagittal-meniscus",
    "coronal-confirmation",
    "axial-confirmation",
    "anatomy-correlation",
)


def _numbers(value: Any, length: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if len(result) == length else None


def _coordinate(dataset: pydicom.Dataset) -> float | None:
    orientation = _numbers(getattr(dataset, "ImageOrientationPatient", None), 6)
    position = _numbers(getattr(dataset, "ImagePositionPatient", None), 3)
    if not orientation or not position:
        return None
    row, column = orientation[:3], orientation[3:]
    normal = (
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    )
    norm = math.sqrt(sum(value * value for value in normal))
    if norm < 1e-8:
        return None
    return sum(
        value / norm * coordinate
        for value, coordinate in zip(normal, position, strict=True)
    )


def ordered_series_instances(root: Path, role_by_series: dict[str, str]) -> dict[str, dict]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, OSError):
            continue
        series_uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        if series_uid not in role_by_series:
            continue
        sop_uid = str(getattr(dataset, "SOPInstanceUID", ""))
        if not sop_uid:
            raise ValueError("Selected DICOM instance is missing SOP Instance UID.")
        coordinate = _coordinate(dataset)
        try:
            instance_number = int(getattr(dataset, "InstanceNumber", 0) or 0)
        except (TypeError, ValueError):
            instance_number = 0
        grouped[series_uid].append(
            {
                "study_uid": str(dataset.StudyInstanceUID),
                "series_uid": series_uid,
                "sop_instance_uid": sop_uid,
                "geometric_position": coordinate,
                "instance_number": instance_number,
            }
        )
        metadata[series_uid] = {
            "series_uid": series_uid,
            "role": role_by_series[series_uid],
            "series_description": str(getattr(dataset, "SeriesDescription", "")),
        }

    result = {}
    for series_uid, instances in grouped.items():
        geometry_complete = all(item["geometric_position"] is not None for item in instances)
        ordering_method = "dicom_geometry" if geometry_complete else "instance_number_fallback"
        if geometry_complete:
            ordered = sorted(
                instances,
                key=lambda item: (
                    item["geometric_position"],
                    item["instance_number"],
                    item["sop_instance_uid"],
                ),
            )
        else:
            ordered = sorted(
                instances,
                key=lambda item: (item["instance_number"], item["sop_instance_uid"]),
            )
        if len({item["sop_instance_uid"] for item in ordered}) != len(ordered):
            raise ValueError("Duplicate SOP Instance UID in selected series.")
        result[series_uid] = metadata[series_uid] | {
            "ordering_method": ordering_method,
            "ordering_warning": (
                None
                if geometry_complete
                else "Image geometry incomplete; ordered by InstanceNumber then SOPInstanceUID."
            ),
            "instances": ordered,
        }
    return result


def assign_global_evidence(
    series_by_uid: dict[str, dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    by_role = {item["role"]: item for item in series_by_uid.values()}
    evidence_map: dict[str, dict] = {}
    uid_to_evidence: dict[str, dict] = {}
    seen_sop_uids = set()
    next_id = 1
    for role in ROLE_ORDER:
        if role not in by_role:
            raise ValueError(f"Missing required MedGemma role: {role}")
        series = by_role[role]
        for index, instance in enumerate(series["instances"]):
            if instance["sop_instance_uid"] in seen_sop_uids:
                raise ValueError("SOP Instance UID is duplicated across selected series.")
            seen_sop_uids.add(instance["sop_instance_uid"])
            evidence_id = f"image-{next_id:06d}"
            next_id += 1
            record = {
                "evidence_id": evidence_id,
                "study_uid": instance["study_uid"],
                "series_uid": instance["series_uid"],
                "sop_instance_uid": instance["sop_instance_uid"],
                "geometric_index": index,
                "geometric_position": instance["geometric_position"],
                "role": role,
                "window_ids": [],
            }
            evidence_map[evidence_id] = record
            uid_to_evidence[instance["sop_instance_uid"]] = record
            instance["evidence_id"] = evidence_id
            instance["geometric_index"] = index
    return evidence_map, uid_to_evidence


def finalize_evidence_map(evidence_map: dict[str, dict]) -> dict:
    entries = [evidence_map[key] for key in sorted(evidence_map)]
    return {
        "schema_version": 1,
        "evidence_count": len(entries),
        "entries": entries,
        "map_sha256": sha256_bytes(canonical_json(entries)),
    }
