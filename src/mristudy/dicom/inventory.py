from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import pydicom
from pydicom.errors import InvalidDicomError

from ..utils import sha256_file
from .models import InstanceRecord, InventoryReport, Issue, Severity, safe_relative

INVENTORY_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SOPClassUID",
    "Modality",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ImageType",
    "InstanceNumber",
    "Rows",
    "Columns",
    "NumberOfFrames",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "RepetitionTime",
    "EchoTime",
    "Laterality",
    "ImageLaterality",
    "PhotometricInterpretation",
    "PixelDataProviderURL",
]


def _float_tuple(value: Any, length: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        result = tuple(float(part) for part in value)
    except (TypeError, ValueError):
        return None
    return result if len(result) == length else None


def _number(value: Any, cast: type = float) -> Any | None:
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _read(path: Path, allow_force: bool) -> tuple[pydicom.Dataset, bool]:
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, specific_tags=INVENTORY_TAGS), False
    except InvalidDicomError:
        if not allow_force:
            raise
        return pydicom.dcmread(
            path, stop_before_pixels=True, specific_tags=INVENTORY_TAGS, force=True
        ), True


def inventory(root: Path, *, allow_force: bool = False) -> InventoryReport:
    root = root.resolve()
    records: list[InstanceRecord] = []
    issues: list[Issue] = []
    sop_paths: dict[str, list[InstanceRecord]] = defaultdict(list)

    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = safe_relative(path, root)
        try:
            ds, forced = _read(path, allow_force)
        except (InvalidDicomError, OSError, ValueError) as error:
            issues.append(Issue(Severity.INFO, "NOT_DICOM", str(error), relative))
            continue

        if not any(getattr(ds, name, None) for name in ("SOPClassUID", "Rows", "Modality")):
            issues.append(
                Issue(
                    Severity.WARNING,
                    "WEAK_DICOM_IDENTITY",
                    "Missing core DICOM identifiers",
                    relative,
                )
            )
            continue

        file_meta = getattr(ds, "file_meta", None)
        transfer_syntax = getattr(file_meta, "TransferSyntaxUID", None) if file_meta else None
        image_type = getattr(ds, "ImageType", []) or []
        if isinstance(image_type, str):
            image_type = [image_type]
        record = InstanceRecord(
            relative_path=relative,
            file_size=path.stat().st_size,
            sha256=sha256_file(path),
            study_uid=str(getattr(ds, "StudyInstanceUID", "")) or None,
            series_uid=str(getattr(ds, "SeriesInstanceUID", "")) or None,
            sop_instance_uid=str(getattr(ds, "SOPInstanceUID", "")) or None,
            sop_class_uid=str(getattr(ds, "SOPClassUID", "")) or None,
            modality=str(getattr(ds, "Modality", "")) or None,
            transfer_syntax_uid=str(transfer_syntax) if transfer_syntax else None,
            series_description=str(getattr(ds, "SeriesDescription", "")) or None,
            protocol_name=str(getattr(ds, "ProtocolName", "")) or None,
            sequence_name=str(getattr(ds, "SequenceName", "")) or None,
            image_type=tuple(str(item) for item in image_type),
            instance_number=_number(getattr(ds, "InstanceNumber", None), int),
            rows=_number(getattr(ds, "Rows", None), int),
            columns=_number(getattr(ds, "Columns", None), int),
            number_of_frames=_number(getattr(ds, "NumberOfFrames", 1), int) or 1,
            image_position=_float_tuple(getattr(ds, "ImagePositionPatient", None), 3),
            image_orientation=_float_tuple(getattr(ds, "ImageOrientationPatient", None), 6),
            pixel_spacing=_float_tuple(getattr(ds, "PixelSpacing", None), 2),
            slice_thickness=_number(getattr(ds, "SliceThickness", None)),
            spacing_between_slices=_number(getattr(ds, "SpacingBetweenSlices", None)),
            repetition_time=_number(getattr(ds, "RepetitionTime", None)),
            echo_time=_number(getattr(ds, "EchoTime", None)),
            laterality=str(
                getattr(ds, "ImageLaterality", None) or getattr(ds, "Laterality", "")
            ) or None,
            photometric_interpretation=str(
                getattr(ds, "PhotometricInterpretation", "")
            ) or None,
            pixel_data_present=bool(
                getattr(ds, "PixelDataProviderURL", None)
                or (
                    _number(getattr(ds, "Rows", None), int)
                    and _number(getattr(ds, "Columns", None), int)
                    and path.stat().st_size > 512
                )
            ),
            forced_read=forced,
        )
        records.append(record)
        if record.sop_instance_uid:
            sop_paths[record.sop_instance_uid].append(record)
        if forced:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "FORCED_DICOM_READ",
                    "Read required force=True",
                    relative,
                )
            )

    for uid, duplicates in sorted(sop_paths.items()):
        if len(duplicates) > 1:
            hashes = {item.sha256 for item in duplicates}
            severity = Severity.WARNING if len(hashes) == 1 else Severity.BLOCKER
            issues.append(
                Issue(
                    severity,
                    "DUPLICATE_SOP_UID",
                    f"SOP Instance UID occurs {len(duplicates)} times",
                    context={
                        "uid_sha256": hashlib.sha256(uid.encode()).hexdigest(),
                        "identical": len(hashes) == 1,
                    },
                )
            )
    return InventoryReport(str(root), records, issues)
