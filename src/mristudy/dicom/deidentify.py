from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pydicom
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from ..utils import atomic_json

REMOVE_KEYWORDS = {
    "PatientName",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientNames",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "RequestingPhysician",
    "AccessionNumber",
    "StudyID",
    "ImageComments",
    "RequestAttributesSequence",
}
EMPTY_KEYWORDS = {"PatientBirthDate", "PatientSex", "PatientAge"}
UID_KEYWORDS = {
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "FrameOfReferenceUID",
    "SynchronizationFrameOfReferenceUID",
    "DimensionOrganizationUID",
    "ReferencedSOPInstanceUID",
}
DATE_VRS = {"DA", "DT"}
TEXT_VRS = {"AE", "AS", "CS", "LO", "LT", "PN", "SH", "ST", "UC", "UR", "UT"}
SAFE_UID_KEYWORDS = {
    "SOPClassUID",
    "ReferencedSOPClassUID",
    "TransferSyntaxUID",
    "ImplementationClassUID",
}


@dataclass
class DeidContext:
    secret: str
    subject_id: str
    date_shift_days: int
    uid_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(cls, subject_id: str) -> DeidContext:
        secret = os.urandom(32).hex()
        digest = hashlib.sha256(f"{secret}:{subject_id}".encode()).digest()
        shift = int.from_bytes(digest[:2], "big") % 3650 - 1825
        return cls(secret, subject_id, shift)

    def map_uid(self, original: str) -> str:
        if original not in self.uid_map:
            self.uid_map[original] = generate_uid(entropy_srcs=[self.secret, "dicom-uid", original])
        return self.uid_map[original]


def _shift_date(value: str, days: int, is_datetime: bool) -> str:
    if not value:
        return ""
    width = 14 if is_datetime else 8
    prefix = value[:width]
    suffix = value[width:]
    try:
        parsed = (
            datetime.strptime(prefix, "%Y%m%d%H%M%S")
            if is_datetime
            else datetime.strptime(prefix, "%Y%m%d")
        )
    except ValueError:
        return ""
    shifted = parsed + timedelta(days=days)
    return shifted.strftime("%Y%m%d%H%M%S" if is_datetime else "%Y%m%d") + suffix


def deidentify_dataset(source: Dataset, context: DeidContext) -> Dataset:
    ds = copy.deepcopy(source)
    ds.remove_private_tags()

    def callback(dataset: Dataset, element: DataElement) -> None:
        keyword = element.keyword
        if keyword in REMOVE_KEYWORDS:
            del dataset[element.tag]
            return
        if keyword in EMPTY_KEYWORDS:
            element.value = ""
            return
        if keyword == "PatientID":
            element.value = context.subject_id
            return
        if keyword in UID_KEYWORDS or (element.VR == "UI" and keyword not in SAFE_UID_KEYWORDS):
            if element.value:
                if isinstance(element.value, (list, tuple)):
                    element.value = [context.map_uid(str(value)) for value in element.value]
                else:
                    element.value = context.map_uid(str(element.value))
            return
        if element.VR in DATE_VRS:
            if isinstance(element.value, (list, tuple)):
                element.value = [
                    _shift_date(str(value), context.date_shift_days, element.VR == "DT")
                    for value in element.value
                ]
            else:
                element.value = _shift_date(
                    str(element.value), context.date_shift_days, element.VR == "DT"
                )
            return
        if element.VR in TEXT_VRS and keyword and any(
            marker in keyword.lower()
            for marker in (
                "physician",
                "operator",
                "institution",
                "address",
                "telephone",
                "comment",
            )
        ):
            element.value = ""

    ds.walk(callback)
    ds.PatientName = "ANONYMOUS"
    ds.PatientID = context.subject_id
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "Local policy based on DICOM PS3.15 confidentiality profile"
    if "BurnedInAnnotation" not in ds:
        ds.BurnedInAnnotation = "YES"

    if getattr(ds, "file_meta", None):
        ds.file_meta.ImplementationVersionName = "MRISTUDY_010"
        if getattr(ds, "SOPInstanceUID", None):
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    return ds


def deidentify_tree(source_root: Path, destination_root: Path, subject_id: str) -> dict:
    context = DeidContext.create(subject_id)
    results = []
    destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for source in sorted(source_root.rglob("*")):
        if source.is_symlink() or not source.is_file():
            continue
        try:
            ds = pydicom.dcmread(source)
        except (pydicom.errors.InvalidDicomError, OSError):
            continue
        relative = source.resolve().relative_to(source_root.resolve())
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        cleaned = deidentify_dataset(ds, context)
        fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".deid-")
        os.close(fd)
        try:
            cleaned.save_as(temporary, enforce_file_format=True)
            pydicom.dcmread(temporary, stop_before_pixels=True)
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        results.append({"source": relative.as_posix(), "destination": relative.as_posix()})
    mapping_path = destination_root.parent / "state" / "uid-map.json"
    atomic_json(
        mapping_path,
        {
            "subject_id": context.subject_id,
            "secret": context.secret,
            "date_shift_days": context.date_shift_days,
            "uid_map": context.uid_map,
        },
    )
    return {
        "subject_id": subject_id,
        "files_written": len(results),
        "private_tags_removed": True,
        "uid_count": len(context.uid_map),
        "pixel_review_required": True,
        "files": results,
    }
