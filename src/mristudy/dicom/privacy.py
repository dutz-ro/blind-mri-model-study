from __future__ import annotations

from collections import Counter
from pathlib import Path

import pydicom

from .models import Severity

SENSITIVE_KEYWORD_PARTS = (
    "patientname",
    "patientaddress",
    "telephone",
    "accession",
    "institution",
    "physician",
    "operator",
    "medicalrecord",
)


def privacy_scan(root: Path) -> dict:
    issues = []
    files = 0
    private_elements = 0
    burned_in = Counter()
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, OSError):
            continue
        files += 1
        identity_removed = str(getattr(ds, "PatientIdentityRemoved", ""))
        if identity_removed != "YES":
            issues.append(
                {
                    "severity": Severity.BLOCKER.value,
                    "code": "IDENTITY_REMOVED_NOT_DECLARED",
                    "path": path.relative_to(root).as_posix(),
                }
            )
        annotation = str(getattr(ds, "BurnedInAnnotation", "MISSING"))
        burned_in[annotation] += 1
        if annotation != "NO":
            issues.append(
                {
                    "severity": Severity.WARNING.value,
                    "code": "PIXEL_REVIEW_REQUIRED",
                    "path": path.relative_to(root).as_posix(),
                    "value": annotation,
                }
            )
        for element in ds.iterall():
            if element.tag.is_private:
                private_elements += 1
            keyword = element.keyword.lower()
            value = str(element.value).strip()
            if value and any(part in keyword for part in SENSITIVE_KEYWORD_PARTS):
                if element.keyword not in {"PatientIdentityRemoved"}:
                    issues.append(
                        {
                            "severity": Severity.WARNING.value,
                            "code": "SENSITIVE_TEXT_FIELD_NONEMPTY",
                            "path": path.relative_to(root).as_posix(),
                            "tag": str(element.tag),
                            "keyword": element.keyword,
                        }
                    )
    if private_elements:
        issues.append(
            {
                "severity": Severity.BLOCKER.value,
                "code": "PRIVATE_ELEMENTS_REMAIN",
                "count": private_elements,
            }
        )
    return {
        "files_scanned": files,
        "private_elements": private_elements,
        "burned_in_annotation": dict(burned_in),
        "human_pixel_review_required": True,
        "issues": issues,
    }
