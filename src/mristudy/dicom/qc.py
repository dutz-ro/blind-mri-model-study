from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from statistics import median

from .geometry import slice_coordinate, sort_series
from .models import InventoryReport, Issue, Severity


def quality_control(report: InventoryReport) -> dict:
    issues = list(report.issues)
    series: dict[str, list] = defaultdict(list)
    for record in report.instances:
        if not record.study_uid or not record.series_uid or not record.sop_instance_uid:
            issues.append(
                Issue(
                    Severity.BLOCKER,
                    "MISSING_UID",
                    "Required DICOM UID missing",
                    record.relative_path,
                )
            )
        if record.modality != "MR":
            issues.append(
                Issue(
                    Severity.WARNING,
                    "NON_MR_INSTANCE",
                    f"Modality is {record.modality!r}",
                    record.relative_path,
                )
            )
        if not record.rows or not record.columns:
            issues.append(
                Issue(
                    Severity.BLOCKER,
                    "MISSING_MATRIX",
                    "Rows or Columns missing",
                    record.relative_path,
                )
            )
        if record.series_uid:
            series[record.series_uid].append(record)

    summaries = []
    for uid, records in sorted(series.items()):
        ordered = sort_series(records)
        matrices = {(item.rows, item.columns) for item in records}
        orientations = {item.image_orientation for item in records if item.image_orientation}
        spacings = {item.pixel_spacing for item in records if item.pixel_spacing}
        if len(matrices) > 1:
            issues.append(
                Issue(Severity.ERROR, "MIXED_MATRIX", "Series contains mixed image matrices")
            )
        if len(orientations) > 1:
            issues.append(
                Issue(
                    Severity.ERROR,
                    "MIXED_ORIENTATION",
                    "Series contains mixed orientations",
                )
            )
        if len(spacings) > 1:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "MIXED_PIXEL_SPACING",
                    "Series contains mixed pixel spacing",
                )
            )
        coordinates = [
            coordinate
            for item in ordered
            if (coordinate := slice_coordinate(item)) is not None
        ]
        gaps = [
            abs(right - left)
            for left, right in zip(coordinates, coordinates[1:], strict=False)
        ]
        if gaps:
            typical = median(gaps)
            if typical > 0 and any(gap > typical * 1.5 for gap in gaps):
                issues.append(
                    Issue(
                        Severity.ERROR,
                        "SLICE_SPACING_OUTLIER",
                        "Series has a geometric spacing outlier",
                    )
                )
        summaries.append(
            {
                "series_uid": uid,
                "instances": len(records),
                "frames": sum(item.number_of_frames for item in records),
                "series_description": records[0].series_description,
                "protocol_name": records[0].protocol_name,
                "matrix": sorted([list(item) for item in matrices]),
                "geometrically_ordered": len(coordinates) == len(records),
            }
        )
    blockers = sum(issue.severity == Severity.BLOCKER for issue in issues)
    errors = sum(issue.severity == Severity.ERROR for issue in issues)
    return {
        "summary": {
            "instances": len(report.instances),
            "series": len(series),
            "blockers": blockers,
            "errors": errors,
            "passed": blockers == 0,
        },
        "series": summaries,
        "issues": [asdict(issue) for issue in issues],
    }
