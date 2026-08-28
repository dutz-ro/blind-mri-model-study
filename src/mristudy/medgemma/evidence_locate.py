"""Resolve an evidence identifier back to a concrete, checkable image.

A finding is only useful if a human can put the image in front of a clinician.
The evidence map already records the anonymous study, series and SOP Instance
UID plus the geometric index, which is enough to locate the anonymised file. The
link back to the original acquisition additionally requires the local UID map.

Privacy
-------
The UID map file also holds the de-identification secret and the subject
identifier. This module reads only its ``uid_map`` table and never returns,
logs, or copies the secret, the subject identifier, or the date shift. The
original-file reference is produced only when explicitly requested, because it
re-identifies the study and must stay on the authorised machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import MRIStudyError
from ..utils import load_json, sha256_file


def _entries(evidence_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = evidence_manifest.get("entries")
    if not isinstance(entries, list):
        raise MRIStudyError("Evidence manifest has no entries.")
    return {str(item["evidence_id"]): item for item in entries if isinstance(item, dict)}


def _anonymous_uid_to_original(uid_map_path: Path) -> dict[str, str]:
    """Invert the original-to-anonymous table without touching the secret."""
    payload = load_json(uid_map_path)
    table = payload.get("uid_map")
    if not isinstance(table, dict):
        raise MRIStudyError("UID map does not contain a uid_map table.")
    return {str(anonymous): str(original) for original, anonymous in table.items()}


def _window_ids_for(evidence_id: str, run_plan: dict[str, Any] | None) -> list[str]:
    if not run_plan:
        return []
    return [
        window["window_id"]
        for window in run_plan.get("windows") or []
        if evidence_id in (window.get("evidence_ids") or [])
    ]


def _anonymised_file(anonymised_root: Path, sop_instance_uid: str) -> Path | None:
    """Find the de-identified instance carrying this SOP Instance UID."""
    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise MRIStudyError("pydicom is required to locate evidence files.") from error
    for path in sorted(anonymised_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(getattr(dataset, "SOPInstanceUID", "")) == sop_instance_uid:
            return path
    return None


def locate_evidence(
    evidence_id: str,
    evidence_manifest: dict[str, Any],
    anonymised_root: Path,
    *,
    run_plan: dict[str, Any] | None = None,
    uid_map_path: Path | None = None,
    include_original_reference: bool = False,
) -> dict[str, Any]:
    """Return a verifiable reference for one evidence identifier.

    ``include_original_reference`` is opt-in because resolving the original
    acquisition UID re-identifies the study.
    """
    entries = _entries(evidence_manifest)
    entry = entries.get(evidence_id)
    if entry is None:
        raise MRIStudyError(f"Unknown evidence identifier: {evidence_id}")

    reference: dict[str, Any] = {
        "evidence_id": evidence_id,
        "role": entry.get("role"),
        "study_uid": entry.get("study_uid"),
        "series_uid": entry.get("series_uid"),
        "sop_instance_uid": entry.get("sop_instance_uid"),
        "geometric_index": entry.get("geometric_index"),
        "geometric_position": entry.get("geometric_position"),
        "window_ids": _window_ids_for(evidence_id, run_plan),
    }

    located = _anonymised_file(anonymised_root, str(entry.get("sop_instance_uid")))
    if located is not None:
        reference["anonymised_path"] = str(located)
        reference["anonymised_relative_path"] = str(located.relative_to(anonymised_root))
        reference["anonymised_sha256"] = sha256_file(located)
    else:
        reference["anonymised_path"] = None
        reference["locate_warning"] = "No anonymised instance carries this SOP Instance UID."

    if include_original_reference:
        if uid_map_path is None:
            raise MRIStudyError("The original reference requires the local UID map path.")
        inverted = _anonymous_uid_to_original(uid_map_path)
        original_uid = inverted.get(str(entry.get("sop_instance_uid")))
        reference["original_sop_instance_uid"] = original_uid
        if original_uid is None:
            reference["original_reference_warning"] = (
                "The anonymous SOP Instance UID is absent from the local UID map."
            )
    return reference


def locate_many(
    evidence_ids: list[str],
    evidence_manifest: dict[str, Any],
    anonymised_root: Path,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return [
        locate_evidence(evidence_id, evidence_manifest, anonymised_root, **kwargs)
        for evidence_id in evidence_ids
    ]


def export_evidence_images(
    references: list[dict[str, Any]],
    destination: Path,
) -> list[dict[str, Any]]:
    """Render located anonymised instances to PNG for local review.

    Only de-identified pixels are rendered, and only for images already cited by
    a finding. Non-image instances are skipped with a recorded reason rather
    than failing the export.
    """
    from ..dicom.export import render_frame

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise MRIStudyError("Pillow is required to export evidence images.") from error

    destination.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    for reference in references:
        record = dict(reference)
        source = reference.get("anonymised_path")
        if not source:
            record["image_export_skipped"] = "no anonymised instance was located"
            exported.append(record)
            continue
        target = destination / f"{reference['evidence_id']}.png"
        try:
            rendered = render_frame(Path(source))
            Image.fromarray(rendered, mode="L").save(target)
        except Exception as error:
            record["image_export_skipped"] = f"render failed: {type(error).__name__}"
            exported.append(record)
            continue
        record["image_path"] = str(target)
        record["image_file"] = target.name
        record["image_sha256"] = sha256_file(target)
        exported.append(record)
    return exported


def collect_cited_evidence_ids(result: dict[str, Any]) -> list[str]:
    """Ordered, de-duplicated evidence identifiers cited by a result."""
    seen: list[str] = []
    for finding in result.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        for key in ("traceable_evidence", "counter_evidence"):
            for entry in finding.get(key) or []:
                if not isinstance(entry, dict):
                    continue
                export_id = entry.get("export_id")
                if isinstance(export_id, str) and export_id not in seen:
                    seen.append(export_id)
    return seen
