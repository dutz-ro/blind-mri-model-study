from __future__ import annotations

from pathlib import Path

from .dicom.contact_sheet import create_contact_sheets
from .dicom.deidentify import deidentify_tree
from .dicom.export import export_tree
from .dicom.inventory import inventory
from .dicom.privacy import privacy_scan
from .dicom.qc import quality_control
from .errors import MRIStudyError
from .paths import StudyPaths
from .reports import render_json_report
from .state import Stage, StudyState
from .utils import atomic_json


def run_inventory(paths: StudyPaths, state: StudyState) -> dict:
    source = Path(state.input_path)
    report = inventory(source)
    payload = report.to_dict()
    atomic_json(paths.manifests / "inventory.json", payload)
    render_json_report("DICOM Inventory", payload, paths.reports / "01-dicom-inventory.html")
    state.advance(Stage.INVENTORIED)
    return payload


def run_qc(paths: StudyPaths, state: StudyState) -> dict:
    report = inventory(Path(state.input_path))
    payload = quality_control(report)
    atomic_json(paths.manifests / "quality-control.json", payload)
    render_json_report("Local Quality Control", payload, paths.reports / "03-quality-control.html")
    if payload["summary"]["blockers"]:
        raise MRIStudyError("Quality control found blockers; state was not advanced.")
    state.advance(Stage.LOCAL_QC_PASSED)
    return payload


def run_deidentify(paths: StudyPaths, state: StudyState) -> dict:
    if any(paths.anonymized.iterdir()):
        raise MRIStudyError("Anonymized destination is not empty; refusing to overwrite it.")
    payload = deidentify_tree(Path(state.input_path), paths.anonymized, state.study_id)
    atomic_json(paths.manifests / "deidentification.json", payload)
    state.advance(Stage.DEIDENTIFIED)
    return payload


def run_privacy_review(paths: StudyPaths, state: StudyState) -> dict:
    payload = privacy_scan(paths.anonymized)
    payload["contact_sheets"] = create_contact_sheets(
        paths.anonymized, paths.evidence / "privacy-review"
    )
    atomic_json(paths.manifests / "privacy-review.json", payload)
    render_json_report(
        "Anonymization Review",
        payload,
        paths.reports / "02-anonymization-report.html",
    )
    state.advance(Stage.PRIVACY_REVIEW_REQUIRED)
    return payload


def run_export(paths: StudyPaths, state: StudyState) -> dict:
    if any(paths.exports.iterdir()):
        raise MRIStudyError("Export destination is not empty; refusing to overwrite it.")
    payload = export_tree(paths.anonymized, paths.exports)
    atomic_json(paths.manifests / "exports.json", payload)
    state.advance(Stage.EXPORTS_READY)
    return payload


def run_local_until_privacy(paths: StudyPaths, state: StudyState) -> None:
    if state.stage == Stage.NEW:
        run_inventory(paths, state)
    if state.stage == Stage.INVENTORIED:
        run_qc(paths, state)
    if state.stage == Stage.LOCAL_QC_PASSED:
        run_deidentify(paths, state)
    if state.stage == Stage.DEIDENTIFIED:
        run_privacy_review(paths, state)
