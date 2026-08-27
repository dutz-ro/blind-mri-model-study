from pathlib import Path

import pydicom

from mristudy.dicom.contact_sheet import create_contact_sheets
from mristudy.dicom.deidentify import deidentify_tree
from mristudy.dicom.export import export_tree
from mristudy.dicom.fixtures import create_synthetic_mri
from mristudy.dicom.inventory import inventory
from mristudy.dicom.privacy import privacy_scan
from mristudy.dicom.qc import quality_control


def test_synthetic_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "source"
    anonymous = tmp_path / "anonymous"
    exports = tmp_path / "exports"
    create_synthetic_mri(source, slices=4)
    report = inventory(source)
    assert len(report.instances) == 4
    assert quality_control(report)["summary"]["passed"]
    result = deidentify_tree(source, anonymous, "study-test")
    assert result["files_written"] == 4
    scan = privacy_scan(anonymous)
    assert scan["private_elements"] == 0
    first = pydicom.dcmread(next(anonymous.glob("*.dcm")), stop_before_pixels=True)
    assert first.PatientName == "ANONYMOUS"
    assert first.PatientID == "study-test"
    assert first.PatientIdentityRemoved == "YES"
    sheets = create_contact_sheets(anonymous, tmp_path / "review")
    assert sheets["candidate_frames"] == 4
    assert sheets["rendered_frames"] == 4
    assert sheets["pages"]
    exported = export_tree(anonymous, exports)
    assert exported["count"] == 4
