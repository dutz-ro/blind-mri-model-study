from pathlib import Path

from mristudy.dicom.export import render_frame
from mristudy.dicom.fixtures import create_synthetic_mri
from mristudy.medgemma.windows import plan_windows


def test_32_contiguous_synthetic_dicoms_decode_and_plan_as_one_window(tmp_path: Path) -> None:
    create_synthetic_mri(tmp_path, slices=32)
    ordered = []
    for index, path in enumerate(sorted(tmp_path.glob("*.dcm"))):
        rendered = render_frame(path)
        assert rendered.shape == (128, 128)
        ordered.append(
            {
                "sop_instance_uid": f"1.2.3.{index}",
                "evidence_id": f"image-{index + 1:06d}",
            }
        )
    result = plan_windows(ordered)
    assert len(result.windows) == 1
    assert result.windows[0].size == 32


def test_two_overlapping_windows_keep_same_evidence_ids() -> None:
    ordered = [
        {
            "sop_instance_uid": f"1.2.3.{index}",
            "evidence_id": f"image-{index + 1:06d}",
        }
        for index in range(50)
    ]
    result = plan_windows(ordered)
    assert len(result.windows) == 2
    assert result.windows[0].evidence_ids[-8:] == result.windows[1].evidence_ids[:8]
