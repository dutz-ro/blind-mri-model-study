from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixels import apply_modality_lut, apply_voi_lut, pixel_array

from ..utils import sha256_file


@dataclass(frozen=True)
class RenderSpec:
    lower_percentile: float = 0.5
    upper_percentile: float = 99.5
    voi_index: int = 0
    policy_version: str = "mr-gray-v1"


DEFAULT_RENDER_SPEC = RenderSpec()


def render_frame(
    path: Path, frame: int = 0, spec: RenderSpec = DEFAULT_RENDER_SPEC
) -> np.ndarray:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    array = pixel_array(path, index=frame, raw=True).astype(np.float64)
    array = np.asarray(apply_modality_lut(array, ds), dtype=np.float64)
    if "VOILUTSequence" in ds or ("WindowCenter" in ds and "WindowWidth" in ds):
        array = np.asarray(apply_voi_lut(array, ds, index=spec.voi_index), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [spec.lower_percentile, spec.upper_percentile])
    if high <= low:
        rendered = np.zeros(array.shape, dtype=np.uint8)
    else:
        rendered = np.clip((array - low) / (high - low), 0, 1)
        rendered = np.rint(rendered * 255).astype(np.uint8)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        rendered = 255 - rendered
    return rendered


def export_tree(source_root: Path, destination_root: Path) -> dict:
    exports = []
    skipped = []
    destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    index = 0
    for source in sorted(source_root.rglob("*")):
        if source.is_symlink() or not source.is_file():
            continue
        try:
            ds = pydicom.dcmread(source, stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, OSError):
            continue
        if not getattr(ds, "Rows", None) or not getattr(ds, "Columns", None):
            skipped.append(
                {
                    "source_relative_path": source.relative_to(source_root).as_posix(),
                    "reason": "not-an-image-instance",
                }
            )
            continue
        frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        for frame in range(frames):
            index += 1
            output_id = f"image-{index:06d}"
            destination = destination_root / f"{output_id}.png"
            try:
                rendered = render_frame(source, frame)
            except (AttributeError, RuntimeError, ValueError) as error:
                skipped.append(
                    {
                        "source_relative_path": source.relative_to(source_root).as_posix(),
                        "frame": frame + 1,
                        "reason": type(error).__name__,
                    }
                )
                continue
            image = Image.fromarray(rendered, mode="L")
            image.save(destination, format="PNG", optimize=False)
            destination.chmod(0o600)
            exports.append(
                {
                    "export_id": output_id,
                    "source_relative_path": source.relative_to(source_root).as_posix(),
                    "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
                    "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
                    "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
                    "frame": frame + 1,
                    "png": destination.name,
                    "sha256": sha256_file(destination),
                    "render_policy": "mr-gray-v1",
                }
            )
    return {"exports": exports, "count": len(exports), "skipped": skipped}
