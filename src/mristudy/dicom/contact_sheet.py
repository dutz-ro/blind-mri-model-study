from __future__ import annotations

import math
from pathlib import Path

import pydicom
from PIL import Image, ImageDraw

from .export import render_frame


def create_contact_sheets(
    source_root: Path,
    destination_root: Path,
    columns: int = 5,
    tile_size: int = 220,
) -> dict:
    destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    frames: list[tuple[Path, int]] = []
    skipped = []
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, OSError):
            continue
        if not getattr(ds, "Rows", None) or not getattr(ds, "Columns", None):
            skipped.append(
                {
                    "path": path.relative_to(source_root).as_posix(),
                    "reason": "not-an-image-instance",
                }
            )
            continue
        for frame in range(int(getattr(ds, "NumberOfFrames", 1) or 1)):
            frames.append((path, frame))

    page_size = columns * 4
    outputs = []
    for page_start in range(0, len(frames), page_size):
        page_frames = frames[page_start : page_start + page_size]
        rows = math.ceil(len(page_frames) / columns)
        sheet = Image.new("L", (columns * tile_size, rows * (tile_size + 24)), color=0)
        draw = ImageDraw.Draw(sheet)
        for offset, (source, frame) in enumerate(page_frames):
            try:
                rendered = render_frame(source, frame)
            except (AttributeError, RuntimeError, ValueError) as error:
                skipped.append(
                    {
                        "path": source.relative_to(source_root).as_posix(),
                        "frame": frame + 1,
                        "reason": type(error).__name__,
                    }
                )
                continue
            image = Image.fromarray(rendered, mode="L")
            image.thumbnail((tile_size, tile_size))
            x = (offset % columns) * tile_size + (tile_size - image.width) // 2
            y = (offset // columns) * (tile_size + 24)
            sheet.paste(image, (x, y))
            label = f"review-{page_start + offset + 1:06d} frame {frame + 1}"
            draw.text((offset % columns * tile_size + 4, y + tile_size + 4), label, fill=255)
        destination = destination_root / f"privacy-review-{page_start // page_size + 1:04d}.png"
        sheet.save(destination, format="PNG")
        destination.chmod(0o600)
        outputs.append(destination.name)
    return {
        "pages": outputs,
        "candidate_frames": len(frames),
        "rendered_frames": len(frames) - len([item for item in skipped if "frame" in item]),
        "skipped": skipped,
    }
