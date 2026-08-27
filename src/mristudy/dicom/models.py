from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    BLOCKER = "BLOCKER"


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str
    path: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InstanceRecord:
    relative_path: str
    file_size: int
    sha256: str
    study_uid: str | None
    series_uid: str | None
    sop_instance_uid: str | None
    sop_class_uid: str | None
    modality: str | None
    transfer_syntax_uid: str | None
    series_description: str | None
    protocol_name: str | None
    sequence_name: str | None
    image_type: tuple[str, ...]
    instance_number: int | None
    rows: int | None
    columns: int | None
    number_of_frames: int
    image_position: tuple[float, float, float] | None
    image_orientation: tuple[float, ...] | None
    pixel_spacing: tuple[float, float] | None
    slice_thickness: float | None
    spacing_between_slices: float | None
    repetition_time: float | None
    echo_time: float | None
    laterality: str | None
    photometric_interpretation: str | None
    pixel_data_present: bool
    forced_read: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InventoryReport:
    root: str
    instances: list[InstanceRecord]
    issues: list[Issue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "instances": [item.to_dict() for item in self.instances],
            "issues": [asdict(issue) for issue in self.issues],
        }


def safe_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
