from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil
from typing import Any


@dataclass(frozen=True)
class Window:
    window_id: str
    start_index: int
    end_index: int
    instance_uids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    overlap_previous: int
    overlap_next: int

    @property
    def size(self) -> int:
        return self.end_index - self.start_index + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"size": self.size}


@dataclass(frozen=True)
class WindowPlan:
    source_instance_count: int
    window_size: int
    minimum_overlap: int
    windows: tuple[Window, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_instance_count": self.source_instance_count,
            "window_size": self.window_size,
            "minimum_overlap": self.minimum_overlap,
            "windows": [window.to_dict() for window in self.windows],
        }


def _balanced_ranges(count: int, window_size: int, overlap: int) -> list[tuple[int, int]]:
    if count <= window_size:
        return [(0, count)]
    stride = window_size - overlap
    window_count = 1 + ceil((count - window_size) / stride)
    total_memberships = count + overlap * (window_count - 1)

    def cut(index: int) -> int:
        return (2 * index * total_memberships + window_count) // (2 * window_count)

    return [
        (cut(index) - index * overlap, cut(index + 1) - index * overlap)
        for index in range(window_count)
    ]


def plan_windows(
    ordered_instances: Sequence[dict[str, Any]],
    window_size: int = 32,
    minimum_overlap: int = 8,
    *,
    window_prefix: str = "window",
) -> WindowPlan:
    if not ordered_instances:
        raise ValueError("Cannot plan windows for an empty series.")
    if not 1 <= window_size <= 32:
        raise ValueError("window_size must be between 1 and 32.")
    if minimum_overlap < 8:
        raise ValueError("minimum_overlap must be at least 8.")
    if minimum_overlap >= window_size:
        raise ValueError("minimum_overlap must be smaller than window_size.")

    uids = [str(item["sop_instance_uid"]) for item in ordered_instances]
    evidence = [str(item["evidence_id"]) for item in ordered_instances]
    if len(uids) != len(set(uids)):
        raise ValueError("Duplicate SOP Instance UID in ordered series.")
    if len(evidence) != len(set(evidence)):
        raise ValueError("Duplicate evidence ID in ordered series.")

    ranges = _balanced_ranges(len(ordered_instances), window_size, minimum_overlap)
    windows = []
    for index, (start, end) in enumerate(ranges):
        previous_overlap = ranges[index - 1][1] - start if index else 0
        next_overlap = end - ranges[index + 1][0] if index + 1 < len(ranges) else 0
        windows.append(
            Window(
                window_id=f"{window_prefix}-window-{index + 1:04d}",
                start_index=start,
                end_index=end - 1,
                instance_uids=tuple(uids[start:end]),
                evidence_ids=tuple(evidence[start:end]),
                overlap_previous=previous_overlap,
                overlap_next=next_overlap,
            )
        )

    covered = {index for start, end in ranges for index in range(start, end)}
    if covered != set(range(len(ordered_instances))):
        raise ValueError("Window plan does not provide complete coverage.")
    for left, right in zip(windows, windows[1:], strict=False):
        overlap_count = left.end_index - right.start_index + 1
        if overlap_count < minimum_overlap:
            raise ValueError("Adjacent windows do not satisfy minimum overlap.")
    if any(window.size > window_size for window in windows):
        raise ValueError("Window exceeds maximum size.")
    return WindowPlan(len(ordered_instances), window_size, minimum_overlap, tuple(windows))
