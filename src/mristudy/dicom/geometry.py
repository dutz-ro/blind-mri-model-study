from __future__ import annotations

import math

from .models import InstanceRecord


def slice_coordinate(record: InstanceRecord) -> float | None:
    if not record.image_orientation or not record.image_position:
        return None
    row = record.image_orientation[:3]
    column = record.image_orientation[3:]
    normal = (
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    )
    norm = math.sqrt(sum(value * value for value in normal))
    if norm < 1e-8:
        return None
    return sum(
        (value / norm) * position
        for value, position in zip(normal, record.image_position, strict=True)
    )


def sort_series(records: list[InstanceRecord]) -> list[InstanceRecord]:
    def key(record: InstanceRecord) -> tuple[int, float, int, str]:
        coordinate = slice_coordinate(record)
        return (
            1 if coordinate is None else 0,
            coordinate or 0.0,
            record.instance_number or 0,
            record.relative_path,
        )

    return sorted(records, key=key)
