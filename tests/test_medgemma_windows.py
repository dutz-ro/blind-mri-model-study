from itertools import pairwise

import pytest

from mristudy.medgemma.windows import plan_windows


def instances(count: int) -> list[dict]:
    return [
        {"sop_instance_uid": f"1.2.3.{index}", "evidence_id": f"image-{index + 1:06d}"}
        for index in range(count)
    ]


@pytest.mark.parametrize("count,expected", [(430, 18), (50, 2), (60, 3), (27, 1)])
def test_expected_window_counts(count: int, expected: int) -> None:
    assert len(plan_windows(instances(count)).windows) == expected


@pytest.mark.parametrize("count", [1, 27, 32, 33, 50, 56, 57, 60, 430, 617])
def test_complete_balanced_deterministic_coverage(count: int) -> None:
    first = plan_windows(instances(count))
    second = plan_windows(instances(count))
    assert first == second
    assert first.windows[0].start_index == 0
    assert first.windows[-1].end_index == count - 1
    assert all(window.size <= 32 for window in first.windows)
    covered = {
        index
        for window in first.windows
        for index in range(window.start_index, window.end_index + 1)
    }
    assert covered == set(range(count))
    if len(first.windows) > 1:
        assert max(window.size for window in first.windows) - min(
            window.size for window in first.windows
        ) <= 1
        for left, right in pairwise(first.windows):
            assert left.end_index - right.start_index + 1 >= 8


def test_overlap_reuses_same_global_evidence_ids() -> None:
    plan = plan_windows(instances(50))
    assert plan.windows[0].evidence_ids[-8:] == plan.windows[1].evidence_ids[:8]


def test_window_constraints_are_enforced() -> None:
    with pytest.raises(ValueError):
        plan_windows(instances(10), window_size=33)
    with pytest.raises(ValueError):
        plan_windows(instances(10), minimum_overlap=7)
    with pytest.raises(ValueError):
        plan_windows([])
