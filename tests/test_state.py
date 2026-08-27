from pathlib import Path

import pytest

from mristudy.errors import StateError
from mristudy.state import Stage, StudyState


def test_state_requires_ordered_transitions(tmp_path: Path) -> None:
    state = StudyState.create(tmp_path / "status.json", "study-abc", "profile", "/input")
    with pytest.raises(StateError):
        state.advance(Stage.DEIDENTIFIED)
    state.advance(Stage.INVENTORIED)
    assert StudyState.load(tmp_path / "status.json").stage == Stage.INVENTORIED


def test_backend_approval_can_be_invalidated_without_skipping_gate(tmp_path: Path) -> None:
    state = StudyState.create(tmp_path / "status.json", "study-abc", "profile", "/input")
    state.stage = Stage.CLOUD_UPLOAD_APPROVED
    state.save()
    state.invalidate_backend_approval()
    assert state.stage == Stage.CLOUD_PLAN_READY


def test_verified_backend_can_return_to_plan_before_medgemma(tmp_path: Path) -> None:
    state = StudyState.create(tmp_path / "status.json", "study-abc", "profile", "/input")
    state.stage = Stage.DICOM_STORE_VERIFIED
    state.save()
    state.invalidate_backend_approval()
    assert state.stage == Stage.CLOUD_PLAN_READY
