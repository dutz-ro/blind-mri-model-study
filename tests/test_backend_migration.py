from pathlib import Path

import pytest

from mristudy.approvals import require_approval, write_approval
from mristudy.errors import ApprovalError
from mristudy.state import Stage, StudyState


def test_backend_metadata_preserves_cloud_plan_stage(tmp_path: Path) -> None:
    state = StudyState.create(tmp_path / "status.json", "study-test", "profile", "/input")
    state.stage = Stage.CLOUD_PLAN_READY
    state.save()
    state.set_backend("scaleway", "backend-operation-plan.json", "abc")
    loaded = StudyState.load(tmp_path / "status.json")
    assert loaded.stage == Stage.CLOUD_PLAN_READY
    assert loaded.schema_version == 2
    assert loaded.backend == {
        "provider": "scaleway",
        "operation_plan": "backend-operation-plan.json",
        "operation_plan_sha256": "abc",
    }


def test_plan_change_invalidates_approval(tmp_path: Path) -> None:
    approval_path = tmp_path / "approval.json"
    original = {"plan_sha256": "original", "runtime": "sha256:a"}
    changed = {"plan_sha256": "changed", "runtime": "sha256:b"}
    write_approval(approval_path, "backend-egress", original)
    with pytest.raises(ApprovalError):
        require_approval(approval_path, "backend-egress", changed)
