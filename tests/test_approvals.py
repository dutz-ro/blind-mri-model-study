from pathlib import Path

import pytest

from mristudy.approvals import require_approval, write_approval
from mristudy.errors import ApprovalError


def test_approval_is_bound_to_artifact(tmp_path: Path) -> None:
    path = tmp_path / "approval.json"
    write_approval(path, "privacy-review", {"a": 1})
    require_approval(path, "privacy-review", {"a": 1})
    with pytest.raises(ApprovalError):
        require_approval(path, "privacy-review", {"a": 2})
