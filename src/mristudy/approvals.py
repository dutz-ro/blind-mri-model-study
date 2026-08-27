from __future__ import annotations

import getpass
from dataclasses import dataclass
from pathlib import Path

from .errors import ApprovalError
from .utils import atomic_json, canonical_json, load_json, sha256_bytes, utc_now


@dataclass(frozen=True)
class Approval:
    scope: str
    actor: str
    artifact_sha256: str
    approved_at: str
    note: str


def artifact_digest(data: dict) -> str:
    return sha256_bytes(canonical_json(data))


def write_approval(path: Path, scope: str, artifact: dict, note: str = "") -> Approval:
    approval = Approval(scope, getpass.getuser(), artifact_digest(artifact), utc_now(), note)
    atomic_json(path, approval.__dict__)
    return approval


def require_approval(path: Path, scope: str, artifact: dict) -> Approval:
    if not path.exists():
        raise ApprovalError(f"Missing approval for {scope}: {path}")
    data = load_json(path)
    approval = Approval(**data)
    if approval.scope != scope or approval.artifact_sha256 != artifact_digest(artifact):
        raise ApprovalError("Approval does not match the current operation plan.")
    return approval
