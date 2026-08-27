from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .errors import MRIStudyError

STUDY_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


@dataclass(frozen=True)
class StudyPaths:
    root: Path

    @property
    def original(self) -> Path:
        return self.root / "original"

    @property
    def anonymized(self) -> Path:
        return self.root / "anonymized-dicom"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def temporary(self) -> Path:
        return self.root / "tmp"

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        self.root.chmod(0o700)
        for path in (
            self.original,
            self.anonymized,
            self.exports,
            self.manifests,
            self.runs,
            self.evidence,
            self.reports,
            self.logs,
            self.state,
            self.temporary,
        ):
            path.mkdir(mode=0o700)


def study_paths(data_root: Path, study_id: str) -> StudyPaths:
    if not STUDY_ID.fullmatch(study_id):
        raise MRIStudyError("Study ID must contain 3-64 lowercase letters, digits, or hyphens.")
    return StudyPaths(data_root / "studies" / study_id)
