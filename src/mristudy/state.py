from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .errors import StateError
from .utils import atomic_json, load_json, utc_now


class Stage(StrEnum):
    NEW = "NEW"
    INVENTORIED = "INVENTORIED"
    LOCAL_QC_PASSED = "LOCAL_QC_PASSED"
    DEIDENTIFIED = "DEIDENTIFIED"
    PRIVACY_REVIEW_REQUIRED = "PRIVACY_REVIEW_REQUIRED"
    PRIVACY_APPROVED = "PRIVACY_APPROVED"
    EXPORTS_READY = "EXPORTS_READY"
    PROVIDERS_PROBED = "PROVIDERS_PROBED"
    PROVIDER_PLAN_READY = "PROVIDER_PLAN_READY"
    PROVIDER_EGRESS_APPROVED = "PROVIDER_EGRESS_APPROVED"
    GENERALIST_RESULTS_LOCKED = "GENERALIST_RESULTS_LOCKED"
    CLOUD_PLAN_READY = "CLOUD_PLAN_READY"
    CLOUD_UPLOAD_APPROVED = "CLOUD_UPLOAD_APPROVED"
    DICOM_STORE_VERIFIED = "DICOM_STORE_VERIFIED"
    MEDGEMMA_COMPLETE = "MEDGEMMA_COMPLETE"
    CLOUD_CLEANED = "CLOUD_CLEANED"
    COMPARISON_COMPLETE = "COMPARISON_COMPLETE"
    REPORT_COMPLETE = "REPORT_COMPLETE"


NEXT = {
    Stage.NEW: Stage.INVENTORIED,
    Stage.INVENTORIED: Stage.LOCAL_QC_PASSED,
    Stage.LOCAL_QC_PASSED: Stage.DEIDENTIFIED,
    Stage.DEIDENTIFIED: Stage.PRIVACY_REVIEW_REQUIRED,
    Stage.PRIVACY_REVIEW_REQUIRED: Stage.PRIVACY_APPROVED,
    Stage.PRIVACY_APPROVED: Stage.EXPORTS_READY,
    Stage.EXPORTS_READY: Stage.PROVIDERS_PROBED,
    Stage.PROVIDERS_PROBED: Stage.PROVIDER_PLAN_READY,
    Stage.PROVIDER_PLAN_READY: Stage.PROVIDER_EGRESS_APPROVED,
    Stage.PROVIDER_EGRESS_APPROVED: Stage.GENERALIST_RESULTS_LOCKED,
    Stage.GENERALIST_RESULTS_LOCKED: Stage.CLOUD_PLAN_READY,
    Stage.CLOUD_PLAN_READY: Stage.CLOUD_UPLOAD_APPROVED,
    Stage.CLOUD_UPLOAD_APPROVED: Stage.DICOM_STORE_VERIFIED,
    Stage.DICOM_STORE_VERIFIED: Stage.MEDGEMMA_COMPLETE,
    Stage.MEDGEMMA_COMPLETE: Stage.CLOUD_CLEANED,
    Stage.CLOUD_CLEANED: Stage.COMPARISON_COMPLETE,
    Stage.COMPARISON_COMPLETE: Stage.REPORT_COMPLETE,
}


@dataclass
class StudyState:
    path: Path
    stage: Stage
    study_id: str
    profile: str
    input_path: str
    created_at: str
    updated_at: str
    schema_version: int = 1
    backend: dict | None = None

    @classmethod
    def create(cls, path: Path, study_id: str, profile: str, input_path: str) -> StudyState:
        now = utc_now()
        state = cls(path, Stage.NEW, study_id, profile, input_path, now, now)
        state.save()
        return state

    @classmethod
    def load(cls, path: Path) -> StudyState:
        data = load_json(path)
        return cls(
            path=path,
            stage=Stage(data["stage"]),
            study_id=data["study_id"],
            profile=data["profile"],
            input_path=data["input_path"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            schema_version=int(data.get("schema_version", 1)),
            backend=data.get("backend"),
        )

    def save(self) -> None:
        atomic_json(
            self.path,
            {
                "study_id": self.study_id,
                "profile": self.profile,
                "input_path": self.input_path,
                "stage": self.stage.value,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "schema_version": self.schema_version,
                "backend": self.backend,
            },
        )

    def advance(self, target: Stage) -> None:
        if NEXT.get(self.stage) != target:
            raise StateError(f"Invalid transition: {self.stage.value} -> {target.value}")
        self.stage = target
        self.updated_at = utc_now()
        self.save()

    def set_backend(self, provider: str, plan_path: str, plan_sha256: str) -> None:
        self.schema_version = 2
        self.backend = {
            "provider": provider,
            "operation_plan": plan_path,
            "operation_plan_sha256": plan_sha256,
        }
        self.updated_at = utc_now()
        self.save()

    def invalidate_backend_approval(self) -> None:
        if self.stage not in {Stage.CLOUD_UPLOAD_APPROVED, Stage.DICOM_STORE_VERIFIED}:
            raise StateError(
                "Backend approval can only be invalidated before MedGemma completion."
            )
        self.stage = Stage.CLOUD_PLAN_READY
        self.updated_at = utc_now()
        self.save()


NEXT_ACTION = {
    Stage.NEW: "Run inventory.",
    Stage.INVENTORIED: "Run local quality control.",
    Stage.LOCAL_QC_PASSED: "Create the de-identified copy.",
    Stage.DEIDENTIFIED: "Generate privacy review artifacts.",
    Stage.PRIVACY_REVIEW_REQUIRED: "A human must review and approve anonymization.",
    Stage.PRIVACY_APPROVED: "Generate canonical exports.",
    Stage.EXPORTS_READY: "Probe providers using synthetic data only.",
    Stage.PROVIDERS_PROBED: "Create the exact generalist model egress plan.",
    Stage.PROVIDER_PLAN_READY: "Approve the exact Routera image egress plan.",
    Stage.PROVIDER_EGRESS_APPROVED: "Run and lock generalist model results.",
    Stage.GENERALIST_RESULTS_LOCKED: "Create a DICOM-aware backend operation plan.",
    Stage.CLOUD_PLAN_READY: "Approve the exact DICOM-aware backend operation plan.",
    Stage.CLOUD_UPLOAD_APPROVED: "Provision the approved backend and verify DICOM access.",
    Stage.DICOM_STORE_VERIFIED: "Run MedGemma.",
    Stage.MEDGEMMA_COMPLETE: "Destroy and verify temporary backend resources.",
    Stage.CLOUD_CLEANED: "Compare locked model results.",
    Stage.COMPARISON_COMPLETE: "Render final reports.",
    Stage.REPORT_COMPLETE: "Study complete.",
}
