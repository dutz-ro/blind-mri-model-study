from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    data_root: Path
    max_cloud_cost_usd: float = 50.0
    litellm_base_url: str | None = None
    litellm_api_key: str | None = None
    gpt_model: str | None = None
    grok_model: str | None = None
    google_project: str | None = None
    google_region: str = "europe-west4"
    google_dataset: str | None = None
    google_dicom_store: str | None = None
    vertex_endpoint_id: str | None = None
    backend: str = "scaleway"
    max_backend_cost_eur: float = 50.0
    orthanc_dicomweb_url: str = "http://127.0.0.1:18042/dicom-web"
    orthanc_username: str | None = None
    orthanc_password: str | None = None
    orthanc_deployment_id: str = "flow-local-orthanc-v1"
    ohif_url: str = "http://127.0.0.1:13000"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_root=Path(
                os.environ.get("MRI_STUDY_DATA_ROOT", "~/mri-study-data")
            ).expanduser(),
            max_cloud_cost_usd=float(os.environ.get("MRI_STUDY_MAX_CLOUD_COST_USD", "50")),
            litellm_base_url=os.environ.get("LITELLM_BASE_URL") or None,
            litellm_api_key=os.environ.get("LITELLM_API_KEY") or None,
            gpt_model=os.environ.get("MRI_STUDY_GPT_MODEL") or None,
            grok_model=os.environ.get("MRI_STUDY_GROK_MODEL") or None,
            google_project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None,
            google_region=os.environ.get("GOOGLE_CLOUD_REGION", "europe-west4"),
            google_dataset=os.environ.get("GOOGLE_HEALTHCARE_DATASET") or None,
            google_dicom_store=os.environ.get("GOOGLE_DICOM_STORE") or None,
            vertex_endpoint_id=os.environ.get("GOOGLE_VERTEX_ENDPOINT_ID") or None,
            backend=os.environ.get("MRI_STUDY_BACKEND", "scaleway"),
            max_backend_cost_eur=float(
                os.environ.get("MRI_STUDY_MAX_BACKEND_COST_EUR", "50")
            ),
            orthanc_dicomweb_url=os.environ.get(
                "MRI_STUDY_ORTHANC_DICOMWEB_URL",
                "http://127.0.0.1:18042/dicom-web",
            ),
            orthanc_username=os.environ.get("MRI_STUDY_ORTHANC_USERNAME") or None,
            orthanc_password=os.environ.get("MRI_STUDY_ORTHANC_PASSWORD") or None,
            orthanc_deployment_id=os.environ.get(
                "MRI_STUDY_ORTHANC_DEPLOYMENT_ID", "flow-local-orthanc-v1"
            ),
            ohif_url=os.environ.get("MRI_STUDY_OHIF_URL", "http://127.0.0.1:13000"),
        )


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
