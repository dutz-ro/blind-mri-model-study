from __future__ import annotations

import json
from dataclasses import dataclass

from ..errors import MRIStudyError
from .dicomweb import DicomStoreRef, validate_uid


def series_url(store: DicomStoreRef, study_uid: str, series_uid: str) -> str:
    return (
        f"{store.dicomweb_url}/studies/{validate_uid(study_uid)}"
        f"/series/{validate_uid(series_uid)}"
    )


def build_dicom_request(dicom_source: str, prompt: str, max_tokens: int = 4096) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_dicom",
                        "image_dicom": {
                            "dicom_source": dicom_source,
                            "access_credential": "application_default",
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_schema": "knee-meniscus-finding-v2",
    }


@dataclass(frozen=True)
class VertexEndpointRef:
    project: str
    location: str
    endpoint_id: str
    dedicated: bool = False


class VertexMedGemmaClient:
    def __init__(self, ref: VertexEndpointRef) -> None:
        self.ref = ref

    def infer(self, request: dict) -> dict:
        try:
            from google.cloud import aiplatform
        except ImportError as error:
            raise MRIStudyError("Install the cloud extra to use Vertex AI.") from error
        aiplatform.init(project=self.ref.project, location=self.ref.location)
        endpoint = aiplatform.Endpoint(self.ref.endpoint_id)
        try:
            response = endpoint.raw_predict(
                body=json.dumps(request, sort_keys=True).encode(),
                headers={"Content-Type": "application/json"},
                use_dedicated_endpoint=self.ref.dedicated,
            )
            return response.json()
        except Exception as error:
            raise MRIStudyError(f"Vertex inference failed: {type(error).__name__}") from error
