from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import google.auth
import httpx
from google.auth.transport.requests import Request

from ..errors import MRIStudyError
from ..utils import sha256_file

UID_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")


def validate_uid(uid: str) -> str:
    if len(uid) > 64 or not UID_RE.fullmatch(uid):
        raise MRIStudyError("Invalid DICOM UID in DICOMweb operation.")
    return uid


@dataclass(frozen=True)
class DicomStoreRef:
    project: str
    location: str
    dataset: str
    store: str

    @property
    def resource_name(self) -> str:
        return (
            f"projects/{self.project}/locations/{self.location}/datasets/{self.dataset}"
            f"/dicomStores/{self.store}"
        )

    @property
    def dicomweb_url(self) -> str:
        return f"https://healthcare.googleapis.com/v1/{self.resource_name}/dicomWeb"


class GoogleTokenProvider:
    def __init__(self) -> None:
        self.credentials = None

    def header(self) -> str:
        if self.credentials is None:
            self.credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-healthcare"]
            )
        if not self.credentials.valid:
            self.credentials.refresh(Request())
        return f"Bearer {self.credentials.token}"


class DicomWebClient:
    def __init__(
        self,
        store: DicomStoreRef,
        token_provider: GoogleTokenProvider | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.store = store
        self.token_provider = token_provider or GoogleTokenProvider()
        self.transport = transport
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = self.token_provider.header()
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
                response = client.request(
                    method, f"{self.store.dicomweb_url}{path}", headers=headers, **kwargs
                )
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            raise MRIStudyError(f"DICOMweb request failed: {type(error).__name__}") from error

    def search_studies(self, params: dict[str, str] | None = None) -> list[dict]:
        return self._request(
            "GET", "/studies", params=params, headers={"Accept": "application/dicom+json"}
        ).json()

    def search_series(self, study_uid: str) -> list[dict]:
        return self._request(
            "GET",
            f"/studies/{validate_uid(study_uid)}/series",
            headers={"Accept": "application/dicom+json"},
        ).json()

    def search_instances(self, study_uid: str, series_uid: str) -> list[dict]:
        return self._request(
            "GET",
            f"/studies/{validate_uid(study_uid)}/series/{validate_uid(series_uid)}/instances",
            headers={"Accept": "application/dicom+json"},
        ).json()

    def store_instance(self, path: Path, study_uid: str | None = None) -> dict:
        target = f"/studies/{validate_uid(study_uid)}" if study_uid else "/studies"
        return self._request(
            "POST",
            target,
            headers={"Content-Type": "application/dicom", "Accept": "application/dicom+json"},
            content=path.read_bytes(),
        ).json()

    def retrieve_instance(
        self, study_uid: str, series_uid: str, instance_uid: str, destination: Path
    ) -> dict:
        response = self._request(
            "GET",
            "/studies/"
            f"{validate_uid(study_uid)}/series/{validate_uid(series_uid)}/instances/"
            f"{validate_uid(instance_uid)}",
            headers={"Accept": "application/dicom; transfer-syntax=*"},
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(response.content)
        temporary.chmod(0o600)
        temporary.replace(destination)
        return {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }

    def delete_instance(self, study_uid: str, series_uid: str, instance_uid: str) -> None:
        self._request(
            "DELETE",
            "/studies/"
            f"{validate_uid(study_uid)}/series/{validate_uid(series_uid)}/instances/"
            f"{validate_uid(instance_uid)}",
        )


def dicom_json_value(item: dict, tag: str) -> str | None:
    element = item.get(tag) or {}
    values = element.get("Value") or []
    return str(values[0]) if values else None
