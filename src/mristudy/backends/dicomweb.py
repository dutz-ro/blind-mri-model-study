from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from ..errors import MRIStudyError

UID_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")


def validate_uid(uid: str) -> str:
    if len(uid) > 64 or not UID_RE.fullmatch(uid):
        raise MRIStudyError("Invalid DICOM UID in DICOMweb operation.")
    return uid


class AuthProvider(Protocol):
    def headers(self) -> dict[str, str]: ...


@dataclass(frozen=True)
class NoAuth:
    def headers(self) -> dict[str, str]:
        return {}


@dataclass(frozen=True)
class BasicAuth:
    username: str
    password: str

    def headers(self) -> dict[str, str]:
        encoded = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        return {"Authorization": f"Basic {encoded}"}


@dataclass(frozen=True)
class BearerAuth:
    token: str

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


@dataclass(frozen=True)
class DicomWebEndpoint:
    base_url: str
    provider: str
    deployment_id: str

    @property
    def normalized_url(self) -> str:
        return self.base_url.rstrip("/")


def dicom_json_value(item: dict, tag: str) -> str | None:
    values = (item.get(tag) or {}).get("Value") or []
    return str(values[0]) if values else None


class DicomWebClient:
    def __init__(
        self,
        endpoint: DicomWebEndpoint,
        auth: AuthProvider | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.endpoint = endpoint
        self.auth = auth or NoAuth()
        self.transport = transport
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = self.auth.headers() | kwargs.pop("headers", {})
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
                response = client.request(
                    method,
                    f"{self.endpoint.normalized_url}{path}",
                    headers=headers,
                    **kwargs,
                )
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            raise MRIStudyError(f"DICOMweb request failed: {type(error).__name__}") from error

    def probe(self) -> dict:
        studies = self.search_studies(limit=1)
        return {
            "provider": self.endpoint.provider,
            "deployment_id": self.endpoint.deployment_id,
            "base_url": self.endpoint.normalized_url,
            "qido": True,
            "study_sample_count": len(studies),
        }

    def search_studies(self, limit: int | None = None) -> list[dict]:
        params = {"limit": str(limit)} if limit else None
        return self._request(
            "GET",
            "/studies",
            params=params,
            headers={"Accept": "application/dicom+json"},
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
        boundary = "mristudy-dicom-boundary"
        body = (
            f"--{boundary}\r\nContent-Type: application/dicom\r\n\r\n".encode()
            + path.read_bytes()
            + f"\r\n--{boundary}--\r\n".encode()
        )
        return self._request(
            "POST",
            target,
            headers={
                "Content-Type": f'multipart/related; type="application/dicom"; boundary={boundary}',
                "Accept": "application/dicom+json",
            },
            content=body,
        ).json()

    def delete_instance(self, study_uid: str, series_uid: str, instance_uid: str) -> None:
        self._request(
            "DELETE",
            f"/studies/{validate_uid(study_uid)}/series/{validate_uid(series_uid)}"
            f"/instances/{validate_uid(instance_uid)}",
        )
