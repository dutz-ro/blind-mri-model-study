from __future__ import annotations

from typing import Any

import httpx

from ..errors import MRIStudyError
from ..utils import sha256_bytes


class MedGemmaHttpClient:
    def __init__(self, base_url: str = "http://127.0.0.1:18080", timeout: float = 1800):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = httpx.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as error:
            raise MRIStudyError(
                f"MedGemma runtime request failed: {type(error).__name__}"
            ) from error

    def health(self) -> dict[str, Any]:
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=30)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as error:
            raise MRIStudyError(f"MedGemma health check failed: {type(error).__name__}") from error

    def synthetic_probe(self, image_count: int) -> dict[str, Any]:
        return self._post("/synthetic-probe", {"image_count": image_count})

    def synthetic_dicom_probe(self) -> dict[str, Any]:
        return self._post("/synthetic-dicom-probe", {"image_count": 32})

    def synthetic_json_probe(self) -> dict[str, Any]:
        return self._post("/synthetic-json-probe", {"image_count": 1})

    def probe_dicomweb(self, study_uid: str, series_uid: str) -> dict[str, Any]:
        return self._post(
            "/probe-dicomweb",
            {
                "study_uid": study_uid,
                "series_uid": series_uid,
            },
        )

    def analyze_window(
        self,
        window: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        prompt_sha256 = sha256_bytes(prompt.encode())
        return self._post(
            "/analyze-window",
            {
                "study_uid": window["study_uid"],
                "series_uid": window["series_uid"],
                "role": window["role"],
                "window_id": window["window_id"],
                "instance_uids": window["instance_uids"],
                "evidence_ids": window["evidence_ids"],
                "prompt": prompt,
                "prompt_sha256": prompt_sha256,
                "output_schema": window["output_schema"],
                "max_new_tokens": window["max_new_tokens"],
                "generation": window["generation"],
                "decoder": window["decoder"],
                "output_contract": window.get("output_contract", {}),
            },
        )
