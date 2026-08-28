from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from ..errors import MRIStudyError


class LiteLLMClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        retry_delays = (0, 15, 30, 60, 120)
        last_error: httpx.HTTPError | None = None
        attempts = 0
        for attempt, delay in enumerate(retry_delays):
            attempts += 1
            if delay:
                time.sleep(delay)
            try:
                response = httpx.request(
                    method,
                    f"{self.base_url}{path}",
                    headers=self._headers(),
                    timeout=self.timeout,
                    **kwargs,
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as error:
                last_error = error
                if error.response.status_code not in {429, 502, 503, 504}:
                    break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as error:
                last_error = error
            if attempt == len(retry_delays) - 1:
                break
        assert last_error is not None
        raise MRIStudyError(
            f"LiteLLM request failed after {attempts} attempts: "
            f"{type(last_error).__name__}"
        ) from last_error

    def models(self) -> list[dict]:
        payload = self._request("GET", "/v1/models").json()
        return payload.get("data", [])

    def model_info(self) -> dict:
        return self._request("GET", "/model/info").json()

    def verify_upstream_host(self, models: list[str], required_host: str) -> dict[str, dict]:
        payload = self.model_info()
        entries = payload.get("data", payload if isinstance(payload, list) else [])
        verified = {}
        for model in models:
            matches = [item for item in entries if item.get("model_name") == model]
            if not matches:
                raise MRIStudyError(f"No LiteLLM model info found for {model}.")
            item = matches[0]
            params = item.get("litellm_params") or {}
            base = params.get("api_base") or params.get("base_url") or params.get("api_base_url")
            host = urlparse(base).hostname if isinstance(base, str) else None
            if host != required_host:
                raise MRIStudyError(
                    f"Refusing model {model}: upstream host is {host!r}, "
                    f"expected {required_host!r}."
                )
            verified[model] = {
                "alias": model,
                "declared_model": params.get("model"),
                "upstream_host": host,
                "upstream_scheme": urlparse(base).scheme,
            }
        return verified

    def liveness(self) -> dict:
        return self._request("GET", "/health/liveliness").json()

    def readiness(self) -> dict:
        return self._request("GET", "/health/readiness").json()

    def synthetic_text_probe(self, model: str) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly OK."}],
            "temperature": 0,
            "max_tokens": 8,
        }
        response = self._request("POST", "/v1/chat/completions", json=payload).json()
        choice = (response.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content") or "").strip()
        return {
            "passed": content == "OK",
            "response_model": response.get("model"),
            "response_id": response.get("id"),
            "usage": response.get("usage"),
        }

    def synthetic_vision_probe(self, model: str, image_path: Path) -> dict:
        encoded = base64.b64encode(image_path.read_bytes()).decode()
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with exactly SYNTHETIC."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 12,
        }
        response = self._request("POST", "/v1/chat/completions", json=payload).json()
        choice = (response.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content") or "").strip()
        return {
            "passed": content == "SYNTHETIC",
            "response_model": response.get("model"),
            "response_id": response.get("id"),
            "usage": response.get("usage"),
        }

    def analyze_images(self, model: str, prompt: str, images: list[Path]) -> dict:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            encoded = base64.b64encode(image.read_bytes()).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 4096,
            "timeout": int(self.timeout),
        }
        return self._request("POST", "/v1/chat/completions", json=payload).json()
