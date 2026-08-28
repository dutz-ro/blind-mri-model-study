from __future__ import annotations

import httpx
import pytest

from mristudy.errors import MRIStudyError
from mristudy.models.litellm import LiteLLMClient


def _response(status: int) -> httpx.Response:
    request = httpx.Request("POST", "http://litellm.test/v1/chat/completions")
    return httpx.Response(status, request=request, json={"status": status})


def test_transient_503_is_retried_without_changing_the_request(monkeypatch) -> None:
    calls = []
    responses = [_response(503), _response(200)]

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs.get("json")))
        return responses.pop(0)

    monkeypatch.setattr(httpx, "request", request)
    monkeypatch.setattr("mristudy.models.litellm.time.sleep", lambda _: None)
    client = LiteLLMClient("http://litellm.test", "secret")
    result = client._request("POST", "/v1/chat/completions", json={"model": "grok-4.5"})

    assert result.status_code == 200
    assert len(calls) == 2
    assert calls[0] == calls[1]


def test_non_transient_400_is_not_retried(monkeypatch) -> None:
    calls = 0

    def request(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _response(400)

    monkeypatch.setattr(httpx, "request", request)
    client = LiteLLMClient("http://litellm.test")
    with pytest.raises(MRIStudyError, match="after 1 attempts"):
        client._request("POST", "/v1/chat/completions", json={})
    assert calls == 1


def test_persistent_503_stops_after_bounded_attempts(monkeypatch) -> None:
    calls = 0
    sleeps = []

    def request(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _response(503)

    monkeypatch.setattr(httpx, "request", request)
    monkeypatch.setattr("mristudy.models.litellm.time.sleep", sleeps.append)
    client = LiteLLMClient("http://litellm.test")
    with pytest.raises(MRIStudyError, match="after 5 attempts"):
        client._request("POST", "/v1/chat/completions", json={})

    assert calls == 5
    assert sleeps == [15, 30, 60, 120]
