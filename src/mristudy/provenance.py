from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from .utils import utc_now

SECRET_KEYS = {"authorization", "api_key", "access_credential", "token", "password"}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in SECRET_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def append_event(path: Path, operation: str, status: str, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "schema_version": 1,
        "event_id": str(uuid.uuid4()),
        "occurred_at": utc_now(),
        "operation": operation,
        "status": status,
        "metadata": redact(metadata or {}),
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")
