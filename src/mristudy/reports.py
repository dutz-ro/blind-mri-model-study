from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_json_report(title: str, data: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = html.escape(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True))
    destination.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:15px/1.5 system-ui;max-width:1100px;margin:40px auto;padding:0 20px}"
        "pre{white-space:pre-wrap;background:#f4f4f0;padding:20px;border-left:5px solid #384a3f}"
        "h1{font-family:Georgia,serif}</style></head><body>"
        f"<h1>{html.escape(title)}</h1><pre>{body}</pre></body></html>",
        encoding="utf-8",
    )
    destination.chmod(0o600)
