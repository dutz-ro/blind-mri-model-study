"""Prove the refactor did not change knee behaviour.

Re-runs series and multiplanar integration over the completed run's window
results using the refactored, profile-aware code with no profile supplied, and
compares the output with the integration artifacts produced before the change.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from mristudy.medgemma.integration import (  # noqa: E402
    integrate_multiplanar,
    integrate_series,
)

RUN = Path(
    "/home/openclaw/mri-study-data/studies/lex-knee-mri/runs/medgemma-schema-v2-46e6e22f30f0"
)
GOLDEN = sorted(Path("/home/openclaw/mri-study-data/golden-snapshots").glob("*"))[-1]
GOLDEN_RUN = GOLDEN / "runs" / RUN.name

plan = json.loads((RUN / "run-plan.json").read_text())
windows = {item["window_id"]: item for item in plan["windows"]}

by_role: dict[str, list] = {}
for path in sorted((RUN / "windows").glob("*.structured.json")):
    window_id = path.name.replace(".structured.json", "")
    window = windows[window_id]
    by_role.setdefault(window["role"], []).append(
        {
            "window_id": window_id,
            "role": window["role"],
            "series_uid": window["series_uid"],
            "start_index": window["start_index"],
            "end_index": window["end_index"],
            "result": json.loads(path.read_text()),
        }
    )

ROLE_ORDER = [
    "primary-volume",
    "sagittal-meniscus",
    "coronal-confirmation",
    "axial-confirmation",
    "anatomy-correlation",
]

mismatches = []
series_results = []
for role in ROLE_ORDER:
    items = by_role.get(role)
    if not items:
        continue
    result, provenance = integrate_series(role, items[0]["series_uid"], items)
    series_results.append({"role": role, "result": result, "provenance": provenance})
    golden_path = GOLDEN_RUN / "series" / f"{role}.json"
    if golden_path.exists():
        if json.loads(golden_path.read_text()) != result:
            mismatches.append(f"series/{role}.json")

final, final_provenance = integrate_multiplanar(series_results)
golden_final = GOLDEN_RUN / "integration" / "multiplanar.json"
if golden_final.exists():
    golden_obj = json.loads(golden_final.read_text())
    if golden_obj != final:
        mismatches.append("integration/multiplanar.json")

print("golden snapshot :", GOLDEN.name)
print("series compared :", len(series_results))
print("mismatches      :", mismatches if mismatches else "none")
print()
print("RESULT:", "IDENTICAL to pre-refactor output" if not mismatches else "DIFFERS")
raise SystemExit(0 if not mismatches else 1)
