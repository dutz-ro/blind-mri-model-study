from __future__ import annotations

from collections import Counter

GENERALIST_SERIES_ROLES = {
    "mpr_sagittal": ("mpr", "sag", "dp", "fs"),
    "mpr_coronal": ("mpr", "cor", "dp", "fs"),
    "mpr_axial": ("mpr", "ax", "dp", "fs"),
    "sagittal_t1": ("sag", "t1"),
}


def _normalized(value: str | None) -> str:
    return " ".join((value or "").lower().replace("-", " ").split())


def select_generalist_exports(inventory: dict, exports_manifest: dict) -> dict:
    series = {}
    role_by_path = {}
    for instance in inventory["instances"]:
        uid = instance.get("series_uid")
        if not uid:
            continue
        if uid in series:
            role = series[uid]["role"]
        else:
            description = _normalized(instance.get("series_description"))
            protocol = _normalized(instance.get("protocol_name"))
            searchable = f"{description} {protocol}"
            role = next(
                (
                    candidate_role
                    for candidate_role, tokens in GENERALIST_SERIES_ROLES.items()
                    if all(token in searchable for token in tokens)
                ),
                None,
            )
        if role:
            if uid not in series:
                series[uid] = {
                    "series_uid": uid,
                    "role": role,
                    "series_description": instance.get("series_description"),
                    "protocol_name": instance.get("protocol_name"),
                }
            role_by_path[instance["relative_path"]] = role

    selected = [
        item
        for item in exports_manifest["exports"]
        if item.get("source_relative_path") in role_by_path
    ]
    counts = Counter(role_by_path[item["source_relative_path"]] for item in selected)
    missing = sorted(set(GENERALIST_SERIES_ROLES) - set(counts))
    if missing:
        raise ValueError(f"Missing required generalist series roles: {', '.join(missing)}")
    return {
        "policy": "knee-meniscus-generalist-v1",
        "selected_series": sorted(series.values(), key=lambda item: item["role"]),
        "selected_exports": [item["export_id"] for item in selected],
        "counts_by_role": dict(sorted(counts.items())),
        "total_exports": len(selected),
        "excluded_exports": len(exports_manifest["exports"]) - len(selected),
    }
