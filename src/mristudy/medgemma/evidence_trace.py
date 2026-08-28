"""Grade and repair the traceability of model findings to source images.

Why this exists
---------------
A completed medical run produced 29 findings across 26 windows in which every
single ``traceable_evidence`` list was empty, while 25 findings carried an
``image-NNNNNN`` identifier in ``finding_id``. The model had put the citation in
the wrong field. Schema validation passed because an empty evidence list is
structurally legal and ``finding_id`` had no format constraint, so the final
report was structurally valid yet visually untraceable.

This module makes traceability an explicit, graded, auditable property:

- a citation misplaced into ``finding_id`` is recovered into the evidence list,
  never invented;
- a citation belonging to an overlapping neighbour window is accepted but
  labelled, because window overlap is real but is not independent corroboration;
- a finding that cannot be traced is preserved and labelled ``unverified``
  rather than discarded, so clinical content is never silently lost.

Every repair is recorded as an auditable action. This module never fabricates an
evidence identifier and never upgrades a weaker status into a stronger one.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

EVIDENCE_TRACE_POLICY = "evidence-traceability-v2"

EVIDENCE_ID_PATTERN = re.compile(r"^image-[0-9]{6}$")

STATUS_VERIFIED = "verified"
STATUS_RECOVERED = "recovered"
STATUS_CROSS_WINDOW = "cross_window"
STATUS_UNVERIFIED = "unverified"

SCOPE_CURRENT_WINDOW = "current_window"
SCOPE_ADJACENT_SAME_SERIES = "adjacent_same_series"
SCOPE_DISTANT_SAME_SERIES = "distant_same_series"
SCOPE_OTHER_SERIES = "other_series"
SCOPE_UNKNOWN = "unknown"

# Ordered from strongest to weakest. A finding is graded by its weakest usable
# citation, so a single cross-window reference cannot present itself as verified.
STATUS_RANK = {
    STATUS_VERIFIED: 0,
    STATUS_RECOVERED: 1,
    STATUS_CROSS_WINDOW: 2,
    STATUS_UNVERIFIED: 3,
}


def _weaker(left: str, right: str) -> str:
    return left if STATUS_RANK[left] >= STATUS_RANK[right] else right


def is_evidence_id(value: Any) -> bool:
    return isinstance(value, str) and bool(EVIDENCE_ID_PATTERN.fullmatch(value))


def _action(
    actions: list[dict[str, Any]],
    rule_id: str,
    path: str,
    operation: str,
    old_value: Any,
    new_value: Any,
    detail: dict[str, Any] | None = None,
) -> None:
    entry = {
        "policy": EVIDENCE_TRACE_POLICY,
        "rule_id": rule_id,
        "path": path,
        "operation": operation,
        "old_value": old_value,
        "new_value": new_value,
    }
    if detail:
        entry["detail"] = detail
    actions.append(entry)


def _cited_ids(finding: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for key in ("traceable_evidence", "counter_evidence"):
        entries = finding.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and is_evidence_id(entry.get("export_id")):
                found.append(str(entry["export_id"]))
    return found


def classify_evidence_id(
    export_id: str,
    window_evidence_ids: set[str],
    plan_evidence_ids: set[str] | None,
) -> str:
    """Grade one citation against the window it was produced in.

    ``plan_evidence_ids`` is the union across the whole arm. An identifier that
    is outside the current window but present elsewhere in the plan is a real
    slice from an overlapping or neighbouring window, which is materially
    different from an identifier that exists nowhere.
    """
    if export_id in window_evidence_ids:
        return STATUS_VERIFIED
    if plan_evidence_ids is not None and export_id in plan_evidence_ids:
        return STATUS_CROSS_WINDOW
    return STATUS_UNVERIFIED


def classify_evidence_scope(
    export_id: str,
    window: dict[str, Any],
    evidence_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve one citation and determine whether it was supplied to this call."""
    entry = evidence_map.get(export_id)
    if entry is None:
        return {
            "export_id": export_id,
            "resolution_status": "unknown",
            "scope_status": SCOPE_UNKNOWN,
            "integration_link_eligible": False,
            "retained_in_integration_result": False,
        }
    current = set(window.get("evidence_ids") or [])
    if export_id in current:
        scope = SCOPE_CURRENT_WINDOW
        eligible = True
    elif entry.get("series_uid") != window.get("series_uid"):
        scope = SCOPE_OTHER_SERIES
        eligible = False
    else:
        index = entry.get("geometric_index")
        start = window.get("start_index")
        end = window.get("end_index")
        adjacent = (
            isinstance(index, int)
            and isinstance(start, int)
            and isinstance(end, int)
            and (index == start - 1 or index == end + 1)
        )
        scope = SCOPE_ADJACENT_SAME_SERIES if adjacent else SCOPE_DISTANT_SAME_SERIES
        eligible = False
    return {
        "export_id": export_id,
        "resolution_status": "resolved",
        "scope_status": scope,
        "integration_link_eligible": eligible,
        "retained_in_integration_result": True,
        "series_uid": entry.get("series_uid"),
        "geometric_index": entry.get("geometric_index"),
        "window_ids": list(entry.get("window_ids") or []),
    }


def project_result_evidence(
    result: dict[str, Any],
    window: dict[str, Any],
    evidence_map: dict[str, dict[str, Any]],
    *,
    recover: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Create an integration-safe result while preserving every clinical claim.

    Known global evidence remains visible, but only evidence actually supplied
    to the current invocation may link observations across windows. Unknown
    evidence is quarantined from the safe result; the finding itself remains.
    """
    safe = deepcopy(result)
    actions: list[dict[str, Any]] = []
    findings = safe.get("findings")
    if not isinstance(findings, list):
        summary = _summary([], set(window.get("evidence_ids") or []))
        summary["findings"] = []
        return safe, actions, summary

    current_ids = set(window.get("evidence_ids") or [])
    plan_ids = set(evidence_map)
    statuses: list[str] = []
    grades: list[dict[str, Any]] = []
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue
        if recover:
            recover_finding_evidence(finding, index, current_ids, plan_ids, actions)

        dispositions: list[dict[str, Any]] = []
        link_eligible: list[str] = []
        all_resolved: list[str] = []
        for key in ("traceable_evidence", "counter_evidence"):
            entries = finding.get(key)
            if not isinstance(entries, list):
                continue
            retained = []
            for evidence_index, item in enumerate(entries):
                if not isinstance(item, dict):
                    retained.append(item)
                    continue
                export_id = item.get("export_id")
                if not is_evidence_id(export_id):
                    retained.append(item)
                    continue
                disposition = classify_evidence_scope(export_id, window, evidence_map)
                disposition.update(
                    {
                        "field": key,
                        "evidence_index": evidence_index,
                        "source_status": (
                            "recovered_from_finding_id"
                            if any(
                                action["rule_id"] == "evidence-recovered-from-finding-id-v1"
                                and action["path"].startswith(f"findings.{index}.")
                                and export_id in (action.get("new_value") or [])
                                for action in actions
                            )
                            else "explicit"
                        ),
                    }
                )
                dispositions.append(disposition)
                if disposition["retained_in_integration_result"]:
                    retained.append(item)
                    all_resolved.append(export_id)
                else:
                    _action(
                        actions,
                        "unknown-evidence-quarantined-v1",
                        f"findings.{index}.{key}.{evidence_index}",
                        "quarantined_unknown_citation",
                        item,
                        None,
                        {"export_id": export_id},
                    )
                if disposition["integration_link_eligible"]:
                    link_eligible.append(export_id)
            finding[key] = retained

        scopes = {item["scope_status"] for item in dispositions}
        recovered = any(
            item["source_status"] == "recovered_from_finding_id"
            for item in dispositions
        )
        if not dispositions or SCOPE_UNKNOWN in scopes:
            status = STATUS_UNVERIFIED
        elif scopes == {SCOPE_CURRENT_WINDOW}:
            status = STATUS_RECOVERED if recovered else STATUS_VERIFIED
        else:
            status = STATUS_CROSS_WINDOW
        statuses.append(status)
        grades.append(
            {
                "index": index,
                "finding_id": finding.get("finding_id"),
                "evidence_status": status,
                "dispositions": dispositions,
                "all_resolved_evidence_ids": sorted(set(all_resolved)),
                "link_eligible_evidence_ids": sorted(set(link_eligible)),
            }
        )

    summary = _summary(statuses, current_ids)
    summary["findings"] = grades
    summary["scope_counts"] = {
        scope: sum(
            item["scope_status"] == scope
            for grade in grades
            for item in grade["dispositions"]
        )
        for scope in (
            SCOPE_CURRENT_WINDOW,
            SCOPE_ADJACENT_SAME_SERIES,
            SCOPE_DISTANT_SAME_SERIES,
            SCOPE_OTHER_SERIES,
            SCOPE_UNKNOWN,
        )
    }
    return safe, actions, summary


def verify_evidence_projection(
    source: dict[str, Any],
    safe: dict[str, Any],
    actions: list[dict[str, Any]],
) -> list[str]:
    """Reject any unexplained clinical change in the evidence projection."""
    errors: list[str] = []
    allowed_actions = [
        action
        for action in actions
        if action["operation"] in {"recovered_citation", "quarantined_unknown_citation"}
    ]
    source_copy = deepcopy(source)
    safe_copy = deepcopy(safe)
    for index, (before, after) in enumerate(
        zip(source_copy.get("findings") or [], safe_copy.get("findings") or [], strict=False)
    ):
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        for key in ("traceable_evidence", "counter_evidence"):
            before.pop(key, None)
            after.pop(key, None)
        if before != after:
            errors.append(f"Clinical fields changed during evidence projection: findings.{index}")
    if len(source_copy.get("findings") or []) != len(safe_copy.get("findings") or []):
        errors.append("Finding count changed during evidence projection.")
    if not allowed_actions and source != safe:
        errors.append("Evidence projection changed without an audited action.")
    expected_quarantines = [
        action
        for action in allowed_actions
        if action["operation"] == "quarantined_unknown_citation"
    ]
    expected_recoveries = [
        action for action in allowed_actions if action["operation"] == "recovered_citation"
    ]
    source_count = sum(
        len(finding.get(key) or [])
        for finding in source.get("findings") or []
        if isinstance(finding, dict)
        for key in ("traceable_evidence", "counter_evidence")
    )
    safe_count = sum(
        len(finding.get(key) or [])
        for finding in safe.get("findings") or []
        if isinstance(finding, dict)
        for key in ("traceable_evidence", "counter_evidence")
    )
    if safe_count != source_count - len(expected_quarantines) + len(expected_recoveries):
        errors.append("Evidence count change does not match the audited actions.")
    expected = deepcopy(source)
    # Quarantine indexes refer to the source arrays, so remove from highest to
    # lowest index within each array to keep the paths stable.
    for action in sorted(
        expected_quarantines,
        key=lambda item: tuple(
            int(part) if part.isdigit() else part for part in item["path"].split(".")
        ),
        reverse=True,
    ):
        parts = action["path"].split(".")
        finding_index = int(parts[1])
        field = parts[2]
        evidence_index = int(parts[3])
        entries = expected["findings"][finding_index][field]
        if evidence_index >= len(entries) or entries[evidence_index] != action["old_value"]:
            errors.append(f"Quarantine action does not match its source path: {action['path']}")
            continue
        entries.pop(evidence_index)
    for action in expected_recoveries:
        parts = action["path"].split(".")
        finding_index = int(parts[1])
        export_ids = action.get("new_value") or []
        for export_id in export_ids:
            expected["findings"][finding_index].setdefault("traceable_evidence", []).append(
                {
                    "export_id": export_id,
                    "role": "supporting",
                    "description": (
                        "Recovered from finding_id; the model placed the citation there."
                    ),
                }
            )
    if expected != safe:
        errors.append("Integration-safe result differs from the audited evidence actions.")
    return errors


def recover_finding_evidence(
    finding: dict[str, Any],
    index: int,
    window_evidence_ids: set[str],
    plan_evidence_ids: set[str] | None,
    actions: list[dict[str, Any]],
) -> None:
    """Move a citation misplaced in ``finding_id`` into ``traceable_evidence``.

    Applied only when the finding cites nothing at all, so a model that cited
    correctly is never modified. An identifier that exists nowhere in the plan is
    left alone: it is not evidence and must not be presented as such.
    """
    if _cited_ids(finding):
        return
    candidate = finding.get("finding_id")
    if not is_evidence_id(candidate):
        return
    known = candidate in window_evidence_ids or (
        plan_evidence_ids is not None and candidate in plan_evidence_ids
    )
    if not known:
        _action(
            actions,
            "unknown-finding-id-not-recovered-v1",
            f"findings.{index}.finding_id",
            "left_unchanged",
            candidate,
            candidate,
            {"reason": "identifier is absent from the approved plan"},
        )
        return
    entries = finding.get("traceable_evidence")
    if not isinstance(entries, list):
        entries = []
    entries.append(
        {
            "export_id": candidate,
            "role": "supporting",
            "description": "Recovered from finding_id; the model placed the citation there.",
        }
    )
    finding["traceable_evidence"] = entries
    _action(
        actions,
        "evidence-recovered-from-finding-id-v1",
        f"findings.{index}.traceable_evidence",
        "recovered_citation",
        [],
        [candidate],
        {
            "source_field": "finding_id",
            "in_current_window": candidate in window_evidence_ids,
        },
    )


def grade_result_evidence(
    result: dict[str, Any],
    window_evidence_ids: list[str] | set[str],
    plan_evidence_ids: list[str] | set[str] | None = None,
    *,
    recover: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Return a graded copy of ``result`` plus audit actions and a summary.

    The result is always returned. Findings that cannot be traced are preserved
    and marked ``unverified`` so that clinical content is never lost, while the
    summary makes the weakness impossible to overlook.
    """
    graded = deepcopy(result)
    actions: list[dict[str, Any]] = []
    window_ids = set(window_evidence_ids)
    plan_ids = set(plan_evidence_ids) if plan_evidence_ids is not None else None

    findings = graded.get("findings")
    if not isinstance(findings, list):
        return graded, actions, _summary([], window_ids)

    statuses: list[str] = []
    grades: list[dict[str, Any]] = []
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue
        if recover:
            recover_finding_evidence(finding, index, window_ids, plan_ids, actions)
        cited = _cited_ids(finding)
        if not cited:
            status = STATUS_UNVERIFIED
        else:
            status = STATUS_VERIFIED
            for export_id in cited:
                status = _weaker(status, classify_evidence_id(export_id, window_ids, plan_ids))
        _action(
            actions,
            "evidence-status-graded-v1",
            f"findings.{index}",
            "graded",
            None,
            status,
            {"cited": cited},
        )
        # The grade is kept beside the result, not inside the finding: the
        # canonical schema sets additionalProperties false, so an extra field
        # would make an otherwise valid clinical result fail validation.
        grades.append(
            {
                "index": index,
                "finding_id": finding.get("finding_id"),
                "evidence_status": status,
                "cited": cited,
            }
        )
        statuses.append(status)

    summary = _summary(statuses, window_ids)
    summary["findings"] = grades
    return graded, actions, summary


def status_by_finding_id(summary: dict[str, Any]) -> dict[str, str]:
    """Map finding identifiers to their traceability grade."""
    return {
        str(item.get("finding_id")): str(item.get("evidence_status"))
        for item in summary.get("findings") or []
        if item.get("finding_id") is not None
    }


def _summary(statuses: list[str], window_ids: set[str]) -> dict[str, Any]:
    counts = {name: 0 for name in STATUS_RANK}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    total = len(statuses)
    traceable = total - counts[STATUS_UNVERIFIED]
    return {
        "policy": EVIDENCE_TRACE_POLICY,
        "total_findings": total,
        "counts": counts,
        "traceable_findings": traceable,
        "evidence_coverage_percent": round(100.0 * traceable / total, 2) if total else 100.0,
        "window_evidence_available": len(window_ids),
    }


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-window summaries into one arm-level view."""
    counts = {name: 0 for name in STATUS_RANK}
    total = 0
    for summary in summaries:
        total += int(summary.get("total_findings", 0))
        for name, value in (summary.get("counts") or {}).items():
            counts[name] = counts.get(name, 0) + int(value)
    traceable = total - counts[STATUS_UNVERIFIED]
    return {
        "policy": EVIDENCE_TRACE_POLICY,
        "total_findings": total,
        "counts": counts,
        "traceable_findings": traceable,
        "evidence_coverage_percent": round(100.0 * traceable / total, 2) if total else 100.0,
    }


def coverage_warning(summary: dict[str, Any], minimum_percent: float) -> str | None:
    """Describe insufficient traceability without blocking the run.

    Losing a produced clinical result is worse than reporting it with an explicit
    caveat, so this returns a warning rather than raising.
    """
    total = int(summary.get("total_findings", 0))
    if not total:
        return None
    coverage = float(summary.get("evidence_coverage_percent", 0.0))
    if coverage >= minimum_percent:
        return None
    counts = summary.get("counts") or {}
    return (
        f"Evidence traceability is {coverage:.2f}%, below the {minimum_percent:.2f}% "
        f"expectation. {counts.get(STATUS_UNVERIFIED, 0)} of {total} findings have no "
        "usable image citation. Findings are retained but must be read as unverified."
    )
