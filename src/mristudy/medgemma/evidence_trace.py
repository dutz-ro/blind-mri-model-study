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

EVIDENCE_TRACE_POLICY = "evidence-traceability-v1"

EVIDENCE_ID_PATTERN = re.compile(r"^image-[0-9]{6}$")

STATUS_VERIFIED = "verified"
STATUS_RECOVERED = "recovered"
STATUS_CROSS_WINDOW = "cross_window"
STATUS_UNVERIFIED = "unverified"

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
        summary = _summary([], window_ids)
        graded["evidence_summary"] = summary
        return graded, actions, summary

    statuses: list[str] = []
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
        previous = finding.get("evidence_status")
        finding["evidence_status"] = status
        if previous != status:
            _action(
                actions,
                "evidence-status-graded-v1",
                f"findings.{index}.evidence_status",
                "graded",
                previous,
                status,
                {"cited": cited},
            )
        statuses.append(status)

    summary = _summary(statuses, window_ids)
    graded["evidence_summary"] = summary
    return graded, actions, summary


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
