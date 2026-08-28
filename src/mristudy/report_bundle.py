"""Build the local review bundle consumed by the private report viewer.

The viewer presents each model independently and never merges results into a
shared verdict. This module produces exactly what it needs: per-model findings,
the images each finding cites, and an explicit traceability label so a reader
can tell a claim backed by a visible slice from one that is not.

Privacy
-------
The bundle is written under the study reports directory, which is outside Git.
It carries anonymous identifiers and de-identified renderings only. Nothing here
resolves an original acquisition UID.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .medgemma.evidence_locate import export_evidence_images, locate_many
from .medgemma.evidence_trace import (
    STATUS_CROSS_WINDOW,
    STATUS_RECOVERED,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    status_by_finding_id,
)
from .utils import load_json, utc_now

# Vocabulary already used by the viewer legend.
ROLE_LABELS = {
    "supporting": "susținere",
    "counter": "contra-dovadă",
    "context": "context",
}

STATUS_LABELS = {
    STATUS_VERIFIED: "dovadă verificată",
    STATUS_RECOVERED: "dovadă recuperată",
    STATUS_CROSS_WINDOW: "dovadă din fereastră vecină",
    STATUS_UNVERIFIED: "proveniență neconfirmată",
}


def evidence_summary_from_multiplanar_provenance(
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Derive per-output finding grades from integrated evidence provenance."""
    findings = []
    counts = {
        STATUS_VERIFIED: 0,
        STATUS_RECOVERED: 0,
        STATUS_CROSS_WINDOW: 0,
        STATUS_UNVERIFIED: 0,
    }
    for group in provenance.get("groups") or []:
        for output in group.get("output_finding_provenance") or []:
            dispositions = output.get("evidence_dispositions") or []
            scopes = {item.get("scope_status") for item in dispositions}
            recovered = any(
                item.get("source_status") == "recovered_from_finding_id"
                for item in dispositions
            )
            if not dispositions or "unknown" in scopes:
                status = STATUS_UNVERIFIED
            elif scopes == {"current_window"}:
                status = STATUS_RECOVERED if recovered else STATUS_VERIFIED
            else:
                status = STATUS_CROSS_WINDOW
            counts[status] += 1
            findings.append(
                {
                    "finding_id": output.get("output_finding_id"),
                    "evidence_status": status,
                    "dispositions": dispositions,
                    "link_eligible_evidence_ids": output.get(
                        "link_eligible_evidence_ids"
                    )
                    or [],
                }
            )
    total = len(findings)
    traceable = total - counts[STATUS_UNVERIFIED]
    return {
        "policy": "integrated-evidence-provenance-v1",
        "total_findings": total,
        "counts": counts,
        "traceable_findings": traceable,
        "evidence_coverage_percent": round(100.0 * traceable / total, 2)
        if total
        else 100.0,
        "findings": findings,
    }


def _derive_status(finding: dict[str, Any]) -> str:
    """Fall back to presence of citations when no grade was supplied.

    An integrated result carries no window context, so it can only be reported
    as traceable or not. This never invents a stronger grade than the evidence
    supports.
    """
    cited = _finding_evidence(finding)
    return STATUS_VERIFIED if cited else STATUS_UNVERIFIED


def _finding_evidence(finding: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for key, role in (("traceable_evidence", "supporting"), ("counter_evidence", "counter")):
        for item in finding.get(key) or []:
            if not isinstance(item, dict):
                continue
            export_id = item.get("export_id")
            if not isinstance(export_id, str):
                continue
            entries.append(
                {
                    "export_id": export_id,
                    "role": item.get("role") or role,
                    "roleLabel": ROLE_LABELS.get(item.get("role") or role, role),
                    "description": item.get("description") or "",
                }
            )
    return entries


def build_viewer_bundle(
    model_results: dict[str, dict[str, Any]],
    evidence_manifest: dict[str, Any],
    anonymised_root: Path,
    destination: Path,
    *,
    run_plan: dict[str, Any] | None = None,
    export_images: bool = True,
    evidence_summaries: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble one bundle per model without merging models together.

    ``model_results`` maps an arm name to a validated clinical result. Findings
    are emitted in full, including those graded ``unverified``: a produced
    clinical statement is retained and labelled rather than hidden, so the
    reader sees what the model said and how well it is supported.
    """
    destination.mkdir(parents=True, exist_ok=True)
    images_dir = destination / "evidence"

    wanted: list[str] = []
    for result in model_results.values():
        for finding in result.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            for entry in _finding_evidence(finding):
                if entry["export_id"] not in wanted:
                    wanted.append(entry["export_id"])

    references = locate_many(
        wanted,
        evidence_manifest,
        anonymised_root,
        run_plan=run_plan,
    )
    if export_images:
        references = export_evidence_images(references, images_dir)
    by_id = {item["evidence_id"]: item for item in references}

    summaries = evidence_summaries or {}
    models: list[dict[str, Any]] = []
    for name, result in sorted(model_results.items()):
        graded_status = status_by_finding_id(summaries.get(name, {}))
        findings = []
        for finding in result.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            status = graded_status.get(str(finding.get("finding_id")))
            if status is None:
                status = _derive_status(finding)
            evidence = []
            for entry in _finding_evidence(finding):
                reference = by_id.get(entry["export_id"], {})
                evidence.append(
                    {
                        **entry,
                        "image": reference.get("image_file"),
                        "seriesRole": reference.get("role"),
                        "geometricIndex": reference.get("geometric_index"),
                        "windowIds": reference.get("window_ids") or [],
                        "sopInstanceUid": reference.get("sop_instance_uid"),
                        "available": bool(reference.get("image_file")),
                    }
                )
            findings.append(
                {
                    "findingId": finding.get("finding_id"),
                    "confidence": finding.get("confidence"),
                    "limitations": finding.get("limitations") or [],
                    "evidenceStatus": status,
                    "evidenceStatusLabel": STATUS_LABELS.get(status, status),
                    "evidence": evidence,
                    "clinical": {
                        key: value
                        for key, value in finding.items()
                        if key
                        not in {
                            "finding_id",
                            "confidence",
                            "limitations",
                            "traceable_evidence",
                            "counter_evidence",
                            "evidence_status",
                        }
                    },
                }
            )
        models.append(
            {
                "model": name,
                "schemaVersion": result.get("schema_version"),
                "studyPurpose": result.get("study_purpose"),
                "globalLimitations": result.get("global_limitations") or [],
                "treatmentBoundary": result.get("treatment_boundary") or {},
                "evidenceSummary": summaries.get(name, {}),
                "findings": findings,
            }
        )

    bundle = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "non_diagnostic": True,
        "independent_models": True,
        "no_shared_verdict": True,
        "statusLabels": STATUS_LABELS,
        "roleLabels": ROLE_LABELS,
        "models": models,
        "imagesDirectory": "evidence",
        "notice": (
            "Exploratory and non-diagnostic. Models are presented independently and "
            "must not be combined into a shared conclusion. Markers are indicative "
            "only and are not model-provided coordinates or validated segmentations."
        ),
    }
    return bundle


def load_locked_model_results(manifests: Path, runs: Path) -> dict[str, dict[str, Any]]:
    """Collect locked per-model results that carry structured findings."""
    results: dict[str, dict[str, Any]] = {}
    medgemma_lock = manifests / "medgemma-results-lock.json"
    if medgemma_lock.exists():
        lock = load_json(medgemma_lock)
        run_id = lock.get("run")
        pointer = lock.get("multiplanar_result")
        if run_id and isinstance(pointer, str):
            candidate = runs / run_id / pointer
            if candidate.exists():
                results["medgemma-dicom"] = load_json(candidate)
    return results
