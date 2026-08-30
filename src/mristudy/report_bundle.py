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

from copy import deepcopy
from pathlib import Path
from typing import Any

from .clinical_schema import parse_structured_model_result
from .medgemma.evidence_locate import export_evidence_images, locate_many
from .medgemma.evidence_trace import (
    STATUS_CROSS_WINDOW,
    STATUS_RECOVERED,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    project_result_evidence,
    status_by_finding_id,
)
from .medgemma.integration import integrate_series
from .profiles import StudyProfile, load_profile_by_name, resolve_path
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


LATERALITY_MARKERS = {
    "left": ("gauche", "gaauche", "left", "stang", "stâng"),
    "right": ("droit", "droite", "right", "drept"),
}


def study_metadata(
    lock: dict[str, Any],
    run_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe the acquisition the report is about, without asserting anatomy.

    The report page needs to state which sequences were analysed and how much of
    them, so a reader can see the scope of a claim. Laterality is derived from
    the approved series descriptions and is reported as a derivation, never as an
    independent observation: the study analysed meniscal appearance, not side.
    """
    windows = (run_plan or {}).get("windows") or []
    sequences: list[dict[str, Any]] = []
    for window in windows:
        role = window.get("role")
        description = window.get("series_description")
        existing = next(
            (
                item
                for item in sequences
                if item["role"] == role and item["seriesDescription"] == description
            ),
            None,
        )
        if existing is None:
            sequences.append(
                {"role": role, "seriesDescription": description, "windows": 1}
            )
        else:
            existing["windows"] += 1
    haystack = " ".join(
        str(item["seriesDescription"] or "").lower() for item in sequences
    )
    sides = [
        side
        for side, markers in LATERALITY_MARKERS.items()
        if any(marker in haystack for marker in markers)
    ]
    return {
        "run": lock.get("run"),
        "outputSchema": lock.get("schema_version"),
        "bodyRegion": (run_plan or {}).get("body_region"),
        "profile": (run_plan or {}).get("profile"),
        "coverage": lock.get("coverage") or {},
        "windowCount": lock.get("window_count"),
        "seriesIntegrations": lock.get("series_integrations") or [],
        "sequences": sequences,
        "laterality": sides[0] if len(sides) == 1 else "indeterminate",
        "lateralitySource": "series_description",
        "radiologyReportUsed": bool((run_plan or {}).get("radiology_report_used")),
        "windowEvidenceSummary": lock.get("evidence_traceability") or {},
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


def evidence_summary_from_series_provenance(
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Derive viewer grades from one batch/window integration matrix."""
    return evidence_summary_from_multiplanar_provenance(
        {
            "groups": [
                {
                    "output_finding_provenance": [
                        {
                            "output_finding_id": item.get("integrated_finding_id"),
                            "evidence_dispositions": item.get("evidence_dispositions") or [],
                            "link_eligible_evidence_ids": item.get(
                                "link_eligible_evidence_ids"
                            )
                            or [],
                        }
                        for item in provenance.get("finding_provenance") or []
                    ]
                }
            ]
        }
    )


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
    model_metadata: dict[str, dict[str, Any]] | None = None,
    study: dict[str, Any] | None = None,
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
    metadata = model_metadata or {}
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
                    "provenance": (metadata.get(name, {}).get("findingProvenance") or {}).get(
                        str(finding.get("finding_id")), {}
                    ),
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
                "provenance": {
                    key: value
                    for key, value in metadata.get(name, {}).items()
                    if key != "findingProvenance"
                },
                "findings": findings,
            }
        )

    bundle = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "study": study or {},
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


def _evidence_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["evidence_id"]): item
        for item in manifest.get("entries") or []
        if isinstance(item, dict) and item.get("evidence_id")
    }


def _generalist_run_directory(lock: dict[str, Any], runs: Path) -> Path | None:
    candidates = []
    for candidate in sorted(runs.glob("generalists-*")):
        if all(
            all(
                (candidate / arm / str(response.get("structured_result"))).is_file()
                for response in details.get("responses") or []
                if response.get("structured_result")
            )
            for arm, details in (lock.get("models") or {}).items()
        ):
            candidates.append(candidate)
    return candidates[0] if len(candidates) == 1 else None


def _integrate_generalists(
    lock: dict[str, Any],
    runs: Path,
    provider_plan: dict[str, Any],
    evidence_map: dict[str, dict[str, Any]],
    profile: StudyProfile,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    run_dir = _generalist_run_directory(lock, runs)
    if run_dir is None:
        return {}, {}, {}
    batches = {
        str(item["batch_id"]): item for item in provider_plan.get("batches") or []
    }
    results: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for arm, details in sorted((lock.get("models") or {}).items()):
        windows = []
        response_model = None
        finding_provenance: dict[str, Any] = {}
        for response in details.get("responses") or []:
            structured_name = response.get("structured_result")
            batch_id = str(response.get("batch_id"))
            batch = batches.get(batch_id)
            if not structured_name or batch is None:
                continue
            result = load_json(run_dir / arm / str(structured_name))
            parse_structured_model_result(result)
            safe, _, summary = project_result_evidence(
                result,
                {
                    "window_id": batch_id,
                    "evidence_ids": list(batch.get("exports") or []),
                    "series_uid": f"generalist-{arm}",
                },
                evidence_map,
                recover=True,
            )
            windows.append(
                {"window_id": batch_id, "result": safe, "evidence_summary": summary}
            )
            response_model = response.get("response_model") or response_model
        if not windows:
            continue
        integrated, provenance = integrate_series(
            arm,
            f"generalist-selection-{arm}",
            windows,
            evidence_map=evidence_map,
            policy=profile.integration,
            schema_version=profile.output_schema,
            body_region=profile.body_region,
        )
        for item in provenance.get("finding_provenance") or []:
            finding_provenance[str(item["integrated_finding_id"])] = {
                "sourceType": "generalist_batches",
                "batchIds": item.get("window_ids") or [],
                "sourceFindingIds": item.get("source_finding_ids") or [],
                "repeatedAcrossBatches": bool(item.get("repeated_across_windows")),
            }
        results[arm] = integrated
        summaries[arm] = evidence_summary_from_series_provenance(provenance)
        metadata[arm] = {
            "sourceType": "generalist_exports",
            "configuredModel": details.get("configured_model"),
            "returnedModel": response_model,
            "run": run_dir.name,
            "batchCount": len(windows),
            "imageCount": len(
                {
                    export_id
                    for batch in batches.values()
                    for export_id in batch.get("exports") or []
                }
            ),
            "findingProvenance": finding_provenance,
        }
    return results, summaries, metadata


def _historical_medgemma_variants(
    runs: Path,
    final_run: str | None,
    evidence_map: dict[str, dict[str, Any]],
    profile: StudyProfile,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve positive variants from earlier runs without upgrading them.

    Earlier MedGemma runs are not silently treated as the final lock. Their
    positive variants are retained as historical observations, deduplicated by
    the profile claim key, and labelled with every source run and window. Empty
    citations remain unverified; only a citation the model actually emitted (or
    misplaced in finding_id) can become evidence.
    """
    variants: dict[tuple[Any, ...], dict[str, Any]] = {}
    for run_dir in sorted(runs.glob("medgemma-schema-v2-*")):
        if run_dir.name == final_run:
            continue
        for path in sorted((run_dir / "windows").glob("*.raw.json")):
            raw = load_json(path)
            result = raw.get("result") or {}
            try:
                parse_structured_model_result(result)
            except (ValueError, TypeError, KeyError):
                continue
            positives = [
                finding
                for finding in result.get("findings") or []
                if finding.get("tear_status") in profile.integration.positive_statuses
            ]
            if not positives:
                continue
            projected = {**result, "findings": positives}
            safe, _, summary = project_result_evidence(
                projected,
                {
                    "window_id": raw.get("window_id") or path.stem,
                    "evidence_ids": list(raw.get("evidence_ids") or []),
                    "series_uid": raw.get("series_uid"),
                },
                evidence_map,
                recover=True,
            )
            grades = {
                str(item.get("finding_id")): item
                for item in summary.get("findings") or []
            }
            for finding in safe.get("findings") or []:
                key = tuple(
                    resolve_path(finding, field)
                    for field in profile.integration.claim_key_paths
                )
                record = variants.setdefault(
                    key,
                    {
                        "finding": deepcopy(finding),
                        "sources": [],
                        "dispositions": [],
                        "linkEligible": set(),
                    },
                )
                grade = grades.get(str(finding.get("finding_id")), {})
                record["sources"].append(
                    {
                        "run": run_dir.name,
                        "windowId": raw.get("window_id") or path.stem,
                        "promptSha256": raw.get("prompt_sha256"),
                        "decoderPolicy": (raw.get("decoder") or {}).get("policy"),
                    }
                )
                record["dispositions"].extend(grade.get("dispositions") or [])
                record["linkEligible"].update(
                    grade.get("link_eligible_evidence_ids") or []
                )
                for evidence_key in ("traceable_evidence", "counter_evidence"):
                    merged = {
                        item["export_id"]: item
                        for item in record["finding"].get(evidence_key) or []
                    }
                    for item in finding.get(evidence_key) or []:
                        merged[item["export_id"]] = item
                    record["finding"][evidence_key] = [
                        merged[item] for item in sorted(merged)
                    ]

    findings = []
    provenance: dict[str, Any] = {}
    for index, record in enumerate(variants.values(), start=1):
        finding = record["finding"]
        finding_id = f"historical-medgemma-variant-{index:02d}"
        finding["finding_id"] = finding_id
        findings.append(finding)
        provenance[finding_id] = {
            "sourceType": "historical_medgemma_variant",
            "historical": True,
            "notPartOfFinalLock": True,
            "sources": record["sources"],
            "evidenceDispositions": record["dispositions"],
            "linkEligibleEvidenceIds": sorted(record["linkEligible"]),
        }
    return findings, provenance


def _historical_evidence_summary(
    provenance: dict[str, Any],
) -> dict[str, Any]:
    findings = []
    counts = {
        STATUS_VERIFIED: 0,
        STATUS_RECOVERED: 0,
        STATUS_CROSS_WINDOW: 0,
        STATUS_UNVERIFIED: 0,
    }
    for finding_id, item in provenance.items():
        dispositions = item.get("evidenceDispositions") or []
        scopes = {entry.get("scope_status") for entry in dispositions}
        recovered = any(
            entry.get("source_status") == "recovered_from_finding_id"
            for entry in dispositions
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
                "finding_id": finding_id,
                "evidence_status": status,
                "dispositions": dispositions,
                "link_eligible_evidence_ids": item.get("linkEligibleEvidenceIds") or [],
            }
        )
    return {
        "policy": "historical-medgemma-variant-traceability-v1",
        "total_findings": len(findings),
        "counts": counts,
        "traceable_findings": len(findings) - counts[STATUS_UNVERIFIED],
        "evidence_coverage_percent": round(
            100.0
            * (len(findings) - counts[STATUS_UNVERIFIED])
            / max(1, len(findings)),
            2,
        ),
        "findings": findings,
    }


def load_locked_model_bundle_inputs(
    manifests: Path,
    runs: Path,
    state: Path | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Collect locked arms plus explicitly labelled historical variants."""
    results: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    evidence_manifest_path = manifests / "medgemma-evidence-map.json"
    evidence_manifest = (
        load_json(evidence_manifest_path) if evidence_manifest_path.exists() else {}
    )
    evidence_map = _evidence_entries(evidence_manifest)
    study_manifest_path = manifests / "study.json"
    study_manifest = (
        load_json(study_manifest_path)
        if study_manifest_path.exists()
        else {"profile": "knee-meniscus-v1"}
    )
    profile = load_profile_by_name(str(study_manifest["profile"]))

    generalist_lock_path = manifests / "generalist-results-lock.json"
    provider_plan_path = state / "provider-egress-plan.json" if state else None
    if generalist_lock_path.exists() and provider_plan_path and provider_plan_path.exists():
        generalist_results, generalist_summaries, generalist_metadata = (
            _integrate_generalists(
                load_json(generalist_lock_path),
                runs,
                load_json(provider_plan_path),
                evidence_map,
                profile,
            )
        )
        results.update(generalist_results)
        summaries.update(generalist_summaries)
        metadata.update(generalist_metadata)

    medgemma_lock = manifests / "medgemma-results-lock.json"
    if medgemma_lock.exists():
        lock = load_json(medgemma_lock)
        run_id = lock.get("run")
        pointer = lock.get("multiplanar_result")
        if run_id and isinstance(pointer, str):
            candidate = runs / run_id / pointer
            if candidate.exists():
                result = load_json(candidate)
                historical, historical_provenance = _historical_medgemma_variants(
                    runs, str(run_id), evidence_map, profile
                )
                result["findings"] = list(result.get("findings") or []) + historical
                results["medgemma-dicom"] = result
                metadata["medgemma-dicom"] = {
                    "sourceType": "dicom_windows",
                    "run": run_id,
                    "windowCount": lock.get("window_count"),
                    "imageCount": (lock.get("coverage") or {}).get(
                        "analyzed_unique_slices"
                    ),
                    "historicalVariantCount": len(historical),
                    "historicalVariantsAreNotFinalLock": True,
                    "findingProvenance": historical_provenance,
                }
                summaries["medgemma-dicom-historical"] = _historical_evidence_summary(
                    historical_provenance
                )
    return results, summaries, metadata


def load_locked_model_results(
    manifests: Path, runs: Path, state: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Compatibility wrapper returning only the structured model results."""
    results, _, _ = load_locked_model_bundle_inputs(manifests, runs, state)
    return results
