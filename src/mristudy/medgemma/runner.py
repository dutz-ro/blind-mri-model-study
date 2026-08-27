from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ..clinical_schema import parse_structured_model_result
from ..errors import MRIStudyError
from ..prompts import medgemma_json_suffix_v2
from ..utils import atomic_json, canonical_json, load_json, sha256_bytes, sha256_file, utc_now
from .evidence import ROLE_ORDER
from .integration import integrate_multiplanar, integrate_series, validate_integration_evidence
from .integrity import require_complete_plan
from .output_contract import classify_output
from .validation import validate_window_result


class WindowInferenceClient(Protocol):
    def analyze_window(
        self,
        window: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]: ...


def _artifact_manifest(
    window: dict[str, Any],
    raw_path: Path,
    structured_path: Path,
    response: dict[str, Any],
    attempts: list[str],
    accepted_artifact: str,
    plan_sha256: str,
    provision_plan_sha256: str,
    snapshot_manifest_sha256: str,
) -> dict[str, Any]:
    metrics = response.get("metrics") or {}
    manifest = {
        "schema_version": 1,
        "window_id": window["window_id"],
        "role": window["role"],
        "study_uid": window["study_uid"],
        "series_uid": window["series_uid"],
        "series_description": window["series_description"],
        "start_index": window["start_index"],
        "end_index": window["end_index"],
        "source_instance_count": window["source_instance_count"],
        "plan_sha256": plan_sha256,
        "provision_plan_sha256": provision_plan_sha256,
        "model_snapshot_manifest_sha256": snapshot_manifest_sha256,
        "input_list_sha256": window["input_list_sha256"],
        "model_repository": window["model_repository"],
        "model_revision": window["model_revision"],
        "observed_model": response.get("model"),
        "observed_model_revision": response.get("model_revision"),
        "runtime_digest": window["runtime_digest"],
        "prompt_sha256": window["prompt_sha256"],
        "generation": window["generation"],
        "decoder": window["decoder"],
        "output_schema": window["output_schema"],
        "instance_uids": window["instance_uids"],
        "evidence_ids": window["evidence_ids"],
        "raw_artifact": raw_path.name,
        "raw_sha256": sha256_file(raw_path),
        "accepted_response_artifact": accepted_artifact,
        "accepted_response_sha256": sha256_file(raw_path.parent / accepted_artifact),
        "structured_artifact": structured_path.name,
        "structured_sha256": sha256_file(structured_path),
        "latency_seconds": metrics.get("total_seconds"),
        "encoding_seconds": metrics.get("encoding_seconds"),
        "generation_seconds": metrics.get("generation_seconds"),
        "peak_vram_bytes": metrics.get("gpu_peak_memory_bytes"),
        "tokens": metrics.get("tokens"),
        "oom": metrics.get("oom", False),
        "schema_status": "valid",
        "correction_artifacts": attempts,
        "structural_normalization_actions": response.get("normalization_actions") or [],
        "completed_at": utc_now(),
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest))
    return manifest


def validate_completed_window(
    window: dict[str, Any],
    raw_path: Path,
    structured_path: Path,
    manifest_path: Path,
    plan_sha256: str,
    provision_plan_sha256: str,
    snapshot_manifest_sha256: str,
) -> dict[str, Any]:
    if not (raw_path.exists() and structured_path.exists() and manifest_path.exists()):
        raise MRIStudyError(f"Partial window artifacts for {window['window_id']}.")
    manifest = load_json(manifest_path)
    expected = {
        "window_id": window["window_id"],
        "role": window["role"],
        "study_uid": window["study_uid"],
        "series_uid": window["series_uid"],
        "start_index": window["start_index"],
        "end_index": window["end_index"],
        "plan_sha256": plan_sha256,
        "provision_plan_sha256": provision_plan_sha256,
        "model_snapshot_manifest_sha256": snapshot_manifest_sha256,
        "input_list_sha256": window["input_list_sha256"],
        "instance_uids": window["instance_uids"],
        "evidence_ids": window["evidence_ids"],
        "generation": window["generation"],
        "decoder": window["decoder"],
        "raw_sha256": sha256_file(raw_path),
        "structured_sha256": sha256_file(structured_path),
        "schema_status": "valid",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise MRIStudyError(f"Window manifest mismatch for {window['window_id']}: {key}")
    manifest_payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != sha256_bytes(canonical_json(manifest_payload)):
        raise MRIStudyError(f"Window manifest digest is invalid: {window['window_id']}")
    accepted_path = raw_path.parent / manifest["accepted_response_artifact"]
    if not accepted_path.is_file():
        raise MRIStudyError(f"Accepted window response is missing: {window['window_id']}")
    if sha256_file(accepted_path) != manifest["accepted_response_sha256"]:
        raise MRIStudyError(f"Accepted window response changed: {window['window_id']}")
    for artifact in manifest.get("correction_artifacts") or []:
        if not (raw_path.parent / artifact).is_file():
            raise MRIStudyError(f"Window correction artifact is missing: {window['window_id']}")
    listed_artifacts = {
        manifest["raw_artifact"],
        manifest["structured_artifact"],
        manifest_path.name,
        *manifest.get("correction_artifacts", []),
    }
    actual_artifacts = {
        path.name for path in raw_path.parent.glob(f"{window['window_id']}.*.json")
    }
    if actual_artifacts != listed_artifacts:
        raise MRIStudyError(f"Unlisted or missing window artifacts: {window['window_id']}")
    accepted_response = load_json(accepted_path)
    accepted_attempt = accepted_response.get("attempt")
    if accepted_attempt is None:
        raise MRIStudyError(f"Accepted attempt index is missing: {window['window_id']}")
    raw_response = load_json(raw_path)
    runtime_attempt = next(
        (
            item
            for item in raw_response.get("attempts") or []
            if item.get("attempt") == accepted_attempt
        ),
        None,
    )
    if runtime_attempt is None:
        raise MRIStudyError(f"Accepted runtime attempt is missing: {window['window_id']}")
    schema = load_json(Path(__file__).parents[1] / "schemas/knee-meniscus-finding-v2.schema.json")
    replay = classify_output(runtime_attempt["raw_text"], schema, window["evidence_ids"])
    if replay.result is None:
        raise MRIStudyError(f"Accepted runtime output is not reproducible: {window['window_id']}")
    accepted_structured = replay.result
    structured = load_json(structured_path)
    if accepted_structured != structured:
        raise MRIStudyError(
            f"Accepted response differs from structured result: {window['window_id']}"
        )
    errors = validate_window_result(structured, window["evidence_ids"])
    if errors:
        raise MRIStudyError(f"Stored window result is invalid: {window['window_id']}")
    return {
        "window_id": window["window_id"],
        "role": window["role"],
        "series_uid": window["series_uid"],
        "start_index": window["start_index"],
        "end_index": window["end_index"],
        "result": structured,
        "manifest": manifest,
    }


def execute_window(
    client: WindowInferenceClient,
    window: dict[str, Any],
    base_prompt: str,
    windows_dir: Path,
    plan_sha256: str,
    provision_plan_sha256: str,
    snapshot_manifest_sha256: str,
) -> dict[str, Any]:
    raw_path = windows_dir / f"{window['window_id']}.raw.json"
    structured_path = windows_dir / f"{window['window_id']}.structured.json"
    manifest_path = windows_dir / f"{window['window_id']}.manifest.json"
    existing = list(windows_dir.glob(f"{window['window_id']}.*.json"))
    if existing:
        return validate_completed_window(
            window,
            raw_path,
            structured_path,
            manifest_path,
            plan_sha256,
            provision_plan_sha256,
            snapshot_manifest_sha256,
        )

    response = client.analyze_window(window, base_prompt)
    accepted_attempt = response.get("accepted_attempt")
    accepted_details = None
    if accepted_attempt is not None:
        accepted_details = next(
            (
                item
                for item in response.get("attempts") or []
                if item.get("attempt") == accepted_attempt
            ),
            None,
        )
    response["normalization_actions"] = (
        (accepted_details or {}).get("normalization_actions") or []
    )
    atomic_json(raw_path, response)
    attempts = []
    for attempt in response.get("attempts") or []:
        attempt_number = int(attempt["attempt"])
        attempt_path = windows_dir / f"{window['window_id']}.attempt-{attempt_number:02d}.json"
        atomic_json(attempt_path, attempt)
        attempts.append(attempt_path.name)
    expected_provenance = {
        "model": window["model_repository"],
        "model_revision": window["model_revision"],
        "window_id": window["window_id"],
        "role": window["role"],
        "instance_uids": window["instance_uids"],
        "evidence_ids": window["evidence_ids"],
        "output_schema": window["output_schema"],
        "prompt_sha256": window["prompt_sha256"],
        "generation": window["generation"],
        "decoder": window["decoder"],
        "runtime_digest": window["runtime_digest"],
        "run_plan_sha256": plan_sha256,
        "provision_plan_sha256": provision_plan_sha256,
        "model_snapshot_manifest_sha256": snapshot_manifest_sha256,
    }
    mismatches = [
        key
        for key, expected in expected_provenance.items()
        if response.get(key) != expected
    ]
    errors = list(response.get("validation_errors") or [])
    if mismatches:
        errors.append(f"Runtime response provenance mismatch: {mismatches}")
    structured = None
    try:
        structured = parse_structured_model_result(response.get("result") or {})
        errors.extend(validate_window_result(structured, window["evidence_ids"]))
    except (ValueError, TypeError, KeyError) as error:
        errors.append(str(error))
    if accepted_attempt is None or response.get("schema_status") != "valid":
        errors.append("Runtime did not accept a valid structural attempt.")
    accepted_artifact = f"{window['window_id']}.attempt-{int(accepted_attempt or 0):02d}.json"
    if not attempts or accepted_artifact not in attempts:
        errors.append("Accepted runtime attempt artifact is missing.")
    if structured is None or errors:
        invalid_path = windows_dir / f"{window['window_id']}.invalid.json"
        atomic_json(
            invalid_path,
            {
                "window_id": window["window_id"],
                "attempts": attempts,
                "validation_errors": errors,
                "stopped_at": utc_now(),
            },
        )
        raise MRIStudyError(
            f"Window failed after two structural corrections: {window['window_id']}"
        )

    atomic_json(structured_path, structured)
    manifest = _artifact_manifest(
        window,
        raw_path,
        structured_path,
        response,
        attempts,
        accepted_artifact,
        plan_sha256,
        provision_plan_sha256,
        snapshot_manifest_sha256,
    )
    atomic_json(manifest_path, manifest)
    return {
        "window_id": window["window_id"],
        "role": window["role"],
        "series_uid": window["series_uid"],
        "start_index": window["start_index"],
        "end_index": window["end_index"],
        "result": structured,
        "manifest": manifest,
    }


def run_window_plan(
    client: WindowInferenceClient,
    plan: dict[str, Any],
    run_dir: Path,
    prompt: str,
    *,
    resume: bool = False,
    evidence_map: dict[str, dict[str, Any]] | None = None,
    provision_plan_sha256: str,
    snapshot_manifest_sha256: str,
) -> dict[str, Any]:
    require_complete_plan(plan)
    plan_path = run_dir / "run-plan.json"
    if resume:
        if not plan_path.exists():
            raise MRIStudyError("Resume run has no immutable run plan.")
        stored_plan = load_json(plan_path)
        require_complete_plan(stored_plan)
        if stored_plan != plan:
            raise MRIStudyError("Resume plan SHA-256 does not match.")
    else:
        run_dir.mkdir(mode=0o700)
        atomic_json(plan_path, plan)

    windows_dir = run_dir / "windows"
    series_dir = run_dir / "series"
    integration_dir = run_dir / "integration"
    for path in (windows_dir, series_dir, integration_dir):
        path.mkdir(exist_ok=True, mode=0o700)

    window_results = []
    for window in plan["windows"]:
        window_prompt = (
            prompt
            + f"\nThis window has role {window['role']}, ID {window['window_id']}, and "
            + f"series description {window['series_description']!r}. Report only findings "
            + "evaluable from this contiguous window. Use indeterminate for fields requiring "
            + "another plane. Cite only the supplied global evidence IDs.\n"
            + medgemma_json_suffix_v2()
        )
        expected_prompt_sha256 = sha256_bytes(window_prompt.encode())
        if window["prompt_sha256"] != expected_prompt_sha256:
            raise MRIStudyError(f"Window prompt hash mismatch: {window['window_id']}")
        window_results.append(
            execute_window(
                client,
                window,
                window_prompt,
                windows_dir,
                plan["plan_sha256"],
                provision_plan_sha256,
                snapshot_manifest_sha256,
            )
        )

    series_results = []
    planned_roles = {window["role"] for window in plan["windows"]}
    for role in (item for item in ROLE_ORDER if item in planned_roles):
        role_windows = [item for item in window_results if item["role"] == role]
        if not role_windows:
            raise MRIStudyError(f"No completed windows for required role: {role}")
        series_uid = role_windows[0]["series_uid"]
        result, provenance = integrate_series(
            role,
            series_uid,
            role_windows,
            evidence_map=evidence_map,
        )
        if evidence_map is not None:
            validate_integration_evidence(result, evidence_map)
            errors = validate_window_result(result, list(evidence_map))
            if errors:
                raise MRIStudyError(f"Series integration failed validation: {role}")
        parse_structured_model_result(result)
        atomic_json(series_dir / f"{role}.json", result)
        atomic_json(series_dir / f"{role}.provenance.json", provenance)
        series_results.append(
            {
                "role": role,
                "series_uid": series_uid,
                "result": result,
                "provenance": provenance,
            }
        )

    final_result, final_provenance = integrate_multiplanar(series_results)
    if evidence_map is not None:
        validate_integration_evidence(final_result, evidence_map)
        errors = validate_window_result(final_result, list(evidence_map))
        if errors:
            raise MRIStudyError("Multiplanar integration failed evidence validation.")
    parse_structured_model_result(final_result)
    atomic_json(integration_dir / "multiplanar.json", final_result)
    atomic_json(integration_dir / "multiplanar.provenance.json", final_provenance)
    final = {
        "run_plan_sha256": plan["plan_sha256"],
        "coverage": {
            "total_source_slices": plan["total_source_slices"],
            "analyzed_unique_slices": plan["unique_analyzed_slices"],
            "coverage_percent": plan["coverage_percent"],
            "windows": plan["total_windows"],
            "maximum_window_size": plan["maximum_window_size"],
            "minimum_overlap": plan["minimum_overlap"],
            "duplicated_memberships_due_to_overlap": plan["duplicate_window_memberships"],
            "missing_instance_uids": plan["missing_instance_uids"],
            "unexpected_instance_uids": plan["unexpected_instance_uids"],
        },
        "window_count": len(window_results),
        "series_integrations": [item["role"] for item in series_results],
        "multiplanar_result": "integration/multiplanar.json",
        "multiplanar_provenance": "integration/multiplanar.provenance.json",
        "completed_at": utc_now(),
    }
    if final["coverage"]["coverage_percent"] != 100.0:
        raise MRIStudyError("Coverage below 100%; final result cannot be locked.")
    atomic_json(run_dir / "final.json", final)
    return final
