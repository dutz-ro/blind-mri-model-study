from __future__ import annotations

from pathlib import Path

from ..clinical_schema import parse_structured_model_result, structured_validation_error
from ..prompts import knee_meniscus_prompt_v2
from ..utils import atomic_json, canonical_json, sha256_bytes, sha256_file, utc_now
from .litellm import LiteLLMClient

DEFAULT_PROMPT = knee_meniscus_prompt_v2()
MAX_SCHEMA_CORRECTION_ATTEMPTS = 2


def _request_valid_structured_result(
    client: LiteLLMClient,
    model: str,
    prompt: str,
    paths: list[Path],
    arm_dir: Path,
    batch_id: str,
    initial_response: dict | None = None,
) -> tuple[dict, dict, Path]:
    response = initial_response or client.analyze_images(model, prompt, paths)
    raw_path = arm_dir / f"{batch_id}.json"
    if not raw_path.exists():
        atomic_json(raw_path, response)
    error = structured_validation_error(response)
    if not error:
        return response, parse_structured_model_result(response), raw_path

    for attempt in range(1, MAX_SCHEMA_CORRECTION_ATTEMPTS + 1):
        correction_prompt = (
            prompt
            + "\n\nYour previous response failed the required JSON Schema. "
            + "Do not change the image interpretation merely to satisfy validation. "
            + "Map unsupported labels to the closest allowed value, using indeterminate when "
            + "the evidence does not support present or absent. Return only the complete corrected "
            + "JSON object. Validation error: "
            + error[:1500]
        )
        response = client.analyze_images(model, correction_prompt, paths)
        corrected_path = arm_dir / f"{batch_id}.correction-{attempt:02d}.json"
        atomic_json(corrected_path, response)
        error = structured_validation_error(response)
        if not error:
            return response, parse_structured_model_result(response), corrected_path
    raise ValueError(
        f"Model failed schema validation after {MAX_SCHEMA_CORRECTION_ATTEMPTS} corrections: "
        f"{error[:500]}"
    )


def provider_plan(
    exports_manifest: dict,
    models: dict[str, str],
    batch_size: int = 12,
    selected_export_ids: list[str] | None = None,
    transport: str = "litellm-openai-compatible",
    required_upstream_host: str = "api.routera.one",
) -> dict:
    selected = set(selected_export_ids or [])
    exports = [
        item
        for item in exports_manifest["exports"]
        if not selected or item["export_id"] in selected
    ]
    batches = []
    for offset in range(0, len(exports), batch_size):
        batch = exports[offset : offset + batch_size]
        batches.append(
            {
                "batch_id": f"batch-{offset // batch_size + 1:04d}",
                "exports": [item["export_id"] for item in batch],
                "hashes": [item["sha256"] for item in batch],
            }
        )
    plan = {
        "schema_version": 1,
        "operation": "generalist-model-image-egress",
        "created_at": utc_now(),
        "models": models,
        "transport": transport,
        "required_upstream_host": required_upstream_host,
        "prompt_sha256": sha256_bytes(DEFAULT_PROMPT.encode()),
        "output_schema": "knee-meniscus-finding-v2",
        "batch_size": batch_size,
        "batches": batches,
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json(plan))
    return plan


def run_generalists(
    client: LiteLLMClient,
    exports_dir: Path,
    exports_manifest: dict,
    plan: dict,
    runs_dir: Path,
    resume: bool = False,
) -> dict:
    by_id = {item["export_id"]: item for item in exports_manifest["exports"]}
    summary = {"models": {}, "plan_sha256": plan["plan_sha256"]}
    for arm, model in plan["models"].items():
        arm_dir = runs_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=resume)
        responses = []
        for batch in plan["batches"]:
            paths = [exports_dir / by_id[export_id]["png"] for export_id in batch["exports"]]
            prompt = DEFAULT_PROMPT + "\nInput export IDs in order: " + ", ".join(batch["exports"])
            output = arm_dir / f"{batch['batch_id']}.json"
            structured_output = arm_dir / f"{batch['batch_id']}.structured.json"
            structured_path = None
            response_source = output
            if resume and output.exists() and structured_output.exists():
                import json

                with output.open(encoding="utf-8") as handle:
                    raw = json.load(handle)
                parse_structured_model_result(raw)
                structured_path = structured_output.name
            elif resume and output.exists() and not structured_output.exists():
                import json

                with output.open(encoding="utf-8") as handle:
                    initial = json.load(handle)
                raw, structured, response_source = _request_valid_structured_result(
                    client,
                    model,
                    prompt,
                    paths,
                    arm_dir,
                    batch["batch_id"],
                    initial,
                )
                atomic_json(structured_output, structured)
                structured_path = structured_output.name
            elif output.exists() or structured_output.exists():
                raise ValueError(f"Unexpected batch artifacts: {arm}/{batch['batch_id']}")
            else:
                if plan.get("output_schema") == "knee-meniscus-finding-v2":
                    raw, structured, response_source = _request_valid_structured_result(
                        client,
                        model,
                        prompt,
                        paths,
                        arm_dir,
                        batch["batch_id"],
                    )
                else:
                    raw = client.analyze_images(model, prompt, paths)
                    atomic_json(output, raw)
            if plan.get("output_schema") == "knee-meniscus-finding-v2":
                if structured_path is None:
                    structured = parse_structured_model_result(raw)
                if not structured_output.exists():
                    atomic_json(structured_output, structured)
                structured_path = structured_output.name
            responses.append(
                {
                    "batch_id": batch["batch_id"],
                    "response_sha256": sha256_file(response_source),
                    "response_artifact": response_source.name,
                    "response_model": raw.get("model"),
                    "usage": raw.get("usage"),
                    "structured_result": structured_path,
                    "clinical_schema_status": "valid" if structured_path else "not_requested",
                }
            )
        final = {
            "arm": arm,
            "configured_model": model,
            "prompt_sha256": plan["prompt_sha256"],
            "output_schema": plan.get("output_schema"),
            "responses": responses,
        }
        atomic_json(arm_dir / "final.json", final)
        summary["models"][arm] = final
    return summary


def supplemental_schema_plan(
    exports_manifest: dict,
    models: dict[str, str],
    selected_export_ids: list[str],
    batch_size: int = 12,
) -> dict:
    plan = provider_plan(
        exports_manifest,
        models,
        batch_size,
        selected_export_ids,
    )
    plan["operation"] = "generalist-supplemental-clinical-schema-v2"
    plan["does_not_replace_existing_lock"] = True
    plan["output_schema"] = "knee-meniscus-finding-v2"
    plan["plan_sha256"] = sha256_bytes(canonical_json(plan))
    return plan


def run_generalist_supplement(
    client: LiteLLMClient,
    exports_dir: Path,
    exports_manifest: dict,
    plan: dict,
    runs_dir: Path,
    resume: bool = False,
) -> dict:
    return run_generalists(
        client,
        exports_dir,
        exports_manifest,
        plan,
        runs_dir,
        resume=resume,
    )
