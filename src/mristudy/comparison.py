from __future__ import annotations

from pathlib import Path

from .clinical_schema import SCHEMA_VERSION, clinical_summary, parse_structured_model_result
from .utils import load_json


def build_comparison(manifests: Path, runs: Path) -> dict:
    arms = {}
    generalist_lock = manifests / "generalist-results-lock.json"
    if generalist_lock.exists():
        lock = load_json(generalist_lock)
        for name, arm in lock.get("models", {}).items():
            arms[name] = {
                **arm,
                "clinical_schema_status": "legacy_schema",
                "clinical_schema_version": None,
                "note": (
                    "This locked result predates knee-meniscus-finding-v2 and was not "
                    "retroactively converted or rerun."
                ),
            }
    supplement_lock = manifests / "generalist-schema-v2-results-lock.json"
    if supplement_lock.exists():
        supplement = load_json(supplement_lock)
        for name, arm in supplement.get("summary", {}).get("models", {}).items():
            structured_batches = []
            for response in arm.get("responses", []):
                structured_name = response.get("structured_result")
                if not structured_name:
                    continue
                structured_path = runs / supplement["run_id"] / name / structured_name
                structured = load_json(structured_path)
                structured_batches.append(
                    {
                        "batch_id": response["batch_id"],
                        "clinical_summary": clinical_summary(structured),
                    }
                )
            arms[f"{name}-schema-v2"] = {
                **arm,
                "clinical_schema_status": "valid_batches",
                "clinical_schema_version": SCHEMA_VERSION,
                "supplements_legacy_arm": name,
                "structured_batches": structured_batches,
            }
    medgemma_lock = manifests / "medgemma-results-lock.json"
    if medgemma_lock.exists():
        lock = load_json(medgemma_lock)
        arm = {"run": lock["run"], "provider": lock.get("provider")}
        if lock.get("multiplanar_result"):
            result_path = runs / lock["run"] / lock["multiplanar_result"]
            provenance_path = runs / lock["run"] / lock["multiplanar_provenance"]
            structured = load_json(result_path)
            parse_structured_model_result(structured)
            arm["clinical_schema_status"] = "valid"
            arm["clinical_schema_version"] = SCHEMA_VERSION
            arm["clinical_summary"] = clinical_summary(structured)
            arm["multiplanar_provenance"] = load_json(provenance_path)
            arm["coverage"] = lock.get("coverage")
            arm["window_count"] = lock.get("window_count")
            arm["series_integrations"] = lock.get("series_integrations")
            arm["runtime"] = lock.get("runtime")
            arm["run_plan_sha256"] = lock.get("run_plan_sha256")
        elif lock.get("outputs"):
            structured_outputs = []
            for output in lock["outputs"]:
                structured = load_json(runs / lock["run"] / output["structured"])
                parse_structured_model_result(structured)
                structured_outputs.append(
                    {
                        **output,
                        "clinical_summary": clinical_summary(structured),
                    }
                )
            arm["clinical_schema_status"] = "valid"
            arm["clinical_schema_version"] = SCHEMA_VERSION
            arm["structured_outputs"] = structured_outputs
            arm["runtime"] = lock.get("runtime")
        else:
            result_path = runs / lock["run"] / "final.json"
            raw_result = load_json(result_path)
            arm["result"] = raw_result
            try:
                structured = parse_structured_model_result(raw_result)
                arm["clinical_schema_status"] = "valid"
                arm["clinical_schema_version"] = SCHEMA_VERSION
                arm["clinical_summary"] = clinical_summary(structured)
            except (ValueError, TypeError, KeyError) as error:
                arm["clinical_schema_status"] = "invalid"
                arm["clinical_schema_version"] = None
                arm["clinical_schema_error"] = type(error).__name__
        arms["medgemma-dicom"] = arm
    return {
        "purpose": "exploratory_blind_model_comparison",
        "diagnostic_use": False,
        "arms": arms,
        "interpretation": {
            "model_agreement_is_medical_validation": False,
            "disagreement_preserved": True,
            "radiology_report_used": False,
            "vascular_zone_is_anatomic_estimate": True,
            "vascularity_directly_measured": False,
            "repairability_determined": False,
            "treatment_recommendation_provided": False,
        },
        "clinical_schema": {
            "current": SCHEMA_VERSION,
            "required_for_new_model_runs": True,
            "legacy_locked_outputs_preserved": True,
        },
    }
