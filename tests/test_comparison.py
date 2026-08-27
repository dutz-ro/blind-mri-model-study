from pathlib import Path

from mristudy.comparison import build_comparison
from mristudy.utils import atomic_json


def test_comparison_preserves_all_available_arms(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    runs = tmp_path / "runs"
    atomic_json(manifests / "generalist-results-lock.json", {"models": {"gpt": {}, "grok": {}}})
    atomic_json(manifests / "medgemma-results-lock.json", {"run": "medgemma-1"})
    atomic_json(runs / "medgemma-1" / "final.json", {"choices": []})
    result = build_comparison(manifests, runs)
    assert set(result["arms"]) == {"gpt", "grok", "medgemma-dicom"}
    assert result["interpretation"]["disagreement_preserved"]
    assert result["arms"]["gpt"]["clinical_schema_status"] == "legacy_schema"
    assert result["arms"]["medgemma-dicom"]["clinical_schema_status"] == "invalid"
    assert result["interpretation"]["vascular_zone_is_anatomic_estimate"]


def test_comparison_includes_v2_generalist_structured_batches(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    runs = tmp_path / "runs"
    structured = {
        "schema_version": "knee-meniscus-finding-v2",
        "study_purpose": "exploratory_non_diagnostic",
        "findings": [],
        "global_limitations": ["Limited batch context."],
        "treatment_boundary": {
            "treatment_recommendation_provided": False,
            "repairability_determined": False,
            "vascularity_directly_measured": False,
        },
    }
    atomic_json(runs / "run-v2" / "gpt-sol" / "batch-0001.structured.json", structured)
    atomic_json(
        manifests / "generalist-schema-v2-results-lock.json",
        {
            "run_id": "run-v2",
            "schema_version": "knee-meniscus-finding-v2",
            "summary": {
                "models": {
                    "gpt-sol": {
                        "responses": [
                            {
                                "batch_id": "batch-0001",
                                "structured_result": "batch-0001.structured.json",
                            }
                        ]
                    }
                }
            },
        },
    )
    result = build_comparison(manifests, runs)
    arm = result["arms"]["gpt-sol-schema-v2"]
    assert arm["clinical_schema_status"] == "valid_batches"
    assert arm["structured_batches"][0]["clinical_summary"]["finding_count"] == 0


def test_comparison_includes_multi_series_medgemma_outputs(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    runs = tmp_path / "runs"
    structured = {
        "schema_version": "knee-meniscus-finding-v2",
        "study_purpose": "exploratory_non_diagnostic",
        "findings": [],
        "global_limitations": [],
        "treatment_boundary": {
            "treatment_recommendation_provided": False,
            "repairability_determined": False,
            "vascularity_directly_measured": False,
        },
    }
    atomic_json(runs / "med-1" / "primary.structured.json", structured)
    atomic_json(
        manifests / "medgemma-results-lock.json",
        {
            "run": "med-1",
            "provider": "scaleway",
            "outputs": [
                {
                    "role": "primary-volume",
                    "structured": "primary.structured.json",
                }
            ],
        },
    )
    result = build_comparison(manifests, runs)
    arm = result["arms"]["medgemma-dicom"]
    assert arm["clinical_schema_status"] == "valid"
    assert arm["structured_outputs"][0]["role"] == "primary-volume"


def test_comparison_includes_windowed_medgemma_coverage(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    runs = tmp_path / "runs"
    structured = {
        "schema_version": "knee-meniscus-finding-v2",
        "study_purpose": "exploratory_non_diagnostic",
        "findings": [],
        "global_limitations": [],
        "treatment_boundary": {
            "treatment_recommendation_provided": False,
            "repairability_determined": False,
            "vascularity_directly_measured": False,
        },
    }
    atomic_json(runs / "med-window" / "integration/multiplanar.json", structured)
    atomic_json(
        runs / "med-window" / "integration/multiplanar.provenance.json",
        {"groups": [], "contradictions_preserved": True},
    )
    coverage = {
        "total_source_slices": 617,
        "analyzed_unique_slices": 617,
        "coverage_percent": 100.0,
        "windows": 26,
    }
    atomic_json(
        manifests / "medgemma-results-lock.json",
        {
            "run": "med-window",
            "provider": "scaleway",
            "multiplanar_result": "integration/multiplanar.json",
            "multiplanar_provenance": "integration/multiplanar.provenance.json",
            "coverage": coverage,
            "window_count": 26,
            "series_integrations": ["primary-volume"],
            "run_plan_sha256": "plan",
        },
    )
    result = build_comparison(manifests, runs)
    arm = result["arms"]["medgemma-dicom"]
    assert arm["coverage"] == coverage
    assert arm["window_count"] == 26
    assert arm["clinical_schema_status"] == "valid"
