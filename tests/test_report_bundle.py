from mristudy.medgemma.evidence_trace import (
    STATUS_CROSS_WINDOW,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
)
from mristudy.profiles import load_profile_by_name
from mristudy.report_bundle import (
    _historical_medgemma_variants,
    evidence_summary_from_multiplanar_provenance,
    evidence_summary_from_series_provenance,
    study_metadata,
)


def _output(finding_id: str, dispositions: list[dict]) -> dict:
    return {
        "output_finding_id": finding_id,
        "evidence_dispositions": dispositions,
        "link_eligible_evidence_ids": [
            item["export_id"]
            for item in dispositions
            if item.get("integration_link_eligible")
        ],
    }


def test_integrated_viewer_status_preserves_cross_window_warning() -> None:
    provenance = {
        "groups": [
            {
                "output_finding_provenance": [
                    _output(
                        "finding-current",
                        [
                            {
                                "export_id": "image-000071",
                                "scope_status": "current_window",
                                "source_status": "explicit",
                                "integration_link_eligible": True,
                            }
                        ],
                    ),
                    _output(
                        "finding-adjacent",
                        [
                            {
                                "export_id": "image-000070",
                                "scope_status": "adjacent_same_series",
                                "source_status": "explicit",
                                "integration_link_eligible": False,
                            }
                        ],
                    ),
                    _output("finding-none", []),
                ]
            }
        ]
    }
    summary = evidence_summary_from_multiplanar_provenance(provenance)
    statuses = {
        item["finding_id"]: item["evidence_status"] for item in summary["findings"]
    }
    assert statuses == {
        "finding-current": STATUS_VERIFIED,
        "finding-adjacent": STATUS_CROSS_WINDOW,
        "finding-none": STATUS_UNVERIFIED,
    }
    assert summary["evidence_coverage_percent"] == 66.67


def test_unknown_disposition_is_unverified_in_viewer() -> None:
    summary = evidence_summary_from_multiplanar_provenance(
        {
            "groups": [
                {
                    "output_finding_provenance": [
                        _output(
                            "finding-unknown",
                            [
                                {
                                    "export_id": "image-999999",
                                    "scope_status": "unknown",
                                    "source_status": "explicit",
                                    "integration_link_eligible": False,
                                }
                            ],
                        )
                    ]
                }
            ]
        }
    )
    assert summary["findings"][0]["evidence_status"] == STATUS_UNVERIFIED


def test_study_metadata_describes_scope_without_asserting_anatomy() -> None:
    """The report page states what was analysed, so a reader can size a claim.

    Laterality is derived from the approved series descriptions and is labelled
    as such: the study examined meniscal appearance, never the side.
    """
    lock = {
        "run": "medgemma-schema-v2-abc",
        "schema_version": "knee-meniscus-finding-v2",
        "window_count": 2,
        "coverage": {"coverage_percent": 100.0},
        "series_integrations": ["primary-volume"],
        "evidence_traceability": {"total_findings": 3, "evidence_coverage_percent": 66.7},
    }
    plan = {
        "body_region": "knee",
        "profile": "knee-meniscus-v1",
        "radiology_report_used": False,
        "windows": [
            {"role": "primary-volume", "series_description": "SAG 3D DP FS IA GAUCHE"},
            {"role": "primary-volume", "series_description": "SAG 3D DP FS IA GAUCHE"},
            {"role": "coronal-confirmation", "series_description": "MPR COR DP FS"},
        ],
    }
    metadata = study_metadata(lock, plan)
    assert metadata["laterality"] == "left"
    assert metadata["lateralitySource"] == "series_description"
    assert metadata["sequences"][0] == {
        "role": "primary-volume",
        "seriesDescription": "SAG 3D DP FS IA GAUCHE",
        "windows": 2,
    }
    assert metadata["windowEvidenceSummary"]["evidence_coverage_percent"] == 66.7
    assert metadata["radiologyReportUsed"] is False


def test_study_metadata_refuses_a_side_it_cannot_derive() -> None:
    lock = {"run": "run", "coverage": {}}
    plan = {
        "windows": [
            {"role": "primary-volume", "series_description": "SAG DP FS GAUCHE"},
            {"role": "coronal-confirmation", "series_description": "COR DP FS DROITE"},
        ]
    }
    assert study_metadata(lock, plan)["laterality"] == "indeterminate"
    assert study_metadata({"run": "run"}, None)["laterality"] == "indeterminate"


def test_series_summary_preserves_batch_scope() -> None:
    summary = evidence_summary_from_series_provenance(
        {
            "finding_provenance": [
                {
                    "integrated_finding_id": "gpt-finding-1",
                    "evidence_dispositions": [
                        {
                            "export_id": "image-000003",
                            "scope_status": "current_window",
                            "source_status": "explicit",
                            "integration_link_eligible": True,
                        }
                    ],
                    "link_eligible_evidence_ids": ["image-000003"],
                },
                {
                    "integrated_finding_id": "gpt-finding-2",
                    "evidence_dispositions": [
                        {
                            "export_id": "image-000015",
                            "scope_status": "distant_same_series",
                            "source_status": "explicit",
                            "integration_link_eligible": False,
                        }
                    ],
                    "link_eligible_evidence_ids": [],
                },
            ]
        }
    )
    assert [item["evidence_status"] for item in summary["findings"]] == [
        STATUS_VERIFIED,
        STATUS_CROSS_WINDOW,
    ]


def test_historical_medgemma_positive_is_retained_but_not_upgraded(tmp_path) -> None:
    run = tmp_path / "medgemma-schema-v2-old"
    windows = run / "windows"
    windows.mkdir(parents=True)
    result = {
        "schema_version": "knee-meniscus-finding-v2",
        "study_purpose": "exploratory_non_diagnostic",
        "findings": [
            {
                "finding_id": "f1",
                "meniscus": "medial",
                "segment": "posterior_horn",
                "tear_status": "probable",
                "tear_pattern": "horizontal",
                "orientation": "horizontal",
                "vascular_zone_estimate": {
                    "zone": "indeterminate",
                    "directly_observed_vascularity": False,
                    "basis": "insufficient_anatomic_resolution",
                    "confidence": "low",
                    "limitations": [],
                },
                "capsular_relation": {
                    "distance_mm_estimate": None,
                    "measurement_method": "not_measurable",
                    "peripheral_extension": "indeterminate",
                    "meniscocapsular_separation": "indeterminate",
                },
                "surface_extension": {
                    "superior": "indeterminate",
                    "inferior": "indeterminate",
                    "free_edge": "indeterminate",
                },
                "stability_features": {
                    "displaced_fragment": "indeterminate",
                    "flap": "indeterminate",
                    "bucket_handle_pattern": "indeterminate",
                    "imaging_signs_of_mechanical_obstruction": "indeterminate",
                },
                "healing_context": {
                    "extrusion_mm_estimate": None,
                    "parameniscal_cyst": "indeterminate",
                    "root_involvement": "indeterminate",
                    "tissue_quality_imaging_estimate": "indeterminate",
                    "chronicity_imaging_estimate": "indeterminate",
                },
                "imaging_factors_for_specialist_review": [],
                "traceable_evidence": [],
                "counter_evidence": [],
                "limitations": [],
                "confidence": "low",
            }
        ],
        "global_limitations": [],
        "treatment_boundary": {
            "treatment_recommendation_provided": False,
            "repairability_determined": False,
            "vascularity_directly_measured": False,
        },
    }
    import json

    (windows / "primary-volume-window-0005.raw.json").write_text(
        json.dumps(
            {
                "window_id": "primary-volume-window-0005",
                "evidence_ids": ["image-000095"],
                "result": result,
            }
        )
    )
    findings, provenance = _historical_medgemma_variants(
        tmp_path,
        "medgemma-schema-v2-final",
        {
            "image-000095": {
                "evidence_id": "image-000095",
                "series_uid": "series",
                "geometric_index": 94,
            }
        },
        load_profile_by_name("knee-meniscus-v1"),
    )
    assert findings[0]["tear_status"] == "probable"
    finding_id = findings[0]["finding_id"]
    assert provenance[finding_id]["notPartOfFinalLock"] is True
    assert provenance[finding_id]["evidenceDispositions"] == []
