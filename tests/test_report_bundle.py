from mristudy.medgemma.evidence_trace import (
    STATUS_CROSS_WINDOW,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
)
from mristudy.report_bundle import (
    evidence_summary_from_multiplanar_provenance,
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
