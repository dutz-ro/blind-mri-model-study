from mristudy.medgemma.evidence_trace import (
    STATUS_CROSS_WINDOW,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
)
from mristudy.report_bundle import evidence_summary_from_multiplanar_provenance


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
