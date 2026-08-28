"""Traceability grading and audited recovery of misplaced image citations.

These tests encode the defect observed in a completed medical run: every one of
29 findings had an empty ``traceable_evidence`` list while 25 carried an
``image-NNNNNN`` value in ``finding_id``. Schema validation accepted that, so
the report was structurally valid but visually untraceable.
"""

from __future__ import annotations

from typing import Any

from mristudy.medgemma.evidence_trace import (
    STATUS_CROSS_WINDOW,
    STATUS_RECOVERED,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    aggregate_summaries,
    classify_evidence_id,
    coverage_warning,
    grade_result_evidence,
    is_evidence_id,
)


def status_by_index(summary):
    return {item["index"]: item["evidence_status"] for item in summary["findings"]}

WINDOW = [f"image-{index:06d}" for index in range(212, 243)]
NEIGHBOUR = [f"image-{index:06d}" for index in range(181, 212)]
PLAN = set(WINDOW) | set(NEIGHBOUR)


def _finding(finding_id: str, supporting: list[str] | None = None) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "traceable_evidence": [
            {"export_id": item, "role": "supporting", "description": "d"}
            for item in (supporting or [])
        ],
        "counter_evidence": [],
    }


def _result(findings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "knee-meniscus-finding-v2",
        "study_purpose": "exploratory_non_diagnostic",
        "findings": findings,
        "global_limitations": [],
        "treatment_boundary": {
            "treatment_recommendation_provided": False,
            "repairability_determined": False,
            "vascularity_directly_measured": False,
        },
    }


def test_correctly_cited_finding_is_verified_and_untouched() -> None:
    result = _result([_finding("window-0010-01", ["image-000220"])])
    graded, actions, summary = grade_result_evidence(result, WINDOW, PLAN)
    assert status_by_index(summary)[0] == STATUS_VERIFIED
    assert summary["evidence_coverage_percent"] == 100.0
    assert not [item for item in actions if item["operation"] == "recovered_citation"]


def test_citation_misplaced_in_finding_id_is_recovered() -> None:
    """The exact defect: the model used finding_id as the citation field."""
    result = _result([_finding("image-000220")])
    graded, actions, summary = grade_result_evidence(result, WINDOW, PLAN)
    evidence = graded["findings"][0]["traceable_evidence"]
    assert [item["export_id"] for item in evidence] == ["image-000220"]
    assert evidence[0]["role"] == "supporting"
    assert status_by_index(summary)[0] == STATUS_VERIFIED
    recovered = [item for item in actions if item["operation"] == "recovered_citation"]
    assert len(recovered) == 1
    assert recovered[0]["detail"]["in_current_window"] is True


def test_recovered_citation_from_overlapping_window_is_labelled() -> None:
    """Overlap is real, but it is not independent corroboration."""
    result = _result([_finding("image-000211")])
    graded, actions, summary = grade_result_evidence(result, WINDOW, PLAN)
    assert status_by_index(summary)[0] == STATUS_CROSS_WINDOW
    recovered = [item for item in actions if item["operation"] == "recovered_citation"]
    assert recovered[0]["detail"]["in_current_window"] is False
    assert summary["counts"][STATUS_CROSS_WINDOW] == 1


def test_unknown_identifier_is_never_invented_as_evidence() -> None:
    result = _result([_finding("image-999999")])
    graded, actions, summary = grade_result_evidence(result, WINDOW, PLAN)
    assert graded["findings"][0]["traceable_evidence"] == []
    assert status_by_index(summary)[0] == STATUS_UNVERIFIED
    assert any(item["rule_id"] == "unknown-finding-id-not-recovered-v1" for item in actions)
    assert summary["counts"][STATUS_UNVERIFIED] == 1


def test_malformed_identifier_is_left_alone() -> None:
    """Values such as image-00035x were observed and must not be parsed loosely."""
    for malformed in ("image-0160", "image-00035x", "image-00043"):
        result = _result([_finding(malformed)])
        graded, _, summary = grade_result_evidence(result, WINDOW, PLAN)
        assert graded["findings"][0]["traceable_evidence"] == []
        assert status_by_index(summary)[0] == STATUS_UNVERIFIED


def test_untraceable_finding_is_preserved_not_discarded() -> None:
    """Losing a produced clinical result is worse than flagging it."""
    finding = _finding("window-0010-01")
    finding["clinical_note"] = "retained"
    graded, _, summary = grade_result_evidence(_result([finding]), WINDOW, PLAN)
    assert len(graded["findings"]) == 1
    assert graded["findings"][0]["clinical_note"] == "retained"
    assert status_by_index(summary)[0] == STATUS_UNVERIFIED
    assert summary["evidence_coverage_percent"] == 0.0


def test_weakest_citation_determines_status() -> None:
    result = _result([_finding("window-0010-01", ["image-000220", "image-000200"])])
    graded, _, summary = grade_result_evidence(result, WINDOW, PLAN)
    assert status_by_index(summary)[0] == STATUS_CROSS_WINDOW


def test_grading_is_idempotent() -> None:
    result = _result([_finding("image-000211"), _finding("window-0010-02")])
    once, _, first = grade_result_evidence(result, WINDOW, PLAN)
    twice, _, second = grade_result_evidence(once, WINDOW, PLAN)
    assert once == twice
    assert first == second


def test_graded_result_stays_schema_clean() -> None:
    """The canonical schema forbids extra properties, so grades live beside it."""
    result = _result([_finding("image-000220")])
    graded, _, summary = grade_result_evidence(result, WINDOW, PLAN)
    assert "evidence_status" not in graded["findings"][0]
    assert "evidence_summary" not in graded
    assert summary["findings"][0]["evidence_status"] == STATUS_VERIFIED


def test_original_result_is_not_mutated() -> None:
    result = _result([_finding("image-000220")])
    grade_result_evidence(result, WINDOW, PLAN)
    assert result["findings"][0]["traceable_evidence"] == []
    assert "evidence_status" not in result["findings"][0]


def test_classification_distinguishes_window_plan_and_unknown() -> None:
    assert classify_evidence_id("image-000220", set(WINDOW), PLAN) == STATUS_VERIFIED
    assert classify_evidence_id("image-000211", set(WINDOW), PLAN) == STATUS_CROSS_WINDOW
    assert classify_evidence_id("image-999999", set(WINDOW), PLAN) == STATUS_UNVERIFIED


def test_is_evidence_id_requires_exact_shape() -> None:
    assert is_evidence_id("image-000001")
    assert not is_evidence_id("image-0160")
    assert not is_evidence_id("image-00035x")
    assert not is_evidence_id("window-0010-01")
    assert not is_evidence_id(None)


def test_aggregate_and_warning_report_low_traceability() -> None:
    summaries = [
        grade_result_evidence(_result([_finding("w-1")]), WINDOW, PLAN)[2],
        grade_result_evidence(_result([_finding("w-2", ["image-000220"])]), WINDOW, PLAN)[2],
    ]
    overall = aggregate_summaries(summaries)
    assert overall["total_findings"] == 2
    assert overall["evidence_coverage_percent"] == 50.0
    assert coverage_warning(overall, 80.0) is not None
    assert coverage_warning(overall, 25.0) is None


def test_no_findings_reports_full_coverage_and_no_warning() -> None:
    graded, actions, summary = grade_result_evidence(_result([]), WINDOW, PLAN)
    assert graded["findings"] == []
    assert actions == []
    assert summary["evidence_coverage_percent"] == 100.0
    assert coverage_warning(summary, 80.0) is None


def test_recovery_can_be_disabled_for_strict_reporting() -> None:
    result = _result([_finding("image-000220")])
    graded, actions, summary = grade_result_evidence(result, WINDOW, PLAN, recover=False)
    assert graded["findings"][0]["traceable_evidence"] == []
    assert status_by_index(summary)[0] == STATUS_UNVERIFIED
    assert not [item for item in actions if item["operation"] == "recovered_citation"]
    assert summary["counts"][STATUS_RECOVERED] == 0
