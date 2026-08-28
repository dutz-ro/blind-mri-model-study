from pathlib import Path

from mristudy.medgemma.replay import (
    replay_acceptance,
    replay_historical_runs,
    replay_report_digest,
)
from mristudy.utils import atomic_json
from tests.test_clinical_schema import valid_result


def test_replay_indexes_only_hashes_and_classifications(tmp_path: Path) -> None:
    windows = tmp_path / "medgemma-schema-v2-test/windows"
    windows.mkdir(parents=True)
    atomic_json(
        windows / "opaque.raw.json",
        {
            "window_id": "private-window-name",
            "evidence_ids": ["image-000123"],
            "attempts": [
                {
                    "attempt": 0,
                    "raw_text": __import__("json").dumps(valid_result()),
                }
            ],
        },
    )
    output = tmp_path / "replay.json"
    report = replay_historical_runs(tmp_path, output)
    assert report["source_artifacts"] == 1
    assert report["attempts"] == 1
    assert report["classifications"] == {"exact_valid": 1}
    rendered = output.read_text(encoding="utf-8")
    assert "private-window-name" not in rendered
    assert "image-000123" not in rendered
    assert replay_acceptance(report) == {
        "passed": True,
        "historical_attempts": 1,
        "recoverable_attempts": 1,
        "unrecoverable_attempts": 0,
        "truncated_attempts": 0,
        "ambiguous_attempts": 0,
        "unrecoverable_have_no_json_candidates": True,
        "all_json_candidates_recoverable": True,
        "truncated_outputs_require_structural_correction": True,
    }


def test_replay_acceptance_rejects_ambiguity() -> None:
    assert not replay_acceptance(
        {
            "attempts": 1,
            "classifications": {"ambiguous_multiple_valid_results": 1},
            "records": [],
        }
    )["passed"]


def test_replay_acceptance_classifies_truncation_as_correctable() -> None:
    acceptance = replay_acceptance(
        {
            "attempts": 2,
            "classifications": {"exact_valid": 1, "truncated_json": 1},
            "records": [
                {
                    "attempts": [
                        {"status": "exact_valid", "candidate_count": 1},
                        {"status": "truncated_json", "candidate_count": 0},
                    ]
                }
            ],
        }
    )
    assert acceptance["passed"]
    assert acceptance["truncated_attempts"] == 1
    assert acceptance["all_json_candidates_recoverable"]
    assert acceptance["truncated_outputs_require_structural_correction"]


def test_replay_digest_ignores_timestamp_but_binds_records() -> None:
    report = {
        "created_at": "first",
        "records": [{"source_artifact_sha256": "a"}],
        "classifications": {"exact_valid": 1},
    }
    first = replay_report_digest(report)
    report["created_at"] = "second"
    assert replay_report_digest(report) == first
    report["records"][0]["source_artifact_sha256"] = "b"
    assert replay_report_digest(report) != first
