from copy import deepcopy

import pytest

from mristudy.clinical_schema import load_knee_meniscus_schema
from mristudy.errors import MRIStudyError
from mristudy.medgemma.integrity import (
    recompute_plan_sha256,
    require_complete_plan,
    require_valid_evidence_map,
)
from mristudy.medgemma.output_contract import decoder_contract
from mristudy.utils import canonical_json, sha256_bytes


def complete_plan() -> dict:
    decoder = decoder_contract(load_knee_meniscus_schema())
    plan = {
        "total_source_slices": 2,
        "unique_analyzed_slices": 2,
        "coverage_percent": 100.0,
        "total_windows": 1,
        "maximum_window_size": 2,
        "duplicate_window_memberships": 0,
        "missing_instance_uids": [],
        "unexpected_instance_uids": [],
        "minimum_overlap": 8,
        "evidence_map_sha256": "placeholder",
        "historical_replay_sha256": "a" * 64,
        "decoder": decoder,
        "windows": [
            {
                "window_id": "primary-volume-window-0001",
                "role": "primary-volume",
                "series_uid": "1.2.4",
                "instance_uids": ["1.2.5", "1.2.6"],
                "evidence_ids": ["image-000001", "image-000002"],
                "max_new_tokens": 4096,
                "maximum_structural_corrections": 2,
                "generation": {
                    "do_sample": False,
                    "repetition_penalty": 1.05,
                    "no_repeat_ngram_size": 8,
                    "json_schema_enforced": True,
                },
                "decoder": decoder,
                "start_index": 0,
                "end_index": 1,
                "source_instance_count": 2,
                "overlap_previous": 0,
                "overlap_next": 0,
            }
        ],
    }
    entries = [
        {
            "evidence_id": "image-000001",
            "sop_instance_uid": "1.2.5",
            "series_uid": "1.2.4",
            "role": "primary-volume",
            "window_ids": ["primary-volume-window-0001"],
        },
        {
            "evidence_id": "image-000002",
            "sop_instance_uid": "1.2.6",
            "series_uid": "1.2.4",
            "role": "primary-volume",
            "window_ids": ["primary-volume-window-0001"],
        },
    ]
    evidence_sha = sha256_bytes(canonical_json(entries))
    plan["evidence_map_sha256"] = evidence_sha
    plan["plan_sha256"] = recompute_plan_sha256(plan)
    return {
        "plan": plan,
        "manifest": {
            "evidence_count": len(entries),
            "entries": entries,
            "map_sha256": evidence_sha,
        },
    }


def test_plan_digest_and_coverage_are_recomputed() -> None:
    payload = complete_plan()
    require_complete_plan(payload["plan"])
    changed = deepcopy(payload["plan"])
    changed["windows"][0]["instance_uids"].pop()
    with pytest.raises(MRIStudyError, match="SHA-256"):
        require_complete_plan(changed)


def test_evidence_map_digest_and_memberships_are_recomputed() -> None:
    payload = complete_plan()
    require_valid_evidence_map(payload["manifest"], payload["plan"])
    changed = deepcopy(payload["manifest"])
    changed["entries"][0]["window_ids"] = []
    changed["map_sha256"] = sha256_bytes(canonical_json(changed["entries"]))
    changed_plan = deepcopy(payload["plan"])
    changed_plan["evidence_map_sha256"] = changed["map_sha256"]
    changed_plan["plan_sha256"] = recompute_plan_sha256(changed_plan)
    with pytest.raises(MRIStudyError, match="memberships"):
        require_valid_evidence_map(changed, changed_plan)


def test_evidence_id_cannot_change_sop_uid() -> None:
    payload = complete_plan()
    changed = deepcopy(payload["manifest"])
    changed["entries"][0]["sop_instance_uid"] = "1.2.999"
    changed["map_sha256"] = sha256_bytes(canonical_json(changed["entries"]))
    changed_plan = deepcopy(payload["plan"])
    changed_plan["evidence_map_sha256"] = changed["map_sha256"]
    changed_plan["plan_sha256"] = recompute_plan_sha256(changed_plan)
    with pytest.raises(MRIStudyError, match="different SOP"):
        require_valid_evidence_map(changed, changed_plan)
