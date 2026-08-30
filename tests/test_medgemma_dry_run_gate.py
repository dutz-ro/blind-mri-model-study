"""Drive the complete runner path on CPU, with no GPU and no medical images.

Every historical pre-lock failure was discovered on a paid GPU because the
first real medical window exercised code paths no local test had executed end
to end. This gate runs the full sequence -- window execution, artifact and
manifest integrity, resume validation, series integration and multiplanar
integration -- against a fully conformant client.

A failure here means the pipeline or its contract regressed. It says nothing
about model quality: the clinical payload is a fixed, deliberately empty and
conservative result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mristudy.clinical_schema import load_knee_meniscus_schema
from mristudy.errors import MRIStudyError
from mristudy.medgemma.output_contract import decoder_contract, generation_policy, output_contract
from mristudy.medgemma.runner import run_window_plan
from mristudy.prompts import (
    medgemma_knee_meniscus_prompt_v2,
    medgemma_window_prompt,
)
from mristudy.utils import sha256_bytes
from tests.test_medgemma_runner import plan as single_window_plan

PROVISION_SHA = "dry-run-provision-plan"
SNAPSHOT_SHA = "dry-run-snapshot-manifest"


def _valid_result() -> dict[str, Any]:
    """The canonical conservative v2 result: no findings, all boundaries false."""
    return {
        "schema_version": "knee-meniscus-finding-v2",
        "study_purpose": "exploratory_non_diagnostic",
        "findings": [],
        "global_limitations": ["Dry-run contract verification."],
        "treatment_boundary": {
            "treatment_recommendation_provided": False,
            "repairability_determined": False,
            "vascularity_directly_measured": False,
        },
    }


class ConformantClient:
    """Return exactly the provenance the runner requires, sourced from the plan."""

    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.calls: list[str] = []

    def analyze_window(self, window: dict[str, Any], prompt: str) -> dict[str, Any]:
        self.calls.append(window["window_id"])
        result = _valid_result()
        raw_text = json.dumps(result)
        return {
            "model": window["model_repository"],
            "model_revision": window["model_revision"],
            "model_snapshot_manifest_sha256": SNAPSHOT_SHA,
            "runtime_digest": window["runtime_digest"],
            "run_plan_sha256": self.plan["plan_sha256"],
            "provision_plan_sha256": PROVISION_SHA,
            "window_id": window["window_id"],
            "role": window["role"],
            "instance_uids": window["instance_uids"],
            "evidence_ids": window["evidence_ids"],
            "output_schema": window["output_schema"],
            "prompt_sha256": window["prompt_sha256"],
            "generation": window["generation"],
            "decoder": window["decoder"],
            "output_contract": window.get("output_contract", {}),
            "raw_text": raw_text,
            "result": result,
            "schema_status": "valid",
            "validation_errors": [],
            "attempts": [
                {
                    "attempt": 0,
                    "raw_text": raw_text,
                    "result": result,
                    "validation_errors": [],
                }
            ],
            "accepted_attempt": 0,
            "metrics": {
                "total_seconds": 0.0,
                "gpu_peak_memory_bytes": 0,
                "input_images": len(window["instance_uids"]),
                "oom": False,
            },
        }


def _run(plan: dict[str, Any], run_dir: Path, client: ConformantClient) -> dict[str, Any]:
    decoder = decoder_contract(load_knee_meniscus_schema())
    plan["decoder"] = decoder
    plan["output_contract"] = output_contract()
    plan["windows"][0]["generation"] = generation_policy()
    plan["windows"][0]["decoder"] = decoder
    plan["windows"][0]["output_contract"] = output_contract()
    prompt = medgemma_knee_meniscus_prompt_v2()
    plan["windows"][0]["prompt_sha256"] = sha256_bytes(
        medgemma_window_prompt(prompt, plan["windows"][0]).encode()
    )
    from mristudy.medgemma.integrity import recompute_plan_sha256

    plan["plan_sha256"] = recompute_plan_sha256(plan)
    client.plan = plan
    return run_window_plan(
        client,
        plan,
        run_dir,
        prompt,
        provision_plan_sha256=PROVISION_SHA,
        snapshot_manifest_sha256=SNAPSHOT_SHA,
    )


def test_plan_completes_end_to_end_without_a_gpu(tmp_path: Path) -> None:
    plan = single_window_plan()
    client = ConformantClient(plan)
    final = _run(plan, tmp_path / "run", client)
    assert client.calls == [window["window_id"] for window in plan["windows"]]
    assert final["window_count"] == len(plan["windows"])
    assert final["coverage"]["coverage_percent"] == 100.0
    assert final["coverage"]["missing_instance_uids"] == []
    assert final["multiplanar_result"]


def test_every_window_prompt_hash_matches_the_plan() -> None:
    """Prompt drift must fail before inference, not after a paid generation."""
    plan = single_window_plan()
    base = medgemma_knee_meniscus_prompt_v2()
    for window in plan["windows"]:
        expected = medgemma_window_prompt(base, window)
        assert window["prompt_sha256"] == sha256_bytes(expected.encode())


@pytest.mark.parametrize(
    "field", ["decoder", "generation", "output_contract", "runtime_digest"]
)
def test_missing_provenance_field_stops_the_run(tmp_path: Path, field: str) -> None:
    """Reproduce the defect class that invalidated several paid runtimes."""
    plan = single_window_plan()

    class DropsField(ConformantClient):
        def analyze_window(self, window: dict[str, Any], prompt: str) -> dict[str, Any]:
            response = super().analyze_window(window, prompt)
            response.pop(field)
            return response

    with pytest.raises(MRIStudyError):
        _run(plan, tmp_path / "run", DropsField(plan))


def test_completed_windows_are_not_repeated_on_resume(tmp_path: Path) -> None:
    plan = single_window_plan()
    run_dir = tmp_path / "run"
    _run(plan, run_dir, ConformantClient(plan))
    resumed = ConformantClient(plan)
    run_window_plan(
        resumed,
        plan,
        run_dir,
        medgemma_knee_meniscus_prompt_v2(),
        resume=True,
        provision_plan_sha256=PROVISION_SHA,
        snapshot_manifest_sha256=SNAPSHOT_SHA,
    )
    assert resumed.calls == []
