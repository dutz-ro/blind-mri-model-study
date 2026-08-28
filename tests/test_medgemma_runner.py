import json
from pathlib import Path

import pytest

from mristudy.clinical_schema import load_knee_meniscus_schema
from mristudy.errors import MRIStudyError
from mristudy.medgemma.integrity import recompute_plan_sha256
from mristudy.medgemma.output_contract import decoder_contract, output_contract
from mristudy.medgemma.runner import execute_window, run_window_plan
from mristudy.prompts import medgemma_json_suffix_v2, medgemma_knee_meniscus_prompt_v2
from mristudy.utils import canonical_json, sha256_bytes
from tests.medgemma_helpers import result_for_evidence


class FakeClient:
    def __init__(self, invalid_once: bool = False):
        self.calls = []
        self.invalid_once = invalid_once

    def analyze_window(
        self,
        window,
        prompt,
    ):
        self.calls.append(window["window_id"])
        valid = result_for_evidence(window["evidence_ids"][0])
        attempts = []
        if self.invalid_once:
            invalid = result_for_evidence(window["evidence_ids"][0])
            invalid["findings"][0]["stability_features"]["flap"] = "suspected"
            attempts.append(
                {
                    "attempt": 0,
                    "raw_text": json.dumps(invalid),
                    "result": invalid,
                    "validation_errors": ["invalid flap"],
                }
            )
            attempts.append(
                {
                    "attempt": 1,
                    "raw_text": json.dumps(valid),
                    "result": valid,
                    "validation_errors": [],
                }
            )
            accepted_attempt = 1
        else:
            attempts.append(
                {
                    "attempt": 0,
                    "raw_text": json.dumps(valid),
                    "result": valid,
                    "validation_errors": [],
                }
            )
            accepted_attempt = 0
        return {
            "model": "model",
            "model_revision": "revision",
            "window_id": window["window_id"],
            "role": window["role"],
            "instance_uids": window["instance_uids"],
            "evidence_ids": window["evidence_ids"],
            "output_schema": window["output_schema"],
            "prompt_sha256": window["prompt_sha256"],
            "generation": window["generation"],
            "decoder": window["decoder"],
            "output_contract": window.get("output_contract", output_contract()),
            "runtime_digest": window["runtime_digest"],
            "run_plan_sha256": plan()["plan_sha256"],
            "provision_plan_sha256": "provision-plan",
            "model_snapshot_manifest_sha256": "snapshot",
            "result": valid,
            "raw_text": json.dumps(valid),
            "schema_status": "valid",
            "validation_errors": [],
            "attempts": attempts,
            "accepted_attempt": accepted_attempt,
            "normalization_actions": [],
            "metrics": {
                "total_seconds": 1.0,
                "encoding_seconds": 0.2,
                "generation_seconds": 0.8,
                "gpu_peak_memory_bytes": 123,
                "oom": False,
            },
        }


def plan() -> dict:
    decoder = decoder_contract(load_knee_meniscus_schema())
    prompt = (
        medgemma_knee_meniscus_prompt_v2()
        + "\nThis window has role primary-volume, ID primary-volume-window-0001, and "
        + "series description 'primary'. Report only findings evaluable from this contiguous "
        + "window. Use indeterminate for fields requiring another plane. Cite only the supplied "
        + "global evidence IDs.\n"
        + medgemma_json_suffix_v2()
    )
    window = {
        "window_id": "primary-volume-window-0001",
        "role": "primary-volume",
        "study_uid": "1.2.3",
        "series_uid": "1.2.4",
        "series_description": "primary",
        "start_index": 0,
        "end_index": 0,
        "source_instance_count": 1,
        "instance_uids": ["1.2.5"],
        "evidence_ids": ["image-000001"],
        "input_list_sha256": sha256_bytes(
            canonical_json(
                {"instance_uids": ["1.2.5"], "evidence_ids": ["image-000001"]}
            )
        ),
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "output_schema": "knee-meniscus-finding-v2",
        "max_new_tokens": 4096,
        "maximum_structural_corrections": 2,
        "generation": {
            "do_sample": False,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 8,
            "json_schema_enforced": True,
            "canonical_eos_only": True,
        },
        "decoder": decoder,
        "output_contract": output_contract(),
        "model_repository": "model",
        "model_revision": "revision",
        "runtime_digest": "sha256:runtime",
    }
    result = {
        "windows": [window],
        "total_source_slices": 1,
        "unique_analyzed_slices": 1,
        "coverage_percent": 100.0,
        "total_windows": 1,
        "maximum_window_size": 1,
        "minimum_overlap": 8,
        "duplicate_window_memberships": 0,
        "missing_instance_uids": [],
        "unexpected_instance_uids": [],
        "historical_replay_sha256": "a" * 64,
        "decoder": decoder,
        "output_contract": output_contract(),
    }
    result["plan_sha256"] = recompute_plan_sha256(result)
    return result


def test_runner_corrects_invalid_structure_and_preserves_attempt(tmp_path: Path) -> None:
    client = FakeClient(invalid_once=True)
    result = run_window_plan(
        client,
        plan(),
        tmp_path / "run",
        medgemma_knee_meniscus_prompt_v2(),
        provision_plan_sha256="provision-plan",
        snapshot_manifest_sha256="snapshot",
    )
    assert result["coverage"]["coverage_percent"] == 100.0
    assert client.calls == ["primary-volume-window-0001"]
    assert (tmp_path / "run/windows/primary-volume-window-0001.attempt-00.json").exists()
    assert (tmp_path / "run/windows/primary-volume-window-0001.attempt-01.json").exists()


def test_resume_skips_valid_completed_window(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    first = FakeClient()
    run_window_plan(
        first,
        plan(),
        run_dir,
        medgemma_knee_meniscus_prompt_v2(),
        provision_plan_sha256="provision-plan",
        snapshot_manifest_sha256="snapshot",
    )
    second = FakeClient()
    run_window_plan(
        second,
        plan(),
        run_dir,
        medgemma_knee_meniscus_prompt_v2(),
        resume=True,
        provision_plan_sha256="provision-plan",
        snapshot_manifest_sha256="snapshot",
    )
    assert second.calls == []


def test_new_run_creates_its_own_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "new-run"
    assert not run_dir.exists()
    run_window_plan(
        FakeClient(),
        plan(),
        run_dir,
        medgemma_knee_meniscus_prompt_v2(),
        provision_plan_sha256="provision-plan",
        snapshot_manifest_sha256="snapshot",
    )
    assert run_dir.is_dir()
    assert (run_dir / "run-plan.json").is_file()


def test_resume_rejects_partial_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "windows").mkdir(parents=True)
    from mristudy.utils import atomic_json

    atomic_json(run_dir / "run-plan.json", plan())
    atomic_json(run_dir / "windows/primary-volume-window-0001.raw.json", {"partial": True})
    with pytest.raises(MRIStudyError, match="Partial window artifacts"):
        run_window_plan(
            FakeClient(),
            plan(),
            run_dir,
            medgemma_knee_meniscus_prompt_v2(),
            resume=True,
            provision_plan_sha256="provision-plan",
            snapshot_manifest_sha256="snapshot",
        )


def test_resume_rejects_changed_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_window_plan(
        FakeClient(),
        plan(),
        run_dir,
        medgemma_knee_meniscus_prompt_v2(),
        provision_plan_sha256="provision-plan",
        snapshot_manifest_sha256="snapshot",
    )
    changed = dict(plan())
    changed["plan_sha256"] = "changed"
    with pytest.raises(MRIStudyError, match="plan SHA-256"):
        run_window_plan(
            FakeClient(),
            changed,
            run_dir,
            medgemma_knee_meniscus_prompt_v2(),
            resume=True,
            provision_plan_sha256="provision-plan",
            snapshot_manifest_sha256="snapshot",
        )


def test_coverage_below_100_blocks_finalization(tmp_path: Path) -> None:
    incomplete = plan()
    incomplete["coverage_percent"] = 99.0
    incomplete["plan_sha256"] = recompute_plan_sha256(incomplete)
    with pytest.raises(MRIStudyError, match="coverage field is invalid"):
        run_window_plan(
            FakeClient(),
            incomplete,
            tmp_path / "run",
            medgemma_knee_meniscus_prompt_v2(),
            provision_plan_sha256="provision-plan",
            snapshot_manifest_sha256="snapshot",
        )


def test_completed_window_manifest_change_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_window_plan(
        FakeClient(),
        plan(),
        run_dir,
        medgemma_knee_meniscus_prompt_v2(),
        provision_plan_sha256="provision-plan",
        snapshot_manifest_sha256="snapshot",
    )
    from mristudy.utils import atomic_json, load_json

    manifest_path = run_dir / "windows/primary-volume-window-0001.manifest.json"
    manifest = load_json(manifest_path)
    manifest["input_list_sha256"] = "changed"
    atomic_json(manifest_path, manifest)
    with pytest.raises(MRIStudyError, match="manifest mismatch"):
        run_window_plan(
            FakeClient(),
            plan(),
            run_dir,
            medgemma_knee_meniscus_prompt_v2(),
            resume=True,
            provision_plan_sha256="provision-plan",
            snapshot_manifest_sha256="snapshot",
        )


def test_window_manifest_contains_complete_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_window_plan(
        FakeClient(),
        plan(),
        run_dir,
        medgemma_knee_meniscus_prompt_v2(),
        provision_plan_sha256="provision-plan",
        snapshot_manifest_sha256="snapshot",
    )
    from mristudy.utils import load_json

    manifest = load_json(run_dir / "windows/primary-volume-window-0001.manifest.json")
    assert manifest["role"] == "primary-volume"
    assert manifest["study_uid"] == "1.2.3"
    assert manifest["series_uid"] == "1.2.4"
    assert manifest["start_index"] == 0
    assert manifest["end_index"] == 0
    assert manifest["source_instance_count"] == 1
    assert manifest["schema_status"] == "valid"
    assert manifest["correction_artifacts"]
    assert manifest["structural_normalization_actions"] == []


def test_runtime_response_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    client = FakeClient()
    original = client.analyze_window

    def wrong_identity(window, prompt):
        response = original(window, prompt)
        response["model_revision"] = "wrong"
        return response

    client.analyze_window = wrong_identity
    with pytest.raises(MRIStudyError, match="two structural corrections"):
        run_window_plan(
            client,
            plan(),
            tmp_path / "run",
            medgemma_knee_meniscus_prompt_v2(),
            provision_plan_sha256="provision-plan",
            snapshot_manifest_sha256="snapshot",
        )


class FixedResultClient:
    def __init__(self, result: dict):
        self.result = result
        self.calls = 0

    def analyze_window(self, window, prompt):
        self.calls += 1
        raw = json.dumps(self.result)
        return {
            "model": window["model_repository"],
            "model_revision": window["model_revision"],
            "window_id": window["window_id"],
            "role": window["role"],
            "instance_uids": window["instance_uids"],
            "evidence_ids": window["evidence_ids"],
            "output_schema": window["output_schema"],
            "prompt_sha256": window["prompt_sha256"],
            "generation": window["generation"],
            "decoder": window["decoder"],
            "output_contract": window["output_contract"],
            "runtime_digest": window["runtime_digest"],
            "run_plan_sha256": "plan",
            "provision_plan_sha256": "provision",
            "model_snapshot_manifest_sha256": "snapshot",
            "result": self.result,
            "raw_text": raw,
            "schema_status": "valid",
            "validation_errors": [],
            "attempts": [
                {
                    "attempt": 0,
                    "raw_text": raw,
                    "result": self.result,
                    "normalization_actions": [],
                    "validation_errors": [],
                }
            ],
            "accepted_attempt": 0,
            "metrics": {"total_seconds": 0.0, "oom": False},
        }


def _runner_evidence_map() -> dict:
    return {
        "image-000001": {
            "evidence_id": "image-000001",
            "series_uid": "1.2.4",
            "geometric_index": 0,
            "window_ids": ["primary-volume-window-0001"],
        },
        "image-000002": {
            "evidence_id": "image-000002",
            "series_uid": "1.2.4",
            "geometric_index": 1,
            "window_ids": ["primary-volume-window-0002"],
        },
    }


def test_runner_preserves_adjacent_evidence_without_requesting_correction(
    tmp_path: Path,
) -> None:
    window = plan()["windows"][0]
    result = result_for_evidence("image-000002")
    client = FixedResultClient(result)
    completed = execute_window(
        client,
        window,
        medgemma_knee_meniscus_prompt_v2(),
        tmp_path,
        "plan",
        "provision",
        "snapshot",
        _runner_evidence_map(),
        "evidence-map",
    )

    assert client.calls == 1
    assert completed["canonical_result"] == result
    assert completed["result"] == result
    assert completed["evidence_summary"]["scope_counts"]["adjacent_same_series"] == 1
    assert completed["evidence_summary"]["findings"][0][
        "link_eligible_evidence_ids"
    ] == []
    assert (tmp_path / f"{window['window_id']}.integration-safe.json").exists()
    assert (tmp_path / f"{window['window_id']}.evidence-trace.json").exists()


def test_runner_quarantines_unknown_evidence_but_keeps_the_finding(tmp_path: Path) -> None:
    window = plan()["windows"][0]
    result = result_for_evidence("image-999999")
    completed = execute_window(
        FixedResultClient(result),
        window,
        medgemma_knee_meniscus_prompt_v2(),
        tmp_path,
        "plan",
        "provision",
        "snapshot",
        _runner_evidence_map(),
        "evidence-map",
    )

    assert completed["canonical_result"]["findings"][0]["traceable_evidence"][0][
        "export_id"
    ] == "image-999999"
    assert len(completed["result"]["findings"]) == 1
    assert completed["result"]["findings"][0]["traceable_evidence"] == []
    assert completed["evidence_summary"]["counts"]["unverified"] == 1


def test_resume_rederives_and_rejects_changed_integration_safe_result(tmp_path: Path) -> None:
    window = plan()["windows"][0]
    result = result_for_evidence("image-000002")
    client = FixedResultClient(result)
    execute_window(
        client,
        window,
        medgemma_knee_meniscus_prompt_v2(),
        tmp_path,
        "plan",
        "provision",
        "snapshot",
        _runner_evidence_map(),
        "evidence-map",
    )
    safe_path = tmp_path / f"{window['window_id']}.integration-safe.json"
    safe = json.loads(safe_path.read_text(encoding="utf-8"))
    safe["global_limitations"] = ["changed"]
    safe_path.write_text(json.dumps(safe), encoding="utf-8")

    with pytest.raises(MRIStudyError):
        execute_window(
            client,
            window,
            medgemma_knee_meniscus_prompt_v2(),
            tmp_path,
            "plan",
            "provision",
            "snapshot",
            _runner_evidence_map(),
            "evidence-map",
        )
    assert client.calls == 1


def test_initial_run_rejects_runtime_result_that_differs_from_raw_replay(
    tmp_path: Path,
) -> None:
    window = plan()["windows"][0]
    canonical = result_for_evidence("image-000001")
    client = FixedResultClient(canonical)
    original = client.analyze_window

    def mismatched(window, prompt):
        response = original(window, prompt)
        response["result"] = result_for_evidence("image-000002")
        return response

    client.analyze_window = mismatched
    with pytest.raises(MRIStudyError, match="two structural corrections"):
        execute_window(
            client,
            window,
            medgemma_knee_meniscus_prompt_v2(),
            tmp_path,
            "plan",
            "provision",
            "snapshot",
            _runner_evidence_map(),
            "evidence-map",
        )
