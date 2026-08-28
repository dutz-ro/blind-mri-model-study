from __future__ import annotations

from pathlib import Path

from mristudy.clinical_schema import empty_knee_meniscus_result
from mristudy.models.runs import run_generalists
from mristudy.utils import atomic_json, sha256_file


class NoCallClient:
    def analyze_images(self, *args, **kwargs):
        raise AssertionError("A completed batch must not be sent again during resume.")


def test_resume_validates_the_accepted_structured_result_not_initial_raw(
    tmp_path: Path,
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    image = exports / "image-000001.png"
    image.write_bytes(b"not-read-during-resume")
    manifest = {
        "exports": [
            {
                "export_id": "image-000001",
                "png": image.name,
                "sha256": sha256_file(image),
            }
        ]
    }
    plan = {
        "plan_sha256": "plan",
        "prompt_sha256": "prompt",
        "output_schema": "knee-meniscus-finding-v2",
        "models": {"grok": "grok-4.5"},
        "batches": [{"batch_id": "batch-0001", "exports": ["image-000001"]}],
    }
    arm = tmp_path / "runs" / "grok"
    arm.mkdir(parents=True)

    # The initial provider response was structurally invalid, then a correction
    # produced the valid structured result stored beside it.
    atomic_json(arm / "batch-0001.json", {"model": "grok-4.5", "choices": []})
    correction = arm / "batch-0001.correction-01.json"
    atomic_json(correction, {"model": "grok-4.5", "accepted": True})
    atomic_json(arm / "batch-0001.structured.json", empty_knee_meniscus_result())

    result = run_generalists(
        NoCallClient(),
        exports,
        manifest,
        plan,
        tmp_path / "runs",
        resume=True,
    )

    response = result["models"]["grok"]["responses"][0]
    assert response["structured_result"] == "batch-0001.structured.json"
    assert response["response_artifact"] == correction.name
    assert response["response_sha256"] == sha256_file(correction)
