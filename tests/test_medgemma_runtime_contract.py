import ast
from pathlib import Path

from mristudy.clinical_schema import load_knee_meniscus_schema
from mristudy.medgemma.output_contract import classify_output

RUNTIME = Path(__file__).parents[1] / "deploy/scaleway/medgemma/app.py"
CANONICAL_SCHEMA = (
    Path(__file__).parents[1]
    / "src/mristudy/schemas/knee-meniscus-finding-v2.schema.json"
)


def test_runtime_is_syntax_valid_and_has_required_private_endpoints() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    ast.parse(source)
    assert '@app.get("/health")' in source
    assert '@app.post("/synthetic-probe")' in source
    assert '@app.post("/synthetic-dicom-probe")' in source
    assert '@app.post("/synthetic-json-probe")' in source
    assert '@app.post("/probe-dicomweb")' in source
    assert '@app.post("/analyze-window")' in source
    assert '@app.post("/analyze")' not in source
    assert '@app.post("/synthetic-dicom-probe")' in source
    assert 'uvicorn.run(app, host="0.0.0.0", port=8000)' in source


def test_runtime_pins_schema_and_supports_offline_model_snapshot() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    # The runtime resolves its schema version from the mounted schema rather
    # than pinning one literal, so a different body region can run.
    assert "def _mounted_schema_version()" in source
    assert 'PROMPT_SCHEMA_VERSION = _mounted_schema_version()' in source
    assert 'MODEL_PATH = os.environ.get("MEDGEMMA_MODEL_PATH", "")' in source
    assert '"local_files_only": True' in source
    assert '"MEDGEMMA_MODEL_PATH is required."' in source
    assert "Draft202012Validator" in source
    assert "APPROVED_PLAN_SHA256" in source
    assert "PROBE_AUTHORIZATION_PATH" in source
    assert "_require_probe_authorization" in source
    assert '"run_plan_sha256": APPROVED_PLAN_SHA256' in source
    assert '"runtime_digest": RUNTIME_DIGEST' in source
    # Decoder provenance is asserted structurally in
    # tests/test_runtime_response_contract.py. Counting occurrences of a source
    # string cannot distinguish a returned response field from an unrelated
    # mention, and it breaks whenever a legitimate new response path is added.
    assert '"decoder": _decoder_identity()' in source
    assert "def _decoder_identity()" in source
    assert "schema_errors = structural_correction_errors(decision)" in source
    assert 'maximum_corrections = int(approved["maximum_structural_corrections"])' in source
    assert "for attempt in range(1 + maximum_corrections)" in source
    assert "if attempt < maximum_corrections" in source
    assert "for attempt in range(3)" not in source


def test_runtime_orders_dicom_by_geometry_and_maps_evidence_ids() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    assert "request.evidence_ids" in source
    assert "request.instance_uids" in source
    assert "def _verify_window_membership" in source
    assert "classify_output" in source
    assert '"multipart/related; type=application/dicom; transfer-syntax=*"' in source
    assert "sampling" not in source
    assert "max_slices" not in source
    assert "APPROVED_PLAN_PATH" in source
    assert "_load_approved_window" in source
    assert "validate_window_payload" in source


def test_runtime_schema_is_the_canonical_packaged_schema() -> None:
    dockerfile = (
        Path(__file__).parents[1] / "deploy/scaleway/medgemma/Dockerfile"
    ).read_text(encoding="utf-8")
    assert "src/mristudy/schemas/knee-meniscus-finding-v2.schema.json" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile


def test_runtime_parser_accepts_model_info_prefix_before_json_fence() -> None:
    decision = classify_output(
        'info\n```json\n{"schema_version":"knee-meniscus-finding-v2",'
        '"study_purpose":"exploratory_non_diagnostic","findings":[],'
        '"global_limitations":[],"treatment_boundary":{'
        '"treatment_recommendation_provided":false,"repairability_determined":false,'
        '"vascularity_directly_measured":false}}\n```',
        load_knee_meniscus_schema(),
        [],
    )
    assert decision.status == "exact_valid"


def test_runtime_parser_accepts_prose_before_final_json_fence() -> None:
    response = (
        "FINDINGS: synthetic prose\n```json\n"
        '{"schema_version":"knee-meniscus-finding-v2",'
        '"study_purpose":"exploratory_non_diagnostic","findings":[],'
        '"global_limitations":[],"treatment_boundary":{'
        '"treatment_recommendation_provided":false,"repairability_determined":false,'
        '"vascularity_directly_measured":false}}\n```'
    )
    decision = classify_output(response, load_knee_meniscus_schema(), [])
    assert decision.status == "exact_valid"


def test_synthetic_probe_uses_exact_capacity_marker() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    assert 'marker = f"MRI_STUDY_SYNTHETIC_OK_{len(images)}"' in source
    assert '"passed": model_response == marker' in source
    assert '"validated_result": expected' in source
    assert "def _is_valid_empty_synthetic_result" in source
    assert '"passed": _is_valid_empty_synthetic_result(decision.result)' in source


def test_medical_output_contract_is_placed_after_images() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    assert 'prompt.rsplit("Return JSON now.", maxsplit=1)' in source
    assert '"text": "Return JSON now." + prompt_parts[1]' in source
    assert '"repetition_penalty": request.generation["repetition_penalty"]' in source
    assert '"no_repeat_ngram_size": request.generation["no_repeat_ngram_size"]' in source
    assert '"prefix_allowed_tokens_fn": _schema_prefix_allowed_tokens_fn()' in source
    assert "generate_kwargs={" in source
    assert "JsonSchemaParser(" in source
    assert "constrained_schema," in source
    assert "decoder_contract(schema)" in source
    assert "CharacterLevelParserConfig(**DECODER_CONFIG)" in source
    assert "Constrained decoder allowed a non-JSON prefix" in source
    assert '"starts_with_json_object": raw_text.lstrip().startswith("{")' in source
    assert '"generation": request.generation' in source
    assert "Do not inspect or reinterpret the images" in source
    assert 'else:\n                content = [{"type": "text", "text": prompt}]' in source


def test_all_runtime_generation_controls_are_nested() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    tree = ast.parse(source)
    pipeline_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Call)
        and isinstance(node.func.func, ast.Name)
        and node.func.func.id == "_model_pipeline"
    ]
    assert pipeline_calls
    for call in pipeline_calls:
        direct_names = {keyword.arg for keyword in call.keywords}
        assert "prefix_allowed_tokens_fn" not in direct_names
        assert "repetition_penalty" not in direct_names
        assert "no_repeat_ngram_size" not in direct_names
        if any(name in direct_names for name in {"max_new_tokens", "do_sample"}):
            assert "generate_kwargs" not in direct_names


def test_failed_pre_lock_scaleway_run_has_an_approved_cleanup_path() -> None:
    backend_source = (
        Path(__file__).parents[1] / "src/mristudy/backend_cli.py"
    ).read_text(encoding="utf-8")
    cli_source = (Path(__file__).parents[1] / "src/mristudy/cli.py").read_text(
        encoding="utf-8"
    )
    for source in (backend_source, cli_source):
        assert "Stage.CLOUD_UPLOAD_APPROVED" in source
        assert "Stage.DICOM_STORE_VERIFIED" in source
        assert 'medgemma-results-lock.json' in source
        assert "failed_pre_lock_run" in source
    assert "state.invalidate_backend_approval()" in backend_source
