from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import threading
import time
from email.parser import BytesParser
from email.policy import default
from typing import Literal

import httpx
import jsonschema
import numpy as np
import pydicom
import torch
from contracts import (
    canonical_eos_guard,
    generation_budget_exhausted,
    require_finite_logits,
    restore_decoder_config,
    validate_window_payload,
)
from fastapi import FastAPI, HTTPException
from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
    build_transformers_prefix_allowed_tokens_fn,
)
from output_contract import (
    DECODER_CONFIG,
    DECODER_VERSION,
    LONG_OUTPUT_PROBE_MIN_TOKENS,
    LONG_OUTPUT_PROBE_RECORDS,
    MAX_NEW_TOKENS,
    OUTPUT_BUDGET_MARGIN_TOKENS,
    classify_output,
    decoder_acceptance,
    decoder_contract,
    decoder_output_bounds,
    decoder_schema,
    generation_policy,
    maximum_structural_output,
    output_contract,
    strict_json_loads,
    structural_correction_errors,
    structural_correction_prompt,
    validate_window_finding_semantics,
)
from PIL import Image
from pydantic import BaseModel
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from transformers import LogitsProcessor, pipeline

app = FastAPI(title="MRIStudy MedGemma DICOM-aware runtime")

ORTHANC_DICOMWEB_URL = os.environ.get(
    "ORTHANC_DICOMWEB_URL", "http://127.0.0.1:18042/dicom-web"
).rstrip("/")
MODEL_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")
MODEL_REVISION = os.environ.get("MEDGEMMA_MODEL_REVISION", "")
MODEL_PATH = os.environ.get("MEDGEMMA_MODEL_PATH", "")
RESULT_SCHEMA_PATH = os.environ.get(
    "RESULT_SCHEMA_PATH", "/app/knee-meniscus-finding-v2.schema.json"
)


def _mounted_schema_version() -> str:
    """Read the schema version the container was actually given.

    The runtime used to pin one literal, so a study of a different body region
    could not run at all. The mounted schema is authoritative and the approved
    plan must agree with it.
    """
    try:
        with open(RESULT_SCHEMA_PATH, encoding="utf-8") as handle:
            schema = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return ""
    declared = (schema.get("properties") or {}).get("schema_version", {}).get("const")
    return declared if isinstance(declared, str) else ""


PROMPT_SCHEMA_VERSION = _mounted_schema_version() or "knee-meniscus-finding-v2"
APPROVED_PLAN_PATH = os.environ.get("APPROVED_PLAN_PATH", "/run/mristudy/medgemma-run-plan.json")
APPROVED_PLAN_SHA256 = os.environ.get("APPROVED_PLAN_SHA256", "")
PROBE_AUTHORIZATION_PATH = os.environ.get(
    "PROBE_AUTHORIZATION_PATH", "/run/mristudy/medgemma-window-probes.json"
)
PROVISION_PLAN_SHA256 = os.environ.get("PROVISION_PLAN_SHA256", "")
RUNTIME_DIGEST = os.environ.get("RUNTIME_DIGEST", "")
MODEL_SNAPSHOT_MANIFEST_SHA256 = os.environ.get("MODEL_SNAPSHOT_MANIFEST_SHA256", "")
ORTHANC_USERNAME = os.environ.get("ORTHANC_USERNAME", "")
ORTHANC_PASSWORD = os.environ.get("ORTHANC_PASSWORD", "")
_pipeline = None
_pipeline_lock = threading.Lock()
_tokenizer_data = None
_decoder_metadata = None


def _decoder_identity() -> dict:
    """Resolve the decoder contract without loading the model.

    Decoder identity depends only on the canonical schema and the installed
    enforcer version, not on the pipeline. Deriving it lazily here lets the
    health endpoint and the capacity probes report the same decoder that
    medical generation will use, instead of reporting null until the first
    constrained-JSON call has happened.
    """
    global _decoder_metadata
    if _decoder_metadata is None:
        installed_version = importlib.metadata.version("lm-format-enforcer")
        if installed_version != DECODER_VERSION:
            raise RuntimeError(
                "Installed constrained-decoder version differs from the approved version."
            )
        with open(RESULT_SCHEMA_PATH, encoding="utf-8") as handle:
            _decoder_metadata = decoder_contract(json.load(handle))
    return _decoder_metadata


def _health_decoder_identity() -> dict | None:
    """Health must stay reachable for diagnosis even when the decoder is unusable."""
    try:
        return _decoder_identity()
    except Exception:
        return None


class DicomWebProbeRequest(BaseModel):
    study_uid: str
    series_uid: str


class AnalyzeWindowRequest(BaseModel):
    study_uid: str
    series_uid: str
    role: str
    window_id: str
    instance_uids: list[str]
    evidence_ids: list[str]
    prompt: str
    prompt_sha256: str
    output_schema: str
    max_new_tokens: int = MAX_NEW_TOKENS
    generation: dict[str, object]
    decoder: dict[str, object]
    output_contract: dict[str, object]


class SyntheticProbeRequest(BaseModel):
    image_count: Literal[1, 8, 16, 32]


class SyntheticDicomProbeRequest(BaseModel):
    image_count: Literal[32] = 32


class SyntheticJsonProbeRequest(BaseModel):
    image_count: Literal[1] = 1


class SyntheticLivenessProbeRequest(BaseModel):
    record_count: Literal[96] = LONG_OUTPUT_PROBE_RECORDS


@app.get("/health")
def health() -> dict:
    return {
        "status": (
            "ready-for-probes"
            if MODEL_REVISION and MODEL_PATH and APPROVED_PLAN_SHA256
            else "runtime-configuration-incomplete"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION or None,
        "model_snapshot_manifest_sha256": MODEL_SNAPSHOT_MANIFEST_SHA256 or None,
        "runtime_digest": RUNTIME_DIGEST or None,
        "approved_plan_sha256": APPROVED_PLAN_SHA256 or None,
        "provision_plan_sha256": PROVISION_PLAN_SHA256 or None,
        "model_source": "local_snapshot" if MODEL_PATH else "huggingface_hub",
        "dicomweb": ORTHANC_DICOMWEB_URL,
        "output_schema": PROMPT_SCHEMA_VERSION,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": _pipeline is not None,
        "decoder": _health_decoder_identity(),
        "output_contract": output_contract(),
    }


@app.post("/probe-dicomweb")
def probe_dicomweb(request: DicomWebProbeRequest) -> dict:
    url = (
        f"{ORTHANC_DICOMWEB_URL}/studies/{request.study_uid}"
        f"/series/{request.series_uid}/instances"
    )
    try:
        response = httpx.get(
            url,
            headers={"Accept": "application/dicom+json"},
            auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD) if ORTHANC_USERNAME else None,
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise HTTPException(502, f"DICOMweb probe failed: {type(error).__name__}") from error
    return {"instances": len(response.json()), "model_id": MODEL_ID}


def _dicom_payload(response: httpx.Response) -> bytes:
    content_type = response.headers.get("content-type", "")
    if not content_type.lower().startswith("multipart/"):
        return response.content
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        + response.content
    )
    for part in message.iter_parts():
        if part.get_content_type() == "application/dicom":
            return part.get_payload(decode=True)
    raise ValueError("Multipart WADO response contained no application/dicom part.")


def _render_dicom(data: bytes) -> Image.Image:
    ds = pydicom.dcmread(io.BytesIO(data))
    array = np.asarray(apply_modality_lut(ds.pixel_array, ds), dtype=np.float64)
    if "VOILUTSequence" in ds or ("WindowCenter" in ds and "WindowWidth" in ds):
        array = np.asarray(apply_voi_lut(array, ds), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        rendered = np.zeros(array.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [0.5, 99.5])
        if high <= low:
            rendered = np.zeros(array.shape, dtype=np.uint8)
        else:
            rendered = np.rint(np.clip((array - low) / (high - low), 0, 1) * 255).astype(
                np.uint8
            )
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        rendered = 255 - rendered
    return Image.fromarray(rendered, mode="L").convert("RGB")


def _load_approved_window(window_id: str) -> dict:
    try:
        with open(APPROVED_PLAN_PATH, encoding="utf-8") as handle:
            plan = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(503, "Approved MedGemma run plan is unavailable.") from error
    actual_digest = hashlib.sha256(
        json.dumps(
            {key: value for key, value in plan.items() if key != "plan_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    if not APPROVED_PLAN_SHA256 or plan.get("plan_sha256") != APPROVED_PLAN_SHA256:
        raise HTTPException(503, "Mounted MedGemma run plan is not the approved plan.")
    if actual_digest != APPROVED_PLAN_SHA256:
        raise HTTPException(503, "Mounted MedGemma run plan digest is invalid.")
    matches = [item for item in plan.get("windows", []) if item.get("window_id") == window_id]
    if len(matches) != 1:
        raise HTTPException(422, "Window ID is not uniquely present in the approved plan.")
    return matches[0]


def _validate_window_request(request: AnalyzeWindowRequest, approved: dict) -> None:
    if hashlib.sha256(request.prompt.encode()).hexdigest() != request.prompt_sha256:
        raise HTTPException(422, "Prompt SHA-256 does not match request prompt.")
    errors = validate_window_payload(request.model_dump(), approved)
    if errors:
        raise HTTPException(422, "; ".join(errors))


def _correction_prompt(base_prompt: str, attempt: int, errors: list[str], previous: str) -> str:
    del base_prompt
    return structural_correction_prompt(previous, attempt, errors)


def _require_probe_authorization(plan: dict) -> None:
    try:
        with open(PROBE_AUTHORIZATION_PATH, encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(503, "Synthetic probe authorization is unavailable.") from error
    required = {
        "medical_data_authorized": True,
        "run_plan_sha256": APPROVED_PLAN_SHA256,
        "provision_plan_sha256": PROVISION_PLAN_SHA256,
        "runtime_digest": RUNTIME_DIGEST,
        "model_repository": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot_manifest_sha256": MODEL_SNAPSHOT_MANIFEST_SHA256,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise HTTPException(503, f"Synthetic probe authorization mismatch: {key}")
    if plan.get("plan_sha256") != report.get("run_plan_sha256"):
        raise HTTPException(503, "Probe authorization does not match mounted plan.")


def _verify_window_membership(request: AnalyzeWindowRequest) -> None:
    url = (
        f"{ORTHANC_DICOMWEB_URL}/studies/{request.study_uid}"
        f"/series/{request.series_uid}/instances"
    )
    auth = (ORTHANC_USERNAME, ORTHANC_PASSWORD) if ORTHANC_USERNAME else None
    response = httpx.get(
        url,
        headers={"Accept": "application/dicom+json"},
        auth=auth,
        timeout=60,
    )
    response.raise_for_status()
    available = {
        str(values[0])
        for item in response.json()
        if (values := (item.get("00080018") or {}).get("Value"))
    }
    missing = set(request.instance_uids) - available
    if missing:
        raise HTTPException(422, "Window contains UIDs absent from the requested series.")


def _fetch_window_images(request: AnalyzeWindowRequest) -> tuple[list[Image.Image], float]:
    base = (
        f"{ORTHANC_DICOMWEB_URL}/studies/{request.study_uid}"
        f"/series/{request.series_uid}/instances"
    )
    auth = (ORTHANC_USERNAME, ORTHANC_PASSWORD) if ORTHANC_USERNAME else None
    started = time.monotonic()
    images = []
    for uid in request.instance_uids:
        response = httpx.get(
            f"{base}/{uid}",
            headers={
                "Accept": "multipart/related; type=application/dicom; transfer-syntax=*"
            },
            auth=auth,
            timeout=60,
        )
        response.raise_for_status()
        images.append(_render_dicom(_dicom_payload(response)))
    return images, time.monotonic() - started


def _model_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                if not torch.cuda.is_available():
                    raise HTTPException(503, "CUDA GPU is required.")
                if not MODEL_PATH:
                    raise HTTPException(503, "MEDGEMMA_MODEL_PATH is required.")
                if not os.path.isdir(MODEL_PATH):
                    raise HTTPException(503, "Configured MEDGEMMA_MODEL_PATH does not exist.")
                # Torch 2.13 routes selected eager operations through Triton and
                # compiles a helper extension on first use. The minimal runtime
                # intentionally has no compiler, so keep those operations on
                # their regular eager CUDA implementation.
                torch.backends.python_native.disable_operations("bmm")
                _pipeline = pipeline(
                    "image-text-to-text",
                    model=MODEL_PATH,
                    revision=None,
                    dtype=torch.bfloat16,
                    device="cuda",
                    model_kwargs={"local_files_only": True},
                )
    return _pipeline


class _FiniteLogitsGuard(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return require_finite_logits(input_ids, scores)


def _prefix_allowed_tokens_fn(
    schema: dict,
    evidence_ids: list[str] | None = None,
):
    global _tokenizer_data
    model_pipeline = _model_pipeline()
    if _tokenizer_data is None:
        _tokenizer_data = build_token_enforcer_tokenizer_data(model_pipeline.tokenizer)
    constrained_schema = decoder_schema(schema, evidence_ids)
    installed_version = importlib.metadata.version("lm-format-enforcer")
    if installed_version != DECODER_VERSION:
        raise RuntimeError(
            "Installed constrained-decoder version differs from the approved version."
        )
    base_callback = build_transformers_prefix_allowed_tokens_fn(
        _tokenizer_data,
        JsonSchemaParser(
            constrained_schema,
            config=CharacterLevelParserConfig(**DECODER_CONFIG),
        ),
    )
    restore_decoder_config(
        base_callback,
        CharacterLevelParserConfig(
            alphabet=_tokenizer_data.tokenizer_alphabet,
            **DECODER_CONFIG,
        ),
    )
    tokenizer_eos = model_pipeline.tokenizer.eos_token_id
    if not isinstance(tokenizer_eos, int):
        raise RuntimeError("Tokenizer must provide one canonical EOS token ID.")
    configured_eos = model_pipeline.model.generation_config.eos_token_id
    return canonical_eos_guard(base_callback, tokenizer_eos, configured_eos)


def _schema_prefix_allowed_tokens_fn(evidence_ids: list[str] | None = None):
    global _decoder_metadata
    with open(RESULT_SCHEMA_PATH, encoding="utf-8") as handle:
        schema = json.load(handle)
    _decoder_metadata = decoder_contract(schema)
    return _prefix_allowed_tokens_fn(schema, evidence_ids)


def _canonical_eos_token_id() -> int:
    eos_token_id = _model_pipeline().tokenizer.eos_token_id
    if not isinstance(eos_token_id, int):
        raise RuntimeError("Tokenizer must provide one canonical EOS token ID.")
    return eos_token_id


def _generated_text(output: object) -> str:
    if not isinstance(output, list) or not output:
        raise ValueError("Model pipeline returned no output.")
    generated = output[0].get("generated_text")
    if isinstance(generated, list):
        return str(generated[-1].get("content", ""))
    return str(generated or "")


def _generated_token_count(text: str) -> int:
    return len(_model_pipeline().tokenizer.encode(text, add_special_tokens=False))


def _decoder_bounds_report() -> dict:
    """Prove on this runtime that the largest decodable window output fits the budget.

    The budget used to be a number chosen by hand while the schema bounded no
    array and no free-text string, so no output size was ever guaranteed. This
    recomputes the maximum output the constrained decoder can emit and measures
    it with the tokenizer that will generate the medical windows. It runs before
    any medical access and needs no inference.
    """
    with open(RESULT_SCHEMA_PATH, encoding="utf-8") as handle:
        schema = json.load(handle)
    window_images = _largest_approved_window()
    evidence_ids = [f"image-{index + 1:06d}" for index in range(window_images)]
    maximum = maximum_structural_output(schema, evidence_ids)
    serialized = json.dumps(maximum, indent=2)
    tokens = len(_model_pipeline().tokenizer.encode(serialized, add_special_tokens=False))
    required = tokens + OUTPUT_BUDGET_MARGIN_TOKENS
    schema_errors = jsonschema.Draft202012Validator(schema).iter_errors(maximum)
    canonical_errors = [error.message for error in schema_errors]
    return {
        "passed": required <= MAX_NEW_TOKENS and not canonical_errors,
        "maximum_structural_tokens": tokens,
        "required_tokens_with_margin": required,
        "max_new_tokens": MAX_NEW_TOKENS,
        "margin_tokens": OUTPUT_BUDGET_MARGIN_TOKENS,
        "maximum_structural_characters": len(serialized),
        "maximum_structural_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "canonical_validation_errors": canonical_errors,
        "window_images": window_images,
        "output_bounds": decoder_output_bounds(),
    }


def _largest_approved_window() -> int:
    """Size the bound by the widest approved window, not by an assumed constant."""
    with open(APPROVED_PLAN_PATH, encoding="utf-8") as handle:
        plan = json.load(handle)
    sizes = [len(window.get("evidence_ids") or []) for window in plan.get("windows") or []]
    if not sizes:
        raise RuntimeError("Approved plan declares no window evidence.")
    return max(sizes)


@app.post("/decoder-bounds")
def decoder_bounds() -> dict:
    report = _decoder_bounds_report()
    return {
        **report,
        "decoder": _decoder_identity(),
        "output_contract": output_contract(),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot_manifest_sha256": MODEL_SNAPSHOT_MANIFEST_SHA256,
        "runtime_digest": RUNTIME_DIGEST,
        "approved_plan_sha256": APPROVED_PLAN_SHA256,
        "provision_plan_sha256": PROVISION_PLAN_SHA256,
    }


def _synthetic_expected_result() -> dict:
    return {
        "schema_version": "knee-meniscus-finding-v2",
        "study_purpose": "exploratory_non_diagnostic",
        "findings": [],
        "global_limitations": ["Synthetic non-medical model-capacity probe."],
        "treatment_boundary": {
            "treatment_recommendation_provided": False,
            "repairability_determined": False,
            "vascularity_directly_measured": False,
        },
    }


def _is_valid_empty_synthetic_result(result: object) -> bool:
    if not isinstance(result, dict):
        return False
    return (
        result.get("schema_version") == "knee-meniscus-finding-v2"
        and result.get("study_purpose") == "exploratory_non_diagnostic"
        and result.get("findings") == []
        and isinstance(result.get("global_limitations"), list)
        and result.get("treatment_boundary")
        == {
            "treatment_recommendation_provided": False,
            "repairability_determined": False,
            "vascularity_directly_measured": False,
        }
    )


def _run_synthetic_images(images: list[Image.Image], source_type: str) -> dict:
    started = time.monotonic()
    evidence_ids = [f"image-{index + 1:06d}" for index in range(len(images))]
    expected = _synthetic_expected_result()
    marker = f"MRI_STUDY_SYNTHETIC_OK_{len(images)}"
    content = [
        {
            "type": "text",
            "text": "Synthetic transport and capacity test. Do not analyze the images.",
        }
    ]
    for evidence_id, image in zip(evidence_ids, images, strict=True):
        content.append({"type": "text", "text": evidence_id})
        content.append({"type": "image", "image": image})
    content.append(
        {
            "type": "text",
            "text": f"Ignore all visual content and reply with exactly {marker}",
        }
    )
    encoding_seconds = time.monotonic() - started
    try:
        torch.cuda.reset_peak_memory_stats()
        generation_started = time.monotonic()
        output = _model_pipeline()(
            [{"role": "user", "content": content}],
            generate_kwargs={
                "max_new_tokens": 512,
                "do_sample": False,
            },
        )
        generation_seconds = time.monotonic() - generation_started
        model_response = _generated_text(output).strip()
        with open(RESULT_SCHEMA_PATH, encoding="utf-8") as handle:
            jsonschema.Draft202012Validator(json.load(handle)).validate(expected)
    except Exception as error:
        if isinstance(error, torch.OutOfMemoryError):
            return {
                "passed": False,
                "oom": True,
                "image_count": len(images),
                "source_type": source_type,
                "error_class": type(error).__name__,
            }
        raise HTTPException(500, f"Synthetic inference failed: {type(error).__name__}") from error
    return {
        "passed": model_response == marker,
        "oom": False,
        "image_count": len(images),
        "source_type": source_type,
        "evidence_ids": evidence_ids,
        "schema_valid": True,
        "validated_result": expected,
        "expected_marker": marker,
        "model_response": model_response,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot_manifest_sha256": MODEL_SNAPSHOT_MANIFEST_SHA256,
        "runtime_digest": RUNTIME_DIGEST,
        "decoder": _decoder_identity(),
        "output_contract": output_contract(),
        "approved_plan_sha256": APPROVED_PLAN_SHA256,
        "provision_plan_sha256": PROVISION_PLAN_SHA256,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "metrics": {
            "total_seconds": time.monotonic() - started,
            "encoding_seconds": encoding_seconds,
            "generation_seconds": generation_seconds,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "input_images": len(images),
            "tokens": None,
        },
    }


def _synthetic_dicom_bytes(index: int) -> bytes:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid(
        entropy_srcs=["mristudy-runtime-synthetic", str(index)]
    )
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid(entropy_srcs=["mristudy-runtime-synthetic-study"])
    dataset.SeriesInstanceUID = generate_uid(entropy_srcs=["mristudy-runtime-synthetic-series"])
    dataset.Modality = "MR"
    dataset.InstanceNumber = index + 1
    dataset.ImagePositionPatient = [0.0, 0.0, float(index)]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.Rows = 224
    dataset.Columns = 224
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    array = np.zeros((224, 224), dtype=np.uint16)
    inset = 16 + index % 24
    array[inset : 224 - inset, inset : 224 - inset] = 1200 + index
    dataset.PixelData = array.tobytes()
    output = io.BytesIO()
    dataset.save_as(output, enforce_file_format=True)
    return output.getvalue()


@app.post("/synthetic-probe")
def synthetic_probe(request: SyntheticProbeRequest) -> dict:
    images = []
    for index in range(request.image_count):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        inset = 16 + index % 24
        image[inset : 224 - inset, inset : 224 - inset, :] = 128 + index
        images.append(Image.fromarray(image))
    return _run_synthetic_images(images, "generated_rgb")


@app.post("/synthetic-dicom-probe")
def synthetic_dicom_probe(request: SyntheticDicomProbeRequest) -> dict:
    images = [_render_dicom(_synthetic_dicom_bytes(index)) for index in range(request.image_count)]
    return _run_synthetic_images(images, "contiguous_synthetic_dicom")


@app.post("/synthetic-json-probe")
def synthetic_json_probe(request: SyntheticJsonProbeRequest) -> dict:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[32:192, 32:192, :] = 160
    content = [
        {
            "type": "text",
            "text": (
                "Synthetic non-medical constrained-decoder test. Ignore the image. "
                "Return the empty exploratory result required by the supplied JSON schema."
            ),
        },
        {"type": "text", "text": "image-000001"},
        {"type": "image", "image": Image.fromarray(image)},
        {
            "type": "text",
            "text": "Return exactly one JSON object now, without prose or markdown.",
        },
    ]
    started = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats()
        model_pipeline = _model_pipeline()
        output = model_pipeline(
            [{"role": "user", "content": content}],
            generate_kwargs={
                "max_new_tokens": 512,
                "do_sample": False,
                "eos_token_id": _canonical_eos_token_id(),
                "prefix_allowed_tokens_fn": _schema_prefix_allowed_tokens_fn(
                    ["image-000001"]
                ),
                "logits_processor": [_FiniteLogitsGuard()],
            },
        )
        raw_text = _generated_text(output)
        if not raw_text.lstrip().startswith("{"):
            raise ValueError("Constrained decoder allowed a non-JSON prefix.")
        with open(RESULT_SCHEMA_PATH, encoding="utf-8") as handle:
            decision = classify_output(raw_text, json.load(handle), ["image-000001"])
    except Exception as error:
        if isinstance(error, torch.OutOfMemoryError):
            return {"passed": False, "oom": True, "error_class": type(error).__name__}
        raise HTTPException(
            500, f"Synthetic constrained JSON inference failed: {type(error).__name__}"
        ) from error
    return {
        "passed": _is_valid_empty_synthetic_result(decision.result),
        "oom": False,
        "schema_valid": decision.result is not None,
        "status": decision.status,
        "candidate_count": decision.candidate_count,
        "raw_text_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
        "raw_text_length": len(raw_text),
        "starts_with_json_object": raw_text.lstrip().startswith("{"),
        "selected_semantic_sha256": decision.selected_semantic_sha256,
        "decoder": _decoder_identity(),
        "output_contract": output_contract(),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot_manifest_sha256": MODEL_SNAPSHOT_MANIFEST_SHA256,
        "runtime_digest": RUNTIME_DIGEST,
        "approved_plan_sha256": APPROVED_PLAN_SHA256,
        "provision_plan_sha256": PROVISION_PLAN_SHA256,
        "metrics": {
            "total_seconds": time.monotonic() - started,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "input_images": request.image_count,
        },
    }


def _synthetic_liveness_schema(evidence_ids: list[str], record_count: int) -> dict:
    description = (
        "Synthetic non-medical repeated structural payload used only to prove that constrained "
        "generation remains live and completes beyond the former 4096-token ceiling."
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["probe", "evidence"],
        "properties": {
            "probe": {"enum": ["constrained_decoder_liveness_v1"]},
            "evidence": {
                "type": "array",
                "minItems": record_count,
                "maxItems": record_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["export_id", "role", "description"],
                    "properties": {
                        "export_id": {"enum": evidence_ids},
                        "role": {"enum": ["supporting"]},
                        "description": {"enum": [description]},
                    },
                },
            },
        },
    }


@app.post("/synthetic-liveness-probe")
def synthetic_liveness_probe(request: SyntheticLivenessProbeRequest) -> dict:
    evidence_ids = [f"image-{index:06d}" for index in range(1, 33)]
    expected_ids = [
        evidence_ids[index % len(evidence_ids)] for index in range(request.record_count)
    ]
    schema = _synthetic_liveness_schema(evidence_ids, request.record_count)
    content = [
        {
            "type": "text",
            "text": (
                "Synthetic non-medical 32-image constrained-generation liveness test. "
                "Return one JSON object with probe constrained_decoder_liveness_v1 and exactly "
                f"{request.record_count} evidence objects. Cycle through every supplied image ID "
                "in ascending order until the array is complete. Every role is supporting and "
                "every description must exactly match the schema enum. Ignore all visual content."
            ),
        }
    ]
    for index, evidence_id in enumerate(evidence_ids):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        inset = 16 + index % 24
        image[inset : 224 - inset, inset : 224 - inset, :] = 128 + index
        content.append({"type": "text", "text": evidence_id})
        content.append({"type": "image", "image": Image.fromarray(image)})
    content.append(
        {
            "type": "text",
            "text": "Return the complete long JSON object now and stop after its closing brace.",
        }
    )
    policy = generation_policy()
    started = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats()
        output = _model_pipeline()(
            [{"role": "user", "content": content}],
            generate_kwargs={
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": policy["do_sample"],
                "repetition_penalty": policy["repetition_penalty"],
                "no_repeat_ngram_size": policy["no_repeat_ngram_size"],
                "eos_token_id": _canonical_eos_token_id(),
                "prefix_allowed_tokens_fn": _prefix_allowed_tokens_fn(schema),
                "logits_processor": [_FiniteLogitsGuard()],
            },
        )
        raw_text = _generated_text(output)
        generated_tokens = _generated_token_count(raw_text)
        replay = decoder_acceptance(raw_text, schema)
        parsed = strict_json_loads(raw_text.strip()) if replay["accepted"] else None
        if parsed is not None:
            jsonschema.Draft202012Validator(schema).validate(parsed)
        returned_ids = [
            item.get("export_id")
            for item in (parsed or {}).get("evidence", [])
            if isinstance(item, dict)
        ]
    except Exception as error:
        if isinstance(error, torch.OutOfMemoryError):
            return {"passed": False, "oom": True, "error_class": type(error).__name__}
        raise HTTPException(
            500, f"Synthetic decoder liveness inference failed: {type(error).__name__}"
        ) from error
    return {
        # Acceptance covers what the constrained decoder guarantees: a complete
        # record set, every identifier drawn from the approved enumeration, all
        # of them exercised, and generation that stays live far past the former
        # ceiling. It deliberately does not require the exact ascending cycle.
        # That asks the model to count to 96, which no decoder can enforce; the
        # first real execution produced 96 valid records over 7_426 tokens and
        # was rejected only for ordering, which proves nothing about liveness.
        # The ordering result is still reported, because a collapse onto a single
        # identifier would show up as lost coverage.
        "passed": (
            replay["accepted"]
            and parsed is not None
            and len(returned_ids) == request.record_count
            and set(returned_ids) == set(expected_ids)
            and generated_tokens >= LONG_OUTPUT_PROBE_MIN_TOKENS
        ),
        "oom": False,
        "schema_valid": parsed is not None,
        "record_count": len(returned_ids),
        "evidence_cycle_exact": returned_ids == expected_ids,
        "distinct_evidence_ids": len(set(returned_ids)),
        "evidence_ids_covered": set(returned_ids) == set(expected_ids),
        "generated_tokens": generated_tokens,
        "minimum_required_tokens": LONG_OUTPUT_PROBE_MIN_TOKENS,
        "input_images": len(evidence_ids),
        "decoder_replay": replay,
        "raw_text_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
        "raw_text_length": len(raw_text),
        "generation": policy,
        "decoder": _decoder_identity(),
        "output_contract": output_contract(),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot_manifest_sha256": MODEL_SNAPSHOT_MANIFEST_SHA256,
        "runtime_digest": RUNTIME_DIGEST,
        "approved_plan_sha256": APPROVED_PLAN_SHA256,
        "provision_plan_sha256": PROVISION_PLAN_SHA256,
        "metrics": {
            "total_seconds": time.monotonic() - started,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "input_images": len(evidence_ids),
        },
    }


@app.post("/analyze-window")
def analyze_window(request: AnalyzeWindowRequest) -> dict:
    approved = _load_approved_window(request.window_id)
    with open(APPROVED_PLAN_PATH, encoding="utf-8") as handle:
        approved_plan = json.load(handle)
    _require_probe_authorization(approved_plan)
    _validate_window_request(request, approved)
    _verify_window_membership(request)
    total_started = time.monotonic()
    try:
        images, encoding_seconds = _fetch_window_images(request)
        torch.cuda.reset_peak_memory_stats()
        attempts = []
        prompt = request.prompt
        image_aware_retry = False
        total_generation_seconds = 0.0
        parsed = None
        schema_errors: list[str] = []
        with open(RESULT_SCHEMA_PATH, encoding="utf-8") as handle:
            result_schema = json.load(handle)
        maximum_corrections = int(approved["maximum_structural_corrections"])
        for attempt in range(1 + maximum_corrections):
            if attempt == 0 or image_aware_retry:
                prompt_parts = prompt.rsplit("Return JSON now.", maxsplit=1)
                content = [{"type": "text", "text": prompt_parts[0]}]
                for evidence_id, image in zip(request.evidence_ids, images, strict=True):
                    content.append({"type": "text", "text": evidence_id})
                    content.append({"type": "image", "image": image})
                if len(prompt_parts) == 2:
                    content.append({"type": "text", "text": "Return JSON now." + prompt_parts[1]})
            else:
                content = [{"type": "text", "text": prompt}]
            generation_started = time.monotonic()
            output = _model_pipeline()(
                [{"role": "user", "content": content}],
                generate_kwargs={
                    "max_new_tokens": request.max_new_tokens,
                    "do_sample": request.generation["do_sample"],
                    "repetition_penalty": request.generation["repetition_penalty"],
                    "no_repeat_ngram_size": request.generation["no_repeat_ngram_size"],
                    "eos_token_id": _canonical_eos_token_id(),
                    "prefix_allowed_tokens_fn": _schema_prefix_allowed_tokens_fn(
                        request.evidence_ids
                    ),
                    "logits_processor": [_FiniteLogitsGuard()],
                },
            )
            generation_seconds = time.monotonic() - generation_started
            total_generation_seconds += generation_seconds
            raw_text = _generated_text(output)
            generated_tokens = _generated_token_count(raw_text)
            decoder_replay = decoder_acceptance(
                raw_text,
                result_schema,
                request.evidence_ids,
            )
            decision = classify_output(raw_text, result_schema, request.evidence_ids)
            parsed = decision.result if decoder_replay["accepted"] else None
            schema_errors = (
                structural_correction_errors(decision)
                if decoder_replay["accepted"]
                else ["decoder_contract_violated"]
            )
            if parsed is not None:
                schema_errors.extend(validate_window_finding_semantics(parsed))
            exhausted = generation_budget_exhausted(
                generated_tokens,
                request.max_new_tokens,
            )
            if decision.status == "truncated_json" and exhausted:
                schema_errors = ["token_budget_exhausted"]
            selected_candidate = (
                decision.candidates[decision.selected_candidate_index]
                if decision.selected_candidate_index is not None
                else None
            )
            normalization_actions = (
                list(selected_candidate["actions"]) if selected_candidate else []
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_text": raw_text,
                    "result": parsed,
                    "normalization_actions": normalization_actions,
                    "validation_errors": schema_errors,
                    "generation_seconds": generation_seconds,
                    "generated_tokens": generated_tokens,
                    "token_budget_exhausted": exhausted,
                    "output_contract": {
                        "extraction_policy": decision.extraction_policy,
                        "normalization_policy": decision.normalization_policy,
                        "validation_policy": decision.validation_policy,
                        "status": decision.status,
                        "candidate_count": decision.candidate_count,
                        "selected_candidate_index": decision.selected_candidate_index,
                        "selected_candidate_sha256": decision.selected_candidate_sha256,
                        "selected_semantic_sha256": decision.selected_semantic_sha256,
                        "candidates": list(decision.candidates),
                        "decoder_replay": decoder_replay,
                    },
                }
            )
            if not decoder_replay["accepted"]:
                break
            if exhausted:
                break
            if not schema_errors:
                break
            if any(
                item["raw_text"] == raw_text for item in attempts[:-1]
            ):
                # Greedy decoding is deterministic: an attempt that reproduces an
                # earlier response byte for byte proves the correction prompt
                # changed nothing the model acted on. A real window burned three
                # identical 275-second generations this way before failing.
                schema_errors = ["deterministic_repeat"]
                break
            if attempt < maximum_corrections:
                if any(
                    "absent finding must be omitted" in error
                    or "repeated anatomical claim" in error
                    for error in schema_errors
                ):
                    image_aware_retry = True
                    prompt = (
                        request.prompt
                        + "\nThe previous image-aware response was rejected because it emitted "
                        + "absent findings or repeated the same anatomical claim. Reinspect the "
                        + "supplied images. Return only distinct focal suspected, probable, or "
                        + "genuinely indeterminate observations. If none is supported, return "
                        + '"findings": []. Previous validation errors: '
                        + "; ".join(schema_errors)[:1200]
                    )
                else:
                    image_aware_retry = False
                    prompt = _correction_prompt(
                        request.prompt, attempt + 1, schema_errors, raw_text
                    )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(500, f"Window inference failed: {type(error).__name__}") from error
    return {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot_manifest_sha256": MODEL_SNAPSHOT_MANIFEST_SHA256,
        "runtime_digest": RUNTIME_DIGEST,
        "run_plan_sha256": APPROVED_PLAN_SHA256,
        "provision_plan_sha256": PROVISION_PLAN_SHA256,
        "window_id": request.window_id,
        "role": request.role,
        "instance_uids": request.instance_uids,
        "evidence_ids": request.evidence_ids,
        "output_schema": request.output_schema,
        "prompt_sha256": request.prompt_sha256,
        "generation": request.generation,
        "decoder": _decoder_identity(),
        "output_contract": request.output_contract,
        "raw_text": attempts[-1]["raw_text"],
        "result": parsed,
        "schema_status": "valid" if not schema_errors else "invalid",
        "validation_errors": schema_errors,
        "attempts": attempts,
        "accepted_attempt": attempts[-1]["attempt"] if not schema_errors else None,
        "metrics": {
            "total_seconds": time.monotonic() - total_started,
            "encoding_seconds": encoding_seconds,
            "generation_seconds": total_generation_seconds,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "input_images": len(images),
            "oom": False,
        },
    }


if __name__ == "__main__":
    import uvicorn

    # The container listens on its private interface; the host publishes it only on loopback.
    uvicorn.run(app, host="0.0.0.0", port=8000)
