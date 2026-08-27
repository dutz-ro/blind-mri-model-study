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
from contracts import validate_window_payload
from fastapi import FastAPI, HTTPException
from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
    build_transformers_prefix_allowed_tokens_fn,
)
from output_contract import (
    DECODER_CONFIG,
    DECODER_VERSION,
    classify_output,
    decoder_contract,
    decoder_schema,
)
from PIL import Image
from pydantic import BaseModel
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from transformers import pipeline

app = FastAPI(title="MRIStudy MedGemma DICOM-aware runtime")

ORTHANC_DICOMWEB_URL = os.environ.get(
    "ORTHANC_DICOMWEB_URL", "http://127.0.0.1:18042/dicom-web"
).rstrip("/")
MODEL_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")
MODEL_REVISION = os.environ.get("MEDGEMMA_MODEL_REVISION", "")
MODEL_PATH = os.environ.get("MEDGEMMA_MODEL_PATH", "")
PROMPT_SCHEMA_VERSION = "knee-meniscus-finding-v2"
RESULT_SCHEMA_PATH = os.environ.get(
    "RESULT_SCHEMA_PATH", "/app/knee-meniscus-finding-v2.schema.json"
)
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
    output_schema: Literal["knee-meniscus-finding-v2"]
    max_new_tokens: int = 4096
    generation: dict[str, object]
    decoder: dict[str, object]


class SyntheticProbeRequest(BaseModel):
    image_count: Literal[1, 8, 16, 32]


class SyntheticDicomProbeRequest(BaseModel):
    image_count: Literal[32] = 32


class SyntheticJsonProbeRequest(BaseModel):
    image_count: Literal[1] = 1


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
    return (
        "Structural correction only. Do not inspect or reinterpret the images. "
        "Rewrite the previous "
        "response as exactly one JSON object matching the required schema. Preserve every clinical "
        "claim, uncertainty, evidence reference, and counter-evidence. "
        "Do not add or remove findings. "
        "Correct only missing fields, object structure, and allowed enum aliases. Attempt "
        + str(attempt)
        + ". Validation errors: "
        + "; ".join(errors)[:1500]
        + "\nPrevious response to preserve semantically:\n"
        + previous[:12000]
        + "\n\n"
        + base_prompt.rsplit("Return JSON now.", maxsplit=1)[-1]
    )


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
                _pipeline = pipeline(
                    "image-text-to-text",
                    model=MODEL_PATH,
                    revision=None,
                    dtype=torch.bfloat16,
                    device="cuda",
                    model_kwargs={"local_files_only": True},
                )
    return _pipeline


def _schema_prefix_allowed_tokens_fn():
    global _decoder_metadata, _tokenizer_data
    model_pipeline = _model_pipeline()
    if _tokenizer_data is None:
        _tokenizer_data = build_token_enforcer_tokenizer_data(model_pipeline.tokenizer)
    with open(RESULT_SCHEMA_PATH, encoding="utf-8") as handle:
        schema = json.load(handle)
    constrained_schema = decoder_schema(schema)
    installed_version = importlib.metadata.version("lm-format-enforcer")
    if installed_version != DECODER_VERSION:
        raise RuntimeError(
            "Installed constrained-decoder version differs from the approved version."
        )
    _decoder_metadata = decoder_contract(schema)
    return build_transformers_prefix_allowed_tokens_fn(
        _tokenizer_data,
        JsonSchemaParser(
            constrained_schema,
            config=CharacterLevelParserConfig(**DECODER_CONFIG),
        ),
    )


def _generated_text(output: object) -> str:
    if not isinstance(output, list) or not output:
        raise ValueError("Model pipeline returned no output.")
    generated = output[0].get("generated_text")
    if isinstance(generated, list):
        return str(generated[-1].get("content", ""))
    return str(generated or "")


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
                "prefix_allowed_tokens_fn": _schema_prefix_allowed_tokens_fn(),
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
        total_generation_seconds = 0.0
        parsed = None
        schema_errors: list[str] = []
        with open(RESULT_SCHEMA_PATH, encoding="utf-8") as handle:
            result_schema = json.load(handle)
        for attempt in range(3):
            if attempt == 0:
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
                    "prefix_allowed_tokens_fn": _schema_prefix_allowed_tokens_fn(),
                },
            )
            generation_seconds = time.monotonic() - generation_started
            total_generation_seconds += generation_seconds
            raw_text = _generated_text(output)
            decision = classify_output(raw_text, result_schema, request.evidence_ids)
            parsed = decision.result
            schema_errors = [] if parsed is not None else [decision.status]
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
                    "output_contract": {
                        "extraction_policy": decision.extraction_policy,
                        "normalization_policy": decision.normalization_policy,
                        "status": decision.status,
                        "candidate_count": decision.candidate_count,
                        "selected_candidate_index": decision.selected_candidate_index,
                        "selected_candidate_sha256": decision.selected_candidate_sha256,
                        "selected_semantic_sha256": decision.selected_semantic_sha256,
                        "candidates": list(decision.candidates),
                    },
                }
            )
            if not schema_errors:
                break
            if attempt < 2:
                prompt = _correction_prompt(request.prompt, attempt + 1, schema_errors, raw_text)
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
