from __future__ import annotations

import json
import platform
import shutil
import uuid
from pathlib import Path
from typing import Annotated

import typer

from .approvals import require_approval, write_approval
from .backend_cli import register_backend_commands
from .cloud.dicomweb import DicomStoreRef, DicomWebClient
from .cloud.medgemma import VertexEndpointRef
from .cloud.plans import discover_uploadable_images, infrastructure_plan, stow_plan
from .cloud.workflow import destroy_uploaded_instances, run_medgemma_series, upload_and_verify
from .comparison import build_comparison
from .config import Settings
from .dicom.fixtures import create_synthetic_mri
from .dicom.inventory import inventory
from .errors import MRIStudyError
from .medgemma.evidence_locate import (
    collect_cited_evidence_ids,
    export_evidence_images,
    locate_many,
)
from .models.litellm import LiteLLMClient
from .models.runs import (
    provider_plan,
    run_generalist_supplement,
    run_generalists,
    supplemental_schema_plan,
)
from .models.selection import select_generalist_exports
from .paths import StudyPaths, study_paths
from .prompts import knee_meniscus_prompt_v2
from .provenance import append_event
from .report_bundle import (
    build_viewer_bundle,
    evidence_summary_from_multiplanar_provenance,
    load_locked_model_bundle_inputs,
    study_metadata,
)
from .reports import render_json_report
from .state import NEXT_ACTION, Stage, StudyState
from .utils import atomic_json, canonical_json, load_json, sha256_bytes
from .workflow import run_export, run_local_until_privacy

app = typer.Typer(no_args_is_help=True, help="Privacy-gated blind MRI model study pipeline.")
demo_app = typer.Typer(help="Synthetic, non-clinical demonstrations.")
models_app = typer.Typer(help="Probe and run external model providers.")
cloud_app = typer.Typer(help="Plan and execute Google Cloud operations.")
app.add_typer(demo_app, name="demo")
app.add_typer(models_app, name="models")
app.add_typer(cloud_app, name="cloud")


def _settings() -> Settings:
    return Settings.from_env()


def _paths_and_state(study_id: str) -> tuple[StudyPaths, StudyState]:
    paths = study_paths(_settings().data_root, study_id)
    state_path = paths.state / "status.json"
    if not state_path.exists():
        raise MRIStudyError(f"Unknown study: {study_id}")
    return paths, StudyState.load(state_path)


@app.callback()
def main() -> None:
    pass


@app.command()
def doctor() -> None:
    """Inspect the local runtime without modifying system packages."""
    settings = _settings()
    settings.data_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    free = shutil.disk_usage(settings.data_root).free
    checks = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "data_root": str(settings.data_root),
        "free_gib": round(free / 1024**3, 2),
        "docker": shutil.which("docker"),
        "gcloud": shutil.which("gcloud"),
        "litellm_configured": bool(settings.litellm_base_url),
        "google_project_configured": bool(settings.google_project),
    }
    typer.echo(json.dumps(checks, indent=2))
    if free < 5 * 1024**3:
        raise typer.Exit(1)


@app.command("estimate-space")
def estimate_space(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, file_okay=False, resolve_path=True)
    ],
) -> None:
    total = sum(path.stat().st_size for path in input_path.rglob("*") if path.is_file())
    free = shutil.disk_usage(_settings().data_root).free
    estimated = int(total * 2.8)
    typer.echo(
        json.dumps(
            {
                "input_gib": round(total / 1024**3, 3),
                "estimated_required_gib": round(estimated / 1024**3, 3),
                "free_gib": round(free / 1024**3, 3),
                "sufficient": free >= estimated + 2 * 1024**3,
            },
            indent=2,
        )
    )


@app.command("series")
def list_series(study_id: Annotated[str, typer.Option("--study")]) -> None:
    _, state = _paths_and_state(study_id)
    report = inventory(Path(state.input_path))
    grouped: dict[str, dict] = {}
    for item in report.instances:
        if not item.series_uid:
            continue
        entry = grouped.setdefault(
            item.series_uid,
            {
                "series_uid": item.series_uid,
                "description": item.series_description,
                "protocol": item.protocol_name,
                "instances": 0,
                "frames": 0,
            },
        )
        entry["instances"] += 1
        entry["frames"] += item.number_of_frames
    typer.echo(json.dumps(list(grouped.values()), indent=2))


@app.command("init-study")
def init_study(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, file_okay=False, resolve_path=True)
    ],
    profile: Annotated[str, typer.Option()] = "knee-meniscus-v1",
    study_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    settings = _settings()
    settings.data_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    identifier = study_id or f"study-{uuid.uuid4().hex[:12]}"
    paths = study_paths(settings.data_root, identifier)
    paths.create()
    state = StudyState.create(paths.state / "status.json", identifier, profile, str(input_path))
    atomic_json(
        paths.manifests / "study.json",
        {"study_id": identifier, "profile": profile, "input_path": str(input_path)},
    )
    append_event(paths.logs / "provenance.jsonl", "init-study", "complete", {"profile": profile})
    typer.echo(json.dumps({"study_id": identifier, "stage": state.stage.value}, indent=2))


@app.command()
def status(study_id: Annotated[str, typer.Option("--study")]) -> None:
    paths, state = _paths_and_state(study_id)
    typer.echo(
        json.dumps(
            {
                "study_id": study_id,
                "stage": state.stage.value,
                "next_action": NEXT_ACTION[state.stage],
                "root": str(paths.root),
            },
            indent=2,
        )
    )


@app.command("next")
def next_action(study_id: Annotated[str, typer.Option("--study")]) -> None:
    _, state = _paths_and_state(study_id)
    typer.echo(NEXT_ACTION[state.stage])


@app.command()
def run(
    study_id: Annotated[str, typer.Option("--study")],
    through: Annotated[str, typer.Option()] = "privacy-review",
) -> None:
    if through != "privacy-review":
        raise MRIStudyError("Only --through privacy-review is automated without further approvals.")
    paths, state = _paths_and_state(study_id)
    run_local_until_privacy(paths, state)
    append_event(paths.logs / "provenance.jsonl", "run-local", "complete")
    typer.echo(NEXT_ACTION[state.stage])


@app.command()
def approve(
    gate: Annotated[str, typer.Argument()],
    study_id: Annotated[str, typer.Option("--study")],
    note: Annotated[str, typer.Option()] = "",
) -> None:
    paths, state = _paths_and_state(study_id)
    if gate == "privacy-review":
        if state.stage != Stage.PRIVACY_REVIEW_REQUIRED:
            raise MRIStudyError("Privacy approval is not valid at the current stage.")
        artifact = load_json(paths.manifests / "privacy-review.json")
        write_approval(paths.state / "privacy-review.approval.json", gate, artifact, note)
        state.advance(Stage.PRIVACY_APPROVED)
    elif gate == "cloud-upload":
        if state.stage != Stage.CLOUD_PLAN_READY:
            raise MRIStudyError("Cloud approval is not valid at the current stage.")
        artifact = load_json(paths.state / "cloud-upload-plan.json")
        write_approval(paths.state / "cloud-upload.approval.json", gate, artifact, note)
        state.advance(Stage.CLOUD_UPLOAD_APPROVED)
    elif gate == "provider-egress":
        if state.stage != Stage.PROVIDER_PLAN_READY:
            raise MRIStudyError("Provider egress approval is not valid at the current stage.")
        artifact = load_json(paths.state / "provider-egress-plan.json")
        write_approval(paths.state / "provider-egress.approval.json", gate, artifact, note)
        state.advance(Stage.PROVIDER_EGRESS_APPROVED)
    elif gate == "cloud-destroy":
        artifact = load_json(paths.state / "cloud-destroy-plan.json")
        write_approval(paths.state / "cloud-destroy.approval.json", gate, artifact, note)
    elif gate == "backend-plan-migration":
        if state.stage != Stage.CLOUD_PLAN_READY:
            raise MRIStudyError("Backend migration approval is not valid at this stage.")
        artifact = load_json(paths.state / "backend-plan-migration.json")
        write_approval(
            paths.state / "backend-plan-migration.approval.json", gate, artifact, note
        )
    elif gate == "backend-egress":
        if state.stage != Stage.CLOUD_PLAN_READY or not state.backend:
            raise MRIStudyError("Backend egress approval is not valid at this stage.")
        artifact = load_json(paths.state / "scaleway-provision-plan.json")
        write_approval(paths.state / "backend-egress.approval.json", gate, artifact, note)
        state.advance(Stage.CLOUD_UPLOAD_APPROVED)
    elif gate == "backend-destroy":
        results_locked = (paths.manifests / "medgemma-results-lock.json").exists()
        failed_pre_lock_run = state.stage in {
            Stage.CLOUD_UPLOAD_APPROVED,
            Stage.DICOM_STORE_VERIFIED,
        } and not results_locked
        if state.stage != Stage.MEDGEMMA_COMPLETE and not failed_pre_lock_run:
            raise MRIStudyError("Backend destroy approval is not valid at this stage.")
        artifact = load_json(paths.state / "scaleway-destroy-plan.json")
        if bool(artifact.get("failed_pre_lock_run")) != failed_pre_lock_run:
            raise MRIStudyError("Backend destroy plan no longer matches the study state.")
        write_approval(paths.state / "backend-destroy.approval.json", gate, artifact, note)
    elif gate == "generalist-schema-supplement":
        if state.stage not in {
            Stage.GENERALIST_RESULTS_LOCKED,
            Stage.CLOUD_PLAN_READY,
            Stage.CLOUD_UPLOAD_APPROVED,
            Stage.DICOM_STORE_VERIFIED,
        }:
            raise MRIStudyError("Generalist schema supplement is not valid at this stage.")
        artifact = load_json(paths.state / "generalist-schema-supplement-plan.json")
        write_approval(
            paths.state / "generalist-schema-supplement.approval.json",
            gate,
            artifact,
            note,
        )
    else:
        raise MRIStudyError("Unknown approval gate.")
    append_event(paths.logs / "provenance.jsonl", "approve", "complete", {"gate": gate})
    typer.echo(state.stage.value)


@app.command()
def export(study_id: Annotated[str, typer.Option("--study")]) -> None:
    paths, state = _paths_and_state(study_id)
    if state.stage != Stage.PRIVACY_APPROVED:
        raise MRIStudyError("Exports require an approved privacy review.")
    payload = run_export(paths, state)
    append_event(paths.logs / "provenance.jsonl", "export", "complete", {"count": payload["count"]})
    typer.echo(json.dumps({"count": payload["count"], "stage": state.stage.value}, indent=2))


@demo_app.command("run")
def demo_run() -> None:
    settings = _settings()
    identifier = f"demo-{uuid.uuid4().hex[:8]}"
    paths = study_paths(settings.data_root, identifier)
    paths.create()
    fixture = create_synthetic_mri(paths.original)
    state = StudyState.create(
        paths.state / "status.json", identifier, "knee-meniscus-v1", str(paths.original)
    )
    run_local_until_privacy(paths, state)
    artifact = load_json(paths.manifests / "privacy-review.json")
    write_approval(
        paths.state / "privacy-review.approval.json",
        "privacy-review",
        artifact,
        "Synthetic fixture",
    )
    state.advance(Stage.PRIVACY_APPROVED)
    exports = run_export(paths, state)
    summary = {
        "study_id": identifier,
        "synthetic_files": fixture["files"],
        "exports": exports["count"],
        "stage": state.stage.value,
        "root": str(paths.root),
    }
    render_json_report("Synthetic MRI Demo", summary, paths.reports / "demo.html")
    typer.echo(json.dumps(summary, indent=2))


@models_app.command("probe")
def probe_models(
    study_id: Annotated[str | None, typer.Option("--study")] = None,
    execute: Annotated[
        bool, typer.Option(help="Make synthetic model calls that may incur cost.")
    ] = False,
    models: Annotated[list[str] | None, typer.Option("--model")] = None,
) -> None:
    settings = _settings()
    if not settings.litellm_base_url:
        raise MRIStudyError("LITELLM_BASE_URL is not configured.")
    client = LiteLLMClient(
        settings.litellm_base_url,
        settings.litellm_api_key,
        timeout=300.0,
    )
    result = {"liveness": client.liveness(), "models": client.models()}
    if execute:
        if not models:
            raise MRIStudyError("Provide at least one --model for an executing probe.")
        result["text_probes"] = {model: client.synthetic_text_probe(model) for model in models}
    if study_id:
        paths, state = _paths_and_state(study_id)
        if state.stage != Stage.EXPORTS_READY:
            raise MRIStudyError("Provider probes are expected immediately after canonical exports.")
        atomic_json(paths.manifests / "provider-capabilities.json", result)
        state.advance(Stage.PROVIDERS_PROBED)
    typer.echo(json.dumps(result, indent=2))


@models_app.command("plan")
def plan_models(
    study_id: Annotated[str, typer.Option("--study")],
    batch_size: Annotated[int, typer.Option(min=1, max=40)] = 12,
) -> None:
    settings = _settings()
    paths, state = _paths_and_state(study_id)
    if state.stage not in {Stage.PROVIDERS_PROBED, Stage.PROVIDER_PLAN_READY}:
        raise MRIStudyError("Probe providers before creating the image egress plan.")
    if not settings.gpt_model or not settings.grok_model:
        raise MRIStudyError("Configure MRI_STUDY_GPT_MODEL and MRI_STUDY_GROK_MODEL.")
    selection_path = paths.manifests / "generalist-selection.json"
    if not selection_path.exists():
        raise MRIStudyError("Run models select before creating the egress plan.")
    selection = load_json(selection_path)
    plan = provider_plan(
        load_json(paths.manifests / "exports.json"),
        {"gpt-sol": settings.gpt_model, "grok": settings.grok_model},
        batch_size,
        selection["selected_exports"],
    )
    atomic_json(paths.state / "provider-egress-plan.json", plan)
    if state.stage == Stage.PROVIDERS_PROBED:
        state.advance(Stage.PROVIDER_PLAN_READY)
    typer.echo(
        json.dumps(
            {"plan_sha256": plan["plan_sha256"], "batches": len(plan["batches"])},
            indent=2,
        )
    )


@models_app.command("select")
def select_models_input(study_id: Annotated[str, typer.Option("--study")]) -> None:
    paths, state = _paths_and_state(study_id)
    if state.stage not in {Stage.PROVIDERS_PROBED, Stage.PROVIDER_PLAN_READY}:
        raise MRIStudyError("Input selection follows successful provider discovery.")
    try:
        selection = select_generalist_exports(
            load_json(paths.manifests / "inventory.json"),
            load_json(paths.manifests / "exports.json"),
        )
    except ValueError as error:
        raise MRIStudyError(str(error)) from error
    atomic_json(paths.manifests / "generalist-selection.json", selection)
    typer.echo(json.dumps(selection, indent=2))


@models_app.command("run")
def run_models(
    study_id: Annotated[str, typer.Option("--study")],
    resume_run_id: Annotated[str | None, typer.Option("--resume-run-id")] = None,
) -> None:
    settings = _settings()
    paths, state = _paths_and_state(study_id)
    if state.stage != Stage.PROVIDER_EGRESS_APPROVED:
        raise MRIStudyError("The exact provider egress plan must be approved first.")
    if not settings.litellm_base_url:
        raise MRIStudyError("LITELLM_BASE_URL is not configured.")
    plan = load_json(paths.state / "provider-egress-plan.json")
    require_approval(paths.state / "provider-egress.approval.json", "provider-egress", plan)
    client = LiteLLMClient(
        settings.litellm_base_url,
        settings.litellm_api_key,
        timeout=300.0,
    )
    upstreams = client.verify_upstream_host(
        list(plan["models"].values()),
        plan.get("required_upstream_host", "api.routera.one"),
    )
    atomic_json(paths.manifests / "provider-upstream-verification.json", upstreams)
    run_id = resume_run_id or f"generalists-{uuid.uuid4().hex[:12]}"
    if not run_id.startswith("generalists-"):
        raise MRIStudyError("Invalid generalist run ID.")
    run_dir = paths.runs / run_id
    if resume_run_id and not run_dir.is_dir():
        raise MRIStudyError("Generalist run directory does not exist.")
    summary = run_generalists(
        client,
        paths.exports,
        load_json(paths.manifests / "exports.json"),
        plan,
        run_dir,
        resume=bool(resume_run_id),
    )
    atomic_json(paths.manifests / "generalist-results-lock.json", summary)
    state.advance(Stage.GENERALIST_RESULTS_LOCKED)
    append_event(paths.logs / "provenance.jsonl", "run-generalists", "complete", {"run_id": run_id})
    typer.echo(json.dumps({"run_id": run_id, "stage": state.stage.value}, indent=2))


@models_app.command("supplement-plan")
def supplement_models_plan(
    study_id: Annotated[str, typer.Option("--study")],
    batch_size: Annotated[int, typer.Option(min=1, max=40)] = 12,
) -> None:
    settings = _settings()
    paths, state = _paths_and_state(study_id)
    if state.stage not in {
        Stage.GENERALIST_RESULTS_LOCKED,
        Stage.CLOUD_PLAN_READY,
        Stage.CLOUD_UPLOAD_APPROVED,
        Stage.DICOM_STORE_VERIFIED,
    }:
        raise MRIStudyError("Supplement planning requires locked generalist results.")
    if not settings.gpt_model or not settings.grok_model:
        raise MRIStudyError("Configure MRI_STUDY_GPT_MODEL and MRI_STUDY_GROK_MODEL.")
    selection = load_json(paths.manifests / "generalist-selection.json")
    plan = supplemental_schema_plan(
        load_json(paths.manifests / "exports.json"),
        {"gpt-sol": settings.gpt_model, "grok": settings.grok_model},
        selection["selected_exports"],
        batch_size,
    )
    atomic_json(paths.state / "generalist-schema-supplement-plan.json", plan)
    typer.echo(
        json.dumps(
            {
                "plan_sha256": plan["plan_sha256"],
                "batches": len(plan["batches"]),
                "replaces_existing_lock": False,
                "output_schema": plan["output_schema"],
            },
            indent=2,
        )
    )


@models_app.command("supplement-run")
def supplement_models_run(
    study_id: Annotated[str, typer.Option("--study")],
    resume_run_id: Annotated[str | None, typer.Option("--resume-run-id")] = None,
) -> None:
    settings = _settings()
    paths, state = _paths_and_state(study_id)
    if state.stage not in {
        Stage.GENERALIST_RESULTS_LOCKED,
        Stage.CLOUD_PLAN_READY,
        Stage.CLOUD_UPLOAD_APPROVED,
        Stage.DICOM_STORE_VERIFIED,
    }:
        raise MRIStudyError("Supplement execution requires locked generalist results.")
    if not settings.litellm_base_url:
        raise MRIStudyError("LITELLM_BASE_URL is not configured.")
    plan = load_json(paths.state / "generalist-schema-supplement-plan.json")
    require_approval(
        paths.state / "generalist-schema-supplement.approval.json",
        "generalist-schema-supplement",
        plan,
    )
    client = LiteLLMClient(settings.litellm_base_url, settings.litellm_api_key, timeout=300)
    client.verify_upstream_host(
        list(plan["models"].values()),
        plan.get("required_upstream_host", "api.routera.one"),
    )
    run_id = resume_run_id or f"generalists-schema-v2-{uuid.uuid4().hex[:12]}"
    if not run_id.startswith("generalists-schema-v2-"):
        raise MRIStudyError("Invalid supplemental run ID.")
    run_dir = paths.runs / run_id
    if resume_run_id and not run_dir.is_dir():
        raise MRIStudyError("Supplemental run directory does not exist.")
    summary = run_generalist_supplement(
        client,
        paths.exports,
        load_json(paths.manifests / "exports.json"),
        plan,
        run_dir,
        resume=bool(resume_run_id),
    )
    atomic_json(
        paths.manifests / "generalist-schema-v2-results-lock.json",
        {
            "run_id": run_id,
            "schema_version": "knee-meniscus-finding-v2",
            "original_generalist_lock_preserved": True,
            "summary": summary,
        },
    )
    typer.echo(json.dumps({"run_id": run_id, "stage_unchanged": state.stage.value}, indent=2))


@cloud_app.command("plan")
def cloud_plan(study_id: Annotated[str, typer.Option("--study")]) -> None:
    settings = _settings()
    paths, state = _paths_and_state(study_id)
    if state.stage not in {Stage.GENERALIST_RESULTS_LOCKED, Stage.CLOUD_PLAN_READY}:
        raise MRIStudyError("Cloud planning follows locked generalist results.")
    if state.stage == Stage.CLOUD_PLAN_READY and (
        (paths.state / "cloud-upload.approval.json").exists()
        or (paths.state / "backend-plan-migration.approval.json").exists()
        or (paths.state / "backend-egress.approval.json").exists()
    ):
        raise MRIStudyError("Approved cloud/backend plans cannot be regenerated.")
    required = (
        settings.google_project,
        settings.google_dataset,
        settings.google_dicom_store,
    )
    if not all(required):
        raise MRIStudyError("Google project, dataset, and DICOM store must be configured.")
    store = DicomStoreRef(
        settings.google_project or "",
        settings.google_region,
        settings.google_dataset or "",
        settings.google_dicom_store or "",
    )
    uploadable, skipped = discover_uploadable_images(paths.anonymized)
    plan = stow_plan(uploadable, store)
    plan["skipped"] = skipped
    # `skipped` is part of the persisted operation plan. Recompute the digest
    # after adding it so later migration and approval checks bind the exact
    # object on disk rather than the pre-enrichment version.
    plan["plan_sha256"] = sha256_bytes(
        canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"})
    )
    atomic_json(paths.state / "cloud-upload-plan.json", plan)
    infrastructure = infrastructure_plan(
        project=settings.google_project or "",
        region=settings.google_region,
        study_id=study_id,
        upload_plan=plan,
        existing_budget_usd=5.0,
        approved_max_cost_usd=settings.max_cloud_cost_usd,
        global_gpu_quota=0.0,
    )
    atomic_json(paths.state / "cloud-infrastructure-plan.json", infrastructure)
    if state.stage == Stage.GENERALIST_RESULTS_LOCKED:
        state.advance(Stage.CLOUD_PLAN_READY)
    typer.echo(
        json.dumps(
            {
                "upload_plan_sha256": plan["plan_sha256"],
                "infrastructure_plan_sha256": infrastructure["plan_sha256"],
                "files": len(plan["files"]),
                "skipped": len(skipped),
                "preconditions": infrastructure["preconditions"],
            },
            indent=2,
        )
    )


def _cloud_store(settings: Settings) -> DicomStoreRef:
    if (
        not settings.google_project
        or not settings.google_dataset
        or not settings.google_dicom_store
    ):
        raise MRIStudyError("Google project, dataset, and DICOM store must be configured.")
    return DicomStoreRef(
        settings.google_project,
        settings.google_region,
        settings.google_dataset,
        settings.google_dicom_store,
    )


@cloud_app.command("apply")
def cloud_apply(study_id: Annotated[str, typer.Option("--study")]) -> None:
    settings = _settings()
    paths, state = _paths_and_state(study_id)
    if state.stage != Stage.CLOUD_UPLOAD_APPROVED:
        raise MRIStudyError("Cloud upload must be approved first.")
    plan = load_json(paths.state / "cloud-upload-plan.json")
    result = upload_and_verify(
        DicomWebClient(_cloud_store(settings)),
        plan,
        paths.state / "cloud-upload.approval.json",
        paths.manifests / "cloud-dicom-store.json",
    )
    state.advance(Stage.DICOM_STORE_VERIFIED)
    append_event(paths.logs / "provenance.jsonl", "cloud-upload", "complete", result)
    typer.echo(json.dumps(result, indent=2))


@cloud_app.command("run-medgemma")
def cloud_run_medgemma(
    study_id: Annotated[str, typer.Option("--study")],
    series_uid: Annotated[str, typer.Option("--series-uid")],
) -> None:
    settings = _settings()
    paths, state = _paths_and_state(study_id)
    if state.stage != Stage.DICOM_STORE_VERIFIED:
        raise MRIStudyError("Verified DICOM Store input is required.")
    if not settings.google_project or not settings.vertex_endpoint_id:
        raise MRIStudyError("Google project and Vertex endpoint ID must be configured.")
    cloud_manifest = load_json(paths.manifests / "cloud-dicom-store.json")
    study_uid = cloud_manifest["study_uids"][0]
    prompt = knee_meniscus_prompt_v2()
    result = run_medgemma_series(
        _cloud_store(settings),
        VertexEndpointRef(
            settings.google_project,
            settings.google_region,
            settings.vertex_endpoint_id,
        ),
        study_uid,
        series_uid,
        prompt,
    )
    run_dir = paths.runs / f"medgemma-{uuid.uuid4().hex[:12]}"
    run_dir.mkdir(mode=0o700)
    atomic_json(run_dir / "final.json", result)
    atomic_json(paths.manifests / "medgemma-results-lock.json", {"run": run_dir.name})
    state.advance(Stage.MEDGEMMA_COMPLETE)
    typer.echo(json.dumps({"run": run_dir.name, "stage": state.stage.value}, indent=2))


@cloud_app.command("destroy-plan")
def cloud_destroy_plan(study_id: Annotated[str, typer.Option("--study")]) -> None:
    paths, state = _paths_and_state(study_id)
    if state.stage != Stage.MEDGEMMA_COMPLETE:
        raise MRIStudyError("Destroy planning follows MedGemma completion.")
    upload_plan = load_json(paths.state / "cloud-upload-plan.json")
    plan = {
        "schema_version": 1,
        "operation": "delete-uploaded-dicom-instances",
        "upload_plan_sha256": upload_plan["plan_sha256"],
        "target": upload_plan["target"],
        "instances": [item["sop_instance_uid"] for item in upload_plan["files"]],
    }
    atomic_json(paths.state / "cloud-destroy-plan.json", plan)
    typer.echo(json.dumps({"instances": len(plan["instances"])}, indent=2))


@cloud_app.command("destroy")
def cloud_destroy(study_id: Annotated[str, typer.Option("--study")]) -> None:
    settings = _settings()
    paths, state = _paths_and_state(study_id)
    result = destroy_uploaded_instances(
        DicomWebClient(_cloud_store(settings)),
        load_json(paths.state / "cloud-upload-plan.json"),
        load_json(paths.state / "cloud-destroy-plan.json"),
        paths.state / "cloud-destroy.approval.json",
    )
    atomic_json(paths.manifests / "cloud-cleanup.json", result)
    state.advance(Stage.CLOUD_CLEANED)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def compare(study_id: Annotated[str, typer.Option("--study")]) -> None:
    paths, state = _paths_and_state(study_id)
    if state.stage != Stage.CLOUD_CLEANED:
        raise MRIStudyError("Comparison requires verified cloud cleanup.")
    result = build_comparison(paths.manifests, paths.runs)
    atomic_json(paths.reports / "comparison.json", result)
    render_json_report(
        "Blind Three-Arm Comparison",
        result,
        paths.reports / "10-three-arm-comparison.html",
    )
    state.advance(Stage.COMPARISON_COMPLETE)
    typer.echo(json.dumps({"arms": sorted(result["arms"]), "stage": state.stage.value}, indent=2))


@app.command()
def report(study_id: Annotated[str, typer.Option("--study")]) -> None:
    paths, state = _paths_and_state(study_id)
    if state.stage != Stage.COMPARISON_COMPLETE:
        raise MRIStudyError("Final reporting follows comparison.")
    comparison = load_json(paths.reports / "comparison.json")
    final = {
        "study_id": study_id,
        "profile": state.profile,
        "non_diagnostic": True,
        "clinical_reporting": {
            "vascular_zone_is_anatomic_estimate": True,
            "vascularity_directly_measured": False,
            "repairability_determined": False,
            "treatment_recommendation_provided": False,
            "specialist_review_factors_included": True,
        },
        "medgemma_coverage": comparison.get("arms", {})
        .get("medgemma-dicom", {})
        .get("coverage"),
        "comparison": comparison,
        "artifacts": {
            "inventory": "01-dicom-inventory.html",
            "anonymization": "02-anonymization-report.html",
            "quality_control": "03-quality-control.html",
            "comparison": "10-three-arm-comparison.html",
        },
        "manual_next_step": (
            "A qualified human may compare locked outputs with the radiology report post hoc."
        ),
    }
    atomic_json(paths.reports / "final-report.json", final)
    render_json_report(
        "Final Exploratory MRI Study Report",
        final,
        paths.reports / "final-report.html",
    )
    state.advance(Stage.REPORT_COMPLETE)
    typer.echo(json.dumps({"stage": state.stage.value, "report": "final-report.html"}, indent=2))


@app.command("evidence")
def evidence(
    study_id: Annotated[str, typer.Option("--study")],
    evidence_id: Annotated[str | None, typer.Option("--id")] = None,
    from_result: Annotated[
        Path | None, typer.Option("--from-result", help="Locate every evidence ID a result cites.")
    ] = None,
    export_images: Annotated[bool, typer.Option("--export-images")] = False,
    include_original: Annotated[
        bool,
        typer.Option(
            "--include-original",
            help="Resolve the original SOP Instance UID. This re-identifies the study.",
        ),
    ] = False,
) -> None:
    """Resolve evidence identifiers to checkable images.

    A finding is only useful if the underlying image can be put in front of a
    clinician, so this returns the anonymous UIDs, the de-identified file, its
    digest, and the windows the slice belongs to.
    """
    paths, _ = _paths_and_state(study_id)
    manifest_path = paths.manifests / "medgemma-evidence-map.json"
    if not manifest_path.exists():
        raise MRIStudyError("This study has no MedGemma evidence map.")
    evidence_manifest = load_json(manifest_path)
    plan_path = paths.state / "medgemma-run-plan.json"
    run_plan = load_json(plan_path) if plan_path.exists() else None

    if evidence_id and from_result:
        raise MRIStudyError("Use either --id or --from-result, not both.")
    if from_result:
        requested = collect_cited_evidence_ids(load_json(from_result))
        if not requested:
            raise MRIStudyError("That result cites no evidence identifiers.")
    elif evidence_id:
        requested = [evidence_id]
    else:
        raise MRIStudyError("Provide --id or --from-result.")

    references = locate_many(
        requested,
        evidence_manifest,
        paths.anonymized,
        run_plan=run_plan,
        uid_map_path=paths.state / "uid-map.json" if include_original else None,
        include_original_reference=include_original,
    )
    if export_images:
        references = export_evidence_images(references, paths.reports / "evidence")

    atomic_json(paths.reports / "evidence-references.json", {"references": references})
    typer.echo(
        json.dumps(
            {
                "requested": len(requested),
                "located": sum(1 for item in references if item.get("anonymised_path")),
                "images_exported": sum(1 for item in references if item.get("image_file")),
                "manifest": "reports/evidence-references.json",
            },
            indent=2,
        )
    )


@app.command("viewer-bundle")
def viewer_bundle(
    study_id: Annotated[str, typer.Option("--study")],
    skip_images: Annotated[bool, typer.Option("--skip-images")] = False,
) -> None:
    """Build the local bundle consumed by the private report viewer.

    Each model is emitted independently with its own findings and cited images,
    never merged into a shared verdict. Findings without usable image evidence
    are included and labelled rather than dropped.
    """
    paths, _ = _paths_and_state(study_id)
    manifest_path = paths.manifests / "medgemma-evidence-map.json"
    if not manifest_path.exists():
        raise MRIStudyError("This study has no MedGemma evidence map.")
    model_results, evidence_summaries, model_metadata = load_locked_model_bundle_inputs(
        paths.manifests, paths.runs, paths.state
    )
    if not model_results:
        raise MRIStudyError("No locked structured model result was found.")
    plan_path = paths.state / "medgemma-run-plan.json"
    run_plan = load_json(plan_path) if plan_path.exists() else None
    study = {}
    medgemma_lock = paths.manifests / "medgemma-results-lock.json"
    if medgemma_lock.exists():
        lock = load_json(medgemma_lock)
        study = study_metadata(lock, run_plan)
        run_id = lock.get("run")
        provenance_pointer = lock.get("multiplanar_provenance")
        if run_id and isinstance(provenance_pointer, str):
            provenance_path = paths.runs / run_id / provenance_pointer
            if provenance_path.exists():
                final_summary = evidence_summary_from_multiplanar_provenance(
                    load_json(provenance_path)
                )
                historical_summary = evidence_summaries.pop(
                    "medgemma-dicom-historical", {"findings": []}
                )
                final_summary["findings"].extend(
                    historical_summary.get("findings") or []
                )
                for key, value in (historical_summary.get("counts") or {}).items():
                    final_summary["counts"][key] = (
                        final_summary["counts"].get(key, 0) + value
                    )
                final_summary["total_findings"] += historical_summary.get(
                    "total_findings", 0
                )
                final_summary["traceable_findings"] += historical_summary.get(
                    "traceable_findings", 0
                )
                final_summary["evidence_coverage_percent"] = round(
                    100.0
                    * final_summary["traceable_findings"]
                    / max(1, final_summary["total_findings"]),
                    2,
                )
                final_summary["historical_findings"] = len(
                    historical_summary.get("findings") or []
                )
                evidence_summaries["medgemma-dicom"] = final_summary
    bundle = build_viewer_bundle(
        model_results,
        load_json(manifest_path),
        paths.anonymized,
        paths.reports / "viewer",
        run_plan=run_plan,
        export_images=not skip_images,
        evidence_summaries=evidence_summaries,
        model_metadata=model_metadata,
        study=study,
    )
    atomic_json(paths.reports / "viewer" / "report-bundle.json", bundle)
    typer.echo(
        json.dumps(
            {
                "models": [item["model"] for item in bundle["models"]],
                "findings": sum(len(item["findings"]) for item in bundle["models"]),
                "bundle": "reports/viewer/report-bundle.json",
                "images": "reports/viewer/evidence",
            },
            indent=2,
        )
    )


def run_cli() -> None:
    try:
        app()
    except MRIStudyError as error:
        typer.echo(f"Error: {error}", err=True)
        raise SystemExit(1) from error


register_backend_commands(app, _settings, _paths_and_state)


if __name__ == "__main__":
    run_cli()
