from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import HfApi

from .approvals import require_approval
from .backends.dicomweb import BasicAuth, DicomWebClient, DicomWebEndpoint
from .backends.medgemma_http import MedGemmaHttpClient
from .backends.plans import local_orthanc_plan, scaleway_operation_plan
from .backends.scaleway import audit as scaleway_audit
from .backends.scaleway import (
    build_resource_plan,
    create_resources,
    destroy_resources,
    final_provision_plan,
    inventory,
)
from .backends.workflow import import_local_orthanc
from .config import Settings
from .dicom.dcmtk import validate_tree
from .errors import MRIStudyError
from .medgemma.evidence import ROLE_ORDER, ordered_series_instances
from .medgemma.integrity import (
    require_complete_plan,
    require_valid_evidence_map,
)
from .medgemma.plan import build_medgemma_window_plan
from .medgemma.probes import run_required_probes
from .medgemma.replay import replay_acceptance, replay_historical_runs
from .medgemma.runner import run_window_plan
from .paths import StudyPaths
from .profiles import load_profile_by_name
from .prompts import medgemma_knee_meniscus_prompt_v2
from .provenance import append_event
from .state import Stage, StudyState
from .utils import atomic_json, load_json

backend_app = typer.Typer(help="Plan and operate the approved DICOM-aware backend.")

ContextLoader = Callable[[str], tuple[StudyPaths, StudyState]]


def _anonymous_series_catalog(root: Path) -> list[dict]:
    import pydicom

    series: dict[str, dict] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, OSError):
            continue
        uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        if not uid:
            continue
        item = series.setdefault(
            uid,
            {
                "series_uid": uid,
                "description": str(getattr(dataset, "SeriesDescription", "")),
                "protocol": str(getattr(dataset, "ProtocolName", "")),
                "instances": 0,
            },
        )
        item["instances"] += 1
    return sorted(series.values(), key=lambda item: item["description"])


def _orthanc_client(settings: Settings) -> DicomWebClient:
    if not settings.orthanc_username or not settings.orthanc_password:
        raise MRIStudyError("Orthanc credentials are not configured.")
    return DicomWebClient(
        DicomWebEndpoint(
            settings.orthanc_dicomweb_url,
            "local-orthanc",
            settings.orthanc_deployment_id,
        ),
        BasicAuth(settings.orthanc_username, settings.orthanc_password),
    )


def register_backend_commands(
    app: typer.Typer,
    settings_loader: Callable[[], Settings],
    context_loader: ContextLoader,
) -> None:
    app.add_typer(backend_app, name="backend")

    @backend_app.command("orthanc-plan")
    def orthanc_plan(study_id: Annotated[str, typer.Option("--study")]) -> None:
        settings = settings_loader()
        paths, state = context_loader(study_id)
        if state.stage != Stage.CLOUD_PLAN_READY:
            raise MRIStudyError("Local Orthanc planning requires CLOUD_PLAN_READY.")
        plan = local_orthanc_plan(
            paths.anonymized,
            settings.orthanc_dicomweb_url,
            settings.orthanc_deployment_id,
        )
        atomic_json(paths.state / "local-orthanc-import-plan.json", plan)
        typer.echo(
            json.dumps(
                {
                    "plan_sha256": plan["plan_sha256"],
                    "files": len(plan["files"]),
                    "skipped": len(plan["skipped"]),
                },
                indent=2,
            )
        )

    @backend_app.command("validate-dcmtk")
    def validate_dcmtk(study_id: Annotated[str, typer.Option("--study")]) -> None:
        paths, _ = context_loader(study_id)
        result = validate_tree(
            paths.anonymized,
            paths.manifests / "dcmtk-validation.json",
        )
        if not result["passed"]:
            raise MRIStudyError(
                f"DCMTK validation failed for {len(result['failures'])} objects."
            )
        typer.echo(json.dumps(result, indent=2))

    @backend_app.command("orthanc-probe")
    def orthanc_probe(study_id: Annotated[str, typer.Option("--study")]) -> None:
        paths, _ = context_loader(study_id)
        result = _orthanc_client(settings_loader()).probe()
        atomic_json(paths.manifests / "local-orthanc-capabilities.json", result)
        typer.echo(json.dumps(result, indent=2))

    @backend_app.command("orthanc-import")
    def orthanc_import(study_id: Annotated[str, typer.Option("--study")]) -> None:
        settings = settings_loader()
        paths, state = context_loader(study_id)
        if state.stage != Stage.CLOUD_PLAN_READY:
            raise MRIStudyError("Local Orthanc import requires CLOUD_PLAN_READY.")
        plan = load_json(paths.state / "local-orthanc-import-plan.json")
        result = import_local_orthanc(
            _orthanc_client(settings),
            plan,
            paths.anonymized,
            paths.manifests / "local-orthanc-store.json",
        )
        append_event(paths.logs / "provenance.jsonl", "orthanc-import", "complete", result)
        typer.echo(json.dumps(result, indent=2))

    @backend_app.command("verify-dicom-store")
    def verify_dicom_store(study_id: Annotated[str, typer.Option("--study")]) -> None:
        paths, state = context_loader(study_id)
        if state.stage != Stage.CLOUD_UPLOAD_APPROVED:
            raise MRIStudyError("Backend verification requires CLOUD_UPLOAD_APPROVED.")
        orthanc = load_json(paths.manifests / "local-orthanc-store.json")
        preflight = load_json(paths.manifests / "medgemma-window-probes.json")
        run_plan = load_json(paths.state / "medgemma-run-plan.json")
        provision = load_json(paths.state / "scaleway-provision-plan.json")
        evidence_manifest = load_json(paths.manifests / "medgemma-evidence-map.json")
        require_complete_plan(run_plan)
        require_valid_evidence_map(evidence_manifest, run_plan)
        if preflight.get("run_plan_sha256") != run_plan["plan_sha256"]:
            raise MRIStudyError("Synthetic probes are not bound to the current MedGemma plan.")
        if preflight.get("evidence_map_sha256") != run_plan["evidence_map_sha256"]:
            raise MRIStudyError("Synthetic probes are not bound to the current evidence map.")
        if preflight.get("provision_plan_sha256") != provision["plan_sha256"]:
            raise MRIStudyError("Synthetic probes are not bound to the current provision plan.")
        if provision.get("medgemma_run_plan_sha256") != run_plan["plan_sha256"]:
            raise MRIStudyError("Provision plan is not bound to the current MedGemma plan.")
        if provision.get("evidence_map_sha256") != run_plan["evidence_map_sha256"]:
            raise MRIStudyError("Provision plan is not bound to the current evidence map.")
        if preflight.get("runtime_digest") != run_plan["runtime_digest"]:
            raise MRIStudyError("Synthetic probes used a different runtime digest.")
        if preflight.get("model_repository") != run_plan["model_repository"]:
            raise MRIStudyError("Synthetic probes used a different model repository.")
        if preflight.get("model_revision") != run_plan["model_revision"]:
            raise MRIStudyError("Synthetic probes used a different model revision.")
        if preflight.get("model_snapshot_manifest_sha256") != provision["runtime"][
            "model_snapshot_manifest_sha256"
        ]:
            raise MRIStudyError("Synthetic probes used a different model snapshot.")
        expected = len(load_json(paths.state / "local-orthanc-import-plan.json")["files"])
        if orthanc.get("verified_instances") != expected:
            raise MRIStudyError("The approved Orthanc inventory is incomplete.")
        if not preflight.get("medical_data_authorized"):
            raise MRIStudyError("Synthetic model and DICOMweb preflight did not pass.")
        state.advance(Stage.DICOM_STORE_VERIFIED)
        append_event(
            paths.logs / "provenance.jsonl",
            "backend-dicom-store-verified",
            "complete",
            {
                "provider": "local-orthanc",
                "verified_instances": expected,
                "synthetic_probe": True,
            },
        )
        typer.echo(json.dumps({"stage": state.stage.value}, indent=2))

    @backend_app.command("medgemma-probes")
    def medgemma_probes(study_id: Annotated[str, typer.Option("--study")]) -> None:
        paths, state = context_loader(study_id)
        if state.stage != Stage.CLOUD_UPLOAD_APPROVED:
            raise MRIStudyError("MedGemma probes require CLOUD_UPLOAD_APPROVED.")
        plan = load_json(paths.state / "medgemma-run-plan.json")
        provision = load_json(paths.state / "scaleway-provision-plan.json")
        require_complete_plan(plan)
        if provision.get("medgemma_run_plan_sha256") != plan["plan_sha256"]:
            raise MRIStudyError("Provision plan is not bound to the MedGemma run plan.")
        if provision.get("evidence_map_sha256") != plan["evidence_map_sha256"]:
            raise MRIStudyError("Provision plan is not bound to the MedGemma evidence map.")
        client = MedGemmaHttpClient(timeout=3600)
        report = run_required_probes(
            client,
            paths.manifests / "medgemma-window-probes.json",
            expected_model=plan["model_repository"],
            expected_revision=plan["model_revision"],
            expected_runtime_digest=plan["runtime_digest"],
            expected_snapshot_manifest_sha256=provision["runtime"][
                "model_snapshot_manifest_sha256"
            ],
            expected_run_plan_sha256=plan["plan_sha256"],
            expected_provision_plan_sha256=provision["plan_sha256"],
        )
        primary = next(window for window in plan["windows"] if window["role"] == "primary-volume")
        dicomweb = client.probe_dicomweb(primary["study_uid"], primary["series_uid"])
        report["dicomweb_probe"] = dicomweb
        report["run_plan_sha256"] = plan["plan_sha256"]
        report["evidence_map_sha256"] = plan["evidence_map_sha256"]
        report["provision_plan_sha256"] = provision["plan_sha256"]
        report["runtime_digest"] = plan["runtime_digest"]
        report["model_repository"] = plan["model_repository"]
        report["model_revision"] = plan["model_revision"]
        report["model_snapshot_manifest_sha256"] = provision["runtime"][
            "model_snapshot_manifest_sha256"
        ]
        report["medical_data_authorized"] = bool(
            report["required_32_image_probe_passed"]
            and report["dicom_probe"].get("passed")
            and dicomweb.get("instances") == primary["source_instance_count"]
        )
        atomic_json(paths.manifests / "medgemma-window-probes.json", report)
        if not report["medical_data_authorized"]:
            raise MRIStudyError("Required MedGemma probes did not authorize medical data.")
        typer.echo(json.dumps(report, indent=2))

    @backend_app.command("medgemma-plan")
    def medgemma_plan(
        study_id: Annotated[str, typer.Option("--study")],
        runtime_digest: Annotated[str, typer.Option("--runtime-digest")],
    ) -> None:
        paths, state = context_loader(study_id)
        if state.stage != Stage.CLOUD_PLAN_READY:
            raise MRIStudyError("MedGemma planning requires CLOUD_PLAN_READY.")
        orthanc = load_json(paths.manifests / "local-orthanc-store.json")
        catalog = _anonymous_series_catalog(paths.anonymized)
        # Series selection comes from the study profile, so a different body
        # region needs a profile rather than a code change.
        profile = load_profile_by_name(state.profile)
        role_by_series = {}
        for requested_role in profile.series_roles:
            matches = [
                item for item in catalog if requested_role.matches(item["description"])
            ]
            if len(matches) > 1:
                raise MRIStudyError(
                    f"Series role {requested_role.role} matched {len(matches)} series; "
                    "the profile pattern must identify exactly one."
                )
            if not matches:
                if requested_role.required:
                    raise MRIStudyError(
                        f"No anonymous series matches required role {requested_role.role}."
                    )
                continue
            role_by_series[matches[0]["series_uid"]] = requested_role.role
        series = ordered_series_instances(paths.anonymized, role_by_series)
        selected_uids = {
            instance["sop_instance_uid"]
            for item in series.values()
            for instance in item["instances"]
        }
        orthanc_uids = set(orthanc.get("instance_uids") or [])
        if not orthanc_uids:
            raise MRIStudyError(
                "Orthanc manifest lacks exact instance UIDs; refresh local verification."
            )
        if not selected_uids.issubset(orthanc_uids):
            raise MRIStudyError("A selected MedGemma UID is absent from Orthanc.")
        excluded = [
            {
                "series_uid": item["series_uid"],
                "series_description": item["description"],
                "reason": (
                    "Excluded by profile: not selected for a role in "
                    f"{profile.name}."
                ),
            }
            for item in catalog
            if item["series_uid"] not in role_by_series
        ]
        replay_report = load_json(paths.manifests / "medgemma-historical-replay.json")
        acceptance = replay_acceptance(replay_report)
        if not acceptance["passed"] or not acceptance["all_json_candidates_recoverable"]:
            raise MRIStudyError("Historical MedGemma replay has not passed hardening acceptance.")
        plan, evidence_map = build_medgemma_window_plan(
            series_by_uid=series,
            orthanc_manifest=orthanc,
            model_repository="google/medgemma-1.5-4b-it",
            model_revision="91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
            runtime_digest=runtime_digest,
            historical_replay_sha256=replay_report["report_sha256"],
            excluded_series=excluded,
            profile=profile,
        )
        atomic_json(paths.state / "medgemma-run-plan.json", plan)
        atomic_json(paths.manifests / "medgemma-evidence-map.json", evidence_map)
        typer.echo(
            json.dumps(
                {
                    "plan_sha256": plan["plan_sha256"],
                    "total_source_slices": plan["total_source_slices"],
                    "unique_analyzed_slices": plan["unique_analyzed_slices"],
                    "coverage_percent": plan["coverage_percent"],
                    "total_windows": plan["total_windows"],
                    "duplicate_window_memberships": plan["duplicate_window_memberships"],
                    "windows_by_role": {
                        role: sum(1 for window in plan["windows"] if window["role"] == role)
                        for role in ROLE_ORDER
                    },
                },
                indent=2,
            )
        )

    @backend_app.command("medgemma-replay")
    def medgemma_replay(study_id: Annotated[str, typer.Option("--study")]) -> None:
        paths, _ = context_loader(study_id)
        output = paths.manifests / "medgemma-historical-replay.json"
        report = replay_historical_runs(paths.runs, output)
        acceptance = replay_acceptance(report)
        typer.echo(
            json.dumps(
                {
                    "source_artifacts": report["source_artifacts"],
                    "attempts": report["attempts"],
                    "classifications": report["classifications"],
                    "report_sha256": report["report_sha256"],
                    "acceptance": acceptance,
                },
                indent=2,
            )
        )

    @backend_app.command("run-medgemma")
    def run_medgemma(
        study_id: Annotated[str, typer.Option("--study")],
        resume_run_id: Annotated[str | None, typer.Option("--resume-run-id")] = None,
    ) -> None:
        paths, state = context_loader(study_id)
        if state.stage != Stage.DICOM_STORE_VERIFIED:
            raise MRIStudyError("MedGemma requires a verified DICOM-aware backend.")
        manifest_path = paths.state / "medgemma-run-plan.json"
        if not manifest_path.exists():
            raise MRIStudyError("Missing approved MedGemma run plan.")
        run_plan = load_json(manifest_path)
        require_complete_plan(run_plan)
        provision = load_json(paths.state / "scaleway-provision-plan.json")
        if provision.get("medgemma_run_plan_sha256") != run_plan["plan_sha256"]:
            raise MRIStudyError("Provision plan is not bound to the current MedGemma plan.")
        if provision.get("evidence_map_sha256") != run_plan["evidence_map_sha256"]:
            raise MRIStudyError("Provision plan is not bound to the current evidence map.")

        run_id = resume_run_id or f"medgemma-schema-v2-{uuid.uuid4().hex[:12]}"
        if not run_id.startswith("medgemma-schema-v2-"):
            raise MRIStudyError("Invalid MedGemma run ID.")
        run_dir = paths.runs / run_id
        if resume_run_id:
            if not run_dir.is_dir():
                raise MRIStudyError("MedGemma run directory does not exist.")

        client = MedGemmaHttpClient(timeout=3600)
        evidence_manifest = load_json(paths.manifests / "medgemma-evidence-map.json")
        evidence_map = require_valid_evidence_map(evidence_manifest, run_plan)
        preflight = load_json(paths.manifests / "medgemma-window-probes.json")
        if not preflight.get("medical_data_authorized"):
            raise MRIStudyError("Medical data is not authorized by synthetic probes.")
        if preflight.get("run_plan_sha256") != run_plan["plan_sha256"]:
            raise MRIStudyError("Synthetic probe report is stale for this MedGemma plan.")
        if preflight.get("provision_plan_sha256") != provision["plan_sha256"]:
            raise MRIStudyError("Synthetic probe report is stale for this provision plan.")
        if preflight.get("runtime_digest") != run_plan["runtime_digest"]:
            raise MRIStudyError("Synthetic probe report is stale for this runtime.")
        if preflight.get("model_snapshot_manifest_sha256") != provision["runtime"][
            "model_snapshot_manifest_sha256"
        ]:
            raise MRIStudyError("Synthetic probe report is stale for this model snapshot.")
        final = run_window_plan(
            client,
            run_plan,
            run_dir,
            medgemma_knee_meniscus_prompt_v2(),
            resume=bool(resume_run_id),
            evidence_map=evidence_map,
            provision_plan_sha256=provision["plan_sha256"],
            snapshot_manifest_sha256=provision["runtime"][
                "model_snapshot_manifest_sha256"
            ],
        )
        lock = {
            "run": run_id,
            "schema_version": "knee-meniscus-finding-v2",
            "provider": "scaleway",
            "runtime": load_json(paths.state / "scaleway-provision-plan.json")["runtime"],
            "run_plan_sha256": run_plan["plan_sha256"],
            "coverage": final["coverage"],
            "window_count": final["window_count"],
            "series_integrations": final["series_integrations"],
            "multiplanar_result": final["multiplanar_result"],
            "multiplanar_provenance": final["multiplanar_provenance"],
        }
        atomic_json(paths.manifests / "medgemma-results-lock.json", lock)
        state.advance(Stage.MEDGEMMA_COMPLETE)
        append_event(
            paths.logs / "provenance.jsonl",
            "medgemma-run",
            "complete",
            {"run_id": run_id, "windows": final["window_count"]},
        )
        typer.echo(json.dumps({"run_id": run_id, "coverage": final["coverage"]}, indent=2))

    @backend_app.command("migrate-plan")
    def migrate_plan(
        study_id: Annotated[str, typer.Option("--study")],
        source_public_ip: Annotated[str, typer.Option("--source-public-ip")],
    ) -> None:
        settings = settings_loader()
        paths, state = context_loader(study_id)
        if state.stage != Stage.CLOUD_PLAN_READY:
            raise MRIStudyError("Backend migration requires CLOUD_PLAN_READY.")
        if (paths.state / "cloud-upload.approval.json").exists():
            raise MRIStudyError("A Google upload approval exists; migration is blocked.")
        orthanc = load_json(paths.manifests / "local-orthanc-store.json")
        model_revision = None
        model_access = False
        try:
            model_revision = HfApi().model_info("google/medgemma-1.5-4b-it").sha
            model_access = bool(model_revision)
        except Exception:
            pass
        migration, operation = scaleway_operation_plan(
            study_id=study_id,
            legacy_upload_plan=load_json(paths.state / "cloud-upload-plan.json"),
            legacy_infrastructure_plan=load_json(
                paths.state / "cloud-infrastructure-plan.json"
            ),
            orthanc_manifest=orthanc,
            source_public_ip=source_public_ip,
            cost_limit_eur=settings.max_backend_cost_eur,
            model_revision=model_revision,
            model_access_accepted=model_access,
        )
        atomic_json(paths.state / "backend-plan-migration.json", migration)
        atomic_json(paths.state / "backend-operation-plan.json", operation)
        typer.echo(
            json.dumps(
                {
                    "migration_plan_sha256": migration["plan_sha256"],
                    "backend_operation_plan_sha256": operation["plan_sha256"],
                    "egress_authorized": False,
                },
                indent=2,
            )
        )

    @backend_app.command("apply-plan-migration")
    def apply_plan_migration(study_id: Annotated[str, typer.Option("--study")]) -> None:
        paths, state = context_loader(study_id)
        if state.stage != Stage.CLOUD_PLAN_READY:
            raise MRIStudyError("Backend migration requires CLOUD_PLAN_READY.")
        if (paths.state / "cloud-upload.approval.json").exists():
            raise MRIStudyError("A Google upload approval exists; migration is blocked.")
        migration = load_json(paths.state / "backend-plan-migration.json")
        operation = load_json(paths.state / "backend-operation-plan.json")
        require_approval(
            paths.state / "backend-plan-migration.approval.json",
            "backend-plan-migration",
            migration,
        )
        if migration["backend_operation_plan_sha256"] != operation["plan_sha256"]:
            raise MRIStudyError("Backend operation plan changed after migration planning.")
        legacy_upload = load_json(paths.state / "cloud-upload-plan.json")
        legacy_infrastructure = load_json(paths.state / "cloud-infrastructure-plan.json")
        if migration["legacy_upload_plan_sha256"] != legacy_upload["plan_sha256"]:
            raise MRIStudyError("Legacy upload plan changed after migration planning.")
        if (
            migration["legacy_infrastructure_plan_sha256"]
            != legacy_infrastructure["plan_sha256"]
        ):
            raise MRIStudyError("Legacy infrastructure plan changed after migration planning.")
        state.set_backend("scaleway", "backend-operation-plan.json", operation["plan_sha256"])
        append_event(
            paths.logs / "provenance.jsonl",
            "backend-plan-migration",
            "complete",
            {"provider": "scaleway", "plan_sha256": operation["plan_sha256"]},
        )
        typer.echo(json.dumps({"stage": state.stage.value, "backend": state.backend}, indent=2))

    @backend_app.command("scaleway-audit")
    def scaleway_provider_audit(
        study_id: Annotated[str | None, typer.Option("--study")] = None,
        zone: Annotated[str, typer.Option()] = "fr-par-2",
    ) -> None:
        result = scaleway_audit()
        resources = inventory(zone)
        result["zone_inventory"] = {
            "zone": zone,
            "servers": len(resources.get("servers") or []),
            "security_groups": len(resources.get("security_groups") or []),
            "server_types": len(resources.get("server_types") or []),
        }
        if study_id:
            paths, state = context_loader(study_id)
            if not state.backend or state.backend.get("provider") != "scaleway":
                raise MRIStudyError("Apply the Scaleway backend migration first.")
            operation = load_json(paths.state / "backend-operation-plan.json")
            resource_plan = build_resource_plan(operation, resources)
            atomic_json(paths.state / "scaleway-resource-plan.json", resource_plan)
            result["resource_plan"] = resource_plan
        typer.echo(json.dumps(result, indent=2))

    @backend_app.command("scaleway-plan")
    def scaleway_plan(
        study_id: Annotated[str, typer.Option("--study")],
        project_id: Annotated[str, typer.Option("--project-id")],
        zone: Annotated[str, typer.Option()] = "fr-par-2",
        marketplace_image_id: Annotated[str, typer.Option("--image-id")] = (
            "3307b9e4-3cfa-49b5-896e-ce914e4ef4aa"
        ),
        container_digest: Annotated[str, typer.Option("--container-digest")] = (
            "sha256:bd4c4821510fdcb4860d68ca11ff07e86ecbefd4a4d1d6f16a41dadf52fcefe6"
        ),
        container_tar_path: Annotated[Path, typer.Option("--container-tar")] = Path(
            "/home/openclaw/mri-study-data/services/scaleway/"
            "mristudy-medgemma-scaleway-v1.tar"
        ),
        container_tar_sha256: Annotated[str, typer.Option("--container-tar-sha256")] = (
            "64582a53005596d7c368124f229041ac68e5225fa2a3df8cc47c45b8a556deb6"
        ),
        model_snapshot_path: Annotated[Path | None, typer.Option("--model-snapshot")] = None,
        model_snapshot_manifest_sha256: Annotated[
            str | None, typer.Option("--model-snapshot-manifest-sha256")
        ] = None,
    ) -> None:
        paths, state = context_loader(study_id)
        if state.stage != Stage.CLOUD_PLAN_READY:
            raise MRIStudyError("Scaleway planning requires CLOUD_PLAN_READY.")
        if not state.backend or state.backend.get("provider") != "scaleway":
            raise MRIStudyError("Apply the Scaleway backend migration first.")
        operation = load_json(paths.state / "backend-operation-plan.json")
        medgemma_run_plan = load_json(paths.state / "medgemma-run-plan.json")
        resources = inventory(zone)
        cloud_init = Path(__file__).parents[2] / "deploy/scaleway/cloud-init.yaml"
        plan = final_provision_plan(
            operation_plan=operation,
            project_id=project_id,
            zone=zone,
            server_types=resources["server_types"],
            marketplace_image_id=marketplace_image_id,
            container_digest=container_digest,
            container_tar_path=container_tar_path,
            container_tar_sha256=container_tar_sha256,
            cloud_init_path=cloud_init,
            model_snapshot_path=model_snapshot_path,
            model_snapshot_manifest_sha256=model_snapshot_manifest_sha256,
            medgemma_run_plan_sha256=medgemma_run_plan["plan_sha256"],
            evidence_map_sha256=medgemma_run_plan["evidence_map_sha256"],
        )
        atomic_json(paths.state / "scaleway-provision-plan.json", plan)
        typer.echo(
            json.dumps(
                {
                    "plan_sha256": plan["plan_sha256"],
                    "gpu": plan["resources"]["server"]["type"],
                    "zone": plan["zone"],
                    "hourly_eur": plan["cost"]["l4_hourly"],
                    "six_hour_compute_ceiling_eur": plan["cost"][
                        "l4_compute_ceiling_excluding_storage_and_ip"
                    ],
                    "resource_creation_authorized": False,
                    "medical_egress_authorized": False,
                },
                indent=2,
            )
        )

    @backend_app.command("invalidate-provision-approval")
    def invalidate_provision_approval(
        study_id: Annotated[str, typer.Option("--study")],
        reason: Annotated[str, typer.Option()] = "Approved runtime artifact changed.",
    ) -> None:
        paths, state = context_loader(study_id)
        if not (paths.state / "backend-egress.approval.json").exists():
            raise MRIStudyError("No backend egress approval exists to invalidate.")
        resource_manifest = paths.state / "scaleway-resources.json"
        if resource_manifest.exists():
            resources = load_json(resource_manifest).get("resources") or {}
            if any(resources.get(key) for key in ("server_id", "security_group_id", "ssh_key_id")):
                raise MRIStudyError(
                    "Resource manifest still contains external resources; cleanup must be "
                    "verified first."
                )
        if (paths.manifests / "medgemma-results-lock.json").exists():
            raise MRIStudyError(
                "A MedGemma result is already locked; approval cannot be rolled back."
            )
        invalidated = paths.state / "backend-egress.approval.invalidated.json"
        approval = load_json(paths.state / "backend-egress.approval.json")
        atomic_json(invalidated, {"approval": approval, "reason": reason})
        (paths.state / "backend-egress.approval.json").unlink()
        state.invalidate_backend_approval()
        append_event(
            paths.logs / "provenance.jsonl",
            "backend-egress-approval-invalidated",
            "complete",
            {"reason": reason},
        )
        typer.echo(json.dumps({"stage": state.stage.value, "reason": reason}, indent=2))

    @backend_app.command("scaleway-provision")
    def scaleway_provision(study_id: Annotated[str, typer.Option("--study")]) -> None:
        paths, state = context_loader(study_id)
        if state.stage != Stage.CLOUD_UPLOAD_APPROVED:
            raise MRIStudyError("The final Scaleway provision plan must be approved first.")
        plan = load_json(paths.state / "scaleway-provision-plan.json")
        run_plan = load_json(paths.state / "medgemma-run-plan.json")
        evidence_manifest = load_json(paths.manifests / "medgemma-evidence-map.json")
        require_complete_plan(run_plan)
        require_valid_evidence_map(evidence_manifest, run_plan)
        if plan.get("medgemma_run_plan_sha256") != run_plan["plan_sha256"]:
            raise MRIStudyError("Scaleway plan is not bound to the current MedGemma plan.")
        if plan.get("evidence_map_sha256") != run_plan["evidence_map_sha256"]:
            raise MRIStudyError("Scaleway plan is not bound to the current evidence map.")
        require_approval(
            paths.state / "backend-egress.approval.json",
            "backend-egress",
            plan,
        )
        manifest = create_resources(
            plan,
            paths.state / "scaleway-resources.json",
            paths.state / "scaleway-ssh",
        )
        append_event(
            paths.logs / "provenance.jsonl",
            "scaleway-provision",
            "complete",
            {"plan_sha256": plan["plan_sha256"], "resources": manifest["resources"]},
        )
        typer.echo(json.dumps({"resources": manifest["resources"]}, indent=2))

    @backend_app.command("scaleway-destroy-plan")
    def scaleway_destroy_plan(
        study_id: Annotated[str, typer.Option("--study")],
        reason: Annotated[str, typer.Option()] = "Temporary backend run completed.",
    ) -> None:
        paths, state = context_loader(study_id)
        results_locked = (paths.manifests / "medgemma-results-lock.json").exists()
        failed_pre_lock_run = state.stage in {
            Stage.CLOUD_UPLOAD_APPROVED,
            Stage.DICOM_STORE_VERIFIED,
        } and not results_locked
        if state.stage != Stage.MEDGEMMA_COMPLETE and not failed_pre_lock_run:
            raise MRIStudyError(
                "Destroy planning requires MedGemma completion or a failed pre-lock backend run."
            )
        provision = load_json(paths.state / "scaleway-provision-plan.json")
        resources = load_json(paths.state / "scaleway-resources.json")
        plan = {
            "schema_version": 1,
            "operation": "scaleway-destroy-and-verify",
            "provision_plan_sha256": provision["plan_sha256"],
            "resource_manifest": resources["resources"],
            "zone": resources["zone"],
            "reason": reason,
            "failed_pre_lock_run": failed_pre_lock_run,
            "verify_absence": True,
        }
        from .utils import canonical_json, sha256_bytes

        plan["plan_sha256"] = sha256_bytes(canonical_json(plan))
        atomic_json(paths.state / "scaleway-destroy-plan.json", plan)
        typer.echo(json.dumps({"plan_sha256": plan["plan_sha256"]}, indent=2))

    @backend_app.command("scaleway-destroy")
    def scaleway_destroy(study_id: Annotated[str, typer.Option("--study")]) -> None:
        paths, state = context_loader(study_id)
        results_locked = (paths.manifests / "medgemma-results-lock.json").exists()
        failed_pre_lock_run = state.stage in {
            Stage.CLOUD_UPLOAD_APPROVED,
            Stage.DICOM_STORE_VERIFIED,
        } and not results_locked
        if state.stage != Stage.MEDGEMMA_COMPLETE and not failed_pre_lock_run:
            raise MRIStudyError(
                "Scaleway destroy requires MedGemma completion or a failed pre-lock backend run."
            )
        destroy_plan = load_json(paths.state / "scaleway-destroy-plan.json")
        if bool(destroy_plan.get("failed_pre_lock_run")) != failed_pre_lock_run:
            raise MRIStudyError("Scaleway destroy plan no longer matches the study state.")
        require_approval(
            paths.state / "backend-destroy.approval.json",
            "backend-destroy",
            destroy_plan,
        )
        result = destroy_resources(
            load_json(paths.state / "scaleway-provision-plan.json"),
            load_json(paths.state / "scaleway-resources.json"),
        )
        atomic_json(paths.manifests / "scaleway-cleanup.json", result)
        if failed_pre_lock_run:
            resources = load_json(paths.state / "scaleway-resources.json")
            resources["destroyed_resources"] = resources.get("resources") or {}
            resources["resources"] = {}
            atomic_json(paths.state / "scaleway-resources.json", resources)
            approval = paths.state / "backend-egress.approval.json"
            if approval.exists():
                atomic_json(
                    paths.state / "backend-egress.approval.invalidated.json",
                    {
                        "approval": load_json(approval),
                        "reason": destroy_plan["reason"],
                    },
                )
                approval.unlink()
            state.invalidate_backend_approval()
        else:
            state.advance(Stage.CLOUD_CLEANED)
        typer.echo(json.dumps(result, indent=2))
