from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from ..errors import MRIStudyError
from ..prompts import medgemma_knee_meniscus_prompt_v2
from ..utils import atomic_json, canonical_json, sha256_bytes, sha256_file, utc_now


def credentials_present() -> bool:
    return bool(
        os.environ.get("SCW_ACCESS_KEY")
        and os.environ.get("SCW_SECRET_KEY")
        and os.environ.get("SCW_DEFAULT_PROJECT_ID")
    ) or os.path.exists(os.path.expanduser("~/.config/scw/config.yaml"))


def audit() -> dict:
    if not shutil.which("scw"):
        raise MRIStudyError("Scaleway CLI is not installed.")
    if not credentials_present():
        raise MRIStudyError("Scaleway credentials are not configured. Run scw init.")
    command = ["scw", "info", "-o", "json"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    return {
        "cli": shutil.which("scw"),
        "profile": payload.get("profile"),
        "default_project_id": payload.get("config", {}).get("default_project_id"),
        "default_region": payload.get("config", {}).get("default_region"),
        "default_zone": payload.get("config", {}).get("default_zone"),
    }


def _run_json(*args: str) -> object:
    try:
        result = subprocess.run(
            ["scw", *args, "-o", "json"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise MRIStudyError(f"Scaleway CLI failed: {args[0]} {args[1]}") from error
    return json.loads(result.stdout)


def inventory(zone: str) -> dict:
    return {
        "zone": zone,
        "servers": _run_json("instance", "server", "list", f"zone={zone}"),
        "security_groups": _run_json("instance", "security-group", "list", f"zone={zone}"),
        "server_types": _run_json("instance", "server-type", "list", f"zone={zone}"),
        "volumes": _run_json("instance", "volume", "list", f"zone={zone}"),
        "block_volumes": _run_json("block", "volume", "list", f"zone={zone}"),
        "ips": _run_json("instance", "ip", "list", f"zone={zone}"),
        "ssh_keys": _run_json("iam", "ssh-key", "list"),
    }


def _identifiers(items: object) -> set[str]:
    if not isinstance(items, list):
        return set()
    found: set[str] = set()
    for item in items:
        if isinstance(item, dict) and item.get("id"):
            found.add(str(item["id"]))
    return found


def verify_absence(zone: str, expected_absent: dict[str, object]) -> dict:
    """Prove that every named temporary resource is gone from the provider inventory.

    Absence is established from the provider's own listings, never from local state.
    Volumes are checked in both the legacy instance API and the Block (SBS) API,
    because a GPU instance root volume is an SBS volume that survives server
    deletion and continues to be billed.
    """
    current = inventory(zone)
    present = {
        "servers": _identifiers(current.get("servers")),
        "security_groups": _identifiers(current.get("security_groups")),
        "volumes": _identifiers(current.get("volumes"))
        | _identifiers(current.get("block_volumes")),
        "ips": _identifiers(current.get("ips")),
        "ssh_keys": _identifiers(current.get("ssh_keys")),
    }
    remaining: dict[str, list[str]] = {}
    for category, expected in expected_absent.items():
        if category not in present:
            raise MRIStudyError(f"Absence cannot be verified for unknown category: {category}")
        wanted = [str(value) for value in (expected or []) if value]
        still_present = sorted(value for value in wanted if value in present[category])
        if still_present:
            remaining[category] = still_present
    return {
        "zone": zone,
        "checked_categories": sorted(expected_absent),
        "remaining": remaining,
        "verified_absent": not remaining,
        "verified_at": utc_now(),
    }


def build_resource_plan(operation_plan: dict, inventory_data: dict) -> dict:
    available = inventory_data.get("server_types") or []
    names = {
        str(item.get("name") or item.get("commercial_type") or "")
        for item in available
        if isinstance(item, dict)
    }
    selected = next(
        (gpu for gpu in operation_plan["gpu_preference"] if gpu in names),
        None,
    )
    return {
        "provider": "scaleway",
        "operation_plan_sha256": operation_plan["plan_sha256"],
        "selected_gpu_type": selected,
        "available_preferred_gpu_types": [
            gpu for gpu in operation_plan["gpu_preference"] if gpu in names
        ],
        "zone": inventory_data["zone"],
        "existing_servers": len(inventory_data.get("servers") or []),
        "existing_security_groups": len(inventory_data.get("security_groups") or []),
        "blocked": selected is None,
        "block_reason": None if selected else "No approved GPU type is currently listed.",
    }


def _money_eur(value: dict | None) -> float | None:
    if not value or value.get("currency_code") != "EUR":
        return None
    return float(value.get("units", 0)) + float(value.get("nanos", 0)) / 1_000_000_000


def final_provision_plan(
    *,
    operation_plan: dict,
    project_id: str,
    zone: str,
    server_types: list[dict],
    marketplace_image_id: str,
    container_digest: str,
    container_tar_path: Path,
    container_tar_sha256: str,
    cloud_init_path: Path,
    model_snapshot_path: Path | None = None,
    model_snapshot_manifest_sha256: str | None = None,
    medgemma_run_plan_sha256: str | None = None,
    evidence_map_sha256: str | None = None,
) -> dict:
    if not medgemma_run_plan_sha256 or not evidence_map_sha256:
        raise MRIStudyError("Provision plan must bind MedGemma run and evidence-map digests.")
    if model_snapshot_path is None or not model_snapshot_manifest_sha256:
        raise MRIStudyError("Provision plan requires a pinned offline MedGemma snapshot.")
    if not container_tar_path.is_file():
        raise MRIStudyError("Pinned MedGemma container archive is missing.")
    if sha256_file(container_tar_path) != container_tar_sha256:
        raise MRIStudyError("MedGemma container archive digest does not match the plan input.")
    model_manifest = model_snapshot_path / "manifest.sha256"
    if not model_manifest.is_file():
        raise MRIStudyError("Pinned MedGemma model snapshot manifest is missing.")
    actual_model_manifest_sha256 = sha256_file(model_manifest)
    if actual_model_manifest_sha256 != model_snapshot_manifest_sha256:
        raise MRIStudyError("MedGemma model snapshot manifest digest does not match.")
    subprocess.run(
        ["sha256sum", "-c", str(model_manifest)],
        cwd=model_snapshot_path,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    by_name = {item.get("name"): item for item in server_types}
    l4 = by_name.get("L4-1-24G")
    l40s = by_name.get("L40S-1-48G")
    if not l4 or l4.get("availability") != "available":
        raise MRIStudyError("L4-1-24G is not currently available in the selected zone.")
    l4_hourly = _money_eur(l4.get("hourly_price"))
    l40s_hourly = _money_eur((l40s or {}).get("hourly_price"))
    if l4_hourly is None:
        raise MRIStudyError("Scaleway did not return an EUR hourly price for L4.")
    hard_hours = operation_plan["lifecycle"]["hard_ttl_hours"]
    compute_ceiling = round(l4_hourly * hard_hours, 4)
    name = "mristudy-lex-knee-mri-medgemma"
    plan = {
        "schema_version": 1,
        "operation": "scaleway-gpu-provision",
        "created_at": utc_now(),
        "provider": "scaleway",
        "backend_operation_plan_sha256": operation_plan["plan_sha256"],
        "medgemma_run_plan_sha256": medgemma_run_plan_sha256,
        "evidence_map_sha256": evidence_map_sha256,
        "project_id": project_id,
        "zone": zone,
        "resources": {
            "server": {
                "name": name,
                "type": "L4-1-24G",
                "marketplace_local_image_id": marketplace_image_id,
                "marketplace_label": "ubuntu_noble_gpu_os_12",
                "marketplace_storage_type": "instance_sbs",
                "root_volume": "sbs:125GB:5000",
                "public_ip": "new",
                "tags": ["managed-by=mristudy", "study-id=lex-knee-mri"],
            },
            "fallback": {
                "type": "L40S-1-48G",
                "availability_at_plan_time": (l40s or {}).get("availability"),
                "allowed_only_after_documented_l4_capacity_failure": True,
                "requires_new_plan_and_approval": True,
            },
            "security_group": {
                "name": f"{name}-sg",
                "inbound_default": "drop",
                "outbound_default": "accept",
                "allowed_inbound": [
                    {
                        "protocol": "TCP",
                        "source": operation_plan["security"]["ssh_source_cidr"],
                        "port": 22,
                    }
                ],
            },
            "ssh_key": {
                "name": f"{name}-temporary",
                "algorithm": "ed25519",
                "temporary": True,
            },
        },
        "runtime": {
            "model_repository": operation_plan["medgemma"]["repository"],
            "model_revision": operation_plan["medgemma"]["revision"],
            "model_snapshot_path": str(model_snapshot_path) if model_snapshot_path else None,
            "model_snapshot_manifest_sha256": model_snapshot_manifest_sha256,
            "output_schema": "knee-meniscus-finding-v2",
            "prompt_sha256": sha256_bytes(medgemma_knee_meniscus_prompt_v2().encode()),
            "container_digest": container_digest,
            "container_tar_path": str(container_tar_path),
            "container_tar_sha256": container_tar_sha256,
            "cloud_init_path": str(cloud_init_path),
            "cloud_init_sha256": sha256_file(cloud_init_path),
            "dicom_transport": "reverse-ssh-tunnel",
            "orthanc_remote_bind": "127.0.0.1:18042",
            "medgemma_bind": "127.0.0.1:8000",
            "public_application_ports": [],
            "environment": {
                "TORCH_DISABLE_NATIVE_JIT": "1",
                "MEDGEMMA_MODEL_PATH": "/model",
                "ORTHANC_DICOMWEB_URL": "http://127.0.0.1:18042/dicom-web",
                "APPROVED_PLAN_SHA256": medgemma_run_plan_sha256,
                "PROVISION_PLAN_SHA256_SOURCE": "scaleway-provision-plan.plan_sha256",
                "RUNTIME_DIGEST": container_digest,
                "MODEL_SNAPSHOT_MANIFEST_SHA256": model_snapshot_manifest_sha256,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        },
        "cost": {
            "currency": "EUR",
            "l4_hourly": l4_hourly,
            "l40s_hourly": l40s_hourly,
            "hard_ttl_hours": hard_hours,
            "l4_compute_ceiling_excluding_storage_and_ip": compute_ceiling,
            "approved_total_ceiling": operation_plan["cost_control"]["maximum_eur"],
            "storage_and_ip_pricing_must_fit_total_ceiling": True,
        },
        "preconditions": {
            "backend_egress_approval_required": True,
            "synthetic_probe_before_medical_access": True,
            "host_key_pin_required": True,
            "resource_manifest_required": True,
            "destroy_plan_required": True,
            "runtime_probe_authorization_required": True,
        },
        "destroy_scope": [
            "server",
            "all_attached_volumes",
            "public_ip",
            "security_group",
            "temporary_ssh_key",
            "remote_model_cache",
            "remote_dicom_cache",
        ],
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json(plan))
    return plan


def create_resources(plan: dict, manifest_path: Path, ssh_root: Path) -> dict:
    expected_plan_digest = sha256_bytes(
        canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"})
    )
    if plan.get("plan_sha256") != expected_plan_digest:
        raise MRIStudyError("Scaleway provision plan SHA-256 is invalid.")
    if plan["resources"]["server"]["type"] != "L4-1-24G":
        raise MRIStudyError("Only the approved L4 resource plan can be provisioned.")
    cloud_init_path = Path(plan["runtime"]["cloud_init_path"])
    if sha256_file(cloud_init_path) != plan["runtime"]["cloud_init_sha256"]:
        raise MRIStudyError("Cloud-init changed after approval.")
    container_tar = Path(plan["runtime"]["container_tar_path"])
    if sha256_file(container_tar) != plan["runtime"]["container_tar_sha256"]:
        raise MRIStudyError("Container archive changed after approval.")
    model_snapshot = Path(plan["runtime"]["model_snapshot_path"])
    model_manifest = model_snapshot / "manifest.sha256"
    if sha256_file(model_manifest) != plan["runtime"]["model_snapshot_manifest_sha256"]:
        raise MRIStudyError("Model snapshot manifest changed after approval.")
    subprocess.run(
        ["sha256sum", "-c", str(model_manifest)],
        cwd=model_snapshot,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    ssh_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_key = ssh_root / "id_ed25519"
    public_key = ssh_root / "id_ed25519.pub"
    if manifest_path.exists():
        from ..utils import load_json

        manifest = load_json(manifest_path)
        if manifest.get("plan_sha256") != plan["plan_sha256"]:
            raise MRIStudyError("Existing Scaleway resource manifest belongs to another plan.")
    else:
        manifest = {
            "provider": "scaleway",
            "plan_sha256": plan["plan_sha256"],
            "zone": plan["zone"],
            "project_id": plan["project_id"],
            "created_at": utc_now(),
            "resources": {},
            "ssh_private_key": str(private_key),
        }
        atomic_json(manifest_path, manifest)

    if not private_key.exists() or not public_key.exists():
        if private_key.exists() or public_key.exists():
            raise MRIStudyError("Temporary Scaleway SSH key pair is incomplete.")
        subprocess.run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                plan["resources"]["ssh_key"]["name"],
                "-f",
                str(private_key),
            ],
            check=True,
        )
        private_key.chmod(0o600)

    if not manifest["resources"].get("ssh_key_id"):
        existing_keys = _run_json("iam", "ssh-key", "list")
        key_name = plan["resources"]["ssh_key"]["name"]
        matches = [item for item in existing_keys if item.get("name") == key_name]
        if len(matches) > 1:
            raise MRIStudyError("Multiple temporary Scaleway SSH keys have the planned name.")
        key = matches[0] if matches else _run_json(
            "iam",
            "ssh-key",
            "create",
            f"name={key_name}",
            f"public-key=@{public_key}",
            f"project-id={plan['project_id']}",
        )
        manifest["resources"]["ssh_key_id"] = key["id"]
        atomic_json(manifest_path, manifest)

    if not manifest["resources"].get("security_group_id"):
        sg_name = plan["resources"]["security_group"]["name"]
        existing_groups = _run_json(
            "instance", "security-group", "list", f"zone={plan['zone']}"
        )
        matches = [item for item in existing_groups if item.get("name") == sg_name]
        if len(matches) > 1:
            raise MRIStudyError("Multiple Scaleway security groups have the planned name.")
        response = matches[0] if matches else _run_json(
            "instance",
            "security-group",
            "create",
            f"name={sg_name}",
            "description=Temporary MRIStudy MedGemma GPU",
            "inbound-default-policy=drop",
            "outbound-default-policy=accept",
            "stateful=true",
            f"project-id={plan['project_id']}",
            f"zone={plan['zone']}",
        )
        sg = response.get("security_group", response)
        manifest["resources"]["security_group_id"] = sg["id"]
        atomic_json(manifest_path, manifest)
    sg_id = manifest["resources"]["security_group_id"]
    rule = plan["resources"]["security_group"]["allowed_inbound"][0]
    sg_details = _run_json("instance", "security-group", "get", sg_id, f"zone={plan['zone']}")
    rules = sg_details.get("Rules") or sg_details.get("rules") or []
    has_rule = any(
        item.get("direction") == "inbound"
        and item.get("action") == "accept"
        and item.get("protocol") == "TCP"
        and item.get("ip_range") == rule["source"]
        and item.get("dest_port_from") == rule["port"]
        for item in rules
    )
    if not has_rule:
        _run_json(
            "instance",
            "security-group",
            "create-rule",
            f"security-group-id={sg_id}",
            "protocol=TCP",
            "direction=inbound",
            "action=accept",
            f"ip-range={rule['source']}",
            f"dest-port-from={rule['port']}",
            f"zone={plan['zone']}",
        )

    server = plan["resources"]["server"]
    if not manifest["resources"].get("server_id"):
        existing_servers = _run_json("instance", "server", "list", f"zone={plan['zone']}")
        matches = [item for item in existing_servers if item.get("name") == server["name"]]
        if len(matches) > 1:
            raise MRIStudyError("Multiple Scaleway servers have the planned name.")
        created = matches[0] if matches else _run_json(
            "instance",
            "server",
            "create",
            f"image={server['marketplace_local_image_id']}",
            f"type={server['type']}",
            f"name={server['name']}",
            f"root-volume={server['root_volume']}",
            "ip=new",
            f"security-group-id={sg_id}",
            f"cloud-init=@{cloud_init_path}",
            "tags.0=managed-by=mristudy",
            "tags.1=study-id=lex-knee-mri",
            f"project-id={plan['project_id']}",
            f"zone={plan['zone']}",
        )
        server_data = created.get("server", created)
        manifest["resources"]["server_id"] = server_data["id"]
        manifest["resources"]["server_name"] = server_data.get("name")
        manifest["resources"]["public_ip_id"] = (server_data.get("public_ip") or {}).get("id")
        manifest["resources"]["public_ip"] = (server_data.get("public_ip") or {}).get("address")
        manifest["resources"]["volume_ids"] = sorted(
            (server_data.get("volumes") or {}).keys()
        )
        atomic_json(manifest_path, manifest)
    return manifest


def attached_block_volume_ids(zone: str, server_id: str) -> list[str]:
    """Resolve the SBS volumes a server actually references, from the provider.

    The instance volume map is keyed by slot ("0", "1"), which is not a volume
    identifier. Absence must be proven against real SBS identifiers, so they are
    read from the Block API before the server is deleted.
    """
    found: set[str] = set()
    volumes = _run_json("block", "volume", "list", f"zone={zone}")
    if isinstance(volumes, list):
        for volume in volumes:
            if not isinstance(volume, dict):
                continue
            for reference in volume.get("references") or []:
                if not isinstance(reference, dict):
                    continue
                if reference.get("product_resource_id") == server_id and volume.get("id"):
                    found.add(str(volume["id"]))
    return sorted(found)


def destroy_resources(plan: dict, manifest: dict) -> dict:
    resources = manifest["resources"]
    zone = manifest["zone"]
    server_id = resources.get("server_id")
    if server_id and not resources.get("block_volume_ids"):
        resources["block_volume_ids"] = attached_block_volume_ids(zone, server_id)
    if server_id:
        subprocess.run(
            [
                "scw",
                "instance",
                "server",
                "delete",
                server_id,
                "with-volumes=all",
                "with-ip=true",
                "force-shutdown=true",
                f"zone={zone}",
                "--wait",
            ],
            check=True,
        )
    sg_id = resources.get("security_group_id")
    if sg_id:
        subprocess.run(
            ["scw", "instance", "security-group", "delete", sg_id, f"zone={zone}"],
            check=True,
        )
    key_id = resources.get("ssh_key_id")
    if key_id:
        subprocess.run(["scw", "iam", "ssh-key", "delete", key_id], check=True)
    private_key = Path(manifest["ssh_private_key"])
    for path in (private_key, private_key.with_suffix(".pub")):
        if path.exists():
            path.unlink()
    ip_id = resources.get("public_ip_id")
    if ip_id:
        subprocess.run(
            ["scw", "instance", "ip", "delete", ip_id, f"zone={zone}"],
            check=False,
        )
    block_volume_ids = [
        str(value) for value in (resources.get("block_volume_ids") or []) if value
    ]
    for volume_id in block_volume_ids:
        subprocess.run(
            ["scw", "block", "volume", "delete", volume_id, f"zone={zone}"],
            check=False,
        )
    verification = verify_absence(
        zone,
        {
            "servers": [server_id] if server_id else [],
            "security_groups": [sg_id] if sg_id else [],
            "ips": [ip_id] if ip_id else [],
            "ssh_keys": [key_id] if key_id else [],
            "volumes": block_volume_ids,
        },
    )
    if not verification["verified_absent"]:
        raise MRIStudyError(
            f"Scaleway cleanup verification found remaining resources: {verification['remaining']}"
        )
    return {
        "provider": "scaleway",
        "provision_plan_sha256": plan["plan_sha256"],
        "verified_absent": True,
        "verification": verification,
        "destroyed_at": utc_now(),
    }
