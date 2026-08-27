"""Absence verification must cover every billable temporary resource.

A stopped GPU server still bills its attached SBS root volume and its reserved
public IP. The original verification listed only servers and security groups,
so a cleanup could report ``verified_absent`` while a 125 GB volume and a
public IP remained. These tests pin the categories and prove that a remaining
resource is reported rather than silently accepted.
"""

from __future__ import annotations

import pytest

from mristudy.backends import scaleway
from mristudy.errors import MRIStudyError

SERVER = "31904d6e-0000-0000-0000-000000000000"
VOLUME = "14dda90f-0000-0000-0000-000000000000"
IP = "d967d32f-0000-0000-0000-000000000000"
KEY = "a7f994ef-0000-0000-0000-000000000000"


def _inventory(**overrides: object) -> dict:
    empty: dict[str, object] = {
        "zone": "fr-par-2",
        "servers": [],
        "security_groups": [],
        "server_types": [],
        "volumes": [],
        "block_volumes": [],
        "ips": [],
        "ssh_keys": [],
    }
    empty.update(overrides)
    return empty


def test_absence_is_verified_when_provider_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaleway, "inventory", lambda zone: _inventory())
    result = scaleway.verify_absence(
        "fr-par-2",
        {"servers": [SERVER], "volumes": [VOLUME], "ips": [IP], "ssh_keys": [KEY]},
    )
    assert result["verified_absent"] is True
    assert result["remaining"] == {}


def test_remaining_block_volume_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The root volume of a GPU instance is an SBS volume, not an instance volume."""
    monkeypatch.setattr(
        scaleway,
        "inventory",
        lambda zone: _inventory(block_volumes=[{"id": VOLUME, "status": "available"}]),
    )
    result = scaleway.verify_absence("fr-par-2", {"volumes": [VOLUME]})
    assert result["verified_absent"] is False
    assert result["remaining"] == {"volumes": [VOLUME]}


def test_remaining_public_ip_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaleway, "inventory", lambda zone: _inventory(ips=[{"id": IP}]))
    result = scaleway.verify_absence("fr-par-2", {"ips": [IP]})
    assert result["verified_absent"] is False
    assert result["remaining"] == {"ips": [IP]}


def test_remaining_temporary_ssh_key_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaleway, "inventory", lambda zone: _inventory(ssh_keys=[{"id": KEY}]))
    result = scaleway.verify_absence("fr-par-2", {"ssh_keys": [KEY]})
    assert result["verified_absent"] is False
    assert result["remaining"] == {"ssh_keys": [KEY]}


def test_unknown_category_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaleway, "inventory", lambda zone: _inventory())
    with pytest.raises(MRIStudyError):
        scaleway.verify_absence("fr-par-2", {"snapshots": ["anything"]})


def test_inventory_queries_all_billable_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run_json(*args: str) -> list:
        calls.append(args)
        return []

    monkeypatch.setattr(scaleway, "_run_json", fake_run_json)
    scaleway.inventory("fr-par-2")
    joined = {" ".join(call) for call in calls}
    assert any("instance server list" in item for item in joined)
    assert any("instance volume list" in item for item in joined)
    assert any("block volume list" in item for item in joined)
    assert any("instance ip list" in item for item in joined)
    assert any("iam ssh-key list" in item for item in joined)


def test_attached_block_volumes_resolve_real_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slot keys such as "0" are not volume IDs and cannot prove absence."""
    monkeypatch.setattr(
        scaleway,
        "_run_json",
        lambda *args: [
            {
                "id": VOLUME,
                "references": [{"product_resource_id": SERVER, "type": "exclusive"}],
            },
            {
                "id": "unrelated-volume",
                "references": [{"product_resource_id": "another-server"}],
            },
        ],
    )
    assert scaleway.attached_block_volume_ids("fr-par-2", SERVER) == [VOLUME]
