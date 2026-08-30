"""Study profiles are validated and actually drive the pipeline.

The previous profile file was never read by any code path, so it drifted until
it described behaviour the code did not have. These tests pin that a profile is
authoritative and that an invalid one fails closed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mristudy.errors import MRIStudyError
from mristudy.profiles import (
    load_profile_by_name,
    parse_profile,
    profiles_directory,
    resolve_path,
)
from mristudy.schema_registry import available_schemas


def _valid() -> dict:
    return {
        "profile_schema_version": 1,
        "name": "example-v1",
        "body_region": "elbow",
        "output_schema": "mri-finding-core-v1",
        "series_roles": [
            {"role": "primary-volume", "pattern": "(?i)t1"},
            {"role": "fluid-sensitive", "pattern": "(?i)stir", "required": False},
        ],
        "integration": {
            "anatomy_key": ["localization.region"],
            "claim_key": ["localization.region", "certainty_status"],
            "status_path": "certainty_status",
            "positive_statuses": ["suspected", "probable"],
            "negative_statuses": ["absent"],
        },
    }


def test_valid_profile_exposes_role_order_and_requirements() -> None:
    profile = parse_profile(_valid())
    assert profile.role_order == ("primary-volume", "fluid-sensitive")
    assert profile.required_roles() == ("primary-volume",)
    assert profile.output_schema in available_schemas()


def test_unknown_schema_is_rejected() -> None:
    payload = _valid()
    payload["output_schema"] = "not-a-registered-schema"
    with pytest.raises(MRIStudyError):
        parse_profile(payload)


def test_duplicate_role_is_rejected() -> None:
    payload = _valid()
    payload["series_roles"].append({"role": "primary-volume", "pattern": "(?i)x"})
    with pytest.raises(MRIStudyError):
        parse_profile(payload)


def test_invalid_regular_expression_is_rejected() -> None:
    payload = _valid()
    payload["series_roles"][0]["pattern"] = "(?i)[unclosed"
    with pytest.raises(MRIStudyError):
        parse_profile(payload)


def test_contradictory_status_partition_is_rejected() -> None:
    payload = _valid()
    payload["integration"]["negative_statuses"] = ["suspected"]
    with pytest.raises(MRIStudyError):
        parse_profile(payload)


def test_overlap_must_be_below_window_size() -> None:
    payload = _valid()
    payload["windowing"] = {"maximum_window_size": 8, "minimum_overlap": 8}
    with pytest.raises(MRIStudyError):
        parse_profile(payload)


def test_missing_required_key_names_the_key() -> None:
    payload = _valid()
    del payload["body_region"]
    with pytest.raises(MRIStudyError, match="body_region"):
        parse_profile(payload)


def test_series_pattern_matching_is_case_insensitive_by_pattern() -> None:
    profile = parse_profile(_valid())
    primary = profile.series_roles[0]
    assert primary.matches("AX T1 TSE")
    assert not primary.matches("AX T2 FLAIR")


def test_excluded_series_patterns_apply() -> None:
    payload = _valid()
    payload["excluded_series"] = ["(?i)localizer"]
    profile = parse_profile(payload)
    assert profile.is_excluded("3-plane Localizer")
    assert not profile.is_excluded("AX T1")


def test_resolve_path_reads_nested_values_and_tolerates_absence() -> None:
    finding = {"localization": {"region": "frontal_lobe"}}
    assert resolve_path(finding, "localization.region") == "frontal_lobe"
    assert resolve_path(finding, "localization.subregion") is None
    assert resolve_path(finding, "missing.deeply.nested") is None


def test_shipped_profiles_are_valid() -> None:
    directory = profiles_directory()
    shipped = sorted(directory.glob("*.yaml"))
    assert shipped, "no study profiles are shipped"
    for path in shipped:
        profile = load_profile_by_name(path.stem, directory)
        assert profile.name == path.stem
        assert profile.output_schema in available_schemas()
        assert profile.required_roles()


def test_knee_profile_matches_the_locked_study_configuration() -> None:
    """The knee profile must keep describing the completed run."""
    profile = load_profile_by_name("knee-meniscus-v1")
    assert profile.output_schema == "knee-meniscus-finding-v2"
    assert profile.body_region == "knee"
    assert profile.role_order == (
        "primary-volume",
        "sagittal-meniscus",
        "coronal-confirmation",
        "axial-confirmation",
        "anatomy-correlation",
    )
    assert profile.maximum_window_size == 32
    assert profile.minimum_overlap == 8
    assert profile.integration.anatomy_key_paths == ("meniscus", "segment")
    assert profile.integration.status_path == "tear_status"
    assert profile.integration.positive_statuses == frozenset({"suspected", "probable"})
    assert "Do not emit absent findings" in profile.prompt_region_block
    assert "one finding per image" in profile.prompt_region_block
    assert "empty findings array" in profile.prompt_region_block


def test_brain_profile_uses_the_region_independent_schema() -> None:
    """A second body region is a profile, not a code change."""
    profile = load_profile_by_name("brain-unknown-v1")
    assert profile.output_schema == "mri-finding-core-v1"
    assert profile.body_region == "brain"
    assert profile.integration.anatomy_key_paths[0].startswith("localization.")
    assert profile.integration.status_path == "certainty_status"
    # Sequence naming varies between scanners, so some roles are optional.
    assert len(profile.required_roles()) < len(profile.role_order)


def test_profile_does_not_restate_the_clinical_vocabulary(tmp_path: Path) -> None:
    """Duplicated enums drifted before; the schema stays authoritative."""
    for path in sorted(profiles_directory().glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "statuses" not in payload
        assert "structures" not in payload
        assert "tear_patterns" not in payload
