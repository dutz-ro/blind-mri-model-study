"""Study profiles that actually drive the pipeline.

``configs/profiles/knee-meniscus-v1.yaml`` existed but was never read: no code
called ``load_yaml`` and ``configs/`` appeared nowhere in the package. Every
value it claimed to control was hard-coded in Python, and the file had already
drifted, listing status values the schema rejects and series patterns that do
not match the ones the planner used. A profile that cannot be trusted is worse
than no profile, so profiles are now validated and authoritative.

A profile declares what changes between studies: the body region, which series
to select and under which role, which clinical schema to use, how windows are
formed, and how findings are grouped during integration. Everything else, in
particular coverage, evidence identity, provenance and the safety boundaries,
is invariant and stays in code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import MRIStudyError
from .schema_registry import available_schemas

PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SeriesRole:
    """One requested series and the role it plays in the analysis."""

    role: str
    pattern: str
    required: bool = True

    def matches(self, description: str) -> bool:
        return bool(re.search(self.pattern, description or ""))


@dataclass(frozen=True)
class IntegrationPolicy:
    """How findings are grouped and classified across windows and planes."""

    anatomy_key_paths: tuple[str, ...]
    claim_key_paths: tuple[str, ...]
    status_path: str
    positive_statuses: frozenset[str]
    negative_statuses: frozenset[str]


@dataclass(frozen=True)
class StudyProfile:
    name: str
    body_region: str
    task: str
    output_schema: str
    series_roles: tuple[SeriesRole, ...]
    maximum_window_size: int
    minimum_overlap: int
    integration: IntegrationPolicy
    excluded_series_patterns: tuple[str, ...] = ()
    prompt_region_block: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def role_order(self) -> tuple[str, ...]:
        """Role order fixes global evidence numbering, so it must be stable."""
        return tuple(item.role for item in self.series_roles)

    def required_roles(self) -> tuple[str, ...]:
        return tuple(item.role for item in self.series_roles if item.required)

    def is_excluded(self, description: str) -> bool:
        return any(
            re.search(pattern, description or "")
            for pattern in self.excluded_series_patterns
        )


def _require(payload: dict[str, Any], key: str, source: Path | None) -> Any:
    if key not in payload:
        where = f" in {source}" if source else ""
        raise MRIStudyError(f"Study profile is missing required key {key!r}{where}.")
    return payload[key]


def _compile_patterns(patterns: list[Any], source: Path | None) -> tuple[str, ...]:
    compiled = []
    for pattern in patterns or []:
        try:
            re.compile(str(pattern))
        except re.error as error:
            raise MRIStudyError(f"Invalid regular expression in profile: {pattern!r}") from error
        compiled.append(str(pattern))
    return tuple(compiled)


def parse_profile(payload: dict[str, Any], source: Path | None = None) -> StudyProfile:
    """Validate a profile document and fail closed on anything ambiguous."""
    version = int(payload.get("profile_schema_version", PROFILE_SCHEMA_VERSION))
    if version != PROFILE_SCHEMA_VERSION:
        raise MRIStudyError(f"Unsupported profile schema version: {version}")

    output_schema = str(_require(payload, "output_schema", source))
    if output_schema not in available_schemas():
        raise MRIStudyError(
            f"Profile requests unknown schema {output_schema!r}. "
            f"Registered: {', '.join(available_schemas())}"
        )

    roles: list[SeriesRole] = []
    seen_roles: set[str] = set()
    for entry in _require(payload, "series_roles", source):
        if not isinstance(entry, dict):
            raise MRIStudyError("Each series role must be a mapping.")
        role = str(_require(entry, "role", source))
        if role in seen_roles:
            raise MRIStudyError(f"Duplicate series role in profile: {role}")
        seen_roles.add(role)
        pattern = str(_require(entry, "pattern", source))
        _compile_patterns([pattern], source)
        roles.append(SeriesRole(role, pattern, bool(entry.get("required", True))))
    if not roles:
        raise MRIStudyError("A study profile must declare at least one series role.")

    windowing = payload.get("windowing") or {}
    maximum_window_size = int(windowing.get("maximum_window_size", 32))
    minimum_overlap = int(windowing.get("minimum_overlap", 8))
    if maximum_window_size < 1:
        raise MRIStudyError("maximum_window_size must be at least 1.")
    if minimum_overlap < 0 or minimum_overlap >= maximum_window_size:
        raise MRIStudyError("minimum_overlap must be non-negative and below the window size.")

    integration_payload = _require(payload, "integration", source)
    anatomy = tuple(str(item) for item in _require(integration_payload, "anatomy_key", source))
    claim = tuple(str(item) for item in _require(integration_payload, "claim_key", source))
    if not anatomy or not claim:
        raise MRIStudyError("Integration keys must not be empty.")
    status_path = str(_require(integration_payload, "status_path", source))
    positive = frozenset(str(item) for item in integration_payload.get("positive_statuses") or [])
    negative = frozenset(str(item) for item in integration_payload.get("negative_statuses") or [])
    if not positive:
        raise MRIStudyError("Integration must declare at least one positive status.")
    if positive & negative:
        raise MRIStudyError("A status cannot be both positive and negative.")

    return StudyProfile(
        name=str(_require(payload, "name", source)),
        body_region=str(_require(payload, "body_region", source)),
        task=str(payload.get("task", "exploratory-review")),
        output_schema=output_schema,
        series_roles=tuple(roles),
        maximum_window_size=maximum_window_size,
        minimum_overlap=minimum_overlap,
        integration=IntegrationPolicy(
            anatomy_key_paths=anatomy,
            claim_key_paths=claim,
            status_path=status_path,
            positive_statuses=positive,
            negative_statuses=negative,
        ),
        excluded_series_patterns=_compile_patterns(payload.get("excluded_series"), source),
        prompt_region_block=str(payload.get("prompt_region_block", "")).strip(),
        metadata=dict(payload.get("metadata") or {}),
    )


def load_profile(path: Path) -> StudyProfile:
    if not path.is_file():
        raise MRIStudyError(f"Study profile not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise MRIStudyError(f"Study profile must be a mapping: {path}")
    return parse_profile(payload, path)


def profiles_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "profiles"


def load_profile_by_name(name: str, directory: Path | None = None) -> StudyProfile:
    root = directory or profiles_directory()
    candidate = root / f"{name}.yaml"
    if not candidate.is_file():
        available = sorted(item.stem for item in root.glob("*.yaml"))
        raise MRIStudyError(
            f"Unknown study profile {name!r}. Available: {', '.join(available) or 'none'}"
        )
    return load_profile(candidate)


def resolve_path(finding: dict[str, Any], path: str) -> Any:
    """Read a dotted path from a finding, returning None when absent.

    Integration keys are declared as paths so a profile can group findings for a
    body region whose fields this code has never seen.
    """
    current: Any = finding
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
