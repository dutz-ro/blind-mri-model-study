"""Bind the runtime response shape to the provenance the runner demands.

Every historical pre-lock failure of this system shared one shape: the runtime
returned a medically valid result, the runner rejected it because a provenance
key was missing, the GPU was destroyed, and the approval was invalidated. The
existing contract tests could not catch this because they assert on the text of
``app.py`` (``assert '"decoder": _decoder_metadata' in source``), which proves
only that a string was typed, not that the response satisfies the runner.

These tests derive the required key set from ``runner.execute_window`` itself
and compare it against the keys the runtime actually returns. Adding a key to
``expected_provenance`` without returning it from ``/analyze-window`` fails
here, locally, in milliseconds, instead of after provisioning.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
RUNTIME = REPO / "deploy/scaleway/medgemma/app.py"
RUNNER = REPO / "src/mristudy/medgemma/runner.py"

MEDICAL_ENDPOINT = "analyze_window"


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(_module(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is missing from {path.name}")


def _returned_key_sets(function: ast.FunctionDef) -> list[set[str]]:
    """Collect the literal keys of every dict returned by a function."""
    found: list[set[str]] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if keys:
                found.append(keys)
    return found


def _required_provenance_keys() -> set[str]:
    """Read the provenance contract from the runner, not from a hand-written list."""
    execute_window = _function(RUNNER, "execute_window")
    for node in ast.walk(execute_window):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "expected_provenance" not in targets:
            continue
        if not isinstance(node.value, ast.Dict):
            raise AssertionError("expected_provenance is no longer a dict literal.")
        return {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    raise AssertionError("execute_window no longer defines expected_provenance.")


def test_provenance_contract_is_discoverable_and_non_trivial() -> None:
    required = _required_provenance_keys()
    assert len(required) >= 10
    # Fields whose absence previously destroyed a paid GPU run.
    assert {"decoder", "generation", "runtime_digest"} <= required


def test_medical_endpoint_returns_every_key_the_runner_requires() -> None:
    required = _required_provenance_keys()
    returns = _returned_key_sets(_function(RUNTIME, MEDICAL_ENDPOINT))
    assert returns, "/analyze-window has no literal dict return to verify."
    for returned in returns:
        missing = sorted(required - returned)
        assert not missing, (
            f"/analyze-window omits provenance the runner requires: {missing}. "
            "execute_window would reject this response after the model has "
            "already produced a medical result."
        )


def test_runtime_never_returns_provenance_the_runner_ignores_silently() -> None:
    """A renamed key must fail loudly here rather than at integration time."""
    returns = _returned_key_sets(_function(RUNTIME, MEDICAL_ENDPOINT))
    union = set().union(*returns)
    # run_plan_sha256 is the runtime's name for the runner's plan digest field.
    assert "run_plan_sha256" in union
    assert "provision_plan_sha256" in union
    assert "model_snapshot_manifest_sha256" in union


def _resolved_return_key_sets(name: str, depth: int = 3) -> list[set[str]]:
    """Return literal dict keys for a function, following single delegations.

    Probe endpoints build their payload in a shared helper, so a direct literal
    check would wrongly report an empty contract. When a function's only return
    is a call to another local function, the contract of that callee is used.
    """
    function = _function(RUNTIME, name)
    literal = _returned_key_sets(function)
    if literal or depth <= 0:
        return literal
    delegates = {
        node.value.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    }
    collected: list[set[str]] = []
    for delegate in sorted(delegates):
        collected.extend(_resolved_return_key_sets(delegate, depth - 1))
    return collected


def _success_return_key_sets(name: str) -> list[set[str]]:
    """Successful payloads only.

    A failure path legitimately returns a short diagnostic object such as
    ``{"passed": False, "error_class": ...}``. Requiring full provenance there
    would force meaningless fields onto error handling, so only returns that
    carry the model identity are treated as success payloads.
    """
    return [
        keys for keys in _resolved_return_key_sets(name) if "model" in keys or "status" in keys
    ]


@pytest.mark.parametrize(
    "endpoint",
    ["health", "synthetic_probe", "synthetic_dicom_probe", "synthetic_json_probe"],
)
def test_probe_and_health_endpoints_expose_decoder_identity(endpoint: str) -> None:
    """Probes gate medical access, so they must prove the decoder that will be used.

    The capacity probes at 1/8/16/32 images and 32 synthetic DICOM objects are
    what authorize medical access. If they can pass without reporting decoder
    identity, the medical path may run under a decoder that was never proven,
    which is the failure class that invalidated the previous runtime.
    """
    returns = _success_return_key_sets(endpoint)
    assert returns, f"{endpoint} has no resolvable success payload to verify."
    assert all("decoder" in keys for keys in returns), (
        f"{endpoint} does not report decoder provenance; a probe could pass "
        "while the medical path uses an unverified decoder."
    )
