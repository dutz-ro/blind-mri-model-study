import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/verify-decoder-output-budget.py"


def test_budget_script_reports_only_sizes_and_digests() -> None:
    """The verifier runs on a workstation, so it must never print clinical text."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "maximum_structural_sha256" in source
    assert "maximum_structural_characters" in source
    assert "print(serialized" not in source
    assert '"maximum": maximum' not in source


def test_budget_script_passes_without_a_tokenizer() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["canonical_validation_errors"] == []
    assert report["maximum_structural_characters"] > 0
    assert report["token_budget_proved"] is False
