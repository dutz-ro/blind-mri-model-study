from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_private_replay_script_never_prints_model_content() -> None:
    source = (ROOT / "scripts/verify-private-evidence-policy.py").read_text(encoding="utf-8")
    assert "raw_text_sha256" in source
    assert 'report["raw_text"]' not in source
    assert 'print(raw_text)' not in source
    assert "clinical_findings_preserved" in source
    assert "scope_counts" in source
