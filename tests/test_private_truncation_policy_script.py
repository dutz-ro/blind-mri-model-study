from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_private_truncation_script_is_hash_only() -> None:
    source = (ROOT / "scripts/verify-private-truncation-policy.py").read_text(
        encoding="utf-8"
    )
    assert "raw_text_sha256" in source
    assert "raw_character_count" in source
    assert 'print(raw)' not in source
    assert '"raw_text": raw' not in source
    assert "correction_repeats_schema" in source
    assert "all_classified_truncated" in source
