from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_liveness_script_is_privacy_safe_and_binds_fixed_policy() -> None:
    source = (ROOT / "scripts/verify-constrained-decoder-liveness.py").read_text(
        encoding="utf-8"
    )
    assert "raw_text_sha256" in source
    assert "raw_character_count" in source
    assert "divergence_index" not in source
    assert "decoder_acceptance" in source
    assert 'print(raw)' not in source
    assert '"raw_text": raw' not in source
    assert 'generation_policy()["repetition_penalty"] == 1.0' in source
    assert 'generation_policy()["no_repeat_ngram_size"] == 0' in source
