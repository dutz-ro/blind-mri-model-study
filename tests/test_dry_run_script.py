from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_dry_run_script_is_not_bound_to_one_study() -> None:
    source = (ROOT / "scripts/medgemma-dry-run.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--study", required=True' in source
    assert 'parser.add_argument(\n        "--data-root"' in source
    assert "/home/openclaw/mri-study-data/studies/lex-knee-mri" not in source
