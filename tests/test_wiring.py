"""Source-string wiring asserts for the config-validation feature.

validator.py has no __all__, so these target the source text directly — the
correct mechanism for asserting imports were pruned and the helper is used.
"""

from pathlib import Path


def test_run_gate_uses_helper_and_prunes_imports():
    src = Path("ui/pages/run.py").read_text()
    assert "summarize_config_validation" in src
    # Old inline validators must be fully pruned (ruff F401 otherwise).
    assert "check_file_references" not in src
    assert "check_species_consistency" not in src
    assert "validate_config" not in src  # not a substring of summarize_config_validation


def test_cli_uses_helper_and_prunes_imports():
    src = Path("osmose/cli.py").read_text()
    assert "summarize_config_validation" in src
    assert "check_file_references" not in src
    assert "check_species_consistency" not in src
    assert "validate_config" not in src
