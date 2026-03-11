from unittest.mock import patch
import pytest
from osmose.cli import main


def test_cli_help():
    with patch("sys.argv", ["osmose", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0


def test_cli_validate_missing_file():
    with patch("sys.argv", ["osmose", "validate", "/nonexistent.csv"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0


def test_cli_validate_valid_config(tmp_path):
    cfg = tmp_path / "test.csv"
    cfg.write_text("simulation.nspecies;3\n")
    with patch("sys.argv", ["osmose", "validate", str(cfg)]):
        # Should not raise or should exit 0
        try:
            main()
        except SystemExit as e:
            assert e.code in (0, None)
