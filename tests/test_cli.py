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


def test_cmd_run_missing_config(tmp_path):
    """cmd_run with non-existent config should exit non-zero."""
    fake_jar = tmp_path / "fake.jar"
    fake_jar.write_text("")  # jar exists but config does not
    with patch(
        "sys.argv",
        [
            "osmose",
            "run",
            str(tmp_path / "nonexistent.csv"),
            "--jar",
            str(fake_jar),
        ],
    ):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0


def test_cmd_report_missing_dir(tmp_path):
    """cmd_report with non-existent output dir should exit non-zero."""
    with patch("sys.argv", ["osmose", "report", str(tmp_path / "nonexistent")]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0
