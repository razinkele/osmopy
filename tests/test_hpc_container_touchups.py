"""Tests for HPC container touchups (Task 1-5).

Task 1: calibrate_baltic.py honors OSMOSE_RESULTS_DIR.
Task 2: osmose run --jar defaults to $OSMOSE_JAR.
"""

import importlib


def test_calibrate_baltic_honors_results_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OSMOSE_RESULTS_DIR", str(tmp_path / "rd"))
    import osmose.calibration.checkpoint as cp

    importlib.reload(cp)  # re-evaluate RESULTS_DIR = default_results_dir() with the env set
    cb = importlib.import_module("scripts.calibrate_baltic")
    cb = importlib.reload(cb)
    assert cb.RESULTS_DIR == (tmp_path / "rd")


def test_calibrate_baltic_results_dir_default_without_env(monkeypatch):
    monkeypatch.delenv("OSMOSE_RESULTS_DIR", raising=False)
    import osmose.calibration.checkpoint as cp

    importlib.reload(cp)
    cb = importlib.reload(importlib.import_module("scripts.calibrate_baltic"))
    assert cb.RESULTS_DIR.name == "calibration_results"  # package-root default


def test_jar_from_prefers_arg_then_env(monkeypatch):
    from osmose.cli import _jar_from

    monkeypatch.setenv("OSMOSE_JAR", "/env/osmose.jar")
    assert _jar_from("/cli/x.jar") == "/cli/x.jar"  # explicit --jar wins
    assert _jar_from(None) == "/env/osmose.jar"  # falls back to $OSMOSE_JAR
    monkeypatch.delenv("OSMOSE_JAR", raising=False)
    assert _jar_from(None) is None  # neither -> None


def test_cmd_run_clear_error_when_no_jar(tmp_path, monkeypatch, capsys):
    from argparse import Namespace
    from osmose.cli import cmd_run

    monkeypatch.delenv("OSMOSE_JAR", raising=False)
    cfg = tmp_path / "c.csv"
    cfg.write_text("simulation.nspecies;1\n")
    rc = cmd_run(Namespace(config=str(cfg), jar=None, output=None, java_opts=None, timeout=None))
    assert rc == 1
    assert "jar" in capsys.readouterr().err.lower()  # clear error, NOT an argparse usage error
