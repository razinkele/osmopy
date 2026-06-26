"""Tests for HPC container touchups (Task 1-5).

Task 1: calibrate_baltic.py honors OSMOSE_RESULTS_DIR.
"""

import importlib
import os


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
