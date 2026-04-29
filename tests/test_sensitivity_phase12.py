"""Smoke tests for scripts/sensitivity_phase12.py — checks the analysis path
without running real OSMOSE simulations.

The actual 14k-eval Sobol run is too costly for CI; these tests verify the
script imports cleanly, parses CLI, and that the SALib path on a known
analytical function produces expected sensitivity rankings.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_script_imports():
    """The script must import without side effects."""
    import importlib

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        mod = importlib.import_module("scripts.sensitivity_phase12")
    finally:
        sys.path.pop(0)
    # Sanity: the worker init + eval functions are exposed for pickling
    assert hasattr(mod, "_pool_init")
    assert hasattr(mod, "_eval_one")
    assert hasattr(mod, "main")


def test_dry_run_prints_plan(tmp_path):
    """`--dry-run` should print eval count + ETA without running anything."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        pytest.skip(".venv/bin/python missing")
    result = subprocess.run(
        [
            str(venv_python),
            str(PROJECT_ROOT / "scripts" / "sensitivity_phase12.py"),
            "--dry-run",
            "--n-base", "8",
            "--workers", "4",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "total samples" in result.stdout
    assert "Dry run" in result.stdout
    # n_base=8, D=27 → 8 * (2*27 + 2) = 448 expected samples
    assert "448" in result.stdout


def test_sobol_pipeline_on_known_function(tmp_path):
    """End-to-end: SensitivityAnalyzer + analytical objective recovers the
    right ranking. f(x) = x[0] + 0.1*x[1] should have S1[0] >> S1[1].

    This bypasses the OSMOSE engine but exercises the same Saltelli-sample +
    sobol-analyze path the real script uses. Catches SALib API drift.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from osmose.calibration.sensitivity import SensitivityAnalyzer
    finally:
        sys.path.pop(0)

    a = SensitivityAnalyzer(["a", "b"], [(0.0, 1.0), (0.0, 1.0)])
    X = a.generate_samples(n_base=512)
    Y = X[:, 0] + 0.1 * X[:, 1]
    Si = a.analyze(Y)

    assert Si["ST"][0] > Si["ST"][1] * 5, (
        f"expected param a's ST to dominate b; got ST={Si['ST']}"
    )
    assert 0.0 < Si["S1"][0] < 1.5
    assert -0.05 < Si["S1"][1] < 0.2


def test_resume_csv_round_trip(tmp_path):
    """`_load_existing_y` should pick up where a prior interrupted run left off."""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from scripts.sensitivity_phase12 import _load_existing_y
    finally:
        sys.path.pop(0)

    csv_path = tmp_path / "y.csv"
    with open(csv_path, "w") as f:
        f.write("idx,objective\n")
        f.write("3,1.234\n")
        f.write("7,5.678\n")
        f.write("0,9.000\n")

    Y, done = _load_existing_y(csv_path, n_samples=10)
    assert done == {0, 3, 7}
    assert Y[0] == 9.0
    assert Y[3] == 1.234
    assert Y[7] == 5.678
    assert np.isnan(Y[1])  # untouched index stays NaN
