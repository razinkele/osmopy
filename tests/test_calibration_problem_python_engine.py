"""Tests for OsmoseCalibrationProblem Python-engine path.

Spec: docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md
"""
from __future__ import annotations

import os
import subprocess as _subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from osmose.calibration import FreeParameter, OsmoseCalibrationProblem, Transform
from osmose.schema import build_registry

EXAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"


def _simple_objective(results) -> float:
    """Minimal objective: total biomass across all species at the last time step.

    ``biomass()`` returns a wide DataFrame with ``Time`` + one column per
    species + a ``species`` annotation column ("all" for cross-species output).
    We sum the numeric species columns of the final row.
    """
    df = results.biomass()
    if df.empty:
        return 0.0
    last_row = df.iloc[-1]
    total = 0.0
    for col, val in last_row.items():
        if col in ("Time", "species"):
            continue
        if isinstance(val, (int, float, np.floating)) and not np.isnan(val):
            total += float(val)
    return total


def _make_problem(
    tmp_path: Path,
    *,
    use_java_engine: bool = False,
    jar_path: Path | None = None,
):
    return OsmoseCalibrationProblem(
        free_params=[
            FreeParameter(
                key="mortality.fishing.rate.sp0",
                lower_bound=0.1,
                upper_bound=0.5,
                transform=Transform.LINEAR,
            ),
        ],
        base_config_path=EXAMPLE_CONFIG,
        objective_fns=[_simple_objective],
        registry=build_registry(),
        work_dir=tmp_path,
        use_java_engine=use_java_engine,
        jar_path=jar_path,
        n_parallel=1,
        enable_cache=False,
    )


def test_python_engine_default(tmp_path, monkeypatch):
    """Default OsmoseCalibrationProblem (no use_java_engine) evaluates
    via PythonEngine and does NOT invoke subprocess.run."""

    def _raise_subprocess(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called in Python-engine mode")

    monkeypatch.setattr(_subprocess, "run", _raise_subprocess)
    problem = _make_problem(tmp_path, use_java_engine=False)
    # Two candidates so the >50% failure abort rule doesn't trip on a single
    # transient failure — if both succeed we validate the happy path.
    X = np.array([[0.3], [0.4]])
    out: dict = {}
    problem._evaluate(X, out)
    assert "F" in out
    assert out["F"].shape == (2, 1)
    assert not np.any(np.isinf(out["F"])), f"Python-engine run returned inf: {out['F']}"


def test_java_engine_opt_in(tmp_path, monkeypatch):
    """use_java_engine=True routes through _run_java_subprocess with the
    provided jar_path."""
    fake_jar = tmp_path / "fake.jar"
    fake_jar.write_bytes(b"")
    captured_argv: list = []

    def _fake_subprocess(cmd, *args, **kwargs):
        captured_argv.append(cmd)
        result = mock.MagicMock()
        result.returncode = 1
        result.stderr = b"fake jar"
        return result

    monkeypatch.setattr(_subprocess, "run", _fake_subprocess)
    problem = _make_problem(tmp_path, use_java_engine=True, jar_path=fake_jar)
    # Single call path — call _run_single directly so we don't hit the
    # >50%-candidate-failure abort (which would raise instead of letting us
    # inspect the subprocess argv).
    result = problem._run_single({"mortality.fishing.rate.sp0": "0.3"}, run_id=0)
    assert result == [float("inf")]
    assert captured_argv, "subprocess.run not called"
    argv = captured_argv[0]
    assert str(fake_jar) in argv


@pytest.mark.skipif(
    not os.environ.get("OSMOSE_JAR"),
    reason="OSMOSE_JAR env var not set; cross-engine test skipped",
)
def test_objective_values_match_between_engines(tmp_path):
    """Run the same 3 candidates through both engines; assert objective
    values match within 1 OoM (the project's Python/Java parity tolerance).
    """
    jar_path = Path(os.environ["OSMOSE_JAR"])
    py_problem = _make_problem(tmp_path / "py", use_java_engine=False)
    java_problem = _make_problem(
        tmp_path / "java", use_java_engine=True, jar_path=jar_path
    )

    rng = np.random.default_rng(42)
    X = rng.uniform(0.1, 0.5, size=(3, 1))
    py_out: dict = {}
    java_out: dict = {}
    py_problem._evaluate(X, py_out)
    java_problem._evaluate(X, java_out)

    for i in range(3):
        py_val = py_out["F"][i, 0]
        java_val = java_out["F"][i, 0]
        assert 0.1 <= py_val / java_val <= 10.0, (
            f"Candidate {i}: Python objective {py_val:.4g} vs Java {java_val:.4g} "
            f"differs by >1 OoM"
        )


def test_python_engine_failure_returns_inf(tmp_path, monkeypatch):
    """If the Python engine raises ValueError/KeyError/RuntimeError on a
    bad candidate, _run_single must return [inf]*n_obj (not crash the
    NSGA-II loop). Mirrors the Java path contract where a non-zero exit
    is silently scored as inf.
    """
    problem = _make_problem(tmp_path, use_java_engine=False)
    with mock.patch(
        "osmose.engine.PythonEngine.run_in_memory",
        side_effect=ValueError("bad config"),
    ):
        result = problem._run_single(
            {"mortality.fishing.rate.sp0": "0.3"}, run_id=0
        )
    assert result == [float("inf")], f"Expected [inf], got {result}"
