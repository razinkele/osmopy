"""Tests for multi-phase calibration."""

from unittest.mock import MagicMock

from osmose.calibration.multiphase import CalibrationPhase, MultiPhaseCalibrator
from osmose.calibration.problem import FreeParameter


def _make_phase(name, keys, algorithm="Nelder-Mead"):
    params = [FreeParameter(key=k, lower_bound=0.0, upper_bound=1.0) for k in keys]
    return CalibrationPhase(name=name, free_params=params, algorithm=algorithm)


def test_single_phase():
    phase = _make_phase("phase1", ["mortality.natural.rate.sp0"])
    cal = MultiPhaseCalibrator(phases=[phase])
    cal._optimize_phase = MagicMock(return_value={"mortality.natural.rate.sp0": 0.5})

    results = cal.run(work_dir="/tmp/test", objective_fn=lambda x: 0.0)

    assert len(results) == 1
    assert results[0] == {"mortality.natural.rate.sp0": 0.5}
    cal._optimize_phase.assert_called_once()


def test_two_phases_sequential():
    p1 = _make_phase("phase1", ["param.a"])
    p2 = _make_phase("phase2", ["param.b"])
    cal = MultiPhaseCalibrator(phases=[p1, p2])

    call_log = []

    def mock_optimize(phase, fixed_params, work_dir, objective_fn=None):
        call_log.append((phase.name, dict(fixed_params)))
        if phase.name == "phase1":
            return {"param.a": 0.3}
        return {"param.b": 0.7}

    cal._optimize_phase = mock_optimize

    results = cal.run(work_dir="/tmp/test", objective_fn=lambda x: 0.0)

    assert len(results) == 2
    assert results[0] == {"param.a": 0.3}
    assert results[1] == {"param.b": 0.7}
    # Phase 2 should receive phase 1 results as fixed params
    assert call_log[0] == ("phase1", {})
    assert call_log[1] == ("phase2", {"param.a": 0.3})


def test_three_phases_param_accumulation():
    p1 = _make_phase("p1", ["a"])
    p2 = _make_phase("p2", ["b"])
    p3 = _make_phase("p3", ["c"])
    cal = MultiPhaseCalibrator(phases=[p1, p2, p3])

    call_log = []

    def mock_optimize(phase, fixed_params, work_dir, objective_fn=None):
        call_log.append((phase.name, dict(fixed_params)))
        return {phase.free_params[0].key: 1.0}

    cal._optimize_phase = mock_optimize

    results = cal.run(work_dir="/tmp/test", objective_fn=lambda x: 0.0)

    assert len(results) == 3
    # Phase 3 should have accumulated params from phases 1 and 2
    assert call_log[2] == ("p3", {"a": 1.0, "b": 1.0})


def test_progress_callback():
    phase = _make_phase("phase1", ["x"])
    cal = MultiPhaseCalibrator(phases=[phase])
    cal._optimize_phase = MagicMock(return_value={"x": 0.5})

    progress_calls = []
    cal.run(
        work_dir="/tmp/test",
        objective_fn=lambda x: 0.0,
        on_progress=lambda msg: progress_calls.append(msg),
    )

    assert len(progress_calls) > 0


def test_phase_dataclass_defaults():
    params = [FreeParameter(key="k", lower_bound=0, upper_bound=1)]
    phase = CalibrationPhase(name="test", free_params=params)
    assert phase.algorithm == "Nelder-Mead"
    assert phase.max_iter == 100
    assert phase.n_replicates == 1


def test_optimize_phase_runs_real_objective(tmp_path):
    """_optimize_phase should use the provided objective_fn, not a stub."""
    phase = CalibrationPhase(
        name="test",
        free_params=[FreeParameter(key="x", lower_bound=-5, upper_bound=5)],
        algorithm="Nelder-Mead",
        max_iter=50,
    )

    def real_objective(x):
        return float((x[0] - 2.0) ** 2)

    calibrator = MultiPhaseCalibrator(phases=[phase])
    result = calibrator._optimize_phase(phase, {}, str(tmp_path), objective_fn=real_objective)
    assert abs(result["x"] - 2.0) < 0.5


def test_optimize_phase_differential_evolution(tmp_path):
    """differential_evolution branch should work with provided objective."""
    phase = CalibrationPhase(
        name="test",
        free_params=[FreeParameter(key="x", lower_bound=-5, upper_bound=5)],
        algorithm="differential_evolution",
        max_iter=20,
    )

    def real_objective(x):
        return float((x[0] - 1.0) ** 2)

    calibrator = MultiPhaseCalibrator(phases=[phase])
    result = calibrator._optimize_phase(phase, {}, str(tmp_path), objective_fn=real_objective)
    assert abs(result["x"] - 1.0) < 1.0


def test_optimize_phase_raises_without_objective(tmp_path):
    """_optimize_phase should raise ValueError when no objective_fn is provided."""
    import pytest

    phase = CalibrationPhase(
        name="test",
        free_params=[FreeParameter(key="x", lower_bound=-5, upper_bound=5)],
    )

    calibrator = MultiPhaseCalibrator(phases=[phase])
    with pytest.raises(ValueError, match="objective_fn is required"):
        calibrator._optimize_phase(phase, {}, str(tmp_path))
