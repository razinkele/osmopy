from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem


def test_free_parameter():
    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=0.5)
    assert fp.key == "species.k.sp0"
    assert fp.transform == "linear"


def test_problem_dimensions():
    params = [
        FreeParameter("species.k.sp0", 0.1, 0.5),
        FreeParameter("species.linf.sp0", 10, 200),
    ]
    problem = OsmoseCalibrationProblem(
        free_params=params,
        objective_fns=[lambda r: 0.0, lambda r: 0.0],
        base_config_path=Path("/tmp/fake"),
        jar_path=Path("/tmp/fake.jar"),
        work_dir=Path("/tmp/work"),
    )
    assert problem.n_var == 2
    assert problem.n_obj == 2
    assert np.array_equal(problem.xl, np.array([0.1, 10]))
    assert np.array_equal(problem.xu, np.array([0.5, 200]))


# --- Helper to build a problem for mocked tests ---


def _make_problem(tmp_path, objective_fns=None, free_params=None):
    if free_params is None:
        free_params = [
            FreeParameter("species.k.sp0", 0.1, 0.5),
            FreeParameter("species.linf.sp0", 10, 200),
        ]
    if objective_fns is None:
        objective_fns = [lambda r: 1.0, lambda r: 2.0]
    return OsmoseCalibrationProblem(
        free_params=free_params,
        objective_fns=objective_fns,
        base_config_path=tmp_path / "config",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path / "work",
    )


# --- _run_single tests ---


@patch("subprocess.run")
@patch("osmose.results.OsmoseResults")
def test_run_single_success(mock_results_cls, mock_subprocess, tmp_path):
    """Successful Java run returns computed objective values."""
    mock_subprocess.return_value = MagicMock(returncode=0)
    mock_results_instance = MagicMock()
    mock_results_cls.return_value = mock_results_instance

    problem = _make_problem(tmp_path, objective_fns=[lambda r: 0.5, lambda r: 1.5])
    result = problem._run_single({"species.k.sp0": "0.3"}, run_id=0)

    assert result == [0.5, 1.5]
    mock_subprocess.assert_called_once()
    cmd = mock_subprocess.call_args[0][0]
    assert "-Pspecies.k.sp0=0.3" in cmd
    assert str(tmp_path / "fake.jar") in cmd


@patch("subprocess.run")
def test_run_single_nonzero_returncode(mock_subprocess, tmp_path):
    """Failed Java run returns inf for all objectives."""
    mock_subprocess.return_value = MagicMock(returncode=1)

    problem = _make_problem(tmp_path)
    result = problem._run_single({}, run_id=0)

    assert result == [float("inf"), float("inf")]


@patch("subprocess.run")
@patch("osmose.results.OsmoseResults")
def test_run_single_creates_run_dir(mock_results_cls, mock_subprocess, tmp_path):
    """_run_single creates an isolated run directory."""
    mock_subprocess.return_value = MagicMock(returncode=0)
    mock_results_cls.return_value = MagicMock()

    problem = _make_problem(tmp_path, objective_fns=[lambda r: 0.0])
    problem._run_single({}, run_id=7)

    run_dir = tmp_path / "work" / "run_7"
    assert run_dir.exists()


@patch("subprocess.run")
@patch("osmose.results.OsmoseResults")
def test_run_single_passes_output_dir(mock_results_cls, mock_subprocess, tmp_path):
    """Output dir override is passed to Java and to OsmoseResults."""
    mock_subprocess.return_value = MagicMock(returncode=0)
    mock_results_cls.return_value = MagicMock()

    problem = _make_problem(tmp_path, objective_fns=[lambda r: 0.0])
    problem._run_single({}, run_id=3)

    cmd = mock_subprocess.call_args[0][0]
    expected_output = str(tmp_path / "work" / "run_3" / "output")
    assert f"-Poutput.dir.path={expected_output}" in cmd
    mock_results_cls.assert_called_once_with(Path(expected_output))


# --- _evaluate tests ---


def test_evaluate_linear_params(tmp_path):
    """_evaluate maps linear parameters directly to overrides."""
    problem = _make_problem(tmp_path)
    X = np.array([[0.3, 100.0]])
    out = {}

    with patch.object(problem, "_run_single", return_value=[1.0, 2.0]) as mock_run:
        problem._evaluate(X, out)

    overrides = (
        mock_run.call_args[1]["overrides"]
        if "overrides" in (mock_run.call_args[1] or {})
        else mock_run.call_args[0][0]
    )
    assert overrides == {"species.k.sp0": "0.3", "species.linf.sp0": "100.0"}
    assert np.array_equal(out["F"], np.array([[1.0, 2.0]]))


def test_evaluate_log_transform(tmp_path):
    """Log-transformed parameters are exponentiated (10**val)."""
    params = [FreeParameter("species.k.sp0", -2, 0, transform="log")]
    problem = _make_problem(tmp_path, objective_fns=[lambda r: 0.0], free_params=params)
    X = np.array([[-1.0]])  # 10**-1 = 0.1
    out = {}

    with patch.object(problem, "_run_single", return_value=[0.0]) as mock_run:
        problem._evaluate(X, out)

    overrides = mock_run.call_args[0][0]
    assert float(overrides["species.k.sp0"]) == 0.1


def test_evaluate_multiple_candidates(tmp_path):
    """_evaluate processes all rows in the population matrix."""
    problem = _make_problem(tmp_path, objective_fns=[lambda r: 0.0])
    X = np.array([[0.2, 50.0], [0.3, 100.0], [0.4, 150.0]])
    out = {}

    with patch.object(problem, "_run_single", return_value=[0.0]) as mock_run:
        problem._evaluate(X, out)

    assert mock_run.call_count == 3
    assert out["F"].shape == (3, 1)


def test_evaluate_exception_leaves_inf(tmp_path):
    """If _run_single raises an expected error, that candidate gets inf objectives."""
    problem = _make_problem(tmp_path)
    X = np.array([[0.3, 100.0], [0.4, 150.0]])
    out = {}

    def side_effect(overrides, run_id):
        if run_id == 0:
            raise OSError("disk full")
        return [1.0, 2.0]

    with patch.object(problem, "_run_single", side_effect=side_effect):
        problem._evaluate(X, out)

    assert np.all(np.isinf(out["F"][0]))  # First candidate failed
    assert np.array_equal(out["F"][1], [1.0, 2.0])  # Second succeeded


def test_n_parallel_default(tmp_path):
    """n_parallel defaults to 1."""
    problem = _make_problem(tmp_path)
    assert problem.n_parallel == 1


def test_n_parallel_stored(tmp_path):
    """n_parallel is stored correctly when passed explicitly."""
    params = [
        FreeParameter("species.k.sp0", 0.1, 0.5),
        FreeParameter("species.linf.sp0", 10, 200),
    ]
    problem = OsmoseCalibrationProblem(
        free_params=params,
        objective_fns=[lambda r: 0.0, lambda r: 0.0],
        base_config_path=tmp_path / "config",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path / "work",
        n_parallel=4,
    )
    assert problem.n_parallel == 4


def test_n_parallel_clamps_to_one(tmp_path):
    """n_parallel below 1 is clamped to 1."""
    params = [FreeParameter("species.k.sp0", 0.1, 0.5)]
    problem = OsmoseCalibrationProblem(
        free_params=params,
        objective_fns=[lambda r: 0.0],
        base_config_path=tmp_path / "config",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path / "work",
        n_parallel=0,
    )
    assert problem.n_parallel == 1


def test_evaluate_parallel(tmp_path):
    """Parallel evaluation with n_parallel>1 produces the same results."""
    problem = _make_problem(tmp_path)
    problem.n_parallel = 4
    X = np.array([[0.2, 50.0], [0.3, 100.0], [0.4, 150.0]])
    out = {}

    with patch.object(problem, "_run_single", return_value=[1.0, 2.0]) as mock_run:
        problem._evaluate(X, out)

    assert mock_run.call_count == 3
    assert out["F"].shape == (3, 2)
    assert np.array_equal(out["F"], np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]))


def test_evaluate_logs_candidate_failure(tmp_path, caplog):
    """Silent except:pass should now log failures."""
    import logging

    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=1.0)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 1.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
    )
    with patch.object(problem, "_evaluate_candidate", side_effect=OSError("boom")):
        X = np.array([[0.5]])
        out = {}
        with caplog.at_level(logging.WARNING):
            problem._evaluate(X, out)
        assert np.isinf(out["F"][0, 0])
        assert "boom" in caplog.text


def test_evaluate_propagates_unexpected_exceptions(tmp_path):
    """Unexpected errors (TypeError, etc.) should propagate, not be swallowed."""
    import pytest

    fp = FreeParameter(key="test.param", lower_bound=0, upper_bound=1)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 1.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
    )

    X = np.array([[0.1], [0.5], [0.9]])
    out = {}

    with patch.object(problem, "_evaluate_candidate", side_effect=TypeError("bad objective")):
        with pytest.raises(TypeError):
            problem._evaluate(X, out)


def test_evaluate_tolerates_expected_failures(tmp_path):
    """Expected failures (OSError, etc.) are scored as inf, not propagated."""
    fp = FreeParameter(key="test.param", lower_bound=0, upper_bound=1)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 1.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
    )

    X = np.array([[0.1], [0.5], [0.9]])
    out = {}

    with patch.object(problem, "_evaluate_candidate", side_effect=OSError("disk full")):
        problem._evaluate(X, out)

    assert np.all(np.isinf(out["F"]))


def test_evaluate_parallel_handles_mixed_failures(tmp_path):
    """Parallel evaluation: one candidate fails, others succeed."""
    fp = FreeParameter(key="test.param", lower_bound=0, upper_bound=1)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 1.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
        n_parallel=2,
    )

    X = np.array([[0.1], [0.5], [0.9]])
    out = {}

    def mock_evaluate_candidate(i, params):
        if i == 1:
            raise OSError("disk full")
        return [float(params[0])]

    with patch.object(problem, "_evaluate_candidate", side_effect=mock_evaluate_candidate):
        problem._evaluate(X, out)

    # Candidate 1 should be inf, others should have values
    assert np.isinf(out["F"][1, 0])
    assert not np.isinf(out["F"][0, 0])
    assert not np.isinf(out["F"][2, 0])


def test_run_single_rejects_invalid_override_keys(tmp_path):
    """_run_single raises ValueError for keys that don't match the OSMOSE pattern."""
    import pytest

    fp = FreeParameter(key="valid.key", lower_bound=0, upper_bound=1)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 1.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="[Ii]nvalid"):
        problem._run_single({"../evil": "1.0"}, run_id=0)


def test_run_single_accepts_valid_override_keys(tmp_path):
    """_run_single accepts keys that match the OSMOSE pattern."""
    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=0.5)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 0.5],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result):
        with patch("osmose.results.OsmoseResults", return_value=MagicMock()):
            result = problem._run_single({"species.k.sp0": "0.3"}, run_id=0)
    assert result == [0.5]


def test_run_single_logs_subprocess_stderr(tmp_path, caplog):
    """Subprocess failures should log stderr content."""
    import logging

    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=1.0)
    problem = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[lambda r: 1.0],
        base_config_path=tmp_path / "config.csv",
        jar_path=tmp_path / "fake.jar",
        work_dir=tmp_path,
    )
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = b"Java OutOfMemoryError"
    with patch("subprocess.run", return_value=mock_result):
        with caplog.at_level(logging.WARNING):
            result = problem._run_single({}, run_id=0)
        assert result == [float("inf")]
        assert "OutOfMemoryError" in caplog.text
