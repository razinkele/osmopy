import numpy as np
import pytest

from osmose.calibration.surrogate import SurrogateCalibrator


def test_generate_samples_shape():
    cal = SurrogateCalibrator(param_bounds=[(0, 1), (10, 100)], n_objectives=1)
    samples = cal.generate_samples(n_samples=50)
    assert samples.shape == (50, 2)


def test_generate_samples_within_bounds():
    cal = SurrogateCalibrator(param_bounds=[(0, 1), (10, 100)])
    samples = cal.generate_samples(n_samples=100)
    assert np.all(samples[:, 0] >= 0) and np.all(samples[:, 0] <= 1)
    assert np.all(samples[:, 1] >= 10) and np.all(samples[:, 1] <= 100)


def test_fit_and_predict():
    cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_objectives=1)
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    y = np.sin(X).ravel()
    cal.fit(X, y)
    means, stds = cal.predict(np.array([[5.0]]))
    assert means.shape == (1, 1)
    assert stds.shape == (1, 1)
    # Prediction should be close to sin(5)
    assert abs(means[0, 0] - np.sin(5)) < 0.5


def test_predict_before_fit_raises():
    cal = SurrogateCalibrator(param_bounds=[(0, 1)])
    with pytest.raises(RuntimeError, match="fit"):
        cal.predict(np.array([[0.5]]))


def test_find_optimum():
    cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_objectives=1)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = (X - 5) ** 2  # Minimum at x=5
    cal.fit(X, y.ravel())
    result = cal.find_optimum(n_candidates=1000)
    assert abs(result["params"][0] - 5.0) < 1.0  # Should be near 5


def test_find_optimum_before_fit_raises():
    cal = SurrogateCalibrator(param_bounds=[(0, 1)])
    with pytest.raises(RuntimeError, match="fit"):
        cal.find_optimum()


def test_multi_objective_fit():
    cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_objectives=2)
    X = np.linspace(0, 10, 30).reshape(-1, 1)
    y = np.column_stack([X.ravel() ** 2, (10 - X.ravel()) ** 2])
    cal.fit(X, y)
    means, _stds = cal.predict(np.array([[5.0]]))
    assert means.shape == (1, 2)


def test_multi_objective_find_optimum():
    cal = SurrogateCalibrator(param_bounds=[(0, 10)], n_objectives=2)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    # Both objectives minimize near x=5
    y = np.column_stack([(X.ravel() - 5) ** 2, (X.ravel() - 5) ** 2])
    cal.fit(X, y)
    result = cal.find_optimum(n_candidates=1000)
    assert "params" in result
    assert result["predicted_objectives"].shape == (2,)
    assert result["predicted_uncertainty"].shape == (2,)
    assert abs(result["params"][0] - 5.0) < 1.5
