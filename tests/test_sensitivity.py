import numpy as np

from osmose.calibration.sensitivity import SensitivityAnalyzer


def test_generate_samples_shape():
    sa = SensitivityAnalyzer(
        param_names=["k", "linf"],
        param_bounds=[(0.1, 0.5), (10, 200)],
    )
    samples = sa.generate_samples(n_base=64)
    # Saltelli: n_base * (2 * num_vars + 2) = 64 * 6 = 384
    assert samples.shape == (384, 2)


def test_generate_samples_within_bounds():
    sa = SensitivityAnalyzer(
        param_names=["x"],
        param_bounds=[(0, 10)],
    )
    samples = sa.generate_samples(n_base=32)
    assert np.all(samples >= 0) and np.all(samples <= 10)


def test_analyze_returns_indices():
    sa = SensitivityAnalyzer(
        param_names=["x1", "x2"],
        param_bounds=[(0, 1), (0, 1)],
    )
    samples = sa.generate_samples(n_base=64)
    # Simple test function: y = x1 + 0.1*x2 (x1 should have higher sensitivity)
    Y = samples[:, 0] + 0.1 * samples[:, 1]
    result = sa.analyze(Y)
    assert "S1" in result
    assert "ST" in result
    assert len(result["S1"]) == 2
    # x1 should have higher first-order index than x2
    assert result["S1"][0] > result["S1"][1]
