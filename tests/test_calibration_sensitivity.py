"""Tests for multi-objective Sobol sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.sensitivity import SensitivityAnalyzer


@pytest.fixture()
def analyzer():
    return SensitivityAnalyzer(
        param_names=["a", "b"],
        param_bounds=[(0, 1), (0, 1)],
    )


class TestAnalyze1D:
    """Backward-compatible 1D path."""

    def test_returns_expected_keys(self, analyzer: SensitivityAnalyzer) -> None:
        samples = analyzer.generate_samples(n_base=64)
        Y = np.sum(samples, axis=1)  # Simple function: y = a + b
        result = analyzer.analyze(Y)
        assert "S1" in result
        assert "ST" in result
        assert "S1_conf" in result
        assert "ST_conf" in result
        assert "param_names" in result
        assert "objective_names" not in result  # 1D path — no objective_names key

    def test_s1_shape(self, analyzer: SensitivityAnalyzer) -> None:
        samples = analyzer.generate_samples(n_base=64)
        Y = np.sum(samples, axis=1)
        result = analyzer.analyze(Y)
        assert result["S1"].shape == (2,)  # 2 params
        assert result["ST"].shape == (2,)


class TestAnalyze2D:
    """New multi-objective 2D path."""

    def test_returns_objective_names(self, analyzer: SensitivityAnalyzer) -> None:
        samples = analyzer.generate_samples(n_base=64)
        Y = np.column_stack([
            np.sum(samples, axis=1),
            np.prod(samples, axis=1),
        ])
        result = analyzer.analyze(Y, objective_names=["sum", "product"])
        assert result["objective_names"] == ["sum", "product"]
        assert result["n_objectives"] == 2

    def test_default_objective_names(self, analyzer: SensitivityAnalyzer) -> None:
        samples = analyzer.generate_samples(n_base=64)
        Y = np.column_stack([samples[:, 0], samples[:, 1]])
        result = analyzer.analyze(Y)
        assert result["objective_names"] == ["obj_0", "obj_1"]

    def test_s1_shape_2d(self, analyzer: SensitivityAnalyzer) -> None:
        samples = analyzer.generate_samples(n_base=64)
        Y = np.column_stack([
            np.sum(samples, axis=1),
            samples[:, 0] ** 2,
        ])
        result = analyzer.analyze(Y)
        assert result["S1"].shape == (2, 2)  # (n_obj, n_params)
        assert result["ST"].shape == (2, 2)

    def test_n_objectives_key(self, analyzer: SensitivityAnalyzer) -> None:
        samples = analyzer.generate_samples(n_base=64)
        Y = np.column_stack([samples[:, 0], samples[:, 1], samples[:, 0] + samples[:, 1]])
        result = analyzer.analyze(Y)
        assert result["n_objectives"] == 3
        assert result["S1"].shape == (3, 2)
