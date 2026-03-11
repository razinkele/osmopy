# osmose/calibration/sensitivity.py
"""Sobol sensitivity analysis for OSMOSE parameters."""

from __future__ import annotations

import numpy as np
from SALib.sample import sobol as sobol_sample  # type: ignore[import-untyped]
from SALib.analyze import sobol as sobol_analyze  # type: ignore[import-untyped]


class SensitivityAnalyzer:
    """Sobol sensitivity analysis for OSMOSE parameters.

    Workflow:
    1. Create analyzer with parameter definitions
    2. generate_samples() -- Saltelli sampling
    3. (user evaluates OSMOSE for each sample)
    4. analyze(Y) -- Compute Sobol indices
    """

    def __init__(self, param_names: list[str], param_bounds: list[tuple[float, float]]):
        self.problem = {
            "num_vars": len(param_names),
            "names": param_names,
            "bounds": param_bounds,
        }

    def generate_samples(self, n_base: int = 256) -> np.ndarray:
        """Generate Saltelli samples for Sobol analysis.

        Total samples = n_base * (2 * num_vars + 2).
        """
        return sobol_sample.sample(self.problem, n_base)

    def analyze(self, Y: np.ndarray) -> dict:
        """Compute Sobol sensitivity indices.

        Args:
            Y: Output values for each sample (1D array).

        Returns:
            Dict with 'S1' (first-order), 'ST' (total-order),
            'S1_conf', 'ST_conf', and 'param_names'.
        """
        result = sobol_analyze.analyze(self.problem, Y)
        return {
            "S1": result["S1"],
            "ST": result["ST"],
            "S1_conf": result["S1_conf"],
            "ST_conf": result["ST_conf"],
            "param_names": self.problem["names"],
        }
