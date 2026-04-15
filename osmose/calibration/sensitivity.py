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
    2. generate_samples() -- Sobol sampling (Saltelli's extension)
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
        """Generate Sobol samples for sensitivity analysis.

        Total samples = n_base * (2 * num_vars + 2).
        """
        return sobol_sample.sample(self.problem, n_base)

    def analyze(self, Y: np.ndarray, objective_names: list[str] | None = None) -> dict:
        """Compute Sobol sensitivity indices for one or more objectives.

        Args:
            Y: 1D array (single objective) or 2D (n_samples, n_objectives).
            objective_names: Labels per objective. Defaults to ["obj_0", ...].

        Returns:
            1D: dict with S1, ST, S1_conf, ST_conf, param_names.
            2D: same keys with arrays of shape (n_obj, n_params), plus
                objective_names and n_objectives.
        """
        if Y.ndim == 1:
            result = sobol_analyze.analyze(self.problem, Y)
            return {
                "S1": result["S1"],
                "ST": result["ST"],
                "S1_conf": result["S1_conf"],
                "ST_conf": result["ST_conf"],
                "param_names": self.problem["names"],
            }

        n_obj = Y.shape[1]
        if objective_names is None:
            objective_names = [f"obj_{i}" for i in range(n_obj)]

        all_s1, all_st, all_s1_conf, all_st_conf = [], [], [], []
        for col in range(n_obj):
            result = sobol_analyze.analyze(self.problem, Y[:, col])
            all_s1.append(result["S1"])
            all_st.append(result["ST"])
            all_s1_conf.append(result["S1_conf"])
            all_st_conf.append(result["ST_conf"])

        return {
            "S1": np.array(all_s1),
            "ST": np.array(all_st),
            "S1_conf": np.array(all_s1_conf),
            "ST_conf": np.array(all_st_conf),
            "param_names": self.problem["names"],
            "objective_names": objective_names,
            "n_objectives": n_obj,
        }
