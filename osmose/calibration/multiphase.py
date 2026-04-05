# osmose/calibration/multiphase.py
"""Multi-phase sequential calibration for OSMOSE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize

from osmose.calibration.problem import FreeParameter


@dataclass
class CalibrationPhase:
    """Definition of a single calibration phase."""

    free_params: list[FreeParameter]
    name: str = ""
    algorithm: str = "Nelder-Mead"
    max_iter: int = 100
    n_replicates: int = 1

    def __post_init__(self) -> None:
        if not self.free_params:
            raise ValueError("CalibrationPhase.free_params must not be empty")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")


class MultiPhaseCalibrator:
    """Runs calibration phases sequentially, passing optimized params forward."""

    def __init__(self, phases: list[CalibrationPhase]):
        self.phases = phases

    def run(
        self,
        work_dir: Path | str,
        objective_fn: Callable[[np.ndarray], float] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[dict[str, float]]:
        """Run all phases sequentially.

        Output of phase N becomes fixed params for phase N+1.

        Args:
            work_dir: Working directory for optimization outputs.
            on_progress: Optional callback for progress messages.

        Returns:
            List of dicts, one per phase, with optimized param values.
        """
        results: list[dict[str, float]] = []
        fixed_params: dict[str, float] = {}

        for i, phase in enumerate(self.phases):
            if on_progress:
                on_progress(f"Starting phase {i + 1}/{len(self.phases)}: {phase.name}")

            optimized = self._optimize_phase(phase, fixed_params, work_dir, objective_fn)
            results.append(optimized)
            fixed_params.update(optimized)

            if on_progress:
                on_progress(f"Completed phase {phase.name}: {optimized}")

        return results

    def _optimize_phase(
        self,
        phase: CalibrationPhase,
        fixed_params: dict[str, float],
        work_dir: Path | str,
        objective_fn: Callable[[np.ndarray], float] | None = None,
    ) -> dict[str, float]:
        """Optimize a single phase using scipy.optimize.

        Args:
            phase: The calibration phase to run.
            fixed_params: Parameters fixed from previous phases.
            work_dir: Working directory.

        Returns:
            Dict of optimized parameter values for this phase.
        """
        bounds = [(fp.lower_bound, fp.upper_bound) for fp in phase.free_params]
        x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])

        if objective_fn is None:
            raise ValueError(
                "objective_fn is required. Pass a callable that maps parameter vector to scalar."
            )

        def objective(x):
            return objective_fn(x)

        if phase.algorithm == "differential_evolution":
            result = differential_evolution(objective, bounds, maxiter=phase.max_iter)
        else:
            result = minimize(
                objective,
                x0,
                method=phase.algorithm,
                options={"maxiter": phase.max_iter},
            )

        return {fp.key: float(result.x[j]) for j, fp in enumerate(phase.free_params)}
