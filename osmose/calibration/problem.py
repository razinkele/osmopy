# osmose/calibration/problem.py
"""OSMOSE calibration as a pymoo optimization problem."""

from __future__ import annotations

import enum
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from pymoo.core.problem import Problem  # type: ignore[import-untyped]

from osmose.logging import setup_logging

_OSMOSE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9._]*$")

_log = setup_logging("osmose.calibration")

_expected_errors = (
    subprocess.TimeoutExpired,
    subprocess.CalledProcessError,
    FileNotFoundError,
    OSError,
)


class Transform(enum.Enum):
    """Parameter space transform for calibration."""

    LINEAR = "linear"
    LOG = "log"


@dataclass
class FreeParameter:
    """A parameter to optimize during calibration."""

    key: str  # OSMOSE parameter key
    lower_bound: float
    upper_bound: float
    transform: Transform = Transform.LINEAR

    def __post_init__(self) -> None:
        if not isinstance(self.transform, Transform):
            raise TypeError(
                f"transform must be a Transform enum, got {type(self.transform).__name__}"
            )
        if self.lower_bound >= self.upper_bound:
            raise ValueError(
                f"lower_bound ({self.lower_bound}) must be less than "
                f"upper_bound ({self.upper_bound})"
            )


class OsmoseCalibrationProblem(Problem):
    """Multi-objective optimization problem for OSMOSE.

    Each evaluation:
    1. Maps candidate parameter vector to OSMOSE config overrides
    2. Runs OSMOSE with those overrides
    3. Reads results and computes objective values
    """

    def __init__(
        self,
        free_params: list[FreeParameter],
        objective_fns: list[Callable],
        base_config_path: Path,
        jar_path: Path,
        work_dir: Path,
        java_cmd: str = "java",
        n_parallel: int = 1,
    ):
        self.free_params = free_params
        self.objective_fns = objective_fns
        self.base_config_path = base_config_path
        self.jar_path = jar_path
        self.work_dir = work_dir
        self.java_cmd = java_cmd
        self.n_parallel = max(1, n_parallel)

        xl = np.array([fp.lower_bound for fp in free_params])
        xu = np.array([fp.upper_bound for fp in free_params])

        super().__init__(
            n_var=len(free_params),
            n_obj=len(objective_fns),
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate a population of candidates.

        X has shape (pop_size, n_var). Each row is a candidate.
        If n_parallel > 1, candidates are evaluated concurrently using threads.
        """
        _log.info("Evaluating %d candidates (parallel=%d)", X.shape[0], self.n_parallel)
        F = np.full((X.shape[0], self.n_obj), np.inf)

        if self.n_parallel > 1:
            with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
                futures = {
                    executor.submit(self._evaluate_candidate, i, params): i
                    for i, params in enumerate(X)
                }
                for future in futures:
                    i = futures[future]
                    try:
                        objectives = future.result()
                        for k, obj_val in enumerate(objectives):
                            F[i, k] = obj_val
                    except _expected_errors as exc:
                        _log.warning("Candidate %d failed (expected): %s", i, exc)
        else:
            for i, params in enumerate(X):
                try:
                    objectives = self._evaluate_candidate(i, params)
                    for k, obj_val in enumerate(objectives):
                        F[i, k] = obj_val
                except _expected_errors as exc:
                    _log.warning("Candidate %d failed (expected): %s", i, exc)

        # Abort if >50% of candidates failed (all objectives inf)
        n_inf = np.all(np.isinf(F), axis=1).sum()
        if n_inf > len(F) * 0.5:
            raise RuntimeError(
                f"Calibration aborted: {n_inf}/{len(F)} candidates failed "
                f"(>50% returned inf). Check JAR path and config validity."
            )

        out["F"] = F

    def _evaluate_candidate(self, i: int, params: np.ndarray) -> list[float]:
        """Evaluate a single candidate and return objective values."""
        overrides = {}
        for j, fp in enumerate(self.free_params):
            val = params[j]
            if fp.transform == Transform.LOG:
                val = 10**val
            overrides[fp.key] = str(val)

        return self._run_single(overrides, run_id=i)

    def _run_single(self, overrides: dict[str, str], run_id: int) -> list[float]:
        """Run OSMOSE synchronously with overrides and return objective values.

        Uses subprocess (synchronous) since pymoo evaluates in a loop.
        """
        # Validate override keys before constructing the command
        for key in overrides:
            if not _OSMOSE_KEY_PATTERN.match(key):
                raise ValueError(f"Invalid override key: {key!r}")

        # Create isolated output directory
        run_dir = self.work_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir = run_dir / "output"

        cmd = [self.java_cmd, "-jar", str(self.jar_path), str(self.base_config_path)]
        cmd.append(f"-Poutput.dir.path={output_dir}")
        for key, value in overrides.items():
            cmd.append(f"-P{key}={value}")

        result = subprocess.run(cmd, capture_output=True, timeout=3600)

        if result.returncode != 0:
            stderr_msg = result.stderr.decode(errors="replace")[:500] if result.stderr else ""
            _log.warning(
                "OSMOSE run %d failed (exit %d): %s", run_id, result.returncode, stderr_msg
            )
            return [float("inf")] * self.n_obj

        # Compute objectives
        from osmose.results import OsmoseResults

        results = OsmoseResults(output_dir)
        obj_values = []
        for fn in self.objective_fns:
            obj_values.append(fn(results))

        return obj_values
