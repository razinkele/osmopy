# osmose/calibration/problem.py
"""OSMOSE calibration as a pymoo optimization problem."""

from __future__ import annotations

import enum
import hashlib
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from osmose.schema.registry import ParameterRegistry

import numpy as np
from pymoo.core.problem import Problem  # type: ignore[import-untyped]

from osmose.logging import setup_logging

_OSMOSE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9._]*$")
_OSMOSE_VALUE_PATTERN = re.compile(r"^[\w.+\-eE/]+$")

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
        enable_cache: bool = False,
        cache_dir: Path | None = None,
        registry: "ParameterRegistry | None" = None,
    ):
        self.free_params = free_params
        self.objective_fns = objective_fns
        self.base_config_path = base_config_path
        self.jar_path = jar_path
        self.work_dir = work_dir
        self.java_cmd = java_cmd
        self.n_parallel = max(1, n_parallel)
        self._enable_cache = enable_cache
        self._cache_dir = cache_dir or (self.work_dir / ".cache")
        self._registry = registry
        self._cache_hits = 0
        self._cache_misses = 0
        # Pre-compute base config hash for cache keys
        self._base_config_hash = ""
        if enable_cache and base_config_path.exists():
            self._base_config_hash = hashlib.sha256(base_config_path.read_bytes()).hexdigest()[:16]

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
        for key, value in overrides.items():
            if not _OSMOSE_KEY_PATTERN.match(key):
                raise ValueError(f"Invalid override key: {key!r}")
            val_str = str(value)
            if not _OSMOSE_VALUE_PATTERN.match(val_str):
                raise ValueError(
                    f"Invalid override value for {key!r}: {val_str!r} — "
                    "only alphanumeric, '.', '+', '-', 'e', 'E', '/' allowed"
                )

        # Schema validation
        self._validate_overrides(overrides)

        # Cache check
        if self._enable_cache:
            key = self._cache_key(overrides)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_dir / f"{key}.json"
            if cache_file.exists():
                self._cache_hits += 1
                cached = json.loads(cache_file.read_text())
                return cached["objectives"]

        # Create isolated output directory
        run_dir = self.work_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir = run_dir / "output"

        cmd = [self.java_cmd, "-jar", str(self.jar_path), str(self.base_config_path)]
        cmd.append(f"-Poutput.dir.path={output_dir}")
        for key, value in overrides.items():
            cmd.append(f"-P{key}={value}")

        # 1-hour timeout per evaluation; consider making configurable for long simulations
        result = subprocess.run(cmd, capture_output=True, timeout=3600)

        if result.returncode != 0:
            stderr_msg = result.stderr.decode(errors="replace")[:500] if result.stderr else ""
            _log.warning(
                "OSMOSE run %d failed (exit %d): %s", run_id, result.returncode, stderr_msg
            )
            return [float("inf")] * self.n_obj

        # Compute objectives
        from osmose.results import OsmoseResults

        with OsmoseResults(output_dir, strict=False) as results:
            obj_values = []
            for fn in self.objective_fns:
                obj_values.append(fn(results))

        # Cache write (atomic rename)
        if self._enable_cache:
            key = self._cache_key(overrides)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_dir / f"{key}.json"
            fd, tmp_file = tempfile.mkstemp(dir=str(self._cache_dir), suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump({"objectives": obj_values}, f)
                os.replace(tmp_file, str(cache_file))
            except OSError:
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass
            self._cache_misses += 1

        return obj_values

    def cleanup_run(self, run_id: int) -> None:
        """Remove a completed run directory to reclaim disk space."""
        run_dir = self.work_dir / f"run_{run_id}"
        if run_dir.is_dir():
            import shutil

            shutil.rmtree(run_dir, ignore_errors=True)

    def _cache_key(self, overrides: dict[str, str]) -> str:
        """Deterministic hash of overrides + JAR mtime + base config hash."""
        parts = sorted(overrides.items())
        try:
            jar_mtime = str(self.jar_path.stat().st_mtime)
        except OSError:
            jar_mtime = "missing"
        raw = f"{parts}|{jar_mtime}|{self._base_config_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def cache_stats(self) -> dict:
        """Returns cache hit/miss counts and size."""
        size_mb = 0.0
        if self._cache_dir.is_dir():
            size_mb = sum(f.stat().st_size for f in self._cache_dir.iterdir()) / (1024 * 1024)
        return {"hits": self._cache_hits, "misses": self._cache_misses, "size_mb": size_mb}

    def clear_cache(self) -> None:
        """Remove all cached evaluations."""
        if self._cache_dir.is_dir():
            for f in self._cache_dir.iterdir():
                f.unlink(missing_ok=True)

    def _validate_overrides(self, overrides: dict[str, str]) -> None:
        """Validate overrides against the schema registry.

        Note: overrides values are strings (from OSMOSE config format).
        We must coerce numeric values to float/int before passing to the
        registry, since validate_value() compares with ``<`` / ``>``.
        """
        if self._registry is None:
            return
        from osmose.schema.base import ParamType

        coerced: dict[str, object] = {}
        for k, v in overrides.items():
            field = self._registry.match_field(k)
            if field and field.param_type == ParamType.FLOAT:
                coerced[k] = float(v)
            elif field and field.param_type == ParamType.INT:
                coerced[k] = int(v)
            else:
                coerced[k] = v
        errors = self._registry.validate(coerced)
        if errors:
            raise ValueError(
                f"Override validation failed ({len(errors)} errors):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
