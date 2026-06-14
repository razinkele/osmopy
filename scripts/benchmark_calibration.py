#!/usr/bin/env python3
"""Benchmark NSGA-II calibration throughput: Python vs Java engine.

Runs a small NSGA-II problem on the Baltic example using both engines
and reports wall-clock ratio. Target: >=4x speedup on a 4-thread host.

Usage:
    .venv/bin/python scripts/benchmark_calibration.py [--java JAR_PATH]

If --java is omitted, only the Python-engine run is measured.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

PROJECT_DIR = Path(__file__).resolve().parent.parent
BALTIC_CONFIG = PROJECT_DIR / "data" / "baltic" / "baltic_all-parameters.csv"


class _BenchObjective:
    """Module-level (picklable) total-biomass objective so ``--backend process`` can ship it
    across the ProcessPoolExecutor boundary (a nested local def is NOT picklable)."""

    def __call__(self, results) -> float:
        df = results.biomass()
        # Sum the species-biomass columns at the final time step. `biomass()`
        # returns columns ['Time', <species...>, 'species'] where the final
        # 'species' column is a string label ("all") — select numeric dtypes
        # only and drop 'Time' to get total biomass.
        numeric = df.select_dtypes(include="number").drop(columns=["Time"], errors="ignore")
        return float(numeric.iloc[-1].sum())


def _run_nsga2(
    use_java_engine: bool,
    jar_path: Path | None,
    n_gen: int,
    pop_size: int,
    backend: Literal["thread", "process"] = "thread",
) -> float:
    """Run a small NSGA-II problem; return wall-clock seconds."""
    from osmose.calibration import FreeParameter, Transform
    from osmose.calibration.problem import OsmoseCalibrationProblem
    from osmose.schema import build_registry
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    with tempfile.TemporaryDirectory() as d:
        problem = OsmoseCalibrationProblem(
            free_params=[
                FreeParameter("mortality.fishing.rate.sp0", 0.1, 0.5, Transform.LINEAR),
            ],
            base_config_path=BALTIC_CONFIG,
            objective_fns=[_BenchObjective()],
            registry=build_registry(),
            work_dir=Path(d),
            use_java_engine=use_java_engine,
            jar_path=jar_path,
            n_parallel=4,
            parallel_backend=backend,
            enable_cache=False,
        )

        algorithm = NSGA2(pop_size=pop_size)
        start = time.perf_counter()
        try:
            minimize(problem, algorithm, ("n_gen", n_gen), verbose=False, seed=42)
        finally:
            problem.shutdown_pool()
        return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--java", type=Path, help="Path to OSMOSE jar")
    parser.add_argument("--n-gen", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip the Python run and use --python-wallclock instead (for reusing a previous measurement).",
    )
    parser.add_argument(
        "--python-wallclock",
        type=float,
        help="Provide the Python wall-clock (seconds) when --skip-python is set.",
    )
    parser.add_argument("--backend", choices=["thread", "process"], default="thread")
    args = parser.parse_args()

    backend: Literal["thread", "process"] = "process" if args.backend == "process" else "thread"

    if not BALTIC_CONFIG.exists():
        print(f"ERROR: Baltic config not found at {BALTIC_CONFIG}", file=sys.stderr)
        return 1

    if args.skip_python and args.python_wallclock is None:
        print("ERROR: --skip-python requires --python-wallclock", file=sys.stderr)
        return 1

    print(f"Running {args.n_gen}-gen x {args.pop_size}-pop NSGA-II on Baltic")
    print()
    if args.skip_python:
        t_python = args.python_wallclock
        print(f"Python engine: {t_python:.2f}s (reusing provided measurement)")
    else:
        print(f"Python engine (backend={backend})...")
        t_python = _run_nsga2(
            use_java_engine=False,
            jar_path=None,
            n_gen=args.n_gen,
            pop_size=args.pop_size,
            backend=backend,
        )
        print(f"  Wall-clock: {t_python:.2f}s")

    if args.java is None:
        print()
        print("Skipping Java comparison (--java not supplied).")
        return 0

    print()
    print(f"Java engine ({args.java})...")
    t_java = _run_nsga2(
        use_java_engine=True,
        jar_path=args.java,
        n_gen=args.n_gen,
        pop_size=args.pop_size,
        backend=backend,
    )
    print(f"  Wall-clock: {t_java:.2f}s")

    print()
    speedup = t_java / t_python if t_python > 0 else float("inf")
    print(f"Speedup (Java/Python): {speedup:.2f}x")
    if speedup < 4.0:
        print(f"WARNING: speedup {speedup:.2f}x is below the 4x v0.10.0 release gate")
        return 2
    print("OK: speedup >= 4x (v0.10.0 release gate met)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
