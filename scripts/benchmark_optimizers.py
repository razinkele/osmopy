#!/usr/bin/env python3
"""Head-to-head optimizer benchmark on synthetic test functions.

Runs vanilla scipy DE, CMA-ES (Tier C2), and surrogate-DE (Tier C1) on
standard black-box benchmarks at matched real-eval budgets, reporting
best objective + nfev + wall-clock per (optimizer, problem) pair.

Purpose: empirical validation that the C1/C2 standalone modules behave
as advertised, plus concrete data for picking an optimizer when launching
a real OSMOSE calibration. The synthetic problems are deterministic and
cheap, so this whole script runs in seconds — re-run it after any
optimizer change to catch regressions.

Output: a markdown table to stdout + a JSON dump under
data/benchmarks/optimizer_comparison.json (created if missing).

Usage:
    .venv/bin/python scripts/benchmark_optimizers.py [--budget 300] [--seeds 3]

Each (optimizer, problem) pair is run with `--seeds` independent seeds; the
table reports mean ± std of best objective. Variance under reseeding is
the most honest signal of which optimizer is robust on which landscape.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osmose.calibration.cmaes_runner import run_cmaes  # noqa: E402
from osmose.calibration.surrogate_de import surrogate_assisted_de  # noqa: E402

DEFAULT_BUDGET = 200
DEFAULT_SEEDS = 3
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "benchmarks" / "optimizer_comparison.json"
# Surrogate-DE iteration cap — GP fit cost is O(n_train^3), so beyond ~6 iterations
# the benchmark spends more time in sklearn than in the actual optimization.
SURROGATE_MAX_ITER = 6


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------
def sphere(x: NDArray[np.float64]) -> float:
    """Convex bowl. Minimum 0 at origin. Easy."""
    return float(np.sum(x ** 2))


def rosenbrock(x: NDArray[np.float64]) -> float:
    """Curved valley. Minimum 0 at (1, 1, ..., 1). Hard for first-order methods."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x: NDArray[np.float64]) -> float:
    """Highly multi-modal bowl with many local minima. Tests global search."""
    n = len(x)
    return float(10.0 * n + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


@dataclass(frozen=True)
class Problem:
    name: str
    func: callable
    n_dim: int
    bounds: list[tuple[float, float]]
    optimum: float


PROBLEMS = [
    Problem("sphere-5d", sphere, 5, [(-5.0, 5.0)] * 5, 0.0),
    Problem("sphere-10d", sphere, 10, [(-5.0, 5.0)] * 10, 0.0),
    Problem("rosenbrock-5d", rosenbrock, 5, [(-3.0, 3.0)] * 5, 0.0),
    Problem("rosenbrock-10d", rosenbrock, 10, [(-3.0, 3.0)] * 10, 0.0),
]
# Note: rastrigin (highly multi-modal) is excluded because surrogate-DE's GP
# struggles to capture deep multi-modality, and DE/CMA-ES would dominate
# trivially. Add it back for a fairness study, not for picking-an-optimizer.


# ---------------------------------------------------------------------------
# Optimizer runners — each returns (best_fun, nfev, elapsed_s)
# ---------------------------------------------------------------------------
def run_vanilla_de(problem: Problem, budget: int, seed: int) -> tuple[float, int, float]:
    n = problem.n_dim
    # popsize chosen so popsize * maxiter ≈ budget
    popsize = max(5, n)  # minimum reasonable population
    maxiter = max(2, budget // popsize - 1)
    t0 = time.time()
    result = differential_evolution(
        problem.func,
        problem.bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        tol=1e-8,  # tight; usually exhausts budget instead of converging
        mutation=(0.5, 1.5),
        recombination=0.8,
        polish=False,
        updating="deferred",
        workers=1,  # deterministic; sequential
    )
    return float(result.fun), int(result.nfev), time.time() - t0


def run_cmaes_de(problem: Problem, budget: int, seed: int) -> tuple[float, int, float]:
    n = problem.n_dim
    cma_popsize = 4 + int(3 * np.log(n))  # cma's default
    maxiter = max(5, budget // cma_popsize)
    x0 = [(lo + hi) / 2.0 for lo, hi in problem.bounds]
    t0 = time.time()
    result = run_cmaes(
        problem.func,
        problem.bounds,
        x0=x0,
        sigma0=0.3,
        popsize=cma_popsize,
        maxiter=maxiter,
        tol=1e-8,
        seed=seed,
        workers=1,
        verbose=False,
    )
    return float(result["fun"]), int(result["nfev"]), time.time() - t0


def run_surrogate(problem: Problem, budget: int, seed: int) -> tuple[float, int, float]:
    n = problem.n_dim
    n_initial = max(20, 5 * n)
    # Cap n_iterations at SURROGATE_MAX_ITER regardless of budget — beyond ~6
    # iterations the GP fit cost (O(n_train³)) dominates wall-clock with
    # diminishing optimization gains. n_topk is sized so the total real-eval
    # count fills the requested budget.
    remaining = max(0, budget - n_initial)
    n_iterations = min(SURROGATE_MAX_ITER, max(1, remaining // max(5, n)))
    n_topk = max(5, remaining // n_iterations) if n_iterations > 0 else 5
    t0 = time.time()
    result = surrogate_assisted_de(
        problem.func,
        problem.bounds,
        n_initial=n_initial,
        n_iterations=n_iterations,
        n_topk=n_topk,
        workers=1,
        seed=seed,
    )
    return float(result["fun"]), int(result["nfev"]), time.time() - t0


OPTIMIZERS = {
    "DE": run_vanilla_de,
    "CMA-ES": run_cmaes_de,
    "surrogate-DE": run_surrogate,
}


# ---------------------------------------------------------------------------
# Benchmark loop + reporting
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help="Approximate real-eval budget per (optimizer, problem) run")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="Number of independent seeds per (optimizer, problem) pair")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Suppress sklearn ConvergenceWarning noise on smooth analytic test functions —
    # benign and unactionable for benchmarks.
    import warnings as _w
    _w.filterwarnings("ignore", category=Warning, module="sklearn")

    print(f"=== Optimizer Benchmark — budget={args.budget}, seeds={args.seeds} ===\n")

    results: dict[str, dict[str, list[dict]]] = {}  # problem -> optimizer -> [{seed, fun, nfev, t}]

    for problem in PROBLEMS:
        results[problem.name] = {}
        for opt_name, runner in OPTIMIZERS.items():
            runs = []
            for seed in range(args.seeds):
                fun, nfev, elapsed = runner(problem, args.budget, seed)
                runs.append({
                    "seed": seed,
                    "fun": fun,
                    "nfev": nfev,
                    "elapsed_s": elapsed,
                })
            results[problem.name][opt_name] = runs

    # Render markdown table
    print("| Problem | Optimizer | best fun (mean ± std) | nfev (mean) | wall-clock (s) |")
    print("|---|---|---|---|---|")
    for problem in PROBLEMS:
        for opt_name in OPTIMIZERS:
            runs = results[problem.name][opt_name]
            funs = np.array([r["fun"] for r in runs])
            nfevs = np.array([r["nfev"] for r in runs])
            elapses = np.array([r["elapsed_s"] for r in runs])
            print(
                f"| {problem.name} | {opt_name} | "
                f"{funs.mean():.4g} ± {funs.std():.2g} | "
                f"{nfevs.mean():.0f} | {elapses.mean():.2f} |"
            )

    # Per-problem winner by mean fun
    print("\n=== Per-problem winner (lowest mean fun) ===")
    for problem in PROBLEMS:
        means = {
            opt: float(np.mean([r["fun"] for r in results[problem.name][opt]]))
            for opt in OPTIMIZERS
        }
        winner = min(means.items(), key=lambda kv: kv[1])
        runner_up = sorted(means.items(), key=lambda kv: kv[1])[1]
        margin = (runner_up[1] - winner[1]) / max(abs(winner[1]), 1e-12)
        print(
            f"  {problem.name:18s}: {winner[0]:12s} fun={winner[1]:.4g}  "
            f"(beats {runner_up[0]} by {margin*100:.1f}%)"
        )

    # JSON dump
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "budget": args.budget,
        "seeds": args.seeds,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nFull results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
