"""SP1b — mean-neutral recalibration of cod larval mortality for the SP1 spatial term.

Pure 1-D root finder: coarse grid scan over [0, d0] -> sign-change feasibility gate ->
bisection. Engine-free (the caller injects run_mean_on); see the SP1b design spec.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass
class RecalResult:
    rate: float | None  # solved larva rate, or None when infeasible
    baseline: float  # SP1-off mean cod biomass
    mean_on: float | None  # SP1-on mean at `rate` (None when infeasible)
    rel_err: float | None  # |mean_on - baseline| / baseline
    converged: bool  # reached tol
    feasible: bool  # exactly one sign change (or a near-zero grid hit)
    grid: list[tuple[float, float]]  # (rate, mean) at each grid point
    iters: int  # bisection iterations used
    message: str


def solve_larva_rate(
    baseline: float,
    run_mean_on: Callable[[float], float],
    *,
    grid_points: Sequence[float],
    tol: float = 0.02,
    max_iter: int = 20,
) -> RecalResult:
    """Find the larva rate whose SP1-on mean cod biomass matches `baseline` within `tol`.

    No monotonicity is assumed: the grid measures the shape. Exactly one sign change of
    f(d) = run_mean_on(d) - baseline is required for a solve; zero or >=2 -> infeasible.
    A grid point already within tol short-circuits (rate == max grid point == "no change").
    """
    grid = sorted({float(g) for g in grid_points})
    evals = [(d, float(run_mean_on(d))) for d in grid]

    def rel(m: float) -> float:
        return abs(m - baseline) / baseline

    # (b) near-zero short-circuit BEFORE sign counting (makes f=0 well-posed).
    for d, m in evals:
        if rel(m) <= tol:
            return RecalResult(
                d,
                baseline,
                m,
                rel(m),
                True,
                True,
                evals,
                0,
                f"grid point {d:.4g} already within tol",
            )

    # (c) feasibility gate: count sign changes of f = m - baseline.
    fs = [(d, m - baseline) for d, m in evals]
    crossings = [(fs[i], fs[i + 1]) for i in range(len(fs) - 1) if fs[i][1] * fs[i + 1][1] < 0.0]
    if len(crossings) != 1:
        return RecalResult(
            None,
            baseline,
            None,
            None,
            False,
            False,
            evals,
            0,
            f"{len(crossings)} sign changes on grid (need exactly 1); "
            "baseline unreachable or ambiguous/multi-root",
        )

    # (d) bisection on the single sign-changing sub-interval.
    (a, fa), (b, _fb) = crossings[0]
    iters = 0
    while iters < max_iter:
        mid = 0.5 * (a + b)
        m_mid = float(run_mean_on(mid))
        f_mid = m_mid - baseline
        iters += 1
        if rel(m_mid) <= tol:
            return RecalResult(
                mid, baseline, m_mid, rel(m_mid), True, True, evals, iters, "converged"
            )
        if fa * f_mid < 0.0:
            b = mid
        else:
            a, fa = mid, f_mid

    mid = 0.5 * (a + b)
    m_mid = float(run_mean_on(mid))
    return RecalResult(
        mid,
        baseline,
        m_mid,
        rel(m_mid),
        False,
        True,
        evals,
        iters,
        f"max_iter={max_iter} reached, rel_err={rel(m_mid):.3f} > tol={tol}",
    )
