"""SP1b — mean-neutral recalibration of cod larval mortality for the SP1 spatial term.

Pure 1-D root finder: coarse grid scan over [0, d0] -> sign-change feasibility gate ->
bisection. Engine-free (the caller injects run_mean_on); see the SP1b design spec.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


def mean_cod(cfg: dict[str, str], *, seed: int = 0) -> float:
    """Mean cod biomass over years index [3:15] (finite & >0), matching the SP1 diagnostic."""
    from osmose.engine import PythonEngine

    b = PythonEngine().run_in_memory(cfg, seed=seed).biomass()["cod"].to_numpy()
    w = b[3:15]
    w = w[np.isfinite(w) & (w > 0)]
    return float(w.mean())


def e_clip_first_guess(
    field_path: str | Path, spawn_path: str | Path, d0: float
) -> tuple[float, float]:
    """Analytical first-guess rate d1 = clip(d0 + ln E[clip], 0, d0), and E[clip].

    E[clip] = presence-weighted mean of clip(RV_timemean(cell)/RV_ref) over cod_spawning
    cells. This restores the *instantaneous* egg-weighted average survival; it is only a
    grid seed (the empirical solve finds the true equilibrium root, which — because the
    biomass effect is buffered by density dependence — is usually much closer to d0).
    """
    import xarray as xr

    da = xr.open_dataset(field_path)["reproductive_volume"]
    rv = da.values.mean(axis=0)  # time-mean (nlat, nlon), north-first
    ref = float(da.attrs["RV_ref"])
    spawn = np.flipud(np.genfromtxt(spawn_path, delimiter=";")) > 0
    e_clip = float(np.clip(rv[spawn] / ref, 0.0, 1.0).mean())
    d1 = d0 + math.log(e_clip) if e_clip > 0.0 else 0.0
    return max(0.0, min(d0, d1)), e_clip


RECAL_RATE: float | None = None  # set from scripts/recalibrate_sp1b.py output (SP1b Task 4)


class _UseRecal:
    """Sentinel type for sp1_on_config's default (typed so isinstance narrows the union)."""


_USE_RECAL = _UseRecal()  # read the current module RECAL_RATE at call time

_DET_KEYS = {
    "movement.randomseed.fixed": "true",
    "stochastic.mortality.randomseed.fixed": "true",
}


def with_determinism(cfg: dict[str, str]) -> dict[str, str]:
    """Return a copy of cfg with the two fixed-seed keys set (required for a reproducible
    solve; the runtime numba single-thread pin is set separately by the caller)."""
    return {**cfg, **_DET_KEYS}


def sp1_on_config(
    base_cfg: dict[str, str],
    field_path: str | Path,
    *,
    larva_rate: float | None | _UseRecal = _USE_RECAL,
) -> dict[str, str]:
    """SP1-on config: SP1 flags + determinism keys + recalibrated cod larva rate.

    larva_rate=None omits the rate key (base d0 stands — the infeasible path); a float sets
    it; the default reads the current module RECAL_RATE at call time (so freezing RECAL_RATE
    in Task 4 takes effect without editing this default).
    """
    rate: float | None = RECAL_RATE if isinstance(larva_rate, _UseRecal) else larva_rate
    cfg = with_determinism(base_cfg)
    cfg["reproduction.rv.spatial.enabled"] = "true"
    cfg["reproduction.rv.spatial.field.file"] = str(field_path)
    cfg["reproduction.rv.spatial.species.enabled.sp0"] = "true"
    if rate is not None:
        cfg["mortality.additional.larva.rate.sp0"] = repr(float(rate))  # resolved per-cohort value
    return cfg
