"""SP1b — mean-neutral recalibration of cod larval mortality for the SP1 spatial term.

Pure 1-D root finder: coarse grid scan over [0, d0] -> sign-change feasibility gate ->
bisection. Engine-free (the caller injects run_mean_on); see the SP1b design spec.

Per-stock (cod_west + cod_east, since the 2026-07-25 disaggregation): each SP1-enabled stock
gets its own rate, solved against its own SP1-off mean by a Gauss-Seidel sweep over the
1-D solver (`solve_per_stock`). Every solved rate is frozen TOGETHER WITH the resolved
baseline rate d0 it was solved against (`StockRecal.d0`); `sp1_on_config` refuses to apply a
rate whose d0 no longer matches the live config, because a mean-neutral *offset* is only
meaningful relative to the baseline it offsets (the 2026-07 trap: d0 moved 15 -> 10.16 under a
frozen 14.66 and the overlay silently became a larval-mortality INCREASE).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


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


# ---------------------------------------------------------------------------
# Per-stock sweep
# ---------------------------------------------------------------------------

RateMap = dict[str, float | None]
MeansFn = Callable[[Mapping[str, float | None]], Mapping[str, float]]


@dataclass
class PerStockResult:
    rates: RateMap  # final rate per stock (None = infeasible, key omitted -> d0 stands)
    means: dict[str, float]  # per-stock means at `rates` (joint evaluation)
    rel_errs: dict[str, float]  # |means - baseline| / baseline per stock
    per_stock: dict[str, RecalResult]  # last 1-D solve per stock
    converged: bool  # every stock feasible AND within tol at the joint evaluation
    sweeps: int
    evaluations: int  # distinct engine evaluations actually run


def solve_per_stock(
    baselines: Mapping[str, float],
    run_means_on: MeansFn,
    *,
    order: Sequence[str],
    grids: Mapping[str, Sequence[float]],
    tol: float = 0.02,
    max_iter: int = 20,
    max_sweeps: int = 2,
) -> PerStockResult:
    """Gauss-Seidel sweep of `solve_larva_rate` over coupled stocks.

    Each stock is solved in `order` with every other stock held at its current best rate
    (initially d0 = max of its grid). After a sweep the joint evaluation at the current
    rates is checked; if every feasible stock is within `tol` the sweep stops. Evaluations
    are memoised on the full rate set, so the joint check reuses the last solve's run and a
    deterministic evaluator is never called twice for one rate set.
    """
    cache: dict[tuple[tuple[str, float | None], ...], dict[str, float]] = {}

    def evaluate(rates: Mapping[str, float | None]) -> dict[str, float]:
        key = tuple(sorted(rates.items()))
        if key not in cache:
            cache[key] = dict(run_means_on(dict(rates)))
        return cache[key]

    current: RateMap = {name: float(max(grids[name])) for name in order}
    per_stock: dict[str, RecalResult] = {}
    sweeps = 0
    converged = False
    while sweeps < max_sweeps and not converged:
        sweeps += 1
        for name in order:

            def run_mean_on(d: float, _name: str = name) -> float:
                return evaluate({**current, _name: d})[_name]

            res = solve_larva_rate(
                baselines[name], run_mean_on, grid_points=grids[name], tol=tol, max_iter=max_iter
            )
            per_stock[name] = res
            current[name] = res.rate if res.feasible else None
        joint = evaluate(current)
        rel_errs = {n: abs(joint[n] - baselines[n]) / baselines[n] for n in order}
        converged = all(per_stock[n].feasible and rel_errs[n] <= tol for n in order)
    joint = evaluate(current)
    rel_errs = {n: abs(joint[n] - baselines[n]) / baselines[n] for n in order}
    return PerStockResult(
        rates=dict(current),
        means={n: joint[n] for n in order},
        rel_errs=rel_errs,
        per_stock=per_stock,
        converged=converged,
        sweeps=sweeps,
        evaluations=len(cache),
    )


# ---------------------------------------------------------------------------
# Means
# ---------------------------------------------------------------------------

_WINDOW_YEARS = (3.0, 15.0)  # years [3, 15), matching the SP1 diagnostic
_NON_STOCK_COLUMNS = frozenset({"Time", "species"})  # ResultSet.biomass() metadata columns


def _window_mask(bio: pd.DataFrame) -> np.ndarray:
    """Rows inside the window: by the Time column (years) when recorded, else row index —
    identical for a yearly-recorded config (Baltic: output.recordfrequency.ndt = ndtperyear)."""
    lo, hi = _WINDOW_YEARS
    if "Time" in bio.columns:
        t = bio["Time"].to_numpy(dtype=float)
    else:
        t = np.arange(len(bio), dtype=float)
    return (t >= lo) & (t < hi)


def _window_mean(values: object, mask: np.ndarray) -> float:
    w = np.asarray(values, dtype=float)[mask]
    w = w[np.isfinite(w) & (w > 0)]
    return float(w.mean())


def stock_means_from_biomass(bio: pd.DataFrame) -> dict[str, float]:
    """Mean biomass per species column over years [3, 15) (finite & >0)."""
    mask = _window_mask(bio)
    return {
        str(c): _window_mean(bio[c], mask) for c in bio.columns if str(c) not in _NON_STOCK_COLUMNS
    }


def stock_overshoot_from_biomass(bio: pd.DataFrame) -> dict[str, float]:
    """max/mean per species column over years [3, 15) — the boom/bust index the SP1b
    diagnostic records (measured, never gated)."""
    mask = _window_mask(bio)
    out: dict[str, float] = {}
    for c in bio.columns:
        if str(c) in _NON_STOCK_COLUMNS:
            continue
        w = np.asarray(bio[c], dtype=float)[mask]
        w = w[np.isfinite(w) & (w > 0)]
        out[str(c)] = float(w.max() / w.mean()) if w.size else float("nan")
    return out


def mean_cod_from_biomass(bio: pd.DataFrame) -> float:
    """Total cod = cod_west + cod_east (aggregate 'cod' fallback for undisaggregated configs)."""
    mask = _window_mask(bio)
    if "cod" in bio.columns:
        return _window_mean(bio["cod"], mask)
    return _window_mean(bio["cod_west"] + bio["cod_east"], mask)


def _run_biomass(cfg: dict[str, str], *, seed: int = 0) -> pd.DataFrame:
    from osmose.engine import PythonEngine

    return PythonEngine().run_in_memory(cfg, seed=seed).biomass()


def stock_means(cfg: dict[str, str], *, seed: int = 0) -> dict[str, float]:
    """One engine run -> years-[3:15] mean biomass per focal species."""
    return stock_means_from_biomass(_run_biomass(cfg, seed=seed))


def mean_cod(cfg: dict[str, str], *, seed: int = 0) -> float:
    """Mean total-cod biomass over years index [3:15] (finite & >0), matching the SP1
    diagnostic. Total cod = cod_west + cod_east (aggregate 'cod' fallback)."""
    return mean_cod_from_biomass(_run_biomass(cfg, seed=seed))


def e_clip_first_guess(
    field_path: str | Path, spawn_path: str | Path, d0: float
) -> tuple[float, float]:
    """Analytical first-guess rate d1 = clip(d0 + ln E[clip], 0, d0), and E[clip].

    E[clip] = presence-weighted mean of clip(RV_timemean(cell)/RV_ref) over the stock's
    spawning cells. This restores the *instantaneous* egg-weighted average survival; it is only
    a grid seed (the empirical solve finds the true equilibrium root, which — because the
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


# ---------------------------------------------------------------------------
# Config helpers + the SP1-on overlay
# ---------------------------------------------------------------------------

RATE_KEY = "mortality.additional.larva.rate.sp{i}"
SP1_STOCKS: tuple[str, ...] = ("cod_west", "cod_east")  # stocks SP1 enables, by name


def species_index(cfg: Mapping[str, str], name: str) -> int | None:
    """sp index of the species named `name` (`species.name.sp{i}`), or None if absent."""
    for k, v in cfg.items():
        if k.startswith("species.name.sp") and v == name:
            return int(k[len("species.name.sp") :])
    return None


def resolved_d0(cfg: Mapping[str, str], name: str) -> float:
    """The live resolved per-cohort larva rate of `name` (the baseline d0 a solve offsets)."""
    i = species_index(cfg, name)
    if i is None:
        raise KeyError(f"{name!r} is not a declared species")
    return float(cfg[RATE_KEY.format(i=i)])


@dataclass(frozen=True)
class StockRecal:
    """A solved per-stock rate frozen together with the baseline it was solved against."""

    rate: float | None  # mean-neutral per-cohort larva rate; None = infeasible (d0 stands)
    d0: float  # resolved baseline rate at solve time — the overlay refuses a different one
    baseline: float | None = None  # SP1-off mean (t)
    mean_on: float | None = None  # SP1-on mean at `rate` (t), joint evaluation
    rel_err: float | None = None
    note: str = field(default="")  # solve date / message


# Filled by hand from `scripts/recalibrate_sp1b.py` output. Empty = no solved rate (the
# aggregate-cod solve of 2026-07-02, RECAL_RATE=14.655 at d0=15.0, is retired: see
# docs/diagnostics/sp1b_recalibration.md).
RECAL_RATES: dict[str, StockRecal] = {}


class _UseRecal:
    """Sentinel type for sp1_on_config's default (typed so isinstance narrows the union)."""


_USE_RECAL = _UseRecal()  # read the current module RECAL_RATES at call time

_DET_KEYS = {
    "movement.randomseed.fixed": "true",
    "stochastic.mortality.randomseed.fixed": "true",
}


def with_determinism(cfg: dict[str, str]) -> dict[str, str]:
    """Return a copy of cfg with the two fixed-seed keys set (required for a reproducible
    solve; the runtime numba single-thread pin is set separately by the caller)."""
    return {**cfg, **_DET_KEYS}


def _frozen_rates(cfg: Mapping[str, str]) -> RateMap:
    """RECAL_RATES -> name: rate for stocks the config declares, d0-checked."""
    rates: RateMap = {}
    for name, entry in RECAL_RATES.items():
        if species_index(cfg, name) is None:
            continue
        if entry.rate is not None:
            live = resolved_d0(cfg, name)
            if not math.isclose(live, entry.d0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"SP1b rate for {name} was solved against d0={entry.d0:.3f} but the live "
                    f"config resolves d0={live:.3f}; re-run scripts/recalibrate_sp1b.py "
                    "(a mean-neutral offset is only valid relative to the baseline it offsets)"
                )
        rates[name] = entry.rate
    return rates


def sp1_on_config(
    base_cfg: dict[str, str],
    field_path: str | Path,
    *,
    larva_rates: Mapping[str, float | None] | None | _UseRecal = _USE_RECAL,
    stocks: Sequence[str] = SP1_STOCKS,
) -> dict[str, str]:
    """SP1-on config: SP1 flags (every declared stock in `stocks`) + determinism keys +
    per-stock recalibrated larva rates.

    larva_rates=None omits every rate key (each stock's base d0 stands — the infeasible
    path); a mapping sets the key per stock (None entries omitted); the default reads the
    current module RECAL_RATES at call time and refuses an entry whose solved-against d0
    differs from the live config.
    """
    cfg = with_determinism(base_cfg)
    cfg["reproduction.rv.spatial.enabled"] = "true"
    cfg["reproduction.rv.spatial.field.file"] = str(field_path)
    for name in stocks:
        i = species_index(cfg, name)
        if i is not None:
            cfg[f"reproduction.rv.spatial.species.enabled.sp{i}"] = "true"

    rates: Mapping[str, float | None]
    if isinstance(larva_rates, _UseRecal):
        rates = _frozen_rates(cfg)
    elif larva_rates is None:
        rates = {}
    else:
        rates = larva_rates
    for name, rate in rates.items():
        i = species_index(cfg, name)
        if i is None or rate is None:
            continue
        cfg[RATE_KEY.format(i=i)] = repr(float(rate))  # resolved per-cohort value
    return cfg
