"""Solve one fleet's base fishing rate for a target catch (spec 2026-11-28 fleet-rate recal).

Catch vs rate is domed: past F_MSY more fishing gives less catch, so a catch target can have two
roots. :func:`solve_rate` maps the curve with a coarse log-spaced sweep (extended outward while
the curve's lower end is still past the peak or the upper end still rising below the target),
then bisects in ``log(rate)`` inside the first bracket at or below the peak: always the LOWER-F
root. A target above the peak returns the peak's rate with ``reachable=False``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_EXTEND = 4.0  # factor per outward extension of the sweep
_MAX_EXTEND = 3  # extensions per side


@dataclass(frozen=True)
class Solve:
    rate: float
    catch: float
    target: float
    reachable: bool
    evaluations: list[tuple[float, float]] = field(default_factory=list)
    note: str = ""


def trailing_catch(yield_df: pd.DataFrame, species: str, n_years: int = 5) -> float:
    """Mean of the last ``n_years`` rows of ``species``' yield column (tonnes/yr)."""
    return float(np.mean(yield_df[species].to_numpy(dtype=float)[-n_years:]))


def sweep_rates(current: float, target_ratio: float, n: int = 5) -> list[float]:
    """``n`` log-spaced rates spanning ``current`` and ``current * target_ratio``, with a factor-2
    margin on each side."""
    ratio = target_ratio if np.isfinite(target_ratio) and target_ratio > 0 else 1.0
    ratio = min(max(ratio, 1e-3), 1e3)
    lo = min(current, current * ratio) / 2.0
    hi = max(current, current * ratio) * 2.0
    return [float(x) for x in np.geomspace(lo, hi, n)]


def solve_rate(
    evaluate: Callable[[float], float],
    current: float,
    target: float,
    *,
    tol: float = 0.10,
    max_bisect: int = 8,
    sweep_n: int = 5,
) -> Solve:
    """Rate whose catch is within ``tol`` of ``target`` on the lower-F side of the dome."""
    seen: dict[float, float] = {}

    def ev(r: float) -> float:
        if r not in seen:
            seen[r] = float(evaluate(r))
        return seen[r]

    c0 = ev(current)
    for r in sweep_rates(current, target / max(c0, 1e-9), sweep_n):
        ev(r)

    def pts() -> list[tuple[float, float]]:
        return sorted(seen.items())

    for _ in range(_MAX_EXTEND):  # lower end still past the peak, or still above the target
        (r0, k0), (_, k1) = pts()[0], pts()[1]
        if k0 > target or k0 > k1:
            ev(r0 / _EXTEND)
        else:
            break
    for _ in range(_MAX_EXTEND):  # upper end still rising and still below the target
        (_, k1), (r2, k2) = pts()[-2], pts()[-1]
        if k2 >= k1 and k2 < target and k2 > 0:
            ev(r2 * _EXTEND)
        else:
            break

    p = pts()
    if all(c <= 0 for _, c in p):  # nothing to catch anywhere (extinct stock): leave the rate
        return Solve(current, 0.0, target, False, p, "zero catch at every rate: rate unchanged")
    peak = int(np.argmax([c for _, c in p]))

    bracket = None
    for i in range(peak):
        (ra, ca), (rb, cb) = p[i], p[i + 1]
        if (ca - target) * (cb - target) <= 0:
            bracket = (ra, ca, rb, cb)
            break
    for r, c in p[: peak + 1]:
        if abs(c / target - 1) <= tol:
            return Solve(r, c, target, True, pts(), "sweep point within tolerance")
    if bracket is None:
        pr, pc = p[peak]
        if pc < target:
            return Solve(pr, pc, target, False, pts(), "target above peak: max attainable catch")
        r, c = min(p[: peak + 1], key=lambda rc: abs(rc[1] - target))
        return Solve(r, c, target, False, pts(), "no bracket below the peak: closest point")

    ra, ca, rb, cb = bracket
    r, c = ra, ca
    for _ in range(max_bisect):
        r = math.exp((math.log(ra) + math.log(rb)) / 2)
        c = ev(r)
        if abs(c / target - 1) <= tol:
            return Solve(r, c, target, True, pts(), "bisection")
        if (ca - target) * (c - target) <= 0:
            rb, cb = r, c
        else:
            ra, ca = r, c
    return Solve(r, c, target, False, pts(), "bisection budget spent")
