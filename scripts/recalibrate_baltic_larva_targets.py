#!/usr/bin/env python
"""Targeted larval-rate recalibration of the Baltic config onto its biomass targets.

Written for the 2026-11-08 corrected-LTL recalibration (issue #229). The config carries the
corrected LTL forcing, C2 (cod_east spawning in SD 25) and no herring autumn eggs. Measured with
current parameters, cod_west sat 4% over its ceiling and herring on its floor, so only two larval
mortality rates are re-fitted, onto the biomass_targets.csv target values:

  * herring  mortality.additional.larva.rate.sp1 -> 1.5 Mt (fallback: >= 1.0 Mt if it saturates)
  * cod_west mortality.additional.larva.rate.sp0 -> 10 kt (accept <= 20 kt)
  * cod_east mortality.additional.larva.rate.sp8 -> 70 kt, only if cod_east leaves 60-85 kt
    after a sweep

Objective per stock: the certification's own quantity, the final-decade (last 10 annual values)
mean of a 50-yr single-seed run. Method: geometric bisection on each rate in RESOLVED units (per
step). This 4.4.1 config's reader divides the CSV's per-year value by ndtperyear on read
(osmose/config/reader.py), so a CSV entry is the resolved value x ndtperyear; the final output
prints both. Gauss-Seidel over the stocks, herring first because it is the food,
for 2 sweeps. Tolerance is +-10% on each stock's own axis. Every evaluation is printed and the
record is written to docs/diagnostics/baltic_larva_targets_solve.json. Run alone; it uses the
whole machine.

Usage: PYTHONPATH=. .venv/bin/python scripts/recalibrate_baltic_larva_targets.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
OUT = ROOT / "docs" / "diagnostics" / "baltic_larva_targets_solve.json"
SEED = 42
YEARS = os.environ.get("RECAL_YEARS", "50")  # override only for a smoke run
TOL = 0.10
KEY = "mortality.additional.larva.rate.sp{}"
STOCKS = {  # name: (sp index, target t, acceptable bound as (lo, hi))
    "herring": (1, 1.5e6, (1.0e6, 3.0e6)),
    "cod_west": (0, 1.0e4, (4.0e3, 2.0e4)),
}
COD_EAST = (8, 7.0e4, (6.0e4, 8.5e4))
REPORT = ("cod_west", "herring", "sprat", "flounder", "cod_east", "pikeperch", "stickleback")

base = dict(OsmoseConfigReader().read(str(CONFIG)))
base["simulation.time.nyear"] = YEARS
rates = {sp: float(base[KEY.format(sp)]) for sp in (0, 1, 8)}
evals: list[dict] = []
t0 = time.perf_counter()


def run(r: dict[int, float]) -> dict[str, float]:
    cfg = dict(base)
    for sp, v in r.items():
        cfg[KEY.format(sp)] = repr(float(v))
    bio = PythonEngine().run_in_memory(cfg, seed=SEED).biomass()
    means = {s: float(np.mean(np.asarray(bio[s].values, float)[-10:])) for s in REPORT}
    evals.append({"rates": {str(k): v for k, v in r.items()}, "means": means})
    print(
        f"[{time.perf_counter() - t0:6.0f}s] eval {len(evals):2d}: "
        + " ".join(f"sp{k}={v:.4f}" for k, v in sorted(r.items()))
        + " -> "
        + " ".join(f"{s}={means[s]:.4g}" for s in REPORT),
        flush=True,
    )
    return means


def solve(name: str, sp: int, target: float, current: dict[str, float]) -> float | None:
    """Geometric bisection on rate sp so that `name`'s mean hits target (decreasing in rate).
    Returns None when the target cannot be bracketed within 8 steps of 1.25x (saturation)."""
    r0, f0 = rates[sp], current[name]
    if abs(f0 / target - 1) <= TOL:
        return r0
    step = 1.25 if f0 > target else 1 / 1.25  # mean too high -> raise the rate
    lo_r = hi_r = r0
    f = f0
    for _ in range(8):  # bracket: walk the rate until the mean crosses the target
        r_try = (hi_r if step > 1 else lo_r) * step
        f = run({**rates, sp: r_try})[name]
        if step > 1:
            lo_r, hi_r = hi_r, r_try
        else:
            hi_r, lo_r = lo_r, r_try
        if (f <= target) if step > 1 else (f >= target):
            break
    else:
        print(f"  {name}: target {target:.3g} not bracketed (saturated); last {f:.3g}")
        return None
    best_r, best_err = (hi_r if step > 1 else lo_r), abs(f / target - 1)
    for _ in range(8):  # bisection in log-rate between lo_r (mean high) and hi_r (mean low)
        mid = float(np.sqrt(lo_r * hi_r))
        f = run({**rates, sp: mid})[name]
        if abs(f / target - 1) < best_err:
            best_r, best_err = mid, abs(f / target - 1)
        if best_err <= TOL:
            break
        if f > target:
            lo_r = mid
        else:
            hi_r = mid
    print(f"  {name}: rate {best_r:.4f} (err {best_err:+.1%})")
    return best_r


def main() -> int:
    print(f"start rates: {rates}")
    current = run(dict(rates))
    for sweep in (1, 2):
        for name, (sp, target, bounds) in STOCKS.items():
            r = solve(name, sp, target, current)
            if r is None:  # saturated: fall back to the acceptable bound nearest the target
                fallback = bounds[0] if current[name] < target else bounds[1]
                print(f"  {name}: falling back to {fallback:.3g}")
                r = solve(name, sp, fallback, current)
            if r is not None:
                rates[sp] = r
            current = run(dict(rates))
        ce = current["cod_east"]
        if not (COD_EAST[2][0] <= ce <= COD_EAST[2][1]):
            print(f"  sweep {sweep}: cod_east {ce:.4g} outside 60-85 kt -> solving sp8 to 70 kt")
            r = solve("cod_east", COD_EAST[0], COD_EAST[1], current)
            if r is not None:
                rates[COD_EAST[0]] = r
            current = run(dict(rates))
        ok = {n: b[0] <= current[n] <= b[1] for n, (_, _, b) in STOCKS.items()}
        print(f"sweep {sweep} done: {current} accepted: {ok}")
    ndt = float(base["simulation.time.ndtperyear"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "rates_resolved_per_step": {str(k): v for k, v in rates.items()},
                "rates_csv_per_year": {str(k): v * ndt for k, v in rates.items()},
                "final_means": current,
                "evals": evals,
                "seed": SEED,
                "years": int(YEARS),
                "tol": TOL,
            },
            indent=2,
        )
        + "\n"
    )
    print("\nFINAL rates, resolved per step:", {KEY.format(k): v for k, v in rates.items()})
    print("FINAL rates, CSV per year (x ndt):", {KEY.format(k): v * ndt for k, v in rates.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
