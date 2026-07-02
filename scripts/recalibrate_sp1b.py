#!/usr/bin/env python
"""SP1b — solve cod's larval rate so SP1-on mean cod biomass matches the SP1-off baseline.

Usage: PYTHONPATH=. .venv/bin/python scripts/recalibrate_sp1b.py
Prints the grid, the RecalResult, and a ready-to-paste `RECAL_RATE = ...` line for
osmose/calibration/larva_recal.py. Two-plus 15-yr Baltic runs — foreground only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np

from osmose.calibration.larva_recal import (
    e_clip_first_guess,
    mean_cod,
    solve_larva_rate,
    sp1_on_config,
    with_determinism,
)
from osmose.config import OsmoseConfigReader

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
FIELD = ROOT / "data" / "baltic" / "forcing" / "baltic_rv_field.nc"
SPAWN = ROOT / "data" / "baltic" / "maps" / "cod_spawning.csv"
D0 = 15.0


def _base_cfg() -> dict[str, str]:
    cfg = dict(OsmoseConfigReader().read(CONFIG))
    cfg["simulation.time.nyear"] = "15"
    return cfg


def main() -> int:
    numba.set_num_threads(1)  # runtime determinism pin
    base = _base_cfg()

    # Noise check: under the fixed-seed keys + single thread, f(d) must be reproducible,
    # else the solve chases noise. Require bit-identical repeat means before trusting it.
    off = with_determinism(base)
    baseline = mean_cod(off)
    baseline_again = mean_cod(off)
    if abs(baseline_again - baseline) / baseline > 1e-9:
        print(
            f"NON-DETERMINISTIC baseline ({baseline} vs {baseline_again}) — determinism pins "
            "not effective; fix before solving.",
            file=sys.stderr,
        )
        return 1

    d1, e_clip = e_clip_first_guess(FIELD, SPAWN, D0)
    grid = sorted({0.0, D0, *np.linspace(0.0, D0, 5).tolist(), max(0.0, min(D0, d1))})
    print(
        f"baseline (SP1-off) mean cod = {baseline:.1f}; E[clip]={e_clip:.3f}, "
        f"d1_analytical={d1:.3f}; grid={[round(g, 2) for g in grid]}"
    )

    def run_mean_on(rate: float) -> float:
        return mean_cod(sp1_on_config(base, FIELD, larva_rate=rate))

    res = solve_larva_rate(baseline, run_mean_on, grid_points=grid, tol=0.02)
    print("grid (rate, mean):", [(round(d, 3), round(m, 1)) for d, m in res.grid])
    print(
        f"result: feasible={res.feasible} converged={res.converged} rate={res.rate} "
        f"mean_on={res.mean_on} rel_err={res.rel_err} iters={res.iters} :: {res.message}"
    )
    if res.feasible and res.rate is not None:
        print(
            f"\nPASTE into osmose/calibration/larva_recal.py:\nRECAL_RATE = {res.rate!r}  "
            f"# SP1b solved {res.message}"
        )
    else:
        print("\nINFEASIBLE — leave RECAL_RATE = None; record the grid in the diagnostic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
