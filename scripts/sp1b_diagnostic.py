#!/usr/bin/env python
"""SP1b diagnostic: per stock, the recalibrated rate (with the d0 it was solved against),
achieved rel-err, and the overshoot ratio SP1-on-recalibrated vs SP1-off (measured, not
gated — does mean-neutral spatial egg-survival damp the boom/bust?).

Usage: PYTHONPATH=. .venv/bin/python scripts/sp1b_diagnostic.py   (two 15-yr Baltic runs)
Writes docs/diagnostics/sp1b_recalibration.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numba

from osmose.calibration.larva_recal import (
    JOINT_TOL,
    RECAL_RATES,
    SP1_STOCKS,
    mean_cod_from_biomass,
    resolved_d0,
    sp1_on_config,
    species_index,
    stock_means_from_biomass,
    stock_overshoot_from_biomass,
    with_determinism,
)
from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
FIELD = ROOT / "data" / "baltic" / "forcing" / "baltic_rv_field.nc"
OUT = ROOT / "docs" / "diagnostics" / "sp1b_recalibration.md"


def main() -> int:
    numba.set_num_threads(1)
    base = dict(OsmoseConfigReader().read(CONFIG))
    base["simulation.time.nyear"] = "15"
    stocks = [s for s in SP1_STOCKS if species_index(base, s) is not None]

    bio_off = PythonEngine().run_in_memory(with_determinism(base), seed=0).biomass()
    off = stock_means_from_biomass(bio_off)
    over_off = stock_overshoot_from_biomass(bio_off)
    total_off = mean_cod_from_biomass(bio_off)

    lines = ["# SP1b recalibration diagnostic", ""]
    active = {s: e for s, e in RECAL_RATES.items() if e.rate is not None}
    if not active:
        lines += [
            "RESULT: no solved per-stock rate (RECAL_RATES is empty / all infeasible).",
            "SP1-off baselines: "
            + ", ".join(f"{s}={off[s]:.1f} t (overshoot {over_off[s]:.2f})" for s in stocks),
            "See docs/diagnostics/sp1b_solve.json for the solve grids.",
        ]
    else:
        bio_on = PythonEngine().run_in_memory(sp1_on_config(base, FIELD), seed=0).biomass()
        on = stock_means_from_biomass(bio_on)
        over_on = stock_overshoot_from_biomass(bio_on)
        total_on = mean_cod_from_biomass(bio_on)
        lines += [
            "SP1 (spatial RV egg-survival clip) enabled on: " + ", ".join(stocks) + ".",
            (
                "Each stock's larval rate (resolved per-cohort) is solved so its own SP1-on mean "
                "matches its SP1-off mean: 1-D solver tol 0.02 on the stock's own axis, joint "
                f"band JOINT_TOL = {JOINT_TOL} on the pair (see 'Coupling' below). Rates are "
                "frozen with the d0 they were solved against; `sp1_on_config` refuses a config "
                "whose d0 has moved."
            ),
            "",
            (
                "| stock | d0 | rate | baseline (t) | on_recal (t) | rel_err | overshoot off | "
                "overshoot on |"
            ),
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in stocks:
            e = RECAL_RATES.get(s)
            rate = "— (d0 stands)" if e is None or e.rate is None else f"{e.rate:.4f}"
            lines.append(
                f"| {s} | {resolved_d0(base, s):.4f} | {rate} | {off[s]:.1f} | {on[s]:.1f} | "
                f"{abs(on[s] / off[s] - 1):.4f} | {over_off[s]:.2f} | {over_on[s]:.2f} |"
            )
        lines += [
            "",
            (
                f"total cod: off={total_off:.1f}  on_recal={total_on:.1f}  "
                f"drift={total_on / total_off - 1:+.2%} (measured, not gated)"
            ),
            "",
            "## Overshoot (max/mean over years 3-14) — measured, NOT gated",
        ]
        for s in stocks:
            verdict = "damps" if over_on[s] < over_off[s] else "does not damp"
            lines.append(
                f"{s}: off={over_off[s]:.2f}  on_recal={over_on[s]:.2f}  "
                f"ratio={over_on[s] / over_off[s]:.2f}  ({verdict} the boom/bust)"
            )
        lines += [
            "",
            "## Coupling: why the joint band is wider than the 1-D tol",
            (
                "With its own rate fixed at the root, each stock's mean still moves with the OTHER "
                "stock's rate, and not smoothly: on the 2026-10-01 solve (47 evaluations, two "
                "Gauss-Seidel sweeps, docs/diagnostics/sp1b_solve.log) the half-range within "
                "+-0.1 of the roots was 2.0-2.1% for cod_east across cod_west's rate and 3.8% for "
                "cod_west across cod_east's rate. Sweep 1 ended cod_east -2.9% / cod_west +1.2%, "
                "sweep 2 +4.0% / +0.9%; neither sweep put both inside 2%, and the jitter is as "
                "wide as that band, so a joint 2% is below the coupling's resolution. The frozen "
                "state is the best sweep (1) and the joint band is set above the jitter. A third "
                "sweep could land inside 2% by chance, which would make the drift guard trip on "
                "trajectory sensitivity rather than drift."
            ),
            "",
            "## Caveat: stacked RV terms on cod_east",
            (
                "cod_east carries the temporal RV gate (`reproduction.rv.gate.*`, its dominant "
                "control) AND, under this overlay, the spatial RV clip. The two have not been A/B "
                "gated against each other; the per-stock neutrality above holds for the stacked "
                "pair as a whole, not for either term alone."
            ),
        ]
    lines += [
        "",
        "## History",
        (
            "The 2026-07-02 solve (`RECAL_RATE = 14.6551`, aggregate cod, d0=15.0) was retired "
            "on 2026-10-01: the Baltic baseline was recalibrated on 2026-07-24 (sp0 larva rate "
            "360 -> 243.76 per year, resolved d0 15.0 -> 10.157), so the frozen constant had "
            "silently become a larval-mortality INCREASE; stacked on the spatial clip it drove "
            "cod_west extinct under SP1 (6432 -> 1 t). Issue #131 attributed this to the "
            "2026-07-25 cod E/W split; the split only made it visible. Rates now carry their d0 "
            "so this cannot recur silently."
        ),
    ]
    print("\n".join(lines))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
