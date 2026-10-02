#!/usr/bin/env python
"""SP1b — solve each cod stock's larval rate so its SP1-on mean biomass matches its own
SP1-off baseline (per stock: cod_west and cod_east, Gauss-Seidel over the 1-D solver).

Usage: PYTHONPATH=. .venv/bin/python scripts/recalibrate_sp1b.py
Prints every engine evaluation as it happens, the per-stock grids, the joint result, and a
ready-to-paste `RECAL_RATES = {...}` block for osmose/calibration/larva_recal.py; writes the
full solve record to docs/diagnostics/sp1b_solve.json (provenance). ~20-30 15-yr Baltic runs —
foreground, single thread, nothing else running the engine.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numba
import numpy as np

from osmose.calibration.larva_recal import (
    JOINT_TOL,
    SP1_STOCKS,
    e_clip_first_guess,
    resolved_d0,
    solve_per_stock,
    sp1_on_config,
    species_index,
    stock_means,
    with_determinism,
)
from osmose.config import OsmoseConfigReader

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
FIELD = ROOT / "data" / "baltic" / "forcing" / "baltic_rv_field.nc"
MAPS = ROOT / "data" / "baltic" / "maps"
OUT_JSON = ROOT / "docs" / "diagnostics" / "sp1b_solve.json"
ORDER = ("cod_east", "cod_west")  # the large, RV-gated stock first; the small one adapts
TOL = 0.02


def _base_cfg() -> dict[str, str]:
    cfg = dict(OsmoseConfigReader().read(CONFIG))
    cfg["simulation.time.nyear"] = "15"
    return cfg


def _fmt_rates(rates: Mapping[str, float | None]) -> str:
    return ", ".join(f"{k}={'d0' if v is None else f'{v:.4f}'}" for k, v in sorted(rates.items()))


def main() -> int:
    numba.set_num_threads(1)  # runtime determinism pin
    base = _base_cfg()
    stocks = [s for s in ORDER if species_index(base, s) is not None]
    if not stocks:
        print(f"none of {SP1_STOCKS} declared in {CONFIG}", file=sys.stderr)
        return 1

    # Noise check: under the fixed-seed keys + single thread, f(d) must be reproducible,
    # else the solve chases noise. Require bit-identical repeat means before trusting it.
    off = with_determinism(base)
    t0 = time.time()
    off_means = stock_means(off)
    print(f"[{time.time() - t0:6.0f}s] baseline run: {_fmt_means(off_means, stocks)}", flush=True)
    off_again = stock_means(off)
    for s in stocks:
        if abs(off_again[s] - off_means[s]) / off_means[s] > 1e-9:
            print(
                f"NON-DETERMINISTIC baseline for {s} ({off_means[s]} vs {off_again[s]}) — "
                "determinism pins not effective; fix before solving.",
                file=sys.stderr,
            )
            return 1
    print(f"[{time.time() - t0:6.0f}s] baseline repeat bit-identical", flush=True)
    baselines = {s: off_means[s] for s in stocks}

    grids: dict[str, list[float]] = {}
    d0s: dict[str, float] = {}
    seeds: dict[str, dict[str, float]] = {}
    for s in stocks:
        d0 = resolved_d0(base, s)
        d1, e_clip = e_clip_first_guess(FIELD, MAPS / f"{s}_spawning.csv", d0)
        grids[s] = sorted({0.0, d0, *np.linspace(0.0, d0, 5).tolist(), max(0.0, min(d0, d1))})
        d0s[s] = d0
        seeds[s] = {"e_clip": e_clip, "d1_analytical": d1}
        print(
            f"{s}: baseline={baselines[s]:.1f} t  d0={d0:.4f}  E[clip]={e_clip:.3f}  "
            f"d1={d1:.3f}  grid={[round(g, 3) for g in grids[s]]}",
            flush=True,
        )

    n_eval = 0

    def run_means_on(rates: Mapping[str, float | None]) -> dict[str, float]:
        nonlocal n_eval
        n_eval += 1
        means = stock_means(sp1_on_config(base, FIELD, larva_rates=rates))
        print(
            f"[{time.time() - t0:6.0f}s] eval {n_eval:2d}: {_fmt_rates(rates)} -> "
            f"{_fmt_means(means, stocks)}",
            flush=True,
        )
        return means

    res = solve_per_stock(baselines, run_means_on, order=stocks, grids=grids, tol=TOL, max_sweeps=2)

    print()
    for s in stocks:
        r = res.per_stock[s]
        print(f"{s} grid (rate, mean):", [(round(d, 3), round(m, 1)) for d, m in r.grid])
        print(
            f"{s}: feasible={r.feasible} converged={r.converged} rate={r.rate} "
            f"mean_on={r.mean_on} rel_err={r.rel_err} iters={r.iters} :: {r.message}"
        )
    total_off = sum(baselines.values())
    total_on = sum(res.means.values())
    for i, h in enumerate(res.sweep_history, 1):
        print(f"sweep {i} joint rel_err: " + ", ".join(f"{s}={h[s]:.4f}" for s in stocks))
    print(
        f"\njoint (best sweep): converged={res.converged} sweeps={res.sweeps} "
        f"evaluations={res.evaluations}; "
        + "; ".join(f"{s}: {res.means[s]:.1f} t (rel_err {res.rel_errs[s]:.4f})" for s in stocks)
        + f"; total cod {total_off:.1f} -> {total_on:.1f} t ({total_on / total_off - 1:+.2%})"
    )

    today = datetime.now(tz=UTC).date().isoformat()
    lines = ["RECAL_RATES: dict[str, StockRecal] = {"]
    for s in stocks:
        r = res.per_stock[s]
        rate = res.rates[s]
        note = f"solved {today}: {r.message}" + ("" if res.converged else " (joint NOT converged)")
        lines.append(
            f"    {s!r}: StockRecal(rate={rate!r}, d0={d0s[s]!r}, baseline={baselines[s]!r}, "
            f"mean_on={res.means[s]!r}, rel_err={res.rel_errs[s]!r}, note={note!r}),"
        )
    lines.append("}")
    print("\nPASTE into osmose/calibration/larva_recal.py:\n" + "\n".join(lines))
    if not res.converged:
        print("\nNOT CONVERGED — record the grids in the diagnostic; leave infeasible stocks None.")

    record = {
        "note": (
            "Written by scripts/recalibrate_sp1b.py; each run REPLACES this file (a new solve is "
            "new provenance; prior solves live in git history). 'joint' is the BEST sweep "
            "(smallest worst-stock rel_err), 'per_stock' the 1-D solves of that sweep, "
            "'sweep_history' the joint rel_errs after every sweep. Frozen values in "
            "osmose/calibration/larva_recal.py are pasted from the printed block by hand."
        ),
        "date": today,
        "config": str(CONFIG.relative_to(ROOT)),
        "field": str(FIELD.relative_to(ROOT)),
        "tol": TOL,
        "order": stocks,
        "baselines": baselines,
        "d0": d0s,
        "first_guess": seeds,
        "grids": grids,
        "per_stock": {
            s: {
                "rate": r.rate,
                "feasible": r.feasible,
                "converged": r.converged,
                "mean_on": r.mean_on,
                "rel_err": r.rel_err,
                "iters": r.iters,
                "message": r.message,
                "grid": r.grid,
            }
            for s, r in res.per_stock.items()
        },
        "sweep_history": res.sweep_history,
        "joint": {
            "best_sweep": res.best_sweep,
            "joint_tol": JOINT_TOL,
            "rates": res.rates,
            "means": res.means,
            "rel_errs": res.rel_errs,
            "converged": res.converged,
            "sweeps": res.sweeps,
            "evaluations": res.evaluations,
            "total_cod_off": total_off,
            "total_cod_on": total_on,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nsolve record written to {OUT_JSON.relative_to(ROOT)}")
    return 0


def _fmt_means(means: dict[str, float], stocks: list[str]) -> str:
    return ", ".join(f"{s}={means[s]:.1f}" for s in stocks)


if __name__ == "__main__":
    sys.exit(main())
