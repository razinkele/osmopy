#!/usr/bin/env python3
"""Cell occupancy and predation size-window selectivity on the Baltic (perf plan item 3).

Engine unmodified: a ``step_observer`` reads the end-of-step state. Same definitions as the
2026-09-12 EEC / Bay of Biscay table (docs/plans/2026-05-05-deep-review-remediation-plan.md,
"Exploring the algorithmic improvement"):

- occupied cells: cells holding at least one live, in-domain school with positive length;
- pairs scanned: sum over occupied cells of n_cell^2 (the kernel's Phase 1a loop, per sub-step);
- a pair (p, q) is eligible when q != p and r_min <= L_p / L_q < r_max at p's feeding stage.

The kernel repeats the scan every predation sub-step, so totals are also given x mortality.subdt.

Usage (from the repo root):
    PYTHONPATH=. .venv/bin/python scripts/baltic_cell_occupancy.py --years 10 --seed 42 \\
        --out baltic_cell_occupancy.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BALTIC = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
BINS = (("le15", 0, 15), ("16_50", 16, 50), ("51_100", 51, 100), ("gt100", 101, 10**9))


def _observer(rows: list[dict]):
    def observe(step, state, grid, config, map_sets) -> None:
        live = (state.abundance > 0) & ~state.is_out & (state.length > 0)
        idx = np.where(live)[0]
        if idx.size == 0:
            # Counts are zero; schools-per-cell statistics are undefined and kept out of them.
            rows.append(
                {
                    "step": step,
                    "schools": 0,
                    "cells": 0,
                    "mean": None,
                    "median": None,
                    "max": 0,
                    "scanned": 0,
                    "eligible": 0,
                    "bins": {k: 0 for k, _, _ in BINS},
                    "n_over_100": 0,
                }
            )
            return
        cell = state.cell_y[idx].astype(np.int64) * grid.nx + state.cell_x[idx]
        order = np.argsort(cell, kind="stable")
        idx, cell = idx[order], cell[order]
        groups = np.split(idx, np.flatnonzero(np.diff(cell)) + 1)
        sizes = np.array([len(g) for g in groups])
        sq = sizes.astype(np.int64) ** 2
        eligible = 0
        sp, stg, length = state.species_id, state.feeding_stage, state.length
        for g in groups:
            if len(g) < 2:
                continue
            ratio = length[g][:, None] / length[g][None, :]
            rmin = config.size_ratio_min[sp[g], stg[g]][:, None]
            rmax = config.size_ratio_max[sp[g], stg[g]][:, None]
            ok = (ratio >= rmin) & (ratio < rmax)
            np.fill_diagonal(ok, False)
            eligible += int(ok.sum())
        rows.append(
            {
                "step": step,
                "schools": int(idx.size),
                "cells": len(groups),
                "mean": float(sizes.mean()),
                "median": float(np.median(sizes)),
                "max": int(sizes.max()),
                "scanned": int(sq.sum()),
                "eligible": eligible,
                "bins": {k: int(sq[(sizes >= lo) & (sizes <= hi)].sum()) for k, lo, hi in BINS},
                "n_over_100": int((sizes > 100).sum()),
            }
        )

    return observe


def summarize(rows: list[dict], label: str, subdt: int) -> dict:
    scanned = sum(r["scanned"] for r in rows)
    eligible = sum(r["eligible"] for r in rows)
    occupied = [r for r in rows if r["mean"] is not None]
    return {
        "window": label,
        "steps": len(rows),
        "steps_with_schools": len(occupied),
        "schools_per_step_mean": round(float(np.mean([r["schools"] for r in rows]))),
        "occupied_cells_mean": round(float(np.mean([r["cells"] for r in rows]))),
        # Per-cell statistics over the steps that hold at least one school.
        "schools_per_cell_mean": round(float(np.mean([r["mean"] for r in occupied])), 1),
        "schools_per_cell_median": float(np.median([r["median"] for r in occupied])),
        "schools_per_cell_max": int(max(r["max"] for r in rows)),
        "pairs_scanned": scanned,
        "pairs_eligible": eligible,
        "eligible_fraction": round(eligible / scanned, 4) if scanned else None,
        "pairs_scanned_x_subdt": scanned * subdt,
        "scanned_share_by_cell_size": {
            k: round(sum(r["bins"][k] for r in rows) / scanned, 3) for k, _, _ in BINS
        }
        if scanned
        else None,
        "cells_over_100_per_step_mean": round(float(np.mean([r["n_over_100"] for r in rows])), 1),
    }


def main() -> int:
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    ap = argparse.ArgumentParser(description="Baltic cell occupancy and size-window selectivity.")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = dict(OsmoseConfigReader().read(BALTIC))
    cfg["simulation.time.nyear"] = str(args.years)
    subdt = int(float(cfg.get("mortality.subdt", "10")))
    ndt = int(float(cfg.get("simulation.time.ndtperyear", "24")))
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as d:
        PythonEngine().run(cfg, Path(d), seed=args.seed, step_observer=_observer(rows))
    late = max(0, args.years - 5)
    out = {
        "seed": args.seed,
        "years": args.years,
        "subdt": subdt,
        "year1": summarize([r for r in rows if r["step"] < ndt], "year 1", subdt),
        "final5": summarize(
            [r for r in rows if r["step"] >= late * ndt], f"years {late + 1}-{args.years}", subdt
        ),
        "all": summarize(rows, f"years 1-{args.years}", subdt),
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
