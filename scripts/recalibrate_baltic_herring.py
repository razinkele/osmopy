#!/usr/bin/env python
"""Herring maturity + adult-mortality recalibration of the Baltic config (issue #247).

Evaluates override points in parallel (single-threaded workers), each a 50-yr run of the
production config with any of:
  maturity     species.maturity.size.sp1 (cm)
  m_add        mortality.additional.rate.sp1 (per year; the reader does not migrate it)
  larva_sp1    mortality.additional.larva.rate.sp1 in RESOLVED per-step units (CSV = x 24)
  larva_sp0    mortality.additional.larva.rate.sp0, resolved
and records, per run: every focal species' final-decade mean biomass, herring's final-decade SSB
(engine output, `output.ssb.enabled`), and herring's Q1 numbers-at-age shares by ICES age (read-only
hook, as in scripts/model_herring_length_at_age.py).

Pre-stated design: docs/diagnostics/baltic_herring_recal_prestated_2026-11-12.md.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/recalibrate_baltic_herring.py --grid --out <json>
    PYTHONPATH=. .venv/bin/python scripts/recalibrate_baltic_herring.py --points <points.json> --out <json>
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
N_YEAR, NDT, HERRING = 50, 24, 1
Q1_STEPS = range(2, 6)
MAX_AGE = 8
M_ADD_KEY = "mortality.additional.rate.sp1"
LARVA_KEY = "mortality.additional.larva.rate.sp{}"
MAT_KEY = "species.maturity.size.sp1"
ENVELOPES = {  # assessed stocks, biomass_targets.csv
    "cod_west": (4.0e3, 2.5e4),
    "cod_east": (6.0e4, 8.5e4),
    "herring": (8.0e5, 3.0e6),
    "sprat": (8.0e5, 2.5e6),
    "flounder": (2.0e4, 1.0e5),
}
M_GRID = (0.2, 0.6, 1.1)
L_GRID = (1.2, 2.0, 3.0, 4.1, 5.5, 7.5, 10.0)


def run_point(pt: dict) -> dict:
    import osmose.engine.processes.reproduction as repro
    import osmose.engine.simulate as sim
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    cfg = dict(OsmoseConfigReader().read(str(CONFIG)))
    # Units: the adult rate is used as read (per year); the larval rate arrives resolved (/24).
    assert abs(float(cfg[M_ADD_KEY]) - 2.247192881951114) < 1e-12, cfg[M_ADD_KEY]
    assert abs(float(cfg[LARVA_KEY.format(1)]) - 28.885927824368395 / NDT) < 1e-8, cfg[
        LARVA_KEY.format(1)
    ]
    if pt.get("maturity") is not None:
        cfg[MAT_KEY] = repr(float(pt["maturity"]))
    if pt.get("m_add") is not None:
        cfg[M_ADD_KEY] = repr(float(pt["m_add"]))
    for sp in (0, 1):
        if pt.get(f"larva_sp{sp}") is not None:
            cfg[LARVA_KEY.format(sp)] = repr(float(pt[f"larva_sp{sp}"]))
    if pt.get("ssb_output", True):
        cfg["output.ssb.enabled"] = "true"
    cfg["simulation.time.nyear"] = str(N_YEAR)

    acc = np.zeros(MAX_AGE + 1)
    pending = {}
    real_collect, real_regulate = sim._collect_by_life_stage, repro.regulate_recruitment

    def collect_hook(st, config):
        out = real_collect(st, config)
        m = (st.species_id == HERRING) & (~st.is_egg) & (st.abundance > 0)
        pending["snap"] = (st.age_dt[m].copy(), st.abundance[m].copy())
        return out

    def regulate_hook(n_eggs_linear, ssb, seeded, config, step):
        out = real_regulate(n_eggs_linear, ssb, seeded, config, step)
        snap = pending.pop("snap", None)
        y, s = divmod(int(step), NDT)
        if snap is not None and y >= N_YEAR - 10 and s in Q1_STEPS:
            age_dt, n = snap
            ices_age = np.minimum(y - (int(step) - age_dt) // NDT, MAX_AGE)
            np.add.at(acc, ices_age, n)
        return out

    sim._collect_by_life_stage, repro.regulate_recruitment = collect_hook, regulate_hook
    t0 = time.time()
    try:
        res = PythonEngine().run_in_memory(cfg, seed=int(pt.get("seed", 42)))
        bio = res.biomass()
        ssb = res.ssb() if pt.get("ssb_output", True) else None
    finally:
        sim._collect_by_life_stage, repro.regulate_recruitment = real_collect, real_regulate

    def dm(series) -> float:
        x = np.asarray(series, dtype=float)
        x = x if len(x) == N_YEAR else x.reshape(N_YEAR, -1).mean(axis=1)
        return float(x[-10:].mean())

    species = [c for c in bio.columns if c not in ("Time", "species")]
    out = {
        **pt,
        "resolved_larva_sp1": float(cfg[LARVA_KEY.format(1)]),
        "resolved_larva_sp0": float(cfg[LARVA_KEY.format(0)]),
        "m_add_used": float(cfg[M_ADD_KEY]),
        "maturity_used": float(cfg[MAT_KEY]),
        "biomass": {sp: dm(bio[sp]) for sp in species},
        "herring_ssb": dm(ssb["herring"]) if ssb is not None else None,
        "herring_q1_age_share": (acc / acc.sum()).tolist() if acc.sum() > 0 else None,
        "seconds": round(time.time() - t0, 1),
    }
    out["in_envelope"] = {sp: lo <= out["biomass"][sp] <= hi for sp, (lo, hi) in ENVELOPES.items()}
    return out


def grid_points() -> list[dict]:
    pts = [
        {"label": "control_production_ssb_on"},
        {"label": "ref_mat14.5_current", "maturity": 14.5},
    ]
    for m in M_GRID:
        for lr in L_GRID:
            pts.append({"label": f"m{m}_L{lr}", "maturity": 14.5, "m_add": m, "larva_sp1": lr})
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--points", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    pts = grid_points() if args.grid else json.loads(args.points.read_text())
    print(f"{len(pts)} points x {N_YEAR} yr, {args.workers} workers", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for r in pool.map(run_point, pts):
            results.append(r)
            b = r["biomass"]
            ages = r["herring_q1_age_share"]
            share2 = f"{sum(ages[2:]):5.1%}" if ages else "  n/a"
            ssb = r["herring_ssb"] or 0.0
            ssb_share = f"{ssb / b['herring']:5.1%}" if b["herring"] > 0 else "  n/a"
            print(
                f"[{time.time() - t0:6.0f}s] {r['label']:26s} herring {b['herring'] / 1e3:8.0f} kt"
                f" SSB {ssb / 1e3:7.0f} kt ({ssb_share}) age2+ {share2}"
                f" | codW {b['cod_west'] / 1e3:6.1f} codE {b['cod_east'] / 1e3:6.1f}"
                f" sprat {b['sprat'] / 1e3:6.0f} flo {b['flounder'] / 1e3:5.1f}"
                f" | env {sum(r['in_envelope'].values())}/5 ({r['seconds']}s)",
                flush=True,
            )
            # Save after every run, so a later failure never loses finished results.
            args.out.write_text(json.dumps(results, indent=1, allow_nan=False) + "\n")
    args.out.write_text(json.dumps(results, indent=1, allow_nan=False) + "\n")
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
