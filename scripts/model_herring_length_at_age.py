#!/usr/bin/env python
"""Model herring length-at-age in the BITS survey convention (issue #247), final decade, for the v1.9.0 state (A0)
and the v1.8.8 state (A4), seed 42, 50 yr.

A read-only hook on `_collect_by_life_stage` (paired with `regulate_recruitment` for the step) records,
for herring schools (non-egg, abundance > 0): ICES-convention age = calendar years since birth
(1 Jan birthday) and length, in the BITS windows Q1 = Feb-Mar (steps 2-5 of 24) and Q4 = Oct-Nov
(steps 18-21). Per (quarter, age): N, sum N*L, sum N*L^2, N with L >= 18 (model maturity size), N with
L >= 14.4 (BITS L50), N mature by the engine rule. Writes JSON to argv[1].

Usage: .venv/bin/python scripts/model_herring_length_at_age.py docs/diagnostics/herring_model_length_at_age_2026-11-12.json
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import json
import multiprocessing as mp
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

NDT, N_YEAR, HERRING = 24, 50, 1
Q_STEPS = {"Q1": range(2, 6), "Q4": range(18, 22)}
MAX_AGE = 9


def run(state: str) -> dict:
    import b2_herring_attribution as attr

    import osmose.engine.processes.reproduction as repro
    import osmose.engine.simulate as sim
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    acc = {q: np.zeros((MAX_AGE + 1, 6)) for q in Q_STEPS}  # N, NL, NL2, N>=18, N>=14.4, N mature
    pending = {}
    real_collect, real_regulate = sim._collect_by_life_stage, repro.regulate_recruitment

    def collect_hook(st, config):
        out = real_collect(st, config)
        m = (st.species_id == HERRING) & (~st.is_egg) & (st.abundance > 0)
        pending["snap"] = (
            st.age_dt[m].copy(),
            st.length[m].copy(),
            st.abundance[m].copy(),
            float(config.maturity_size[HERRING]),
            int(config.maturity_age_dt[HERRING]),
        )
        return out

    def regulate_hook(n_eggs_linear, ssb, seeded, config, step):
        out = real_regulate(n_eggs_linear, ssb, seeded, config, step)
        snap = pending.pop("snap", None)
        y, s = divmod(int(step), NDT)
        if snap is None or y < N_YEAR - 10:
            return out
        for q, steps in Q_STEPS.items():
            if s in steps:
                age_dt, length, n, msize, mage = snap
                ices_age = y - (int(step) - age_dt) // NDT
                for a in range(MAX_AGE + 1):
                    k = ices_age == a if a < MAX_AGE else ices_age >= a
                    if not k.any():
                        continue
                    L, N = length[k], n[k]
                    acc[q][a] += [
                        N.sum(),
                        (N * L).sum(),
                        (N * L * L).sum(),
                        N[L >= 18.0].sum(),
                        N[L >= 14.4].sum(),
                        N[(L >= msize) & (age_dt[k] >= mage)].sum(),
                    ]
        return out

    sim._collect_by_life_stage, repro.regulate_recruitment = collect_hook, regulate_hook
    v188 = {f: attr._git_show(f) for f in (attr.F_LTL, attr.F_MORT, attr.F_SEASON, attr.F_CODMAP)}
    tmp = Path(tempfile.mkdtemp(prefix=f"growth_{state}_"))
    try:
        cfg = dict(OsmoseConfigReader().read(str(attr.prepare_arm_dir(state, v188, tmp))))
        cfg["simulation.time.nyear"] = str(N_YEAR)
        res = PythonEngine().run_in_memory(cfg, seed=42)
        herring_fdm = attr.decade_mean(res.biomass()["herring"].to_numpy(dtype=float))
    finally:
        sim._collect_by_life_stage, repro.regulate_recruitment = real_collect, real_regulate
        shutil.rmtree(tmp, ignore_errors=True)

    table = {}
    for q, a_arr in acc.items():
        rows = {}
        for a in range(MAX_AGE + 1):
            N, NL, NL2, n18, n144, nmat = a_arr[a]
            if N <= 0:
                continue
            mean = NL / N
            rows[a] = {
                "mean_L": mean,
                "sd_L": float(np.sqrt(max(NL2 / N - mean * mean, 0.0))),
                "share_N": None,
                "frac_ge_18": n18 / N,
                "frac_ge_14_4": n144 / N,
                "frac_mature_engine": nmat / N,
                "N": N,
            }
        tot = sum(r["N"] for r in rows.values())
        for r in rows.values():
            r["share_N"] = r["N"] / tot
        table[q] = rows
    return {"state": state, "herring_final_decade_biomass": herring_fdm, "length_at_age": table}


if __name__ == "__main__":
    with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as pool:
        out = list(pool.map(run, ["A0", "A4"]))
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    for o in out:
        print(
            f"== {o['state']} (herring final-decade biomass {o['herring_final_decade_biomass']:,.0f} t)"
        )
        for q, rows in o["length_at_age"].items():
            print(
                f"  {q}: "
                + "  ".join(
                    f"a{a}: {r['mean_L']:.1f}cm sd{r['sd_L']:.1f} >=18:{r['frac_ge_18']:.0%} >=14.4:{r['frac_ge_14_4']:.0%} nshare:{r['share_N']:.0%}"
                    for a, r in sorted(rows.items(), key=lambda kv: int(kv[0]))
                    if int(a) <= 6
                )
            )
