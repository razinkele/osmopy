#!/usr/bin/env python
"""Locate C3's length-at-age deficit: is it `c_m`, `Imax`, `beta`, or the fit's objective?

Background
----------
The C3 bioen overlay (`data/baltic/scenarios/c3_bioen/`) fits each species' growth offline to
~8% RMS, yet the coupled engine collapses 5 of 9 stocks
(`docs/baltic_c3_bioen_stage1_2026-09-05.md`, verdict CLOSE BY CHARACTERIZATION). Section 9 of
that document left the locus open between the bioen fit, the predation-accessibility matrix and
the growth-rate structure. This script is the diagnosis that closed it (2026-09-13); it is
committed so the measurement can be repeated after any re-fit.

Three modes, cheapest last-resort first:

``--curve``     No engine run. Runs the fit's OWN forward model
                (`osmose.calibration.bioen_offline.simulate_growth`) with the COMMITTED overlay
                parameters against each species' config vBGF. This is the mode that names the
                cause, and it takes seconds.
``--ingestion`` Engine run. Realized ingestion as a fraction of the `Imax` cap, per species,
                PRE-COLLAPSE. Discriminates food limitation from everything else.
``--budget``    Engine run. `m_share = e_maint/e_gross` vs the fitted target, `phi_T` vs the
                fit's `phiT(Tbar)`, and `m_share` across weight terciles. Discriminates `c_m`,
                the temperature inflation, and `beta` respectively.

What it found (2026-09-13), for comparison against a re-fit
-----------------------------------------------------------
* ingestion/cap 0.92-0.98 for every collapsing stock -> NOT food-limited. Inverted, even: the two
  lowest ratios (pikeperch 0.36, smelt 0.38) are both survivors.
* m_share at or BELOW the 0.30 target for every collapsing stock -> NOT `c_m`.
* m_share flat across weight terciles -> NOT `beta`.
* measured phi_T within 8-19% of the fit's `phiT(Tbar)`, mostly higher -> NOT the inflation.
* fit/vBGF 0.58 (cod_west) and 0.49 (perch) at AGE 1, rising to 1.14-1.17 by age 3+
  -> the fit is short in year one and overshoots later. Its RMS runs over the whole >=1yr range
  on ABSOLUTE lengths, so ages 2-8 dominate the residual and year one is nearly unweighted.

Re-fit target is therefore the juvenile regime (`theta`, `c_rate`, `larvae_thres_dt` -- the
larval cap boost in `simulate_growth`), with early ages weighted in the residual. Pikeperch is a
standing exception: 0.47 at age 1 yet it survives.

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/c3_growth_deficit_diagnosis.py --curve
    PYTHONPATH=. .venv/bin/python scripts/c3_growth_deficit_diagnosis.py --ingestion --years 10
    PYTHONPATH=. .venv/bin/python scripts/c3_growth_deficit_diagnosis.py --budget --years 6

Never run two engine jobs at once on this machine (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OVERLAY_PATH = ROOT / "data" / "baltic" / "scenarios" / "c3_bioen" / "c3_bioen_arm.json"

#: Each species' own habitat-mean temperature, verbatim from the results doc's Sec.3 parameter
#: table (its own depth layer, its own movement-map footprint). Used ONLY by ``--curve``, and
#: deliberately so: the question there is whether the fit reproduces its own target under its own
#: assumptions, so the fit's `Tbar` is the apples-to-apples input. Re-derive from Sec.3 if the
#: overlay is re-fitted.
TBAR_BY_SP = {0: 8.69, 1: 8.14, 2: 8.53, 3: 6.08, 4: 7.32, 5: 9.24, 6: 7.17, 7: 8.09, 8: 5.96}

#: The fit's maintenance share of Imax at the anchor temperature (`BioenFixed().m_share`).
M_SHARE_TARGET = 0.30


def _load():
    """Production Baltic + the committed C3 bioen overlay, assembled exactly as the A/B does."""
    import importlib.util

    from osmose.config import OsmoseConfigReader
    from osmose.demo import osmose_demo

    spec = importlib.util.spec_from_file_location("_ab", ROOT / "scripts" / "baltic_c3_bioen_ab.py")
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)

    tmp = Path(tempfile.mkdtemp(prefix="c3_diag_"))
    demo = osmose_demo("baltic", tmp)
    raw = dict(OsmoseConfigReader().read(str(demo["config_file"])))
    overlay = {k: v for k, v in json.loads(OVERLAY_PATH.read_text()).items() if k != "_meta"}
    return raw, overlay, ab


def _names(raw):
    return [raw[f"species.name.sp{i}"] for i in range(int(raw["simulation.nspecies"]))]


def mode_curve(raw, overlay):
    """The fit's own forward model vs each species' config vBGF, by age. No engine run."""
    from osmose.calibration import bioen_offline as bo

    ndt = int(raw["simulation.time.ndtperyear"])
    fx = bo.BioenFixed()
    names = _names(raw)
    print("\nfit forward model vs config vBGF -- length ratio by age (1.0 = on target)")
    print("the fit's RMS is computed on ABSOLUTE lengths over the whole >=1yr range,")
    print("so late ages dominate it and year one is nearly unweighted.\n")
    print(f"{'species':<14}{'m0':>6}" + "".join(f"{f'age{a}':>8}" for a in range(1, 9)))
    for i, name in enumerate(names):
        linf = float(raw[f"species.linf.sp{i}"])
        k = float(raw[f"species.k.sp{i}"])
        t0 = float(raw[f"species.t0.sp{i}"])
        cf = float(raw[f"species.length2weight.condition.factor.sp{i}"])
        b = float(raw[f"species.length2weight.allometric.power.sp{i}"])
        life = int(float(raw[f"species.lifespan.sp{i}"]))
        m0 = float(overlay[f"species.maturity.m0.sp{i}"])
        w = bo.simulate_growth(
            float(overlay[f"predation.ingestion.rate.max.sp{i}"]),
            float(overlay[f"species.maturity.r.sp{i}"]),
            float(overlay[f"species.bioen.mobilized.tp.sp{i}"]),
            float(overlay[f"species.bioen.maint.energy.c_m.sp{i}"]),
            np.full(ndt, TBAR_BY_SP[i]),
            float(raw[f"species.egg.weight.sp{i}"]),
            ndt * life,
            ndt,
            cf,
            b,
            m0,
            float(overlay[f"species.maturity.m1.sp{i}"]),
            fx,
        )
        cells = ""
        for a in range(1, 9):
            if a > life:
                cells += f"{'--':>8}"
                continue
            lv = linf * (1.0 - np.exp(-k * (a - t0)))
            cells += f"{((w[ndt * a] / cf) ** (1.0 / b)) / lv:>8.2f}"
        print(f"{name:<14}{m0:>6.1f}{cells}")
    print("\n<1 at age 1 rising to >=1 later => the objective under-weights year one")


def mode_ingestion(raw, overlay, ab, years, seed):
    """Realized ingestion / Imax cap, per species, pre-collapse. Engine run."""
    from osmose.engine import PythonEngine
    from osmose.engine import simulate as sim
    from osmose.engine.processes.bioen_predation import per_fish_ingestion_cap

    raw = dict(raw)
    raw["simulation.time.nyear"] = str(years)
    cfg = ab.arm_config(raw, "bioen", overlay)
    names = _names(raw)
    ndt = int(raw["simulation.time.ndtperyear"])
    rec: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    orig = sim._bioen_step

    def instrumented(state, config, temp_data, step, o2_data=None, trait_overrides=None, **kw):
        # Bound is the engine's own `max_eatable = cap_fish * inst_abd` per sub-step
        # (mortality.py). Abundance declines within a step, so summing on END-of-step abundance
        # slightly OVER-states what was available => the ratio is biased LOW (conservative).
        cap = per_fish_ingestion_cap(
            state.weight,
            state.species_id,
            state.age_dt,
            config.bioen_i_max_all,
            config.bioen_beta,
            config.bioen_larvae_thres_dt,
            config.bioen_theta,
            config.bioen_c_rate,
            config.n_species,
            config.n_dt_per_year,
            config.mortality_subdt,
        )
        ok = (state.age_dt >= state.first_feeding_age_dt) & (state.abundance > 0)
        if getattr(state, "is_out", None) is not None:
            ok = ok & ~state.is_out
        for sp in range(config.n_species):
            m = ok & (state.species_id == sp)
            if m.any():
                r = rec[step][sp]
                r[0] += float(np.sum(state.preyed_biomass[m]))
                r[1] += float(np.sum(cap[m] * state.abundance[m])) * config.mortality_subdt
        return orig(state, config, temp_data, step, o2_data, trait_overrides, **kw)

    sim._bioen_step = instrumented
    try:
        PythonEngine().run_in_memory(cfg, seed=seed)
    finally:
        sim._bioen_step = orig

    print(f"\nrealized ingestion / Imax cap -- {years}yr, seed {seed} (ratio biased LOW)")
    print("a value >1.0 is impossible (the cap is a hard bound); seeing none validates the bound\n")
    bands = [(0, 3), (3, 6), (6, years)] if years > 6 else [(0, 2), (2, 4), (4, years)]
    print(f"{'species':<14}" + "".join(f"{f'yr{lo + 1}-{hi}':>10}" for lo, hi in bands))
    for sp, name in enumerate(names):
        cells = ""
        for lo, hi in bands:
            num = sum(rec[s][sp][0] for s in rec if lo * ndt <= s < hi * ndt and sp in rec[s])
            den = sum(rec[s][sp][1] for s in rec if lo * ndt <= s < hi * ndt and sp in rec[s])
            cells += f"{num / den:>10.3f}" if den > 0 else f"{'--':>10}"
        print(f"{name:<14}{cells}")
    print("\n~1 => eating at the cap (parameter/objective locus) | <<1 => food-limited (supply)")


def mode_budget(raw, overlay, ab, years, seed):
    """m_share vs target, phi_T, and m_share by weight tercile. Engine run."""
    from osmose.engine import PythonEngine
    from osmose.engine import simulate as sim
    from osmose.engine.processes import energy_budget as eb

    raw = dict(raw)
    raw["simulation.time.nyear"] = str(years)
    cfg = ab.arm_config(raw, "bioen", overlay)
    names = _names(raw)
    acc: dict = defaultdict(lambda: {"eg": 0.0, "em": 0.0, "phi": [], "wq": [[], [], []]})
    held: dict = {}
    orig_budget = eb.compute_energy_budget
    orig_step = sim._bioen_step

    def step(state, config, temp_data, s, o2_data=None, trait_overrides=None, **kw):
        held["config"] = config
        return orig_step(state, config, temp_data, s, o2_data, trait_overrides, **kw)

    def budget(
        ingestion,
        weight,
        abundance,
        gonad_weight,
        age_dt,
        length,
        temp_c,
        assimilation,
        c_m,
        beta,
        eta,
        r,
        m0,
        m1,
        e_maint_energy,
        phi_t,
        f_o2,
        n_dt_per_year,
        enet_faced,
    ):
        out = orig_budget(
            ingestion,
            weight,
            abundance,
            gonad_weight,
            age_dt,
            length,
            temp_c,
            assimilation,
            c_m,
            beta,
            eta,
            r,
            m0,
            m1,
            e_maint_energy,
            phi_t,
            f_o2,
            n_dt_per_year,
            enet_faced,
        )
        _, _, _, egr, em, _ = out
        cfgo = held.get("config")
        # `compute_energy_budget` is called once per species per step with that species' scalar
        # c_m; match it back to the species index.
        sp = int(np.argmin(np.abs(np.asarray(cfgo.bioen_c_m) - c_m))) if cfgo is not None else -1
        feeding = age_dt >= 1
        if feeding.any() and float(np.sum(egr[feeding])) > 0:
            a = acc[sp]
            a["eg"] += float(np.sum(egr[feeding]))
            a["em"] += float(np.sum(em[feeding]))
            pt = np.asarray(phi_t, dtype=float)
            a["phi"].append(float(np.mean(pt[feeding])) if pt.ndim else float(pt))
            w = weight[feeding] * 1e6
            if len(w) > 9:
                qs = np.quantile(w, [1 / 3, 2 / 3])
                sels = (w <= qs[0], (w > qs[0]) & (w <= qs[1]), w > qs[1])
                for kq, sel in enumerate(sels):
                    g = float(np.sum(egr[feeding][sel]))
                    if g > 0:
                        a["wq"][kq].append(float(np.sum(em[feeding][sel])) / g)
        return out

    sim._bioen_step = step
    eb.compute_energy_budget = budget
    try:
        PythonEngine().run_in_memory(cfg, seed=seed)
    finally:
        eb.compute_energy_budget = orig_budget
        sim._bioen_step = orig_step

    print(f"\nbudget terms -- {years}yr, seed {seed}; fit target m_share = {M_SHARE_TARGET:.2f}\n")
    print(f"{'species':<14}{'m_share':>9}{'phiT':>8}{'m_small':>9}{'m_mid':>8}{'m_large':>9}")
    for sp, name in enumerate(names):
        a = acc.get(sp)
        if not a or a["eg"] <= 0:
            print(f"{name:<14}{'--':>9}")
            continue
        q = [float(np.mean(v)) if v else float("nan") for v in a["wq"]]
        print(
            f"{name:<14}{a['em'] / a['eg']:>9.3f}{float(np.mean(a['phi'])):>8.3f}"
            # `x == x` is the NaN test (NaN != itself): print "--" for NaN, not a typo.
            + "".join(f"{x:>9.2f}" if x == x else f"{'--':>9}" for x in q)  # noqa: PLR0124
        )
    print("\nm_share >> target => c_m | phiT far below Sec.3's phiT(Tbar) => inflation")
    print("m_small != m_large => beta (both e_gross and e_maint scale as w^beta)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--curve", action="store_true", help="fit forward model vs vBGF (no engine)")
    ap.add_argument("--ingestion", action="store_true", help="realized ingestion / cap")
    ap.add_argument("--budget", action="store_true", help="m_share, phi_T, weight terciles")
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not (args.curve or args.ingestion or args.budget):
        ap.error("pick at least one of --curve / --ingestion / --budget")

    warnings.simplefilter("ignore")
    raw, overlay, ab = _load()
    if args.curve:
        mode_curve(raw, overlay)
    if args.ingestion:
        mode_ingestion(raw, overlay, ab, args.years, args.seed)
    if args.budget:
        mode_budget(raw, overlay, ab, args.years, args.seed)


if __name__ == "__main__":
    main()
