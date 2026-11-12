#!/usr/bin/env python
"""Why is the recalibrated herring less sensitive to B2's warming knob? A channel breakdown.

The B2 attribution (``docs/diagnostics/b2_herring_attribution_2026-11-12.md``) found that
herring's ``rcp85_ref`` decline is -26.5% on v1.9.0 (arm A0) and -61.2% with all of #236 reverted
(arm A4, which is v1.8.8 as released). This run takes both states, baseline and ``rcp85_ref``, and
records herring's life cycle with READ-ONLY hooks. No config key is changed:

* ``reproduction.regulate_recruitment``: per step, the engine's spawning biomass (SSB), linear
  eggs, eggs after the Shepherd stock-recruitment curve, and eggs after the knob.
* ``simulate._collect_by_life_stage``: per step, after mortality, herring deaths by cause for each
  class (egg / age-0 post-egg / age 1+), each class's standing stock, and recruits. A recruit is a
  fish at ``age_dt == n_dt_per_year`` (age advances by exactly 1 per step in reproduction, so each
  cohort is counted once).

Channels (warm/base ratios over the final decade; their product is 1 by construction):
  lin    linear eggs / SSB          (fecundity per spawning tonne; expected ~1)
  SR     eggs after SR / linear     (Shepherd compensation at the egg stage)
  knob   eggs after knob / after SR (must equal f exactly on non-seeded steps)
  early  recruits / eggs            (egg -> age-1 survival)
  SPR    SSB / recruits             (spawning biomass per recruit)

Pre-stated criterion: ``docs/diagnostics/herring_compensation_prestated_2026-11-12.md``.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/herring_compensation_breakdown.py \
        --out docs/diagnostics/herring_compensation_2026-11-12.json
"""

from __future__ import annotations

import os

# Single-threaded Numba per worker; must be set before numba is imported anywhere.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import b2_herring_attribution as attr

SEEDS = attr.SEEDS
STATES = ("A0", "A4")
VARIANTS = ("baseline", "rcp85_ref")
HERRING = 1
N_YEAR = 50
NDT = 24
WIN = (40, 50)  # final decade, years 40..49
CAUSES = (
    "PREDATION",
    "STARVATION",
    "ADDITIONAL",
    "FISHING",
    "OUT",
    "FORAGING",
    "DISCARDS",
    "AGING",
)
CLASSES = ("egg", "age0", "age1p")
ATTR_JSON = ROOT / "docs/diagnostics/b2_herring_attribution_2026-11-11.json"
KEEP_FROM = 38  # stored per-year arrays start here (the analysis needs years 39..49)
PER_YEAR = (
    "ssb_annual_mean",
    "e_lin",
    "e_sr",
    "e_fin",
    "seeded_steps",
    "recruits",
    "deaths",
    "n_end",
)


def run_one(job: tuple) -> dict:
    state_arm, variant, seed, v188, overlays, current_mort = job
    import osmose.engine.processes.reproduction as repro
    import osmose.engine.simulate as sim
    from osmose.config import OsmoseConfigReader
    from osmose.engine import PythonEngine

    rec = {
        "ssb": np.zeros((N_YEAR, NDT)),
        "e_lin": np.zeros(N_YEAR),
        "e_sr": np.zeros(N_YEAR),
        "e_fin": np.zeros(N_YEAR),
        "knob_ratio_max_err": 0.0,
        "knob_factor_seen": [],
        "seeded_steps": np.zeros(N_YEAR, dtype=int),
        "recruits": np.zeros(N_YEAR),
        "deaths": np.zeros((N_YEAR, len(CLASSES), len(CAUSES))),
        "n_end": np.zeros((N_YEAR, len(CLASSES))),
    }
    pending: dict = {}

    real_collect = sim._collect_by_life_stage
    real_regulate = repro.regulate_recruitment

    def collect_hook(state, config):
        out = real_collect(state, config)
        sp = state.species_id == HERRING
        if sp.any():
            age = state.age_dt[sp]
            egg = state.is_egg[sp]
            cls = np.where(egg, 0, np.where(age < NDT, 1, 2))
            dead = state.n_dead[sp]
            abd = state.abundance[sp]
            deaths = np.zeros((len(CLASSES), len(CAUSES)))
            n_end = np.zeros(len(CLASSES))
            for k in range(len(CLASSES)):
                m = cls == k
                deaths[k] = dead[m, : len(CAUSES)].sum(axis=0)
                n_end[k] = abd[m].sum()
            pending["deaths"] = deaths
            pending["n_end"] = n_end
            pending["recruits"] = float(abd[age == NDT].sum())
        else:
            pending["deaths"] = np.zeros((len(CLASSES), len(CAUSES)))
            pending["n_end"] = np.zeros(len(CLASSES))
            pending["recruits"] = 0.0
        return out

    def regulate_hook(n_eggs_linear, ssb, seeded_this_step, config, step):
        out = real_regulate(n_eggs_linear, ssb, seeded_this_step, config, step)
        n_sp = config.n_species
        sr = repro.apply_stock_recruitment(
            n_eggs_linear,
            ssb,
            config.recruitment_ssb_half[:n_sp],
            config.recruitment_type[:n_sp],
            config.shepherd_beta[:n_sp],
        )
        y, s = divmod(int(step), NDT)
        if y < N_YEAR:
            rec["ssb"][y, s] = float(ssb[HERRING])
            rec["e_lin"][y] += float(n_eggs_linear[HERRING])
            rec["e_sr"][y] += float(sr[HERRING])
            rec["e_fin"][y] += float(out[HERRING])
            if seeded_this_step[HERRING]:
                rec["seeded_steps"][y] += 1
            elif sr[HERRING] > 0:
                ratio = float(out[HERRING] / sr[HERRING])
                seen = rec["knob_factor_seen"]
                if len(seen) < 4 and (not seen or ratio != seen[-1]):
                    seen.append(ratio)
            if pending:
                rec["deaths"][y] += pending["deaths"]
                rec["n_end"][y] += pending["n_end"]
                rec["recruits"][y] += pending["recruits"]
                pending.clear()
        return out

    sim._collect_by_life_stage = collect_hook
    repro.regulate_recruitment = regulate_hook
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix=f"comp_{state_arm}_{variant}_{seed}_"))
    try:
        cfg = dict(OsmoseConfigReader().read(str(attr.prepare_arm_dir(state_arm, v188, tmp))))
        want = attr.expected_resolved(state_arm, v188, current_mort)
        for sp, v in want.items():
            got = float(cfg[attr.LARVA_KEY.format(i=sp)])
            assert abs(got - v) <= 1e-8 * max(1.0, abs(v)), (state_arm, sp, got, v)
        cfg["simulation.time.nyear"] = str(N_YEAR)
        if variant == "rcp85_ref":
            cfg.update(overlays)
        bio = PythonEngine().run_in_memory(cfg, seed=seed).biomass()
        herring_fdm = attr.decade_mean(bio["herring"].to_numpy(dtype=float))
    finally:
        sim._collect_by_life_stage = real_collect
        repro.regulate_recruitment = real_regulate
        shutil.rmtree(tmp, ignore_errors=True)
    return trim_run(
        {
            "state": state_arm,
            "variant": variant,
            "seed": seed,
            "herring_final_decade_biomass": herring_fdm,
            "ssb_annual_mean": rec["ssb"].mean(axis=1).tolist(),
            "e_lin": rec["e_lin"].tolist(),
            "e_sr": rec["e_sr"].tolist(),
            "e_fin": rec["e_fin"].tolist(),
            "knob_factors_seen": rec["knob_factor_seen"],
            "seeded_steps": rec["seeded_steps"].tolist(),
            "recruits": rec["recruits"].tolist(),
            "deaths": rec["deaths"].tolist(),
            "n_end": rec["n_end"].tolist(),
            "seconds": round(time.time() - t0, 1),
        }
    )


def _w(x, lo=WIN[0], hi=WIN[1], off=0):
    return np.asarray(x, dtype=float)[lo - off : hi - off]


def trim_run(r: dict) -> dict:
    """Keep per-year arrays from KEEP_FROM on, recording the offset (idempotent)."""
    if r.get("year_offset") is not None:
        return r
    out = dict(r)
    for k in PER_YEAR:
        out[k] = list(r[k])[KEEP_FROM:]
    out["year_offset"] = KEEP_FROM
    return out


def summarise_run(r: dict, lagged: bool) -> dict:
    """Final-decade quantities for one run. Lagged: eggs/SSB over 39..48, recruits over 40..49."""
    lo, hi = WIN
    off = r.get("year_offset", 0)
    elo, ehi = (lo - 1, hi - 1) if lagged else (lo, hi)
    ssb = _w(r["ssb_annual_mean"], elo, ehi, off).mean()
    e_lin = _w(r["e_lin"], elo, ehi, off).sum()
    e_sr = _w(r["e_sr"], elo, ehi, off).sum()
    e_fin = _w(r["e_fin"], elo, ehi, off).sum()
    rec = _w(r["recruits"], lo, hi, off).sum()
    return {
        "ssb": ssb,
        "lin": e_lin / ssb,
        "SR": e_sr / e_lin,
        "knob": e_fin / e_sr,
        "early": rec / e_fin,
        "SPR": ssb / (rec / (hi - lo)),
        "biomass": r["herring_final_decade_biomass"],
    }


def hazards_and_fractions(r: dict) -> dict:
    """Per-capita hazard (per step, by cause, by class) and year-0 cohort fractions."""
    off = r.get("year_offset", 0)
    deaths = np.asarray(r["deaths"])[WIN[0] - off : WIN[1] - off].sum(axis=0)  # (class, cause)
    n_end = np.asarray(r["n_end"])[WIN[0] - off : WIN[1] - off].sum(axis=0)  # (class,) over steps
    n_start = n_end + deaths.sum(axis=1)  # survivors + this step's deaths, summed over steps
    hazard = {
        CLASSES[k]: {
            CAUSES[c]: float(deaths[k, c] / n_start[k]) if n_start[k] > 0 else None
            for c in range(len(CAUSES))
        }
        for k in range(len(CLASSES))
    }
    e_fin = _w(r["e_fin"], off=off).sum()
    rec = _w(r["recruits"], off=off).sum()
    year0 = deaths[0] + deaths[1]  # egg + age-0 post-egg deaths by cause
    frac = {CAUSES[c]: float(year0[c] / e_fin) for c in range(len(CAUSES))}
    return {
        "hazard_per_step": hazard,
        "year0_fraction_of_eggs": frac,
        "survival_to_age1": float(rec / e_fin),
        "closure_year0": float((rec + year0.sum()) / e_fin),
    }


def analyse(runs: list[dict]) -> dict:
    by = {(r["state"], r["variant"], r["seed"]): r for r in runs}
    stored = json.loads(ATTR_JSON.read_text())["runs"]
    ref = {(s["arm"], s["variant"], s["seed"]): s["final_decade_means"]["herring"] for s in stored}
    g1 = all(by[k]["herring_final_decade_biomass"] == ref[k] for k in by)
    out: dict = {"gate_G1_hooks_read_only": g1}
    for lagged in (False, True):
        tag = "lagged" if lagged else "same_window"
        res = {}
        for st in STATES:
            mean = {
                v: {
                    q: float(np.mean([summarise_run(by[(st, v, s)], lagged)[q] for s in SEEDS]))
                    for q in ("ssb", "lin", "SR", "knob", "early", "SPR", "biomass")
                }
                for v in VARIANTS
            }
            ratio = {q: mean["rcp85_ref"][q] / mean["baseline"][q] for q in mean["baseline"]}
            ln_ssb = np.log(ratio["ssb"])
            eta = {k: float(np.log(ratio[k]) / ln_ssb) for k in ("lin", "SR", "early", "SPR")}
            res[st] = {
                "means": mean,
                "ratio_warm_over_base": ratio,
                "log_ratio": {k: float(np.log(v)) for k, v in ratio.items()},
                "eta_secant_vs_ssb": eta,
                "product_of_channels": float(
                    ratio["lin"] * ratio["SR"] * ratio["knob"] * ratio["early"] * ratio["SPR"]
                ),
            }
        out[tag] = res
    # knob exactness and seeding in the window
    out["knob_factors_seen"] = {
        f"{r['state']}/{r['variant']}/{r['seed']}": r["knob_factors_seen"] for r in runs
    }
    out["seeded_steps_in_window"] = int(
        sum(sum(_w(r["seeded_steps"], off=r.get("year_offset", 0))) for r in runs)
    )
    hz = {}
    for st in STATES:
        for v in VARIANTS:
            per = [hazards_and_fractions(by[(st, v, s)]) for s in SEEDS]
            hz[f"{st}/{v}"] = {
                "hazard_per_step": {
                    cl: {
                        c: float(np.mean([p["hazard_per_step"][cl][c] or 0.0 for p in per]))
                        for c in CAUSES
                    }
                    for cl in CLASSES
                },
                "year0_fraction_of_eggs": {
                    c: float(np.mean([p["year0_fraction_of_eggs"][c] for p in per])) for c in CAUSES
                },
                "survival_to_age1": float(np.mean([p["survival_to_age1"] for p in per])),
                "closure_year0": [p["closure_year0"] for p in per],
            }
    out["hazards"] = hz
    out["spr_split"] = spr_split(runs)
    return out


def spr_split(runs: list[dict]) -> dict:
    """SSB per recruit = (age-1+ standing stock per recruit) x (SSB per age-1+ fish), plus immature
    biomass per recruit. The first factor is a survival/residence measure; the second is mature
    fraction x weight. Warm/base log ratios of 5-seed means, final decade."""
    lo, hi = WIN

    def one(r):
        off = r.get("year_offset", 0)
        n1p = np.asarray(r["n_end"])[lo - off : hi - off, 2].sum() / ((hi - lo) * NDT)
        rec = _w(r["recruits"], off=off).sum() / (hi - lo)
        ssb = _w(r["ssb_annual_mean"], off=off).mean()
        b = r["herring_final_decade_biomass"]
        return {
            "recruits": rec,
            "age1p_stock": n1p,
            "ssb": ssb,
            "biomass": b,
            "spr": ssb / rec,
            "stock_per_recruit": n1p / rec,
            "ssb_per_age1p_fish": ssb / n1p,
            "immature_biomass_per_recruit": (b - ssb) / rec,
            "mature_share_of_biomass": ssb / b,
        }

    res = {}
    for st in STATES:
        mean = {
            v: {
                k: float(
                    np.mean([one(r)[k] for r in runs if r["state"] == st and r["variant"] == v])
                )
                for k in one(runs[0])
            }
            for v in VARIANTS
        }
        log_ratio = {
            k: float(np.log(mean["rcp85_ref"][k] / mean["baseline"][k])) for k in mean["baseline"]
        }
        res[st] = {
            "means": mean,
            "log_ratio": log_ratio,
            "post_hoc_secant_ssb_per_fish_vs_age1p_density": log_ratio["ssb_per_age1p_fish"]
            / -log_ratio["age1p_stock"],
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument(
        "--reanalyse",
        action="store_true",
        help="recompute the analysis from the runs stored in --out (trimming them); no simulation",
    )
    args = ap.parse_args()

    if args.reanalyse:
        result = json.loads(args.out.read_text())
        result["runs"] = [trim_run(r) for r in result["runs"]]
        f_expected = result["analysis"]["f_expected"]
        result["analysis"] = analyse(result["runs"])
        result["analysis"]["f_expected"] = f_expected
        args.out.write_text(json.dumps(result, allow_nan=False) + "\n")
        print(f"reanalysed {args.out}")
        return 0

    import baltic_b2_scenario_ab as b2

    spec = json.loads(b2.DEFAULT_SPEC_PATH.read_text())
    arm_def = {a["name"]: a for a in spec["arms"]}["rcp85_ref"]
    art_root = Path(tempfile.mkdtemp(prefix="comp_artifacts_"))
    artifacts = b2._b2.write_arm_dir(
        arm_def,
        art_root / "rcp85_ref",
        b2.DEFAULT_O2_PATH,
        b2.DEFAULT_GRID_PATH,
        b2.TREFS,
        b2.BETAS,
    )
    overlays = b2.arm_overlays("rcp85_ref", artifacts, b2.TREFS, b2.BETAS)
    v188 = {f: attr._git_show(f) for f in (attr.F_LTL, attr.F_MORT, attr.F_SEASON, attr.F_CODMAP)}
    current_mort = (ROOT / "data/baltic" / attr.F_MORT).read_text()
    f_expected = b2.expected_knob_factor(
        b2.BETAS[HERRING], b2.TREFS[HERRING], float(arm_def["dT_C"])
    )

    jobs = [
        (st, v, s, v188, overlays, current_mort) for st in STATES for v in VARIANTS for s in SEEDS
    ]
    print(
        f"{len(jobs)} runs x {N_YEAR} yr, {args.workers} workers; f expected {f_expected!r}",
        flush=True,
    )
    t0 = time.time()
    runs = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for r in pool.map(run_one, jobs):
            runs.append(r)
            print(
                f"[{time.time() - t0:6.0f}s] {r['state']} {r['variant']:9s} seed={r['seed']:<5}"
                f" herring {r['herring_final_decade_biomass']:12.0f}  ({r['seconds']}s)",
                flush=True,
            )
    analysis = analyse(runs)
    analysis["f_expected"] = f_expected
    result = {
        "seeds": SEEDS,
        "states": STATES,
        "window_years": WIN,
        "analysis": analysis,
        "runs": runs,
    }
    args.out.write_text(json.dumps(result, allow_nan=False) + "\n")
    print(
        f"G1 hooks read-only (biomass bit-identical to attribution runs): {analysis['gate_G1_hooks_read_only']}"
    )
    print(
        f"knob factor expected {f_expected!r}; seeded herring steps in window: {analysis['seeded_steps_in_window']}"
    )
    for tag in ("same_window", "lagged"):
        print(f"== {tag}")
        for st in STATES:
            a = analysis[tag][st]
            lr = a["log_ratio"]
            print(
                f"  {st}: ln r  biomass {lr['biomass']:+.3f} ssb {lr['ssb']:+.3f} | lin {lr['lin']:+.3f}"
                f" SR {lr['SR']:+.3f} knob {lr['knob']:+.3f} early {lr['early']:+.3f} SPR {lr['SPR']:+.3f}"
                f" | product {a['product_of_channels']:.6f} | eta {json.dumps({k: round(v, 3) for k, v in a['eta_secant_vs_ssb'].items()})}"
            )
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
