#!/usr/bin/env python
"""Is each phase-14 functional-response half-saturation K identifiable? (issue #232)

Phase 14 (``calibrate_baltic.get_phase14_params``) holds the FR shape at type III and tunes
the half-saturation K of each predator in [0.5, 5.0]. So the identifiability question is about
K *within* type III, not about type III vs type I: hold the other predators at the x0 value
K = 1.0, move ONE predator's K from the lower bound to the upper bound, and ask whether the
phase-14 objective (and the assessed stocks it scores) moves by more than the seed-to-seed
spread at the calibration horizon.

The base config is built exactly as ``run_calibration`` builds it for ``--phase 14``:
production Baltic + the frozen phase-13 parameters + Shepherd recruitment on every focal
species + type III on the four FR predators. cod_west (sp0) is run as a POSITIVE CONTROL:
its K is known to move realized predation strongly, so if it reads as unidentifiable the
instrument is blind and no verdict on the background predators can be drawn.

Pre-stated criterion: ``docs/diagnostics/phase14_fr_identifiability_prestated_2026-11-11.md``.

Usage (one engine job; workers run single-threaded):
    PYTHONPATH=. .venv/bin/python scripts/phase14_fr_identifiability.py \
        --out docs/diagnostics/phase14_fr_identifiability_2026-11-11.json
"""

from __future__ import annotations

import os

# Single-threaded Numba per worker; must be set before numba is imported anywhere.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PREDATORS = {0: "cod_west", 15: "GreySeal", 16: "Cormorant"}
CONTROL_SP = 0
K_LEVELS = (0.5, 5.0)
SEEDS = (42, 1, 7, 11)
N_YEARS = 40
ASSESSED = ("cod_west", "cod_east", "herring", "sprat", "flounder")


def build_phase14_base() -> tuple[dict[str, str], list[str]]:
    """The ``--phase 14`` base config, via the calibrator's own ``apply_phase14_base``.

    (The 2026-11-11 run used an inline copy of the same three steps, at ``163da593``.)"""
    import calibrate_baltic as cal

    from osmose.config.reader import OsmoseConfigReader

    keys, _bounds, _x0 = cal.get_phase14_params()
    base = OsmoseConfigReader().read(cal.BALTIC_CONFIG)
    cal.apply_phase14_base(base)
    return base, keys


def _x_for(keys: list[str], sp: int, k: float) -> np.ndarray:
    """log10 parameter vector: every K at the x0 value 1.0 except predator ``sp`` at ``k``."""
    x = np.zeros(len(keys))
    xk = float(np.log10(k))
    # 10 ** log10(5.0) == 5.000000000000001, which the engine's [0.1, 5.0] check rejects. Step
    # one ULP down so the upper bound is the largest K the engine accepts (4.999999999999999).
    while 10.0**xk > k:
        xk = float(np.nextafter(xk, -np.inf))
    x[keys.index(f"predation.functional.response.halfsat.sp{sp}")] = xk
    return x


def run_one(job: tuple[int, float, int]) -> dict:
    sp, k, seed = job
    import calibrate_baltic as cal

    base, keys = build_phase14_base()
    x = _x_for(keys, sp, k)
    t0 = time.time()
    stats = cal.run_simulation(
        base, cal.expand_param_overrides(keys, x, True), n_years=N_YEARS, seed=seed
    )
    objective = cal.make_objective(base, cal.load_targets(), keys, n_years=N_YEARS, seed=seed)
    objective._simulate_and_compute_stats = lambda _x: stats  # score the run we already have
    return {
        "sp": sp,
        "predator": PREDATORS[sp],
        "k": k,
        "seed": seed,
        "objective": float(objective(x)) if stats else None,
        "stats": stats,
        "seconds": round(time.time() - t0, 1),
    }


def analyse(runs: list[dict]) -> dict:
    """Paired bound-to-bound effect vs pooled across-seed SD, per predator."""
    out: dict = {}
    for sp, name in PREDATORS.items():
        by_k = {k: {r["seed"]: r for r in runs if r["sp"] == sp and r["k"] == k} for k in K_LEVELS}
        lo, hi = K_LEVELS
        seeds = sorted(set(by_k[lo]) & set(by_k[hi]))

        def verdict(values_lo: np.ndarray, values_hi: np.ndarray) -> dict:
            delta = values_hi - values_lo
            pooled = float(np.sqrt((np.var(values_lo, ddof=1) + np.var(values_hi, ddof=1)) / 2))
            mean = float(np.mean(delta))
            return {
                "mean_delta": mean,
                "delta_per_seed": [float(d) for d in delta],
                "pooled_seed_sd": pooled,
                # No seed noise at all (e.g. a stock exactly 0 in every run): 0.0 when nothing moved,
                # None (undefined) when it moved deterministically -- never inf, which is not JSON.
                "ratio": abs(mean) / pooled if pooled > 0 else (0.0 if mean == 0 else None),
                "identifiable": bool(abs(mean) > 2 * pooled),
            }

        obj_lo = np.array([by_k[lo][s]["objective"] for s in seeds], dtype=float)
        obj_hi = np.array([by_k[hi][s]["objective"] for s in seeds], dtype=float)
        species = {}
        for sp_name in ASSESSED:
            b_lo = np.array([by_k[lo][s]["stats"][f"{sp_name}_mean"] for s in seeds], dtype=float)
            b_hi = np.array([by_k[hi][s]["stats"][f"{sp_name}_mean"] for s in seeds], dtype=float)
            v = verdict(np.log10(np.maximum(b_lo, 1e-9)), np.log10(np.maximum(b_hi, 1e-9)))
            v["mean_biomass_lo"] = float(b_lo.mean())
            v["mean_biomass_hi"] = float(b_hi.mean())
            species[sp_name] = v
        out[name] = {
            "sp": sp,
            "seeds": seeds,
            "objective": verdict(obj_lo, obj_hi),
            "assessed_log10_biomass": species,
        }
    return out


def gated_outcome(analysis: dict) -> dict:
    """Apply the pre-stated validity gate to the raw per-predator flags.

    If the cod_west control does not pass on the objective, the instrument has not shown that it
    can see a knob known to matter, so every other predator's outcome is "inconclusive". The raw
    criterion flags in ``analysis`` stay as measured.
    """
    control = PREDATORS[CONTROL_SP]
    control_pass = bool(analysis[control]["objective"]["identifiable"])
    outcome = {}
    for name, a in analysis.items():
        if name == control:
            continue
        if not control_pass:
            outcome[name] = "inconclusive"
        else:
            outcome[name] = "identifiable" if a["objective"]["identifiable"] else "not identifiable"
    return {"control": control, "control_pass": control_pass, "predators": outcome}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--reanalyse",
        action="store_true",
        help="recompute the analysis from the runs already stored in --out; no simulation",
    )
    args = ap.parse_args()

    if args.reanalyse:
        result = json.loads(args.out.read_text())
        result["analysis"] = analyse(result["runs"])
        result["outcome"] = gated_outcome(result["analysis"])
        _write_and_report(result, args.out)
        return 0

    jobs = [(sp, k, seed) for sp in PREDATORS for k in K_LEVELS for seed in SEEDS]
    print(f"{len(jobs)} runs x {N_YEARS} yr, {args.workers} workers", flush=True)
    t0 = time.time()
    runs = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for r in pool.map(run_one, jobs):
            runs.append(r)
            print(
                f"[{time.time() - t0:7.0f}s] {r['predator']:9s} K={r['k']:<4} seed={r['seed']:<3}"
                f" obj={r['objective']}  ({r['seconds']}s)",
                flush=True,
            )
    failed = [r for r in runs if not r["stats"]]
    if failed:
        print(f"FAILED runs: {[(r['predator'], r['k'], r['seed']) for r in failed]}")
        return 1
    result = {
        "n_years": N_YEARS,
        "k_levels": K_LEVELS,
        "seeds": SEEDS,
        "control": PREDATORS[CONTROL_SP],
        "criterion": "identifiable iff |mean paired delta| > 2 x pooled across-seed SD",
        "analysis": analyse(runs),
        "runs": runs,
    }
    result["outcome"] = gated_outcome(result["analysis"])
    _write_and_report(result, args.out)
    print(f"({time.time() - t0:.0f}s)")
    return 0


def _fmt_ratio(r: float | None) -> str:
    return "undefined" if r is None else f"{r:.2f}"


def _write_and_report(result: dict, out: Path) -> None:
    out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    for name, a in result["analysis"].items():
        o = a["objective"]
        print(
            f"{name:9s} objective: delta {o['mean_delta']:+.4f}  seed-sd {o['pooled_seed_sd']:.4f}"
            f"  ratio {_fmt_ratio(o['ratio'])}  identifiable={o['identifiable']}"
        )
        for sp_name, v in a["assessed_log10_biomass"].items():
            print(
                f"    {sp_name:9s} dlog10B {v['mean_delta']:+.4f}  sd {v['pooled_seed_sd']:.4f}"
                f"  ratio {_fmt_ratio(v['ratio'])}  {'IDENT' if v['identifiable'] else ''}"
            )
    g = result["outcome"]
    print(f"control {g['control']}: {'PASS' if g['control_pass'] else 'FAIL'} (pre-stated gate)")
    for name, verdict in g["predators"].items():
        print(f"  {name}: {verdict.upper()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
