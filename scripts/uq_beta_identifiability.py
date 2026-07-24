"""Surrogate-Bayesian identifiability check on the calibrated Shepherd shape (beta)
parameters. Self-consistency around the calibrated point (a stable, 5/8-in-range
config): vary each species' beta in a tight box, engine-generate targets at beta*,
recover via run_surrogate_bayes (pass-through gate -> guaranteed posterior), and
report per-beta concentration (post SD / box SD; <<1 = identifiable, ~1 =
prior-dominated) and centeredness (|post mean - beta*| / post SD).

Threading: launch with OMP_NUM_THREADS=1 / NUMBA_NUM_THREADS=1 (1 thread/worker).
Spawn pool -> guarded by if __name__ == "__main__".
"""

from __future__ import annotations

import math
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.targets import BiomassTarget
from osmose.calibration.uq.design import make_engine_evaluator, run_design
from osmose.calibration.uq.gate import evaluate_emulator_calibration
from osmose.calibration.uq.posterior import fit_emulators
from osmose.calibration.uq.predictive import emulator_holdout_coverage, marginal_coverage
from osmose.calibration.uq.run import run_surrogate_bayes

BASE = Path("data/baltic/baltic_all-parameters.csv")
SPECIES = ["cod", "herring", "sprat", "flounder", "perch", "pikeperch", "smelt", "stickleback"]
SAVE = Path("/tmp/claude-1000/-home-razinka-osmopy/d89da751-bed8-4745-b75d-c26886735ab3/scratchpad/beta_identifiability.pkl")

# (species, sp-index, calibrated beta*) — the 5 in-range / well-behaved species.
BETAS = [("cod", 0, 1.952), ("herring", 1, 1.0854), ("sprat", 2, 1.7242),
         ("flounder", 3, 1.397), ("stickleback", 7, 1.3605)]
HALF = 0.15
F = 1.5
NYEAR = 30
NSEED = 5
N_GEN = 10
N_HOLD = 30
NW = 16
N0 = 40

fps, beta_star, box_sd = [], [], []
for _sp, idx, b in BETAS:
    fps.append(FreeParameter(f"stock.recruitment.shape.sp{idx}", b - HALF, b + HALF, Transform.LINEAR))
    beta_star.append(b)
    box_sd.append(HALF / math.sqrt(3.0))
beta_star = np.array(beta_star)
TARGET_SP = [sp for sp, *_ in BETAS]
KEYS = [f"{sp}_biomass_mean" for sp in TARGET_SP]

natural_gate = {}


def passthrough_gate(X, Y, alpha, *, key=None, seed=0):
    rep = evaluate_emulator_calibration(X, Y, alpha, key=key, seed=seed)
    natural_gate[key] = rep
    return replace(rep, passed=True, reasons=[])


def main():
    t0 = time.perf_counter()
    with make_engine_evaluator(fps, BASE, SPECIES, enable_ssb=True, nyear=NYEAR, n_workers=NW) as ev:
        gen = ev.evaluate_batch([(s, beta_star, s) for s in range(N_GEN)])
        targets = []
        for sp in TARGET_SP:
            tbar = float(np.mean([g[f"{sp}_biomass_mean"] for g in gen]))
            targets.append(BiomassTarget(species=sp, target=tbar, lower=tbar / F, upper=tbar * F,
                                         reference_point_type="biomass"))
        print("targets (arith-mean biomass at beta*):",
              {t.species: round(t.target, 1) for t in targets}, flush=True)
        res = run_surrogate_bayes(ev, fps, targets, n_seeds=NSEED, n0=N0, increment=10, n_max=N0,
                                  seed=0, gate_fn=passthrough_gate, k_by_type={"biomass": 1.0},
                                  include_predictive=True)
        holdout = run_design(ev, fps, KEYS, n_points=N_HOLD, n_seeds=NSEED, seed=99,
                             seed_offset=7_000_000)
    dt = time.perf_counter() - t0

    emulators = fit_emulators(res.design)
    with open(SAVE, "wb") as fh:
        pickle.dump({"design": res.design, "sampler_result": res.sampler_result,
                     "posterior_mean": res.posterior_mean, "beta_star": beta_star,
                     "box_sd": box_sd, "keys": KEYS}, fh)

    print(f"\n==== BETA IDENTIFIABILITY ({dt / 60:.1f} min) ====", flush=True)
    print(f"status={res.status!r}  design pts={len(res.design.X)}  n_censored={res.n_censored}")
    print("\n-- gate (real metrics, forced-pass) --")
    for k, r in natural_gate.items():
        print(f"  {k:<22} real_pass={r.passed!s:<5} cov={r.coverage:.3f} mssr={r.mssr:.3f} "
              f"r2={r.r2:.3f} r2_ceiling={r.r2_ceiling:.3f}")
    if res.sampler_result is not None:
        post_sd = res.sampler_result.marginal_sd()
        print(f"\n-- beta recovery (converged={res.sampler_result.converged}, ess={res.sampler_result.ess:.0f}) --")
        print(f"  {'beta param':<34}{'beta*':>8}{'post_mean':>11}{'post_sd':>9}{'conc':>7}{'center':>8}  verdict")
        for j, fp in enumerate(fps):
            pm, ps = res.posterior_mean[j], post_sd[j]
            conc = ps / box_sd[j]
            center = abs(pm - beta_star[j]) / ps if ps > 0 else float("inf")
            verdict = "identifiable" if conc < 0.85 else "prior-dominated"
            print(f"  {fp.key:<34}{beta_star[j]:>8.3f}{pm:>11.3f}{ps:>9.3f}{conc:>7.2f}{center:>8.2f}  {verdict}")
    if res.predictive_ranges is not None:
        print("\n-- marginal_coverage --", marginal_coverage(res.predictive_ranges, targets))
    hcov = emulator_holdout_coverage(emulators, holdout.X, holdout.Y, holdout.alpha, level=0.95)
    print("\n-- held-out OSMOSE coverage @0.95 --")
    for k in KEYS:
        print(f"  {k:<22} coverage={hcov[k]:.3f}  (n_valid={int((~np.isnan(holdout.Y[k])).sum())}/{N_HOLD})")


if __name__ == "__main__":
    main()
