#!/usr/bin/env python
"""Matched-budget comparison of DE, CMA-ES and surrogate-DE on a real objective (perf plan item 5b).

The problem is calibration phase 1b (``calibrate_baltic.get_phase1b_params``: 9 log-space mortality
and ingestion parameters) with its real objective: ``make_objective`` on the production Baltic,
40 years, simulation seed 42 for every candidate. Every optimizer gets the SAME number of real
evaluations (``BUDGET``) and starts from the same point; its configuration is fixed below and
recorded in each run's output. The table compares these configurations, not the algorithms.

Each evaluation is appended, from whichever worker ran it, to a per-run JSONL log (x, f, time), so
best-so-far can be read at any budget and the real evaluation count verified. A run's final best is
then re-ranked on five fresh simulation seeds that exclude 42.

Pre-stated rule: ``docs/perf/2026-11-17-optimizer-comparison-prestated.md``.

Usage (one engine job at a time; each subcommand is its own process):
    PYTHONPATH=. .venv/bin/python scripts/optimizer_comparison.py preflight --out P.json
    PYTHONPATH=. .venv/bin/python scripts/optimizer_comparison.py run --optimizer de --rep 1 \\
        --out runs/de_1.json
    PYTHONPATH=. .venv/bin/python scripts/optimizer_comparison.py rerank --runs runs/*.json \\
        --out rerank.json
"""

from __future__ import annotations

import os

# Before anything imports Numba: CMA-ES and surrogate-DE evaluate through joblib, whose workers
# the worker-thread policy does not reach, and 14 workers at Numba's default thread count would
# oversubscribe the cores (docs/perf/2026-11-14-worker-thread-sweep.md). Each subcommand is its
# own process, so this cannot leak anywhere else.
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

N_YEARS = 40
TRAIN_SEED = 42
WORKERS = 14
BUDGET = 224
RERANK_SEEDS = (1, 2, 3, 4, 5)  # excludes TRAIN_SEED
SENTINEL = 1e6  # make_objective's failed-run penalty

# The fixed configurations compared (all spend exactly BUDGET evaluations at WORKERS = 14).
CONFIGS = {
    "de": {"popsize": 28, "maxiter": 7, "tol": 0.0},  # 28 x (7 + 1) = 224
    "cmaes": {"popsize": 14, "maxiter": 16, "tol": 0.0, "sigma0": 0.3},  # 14 x 16 = 224
    "surrogate-de": {"n_initial": 44, "n_iterations": 6, "n_topk": 30},  # 44 + 6 x 30 = 224
}
# --smoke: every code path at a toy size (2 yr, 4 workers, ~8-12 evaluations). Not a measurement.
SMOKE_CONFIGS = {
    "de": {"popsize": 5, "maxiter": 1, "tol": 0.0},  # scipy needs > 4
    "cmaes": {"popsize": 4, "maxiter": 2, "tol": 0.0, "sigma0": 0.3},
    "surrogate-de": {"n_initial": 4, "n_iterations": 1, "n_topk": 4},
}


def _problem(n_years: int = N_YEARS):
    """(objective, keys, bounds, x0): exactly what ``run_calibration --phase 1b`` builds."""
    import calibrate_baltic as cal

    from osmose.config.reader import OsmoseConfigReader

    keys, bounds, x0 = cal.get_phase1b_params()
    base = OsmoseConfigReader().read(cal.BALTIC_CONFIG)
    objective = cal.make_objective(
        base, cal.load_targets(), keys, n_years=n_years, seed=TRAIN_SEED, use_log_space=True
    )
    return objective, keys, bounds, x0


class _LoggedObjective:
    """Picklable wrapper: evaluates, then appends (x, f, t) to a JSONL log. Each line is one small
    write in append mode, so lines from concurrent workers do not interleave."""

    def __init__(self, objective, log_path: str):
        self.objective = objective
        self.log_path = log_path

    def __call__(self, x):
        f = float(self.objective(np.asarray(x, dtype=float)))
        line = json.dumps({"x": [float(v) for v in x], "f": f, "t": time.time()}) + "\n"
        with open(self.log_path, "a") as fh:
            fh.write(line)
        return f


def _de_init_pop(x0, bounds, popsize: int, seed: int) -> np.ndarray:
    """run_calibration's mixed initial population: x0, 30% within +-15% of it, 70% LHS."""
    from scipy.stats.qmc import LatinHypercube

    rng = np.random.default_rng(seed)
    x0_arr, b = np.asarray(x0), np.asarray(bounds)
    widths = b[:, 1] - b[:, 0]
    n_near = max(1, int(0.3 * popsize))
    pop = np.zeros((popsize, len(x0)))
    pop[0] = x0_arr
    for i in range(1, n_near):
        pop[i] = np.clip(x0_arr + rng.uniform(-0.15, 0.15, len(x0)) * widths, b[:, 0], b[:, 1])
    lhs = LatinHypercube(d=len(x0), seed=rng).random(n=popsize - n_near)
    pop[n_near:] = b[:, 0] + lhs * widths
    return pop


def cmd_run(args) -> int:
    from calibrate_baltic import _dispatch_optimizer

    n_years, workers = (2, 4) if args.smoke else (N_YEARS, WORKERS)
    objective, keys, bounds, x0 = _problem(n_years)
    if args.x0:
        x0 = json.loads(Path(args.x0).read_text())["x0"]
    log = Path(args.out).with_suffix(".evals.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.unlink(missing_ok=True)
    logged = _LoggedObjective(objective, str(log))
    conf = (SMOKE_CONFIGS if args.smoke else CONFIGS)[args.optimizer]
    t0 = time.time()
    kwargs = {
        "objective": logged,
        "bounds": bounds,
        "x0": x0,
        "init_pop": _de_init_pop(x0, bounds, conf.get("popsize", 1), seed=args.rep),
        "maxiter": conf.get("maxiter", 0),
        "popsize": conf.get("popsize", 0),
        "tol": conf.get("tol", 0.0),
        "workers": workers,
        "seed": args.rep,
    }
    if args.optimizer == "surrogate-de":
        from osmose.calibration.surrogate_de import surrogate_assisted_de

        result = surrogate_assisted_de(
            logged,
            bounds,
            x0=x0,
            n_initial=conf["n_initial"],
            n_iterations=conf["n_iterations"],
            n_topk=conf["n_topk"],
            workers=workers,
            seed=args.rep,
            verbose=True,
        )
    elif args.optimizer == "cmaes":
        from osmose.calibration.cmaes_runner import run_cmaes

        result = run_cmaes(
            logged,
            bounds,
            x0=x0,
            sigma0=conf["sigma0"],
            popsize=conf["popsize"],
            maxiter=conf["maxiter"],
            tol=conf["tol"],
            seed=args.rep,
            workers=workers,
            verbose=True,
        )
    else:
        result = _dispatch_optimizer(optimizer="de", **kwargs)
    evals = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    evals.sort(key=lambda e: e["t"])
    out = {
        "optimizer": args.optimizer,
        "rep": args.rep,
        "config": conf,
        "budget": BUDGET,
        "workers": workers,
        "n_years": n_years,
        "smoke": bool(args.smoke),
        "train_seed": TRAIN_SEED,
        "keys": keys,
        "x0": [float(v) for v in x0],
        "nfev_logged": len(evals),
        "n_sentinel": sum(1 for e in evals if e["f"] >= SENTINEL),
        "best_x": result_best(evals)[0],
        "best_f": result_best(evals)[1],
        "optimizer_reported": {"fun": float(result["fun"]), "nfev": int(result.get("nfev", -1))},
        "best_so_far": {
            str(n): best_so_far(evals, n) for n in (BUDGET // 3, 2 * BUDGET // 3, BUDGET)
        },
        "seconds": round(time.time() - t0),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("optimizer", "rep", "nfev_logged", "best_f", "seconds")}))
    return 0


def result_best(evals: list[dict], n: int | None = None) -> tuple[list[float], float]:
    """Best (x, f) among the first ``n`` evaluations in time order (all if None)."""
    pool = evals if n is None else evals[:n]
    best = min(pool, key=lambda e: e["f"])
    return best["x"], best["f"]


def best_so_far(evals: list[dict], n: int) -> float | None:
    return result_best(evals, n)[1] if len(evals) >= n else None


def _eval_on_seed(job):
    """One evaluation in a CHILD process (never the parent: see #260)."""
    x, seed = job
    import calibrate_baltic as cal

    from osmose.config.reader import OsmoseConfigReader

    keys, _bounds, _x0 = cal.get_phase1b_params()
    base = OsmoseConfigReader().read(cal.BALTIC_CONFIG)
    obj = cal.make_objective(
        base, cal.load_targets(), keys, n_years=N_YEARS, seed=seed, use_log_space=True
    )
    return float(obj(np.asarray(x, dtype=float)))


def _evaluate_many(jobs: list[tuple[list[float], int]]) -> list[float]:
    with mp.get_context("spawn").Pool(min(WORKERS, len(jobs))) as pool:
        return pool.map(_eval_on_seed, jobs)


def _production_x(keys) -> list[float]:
    """The production config's own values for ``keys``, as log10 in the objective's (resolved)
    space: the reader has already divided the larval rates by ndtperyear, exactly as the
    objective's overrides expect."""
    import calibrate_baltic as cal

    from osmose.config.reader import OsmoseConfigReader

    base = OsmoseConfigReader().read(cal.BALTIC_CONFIG)
    return [float(np.log10(float(base[k.lower()]))) for k in keys]


def cmd_preflight(args) -> int:
    import calibrate_baltic as cal

    from osmose.engine.config import EngineConfig

    keys, bounds, x0 = cal.get_phase1b_params()
    xp = _production_x(keys)
    jobs = [(list(x0), s) for s in RERANK_SEEDS] + [(xp, s) for s in RERANK_SEEDS]
    vals = _evaluate_many(jobs)
    n = len(RERANK_SEEDS)
    corners = {}
    from osmose.config.reader import OsmoseConfigReader

    base = OsmoseConfigReader().read(cal.BALTIC_CONFIG)
    for name, pick in (("lower", 0), ("upper", 1)):
        cfg = dict(base)
        for k, b in zip(keys, bounds):
            cfg[k.lower()] = str(10 ** b[pick])
        try:
            EngineConfig.from_dict(cfg)
            corners[name] = "builds"
        except Exception as exc:
            corners[name] = f"{type(exc).__name__}: {exc}"
    out = {
        "keys": keys,
        "x0": [float(v) for v in x0],
        "x_production": xp,
        "objective_x0": vals[:n],
        "objective_production": vals[n:],
        "seeds": list(RERANK_SEEDS),
        "bound_corners": corners,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


def cmd_rerank(args) -> int:
    runs = [json.loads(Path(p).read_text()) for p in args.runs]
    jobs = [(r["best_x"], s) for r in runs for s in RERANK_SEEDS]
    vals = _evaluate_many(jobs)
    n = len(RERANK_SEEDS)
    table = []
    for i, r in enumerate(runs):
        v = vals[i * n : (i + 1) * n]
        table.append(
            {
                "optimizer": r["optimizer"],
                "rep": r["rep"],
                "train_best_f": r["best_f"],
                "rerank_mean": float(np.mean(v)),
                "rerank_sd": float(np.std(v, ddof=1)),
                "rerank": v,
                "nfev_logged": r["nfev_logged"],
                "n_sentinel": r["n_sentinel"],
            }
        )
    Path(args.out).write_text(json.dumps({"seeds": list(RERANK_SEEDS), "runs": table}, indent=2))
    for t in table:
        print(
            f"{t['optimizer']:13s} rep {t['rep']}  train {t['train_best_f']:.4f}  "
            f"rerank {t['rerank_mean']:.4f} +- {t['rerank_sd']:.4f}  nfev {t['nfev_logged']}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--optimizer", choices=list(CONFIGS), required=True)
    r.add_argument("--rep", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--x0", help="JSON file with an 'x0' list: start every optimizer there instead")
    r.add_argument("--smoke", action="store_true", help="toy size, to exercise every code path")
    k = sub.add_parser("rerank")
    k.add_argument("--runs", nargs="+", required=True)
    k.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"preflight": cmd_preflight, "run": cmd_run, "rerank": cmd_rerank}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
