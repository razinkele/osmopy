#!/usr/bin/env python3
"""Where does an OSMOSE run spend its time?

`benchmark_engine.py` answers "how long did it take" and is the right tool for
regression detection. This answers "which part", which is what you need before
optimising -- twice during the 2026-09 performance pass an optimisation was
aimed at code that turned out not to be hot, and this script is what settled it.

It reports three things:

1. **Stage split** -- config parse / EngineConfig construction / simulate() /
   write_outputs(). `run_in_memory()` (the calibration path) skips the write
   stage, so its share is the ceiling on what calibration saves by not writing.
2. **Step-loop phase breakdown** -- seconds and share per process (mortality,
   movement, growth, ...), by wrapping the module-level functions in
   `osmose.engine.simulate` with timers. Nothing in the engine is modified;
   the wrappers are installed on the module object for this process only.
3. **Unattributed remainder** -- whatever the phases do not account for, i.e.
   state initialisation and loop overhead.

Timer overhead inflates the total by a few percent, so read the SHARES, and
take absolute timings from `benchmark_engine.py`. Run with a warm numba cache
(execute twice and use the second run): a cold cache cost 45.96s against ~5.6s
warm on EEC 5yr, which swamps everything being measured.

Usage:
    .venv/bin/python scripts/profile_engine.py [--config eec_full] [--years 5]
    .venv/bin/python scripts/profile_engine.py --config baltic --json prof.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
# Reuse the fixture map rather than duplicating it -- these paths drift.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_engine import FIXTURES, resolve_config

import osmose.engine.simulate as sim
from osmose.config.reader import OsmoseConfigReader
from osmose.engine.thread_policy import apply_single_run_threads

# Step-loop callables worth separating. Missing names are skipped, so this
# survives renames in simulate.py without crashing the profiler.
PHASES = [
    "_incoming_flux",
    "_reset_step_variables",
    "_movement",
    "_mortality",
    "_growth",
    "_bioen_step",
    "_aging_mortality",
    "_reproduction",
    "_bioen_reproduction",
    "_collect_biomass_abundance",
    "_collect_mortality",
    "_collect_by_life_stage",
    "_collect_yield",
    "_collect_yield_n",
    "_collect_mean_size",
    "_collect_ssb",
    "_collect_mean_tl",
    "_strip_background",
    "_collect_background_outputs",
    "accumulate_fleet_revenue",
]


def _install_timers(totals: dict[str, float], calls: dict[str, int]) -> list[str]:
    """Wrap each phase on the simulate module; return the names actually wrapped."""
    wrapped = []
    for name in PHASES:
        original = getattr(sim, name, None)
        if original is None:
            continue

        def make(fn, key):
            def timed(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    totals[key] += time.perf_counter() - start
                    calls[key] += 1

            return timed

        setattr(sim, name, make(original, name))
        wrapped.append(name)
    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile an OSMOSE engine run by phase")
    parser.add_argument(
        "--config", default="eec_full", help=f"fixture name {sorted(FIXTURES)} or a path"
    )
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None, help="where to write outputs (default: temp)")
    parser.add_argument("--json", default=None, help="also write the numbers to this JSON file")
    args = parser.parse_args()

    apply_single_run_threads()
    cfg_path = resolve_config(args.config)

    t0 = time.perf_counter()
    config = OsmoseConfigReader().read(cfg_path)
    t_parse = time.perf_counter() - t0
    config["simulation.time.nyear"] = str(args.years)

    # _prepare_run is what PythonEngine.run() actually calls: EngineConfig
    # construction plus grid and RNG setup. Timing it (rather than from_dict
    # alone) keeps the stage split faithful to the production path.
    from osmose.engine import PythonEngine

    engine = PythonEngine()
    t0 = time.perf_counter()
    engine_config, grid, rng, movement_rngs, mortality_rngs = engine._prepare_run(config, args.seed)
    t_engine_cfg = time.perf_counter() - t0

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        import tempfile

        out_dir = Path(tempfile.mkdtemp(prefix="osmose_profile_"))

    totals: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    _install_timers(totals, calls)

    from osmose.engine.output import write_outputs

    t0 = time.perf_counter()
    outputs = sim.simulate(
        engine_config,
        grid,
        rng,
        movement_rngs=movement_rngs,
        mortality_rngs=mortality_rngs,
        output_dir=out_dir,
    )
    t_sim = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_outputs(outputs, out_dir, engine_config, grid=grid)
    t_write = time.perf_counter() - t0

    total = t_parse + t_engine_cfg + t_sim + t_write
    accounted = sum(totals.values())

    print(f"\n{args.config} {args.years}yr  (seed {args.seed})")
    print(f"{'stage':<34}{'seconds':>10}{'share':>9}")
    print("-" * 53)
    for label, secs in (
        ("config parse", t_parse),
        ("engine setup (_prepare_run)", t_engine_cfg),
        ("simulate()", t_sim),
        ("write_outputs()", t_write),
    ):
        print(f"{label:<34}{secs:>10.3f}{secs / total * 100:>8.1f}%")
    print("-" * 53)
    print(f"{'TOTAL':<34}{total:>10.3f}{100.0:>8.1f}%")

    print(f"\n{'step-loop phase':<34}{'seconds':>10}{'share':>9}{'calls':>8}")
    print("-" * 61)
    for name, secs in sorted(totals.items(), key=lambda kv: -kv[1]):
        if secs < 0.001:
            continue
        print(f"{name:<34}{secs:>10.3f}{secs / total * 100:>8.1f}%{calls[name]:>8}")
    print("-" * 61)
    rest = t_sim - accounted
    print(f"{'state init + loop overhead':<34}{rest:>10.3f}{rest / total * 100:>8.1f}%")

    if args.json:
        payload = {
            "config": args.config,
            "n_years": args.years,
            "seed": args.seed,
            "stages_s": {
                "config_parse": t_parse,
                "prepare_run": t_engine_cfg,
                "simulate": t_sim,
                "write_outputs": t_write,
            },
            "total_s": total,
            "phases_s": dict(totals),
            "phase_calls": dict(calls),
            "unattributed_s": rest,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
