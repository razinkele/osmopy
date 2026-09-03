#!/usr/bin/env python3
"""Benchmark the bioen Numba mortality kernel (bioen-Numba-kernel plan Task 3, Step 3).

Three arms, each measured with a short warm-up run first (so Numba's one-time JIT
compile cost is not folded into the timed measurement -- the on-disk ``cache=True``
cache from a prior process run also helps, but is not assumed):

1. ``data/baltic_ev`` bioen ON, kernel dispatch (the production path after Task 3's
   flip -- ``mortality()`` routes to ``_mortality_all_cells_parallel``).
2. ``data/baltic_ev`` bioen ON, ``M._HAS_NUMBA`` forced False for the duration of the
   run (pure-Python per-school reference path, same arithmetic, no kernel).
3. ``data/baltic`` (production Baltic config) bioen OFF, kernel dispatch -- the
   reference point the plan's own numbers were measured against (3.9s / 0.99s per
   simulated year, warm cache, this machine). Task 2 added 19 arguments to
   ``_apply_single_cause`` and two extra full-length per-``mortality()``-call array
   allocations (``cap_fish``/``raw_preyed``) that are live even when ``bioen_enabled``
   is False (they are built unconditionally as ``None`` and read behind a ``bioen``
   guard) -- an inlining failure there would show up here as a regression against the
   plan's recorded number, and would otherwise hide behind a large bioen-arm gain.

``NumbaWarning``/``NumbaPerformanceWarning`` are tracked with
``warnings.catch_warnings(record=True)`` and reported explicitly (including a count of
zero) rather than silenced -- half the Task 3 stop rule is "no such warning was
emitted," so silencing them would hide the very thing being checked.

Usage:
    .venv/bin/python scripts/bench_bioen_kernel.py [--years N] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
BALTIC_EV_CONFIG = PROJECT_DIR / "data" / "baltic_ev" / "baltic_ev_all-parameters.csv"
BALTIC_CONFIG = PROJECT_DIR / "data" / "baltic" / "baltic_all-parameters.csv"


def _load_config(path: Path, n_years: int) -> dict:
    from osmose.config.reader import OsmoseConfigReader

    raw = dict(OsmoseConfigReader().read(str(path)))
    raw["simulation.time.nyear"] = str(n_years)
    return raw


def _numba_warning_filter():
    """A ``warnings.catch_warnings`` category filter that only records Numba's own."""
    from numba.core.errors import NumbaWarning

    return NumbaWarning


def _run_once(raw_config: dict, seed: int, *, force_python: bool) -> tuple[float, list]:
    """Run ``simulate()`` once, returning ``(elapsed_s, numba_warnings)``.

    ``force_python`` toggles ``osmose.engine.processes.mortality._HAS_NUMBA`` for the
    duration of the call -- the same toggle the equivalence harness in
    ``tests/test_engine_bioen_numba_kernel.py`` uses, so "the kernel" and "the Python
    reference" mean exactly the same thing here as they do in the correctness gates.
    """
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.processes import mortality as M
    from osmose.engine.simulate import simulate
    from osmose.engine.thread_policy import apply_single_run_threads

    apply_single_run_threads()

    cfg = EngineConfig.from_dict(raw_config)
    grid_file = raw_config.get("grid.netcdf.file", "")
    if grid_file:
        grid = Grid.from_netcdf(
            Path(str(raw_config.get("_osmose.config.dir", "."))) / grid_file,
            mask_var=raw_config.get("grid.var.mask", "mask"),
        )
    else:
        ny = int(raw_config.get("grid.nline", raw_config.get("grid.nlat", "1")))
        nx = int(raw_config.get("grid.ncolumn", raw_config.get("grid.nlon", "1")))
        grid = Grid.from_dimensions(ny=ny, nx=nx)

    rng = np.random.default_rng(seed)

    NumbaWarning = _numba_warning_filter()
    prev_has_numba = M._HAS_NUMBA
    if force_python:
        M._HAS_NUMBA = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", category=NumbaWarning)
            start = time.perf_counter()
            simulate(cfg, grid, rng)
            elapsed = time.perf_counter() - start
        numba_warnings = [w for w in caught if issubclass(w.category, NumbaWarning)]
    finally:
        M._HAS_NUMBA = prev_has_numba

    return elapsed, numba_warnings


def _bench_arm(label: str, raw_config: dict, seed: int, *, force_python: bool, warmup_years: int):
    """Warm up (untimed, ``warmup_years``) then measure the real run.

    ``warmup_years=0`` skips the warm-up entirely -- used for the pure-Python arm,
    which has no JIT compile cost to amortize.
    """
    warmup_warnings: list = []
    if warmup_years > 0:
        print(f"[{label}] warm-up ({warmup_years} yr, untimed)...", flush=True)
        warm_config = dict(raw_config)
        warm_config["simulation.time.nyear"] = str(warmup_years)
        warmup_elapsed, warmup_warnings = _run_once(warm_config, seed, force_python=force_python)
        print(f"[{label}] warm-up took {warmup_elapsed:.2f}s", flush=True)
    else:
        print(f"[{label}] no warm-up (pure-Python arm has no JIT to prime)", flush=True)

    print(f"[{label}] timed run...", flush=True)
    elapsed, numba_warnings = _run_once(raw_config, seed, force_python=force_python)
    n_years = int(raw_config["simulation.time.nyear"])
    per_year = elapsed / n_years
    print(f"[{label}] {elapsed:.3f}s total, {per_year:.3f}s/yr ({n_years} yr)", flush=True)
    all_warnings = warmup_warnings + numba_warnings
    if all_warnings:
        print(f"[{label}] {len(all_warnings)} Numba warning(s) emitted:", flush=True)
        for w in all_warnings:
            print(f"    {w.category.__name__}: {w.message}", flush=True)
    else:
        print(f"[{label}] 0 Numba warnings emitted", flush=True)

    return {
        "label": label,
        "elapsed_s": round(elapsed, 3),
        "n_years": n_years,
        "per_year_s": round(per_year, 4),
        "numba_warning_count": len(all_warnings),
        "numba_warnings": [f"{w.category.__name__}: {w.message}" for w in all_warnings],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=4, help="Simulated years (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--warmup-years", type=int, default=1, help="Untimed warm-up years (default: 1)"
    )
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    results = []

    print("=" * 70)
    print("Arm A: data/baltic_ev, bioen ON, KERNEL (post-flip production path)")
    print("=" * 70)
    cfg_a = _load_config(BALTIC_EV_CONFIG, args.years)
    assert cfg_a.get("module.bioenergetics.enabled", "").lower() == "true"
    results.append(
        _bench_arm(
            "baltic_ev bioen-ON kernel",
            cfg_a,
            args.seed,
            force_python=False,
            warmup_years=args.warmup_years,
        )
    )

    print()
    print("=" * 70)
    print("Arm B: data/baltic_ev, bioen ON, PYTHON (_HAS_NUMBA=False)")
    print("=" * 70)
    cfg_b = _load_config(BALTIC_EV_CONFIG, args.years)
    results.append(
        _bench_arm(
            "baltic_ev bioen-ON python",
            cfg_b,
            args.seed,
            force_python=True,
            warmup_years=0,  # no JIT to warm up on the pure-Python arm
        )
    )

    print()
    print("=" * 70)
    print("Arm C: data/baltic, bioen OFF, KERNEL (reference point)")
    print("=" * 70)
    cfg_c = _load_config(BALTIC_CONFIG, args.years)
    assert cfg_c.get("module.bioenergetics.enabled", "false").lower() != "true"
    results.append(
        _bench_arm(
            "baltic bioen-OFF kernel",
            cfg_c,
            args.seed,
            force_python=False,
            warmup_years=args.warmup_years,
        )
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    a, b, c = results
    speedup = b["per_year_s"] / a["per_year_s"] if a["per_year_s"] > 0 else float("inf")
    print(f"{a['label']:<32} {a['per_year_s']:.4f} s/yr")
    print(f"{b['label']:<32} {b['per_year_s']:.4f} s/yr")
    print(f"{c['label']:<32} {c['per_year_s']:.4f} s/yr  (reference point)")
    print()
    print(f"bioen kernel speed-up (B / A): {speedup:.1f}x")
    total_warnings = sum(r["numba_warning_count"] for r in results)
    print(f"total Numba warnings across all arms: {total_warnings}")
    print()
    stop_rule_triggered = speedup < 10.0 or total_warnings > 0
    print(
        f"STOP RULE (speed-up < 10x OR any Numba warning): "
        f"{'TRIGGERED -- STOP AND REPORT' if stop_rule_triggered else 'clear'}"
    )

    payload = {
        "years": args.years,
        "seed": args.seed,
        "arms": results,
        "speedup_bioen_kernel_vs_python": round(speedup, 2),
        "total_numba_warnings": total_warnings,
        "stop_rule_triggered": stop_rule_triggered,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
