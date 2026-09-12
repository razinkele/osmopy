#!/usr/bin/env python3
"""Benchmark the Python OSMOSE engine for performance regression detection.

Runs the engine on Bay of Biscay config with timing and outputs structured
JSON results. Use before and after optimization to measure speedup.

Two traps this guards against, both of which produced false results during the
2026-09 performance pass:

* **Cold numba cache.** The first run in a fresh process/worktree pays JIT --
  45.96s against ~5.6s warm on EEC 5yr. Left in the sample it poisons
  median/min/max and once read as a 10% regression. One warm-up run is now
  taken and DISCARDED by default; `--warmup 0` restores the old behaviour.
* **Machine drift between captures.** `--compare` diffs two JSON files that may
  have been recorded minutes or months apart. A change once measured 6.3%
  faster this way and the effect vanished when the control was re-run
  back-to-back. Runs now record host, thread count and timestamp, and compare
  refuses to look confident: it warns on any mismatch and calls a delta
  INCONCLUSIVE when it is smaller than the observed run-to-run spread.

For a real A/B, run both variants back-to-back on one machine and alternate
them. To find out WHICH phase is slow rather than by how much, use
`scripts/profile_engine.py`.

Usage:
    .venv/bin/python scripts/benchmark_engine.py [--years N] [--seed S] [--repeats R]
    .venv/bin/python scripts/benchmark_engine.py --warmup 0        # old behaviour
    .venv/bin/python scripts/benchmark_engine.py --compare baseline.json current.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

from osmose.engine.thread_policy import apply_single_run_threads

PROJECT_DIR = Path(__file__).parent.parent
EXAMPLES_CONFIG = PROJECT_DIR / "data" / "examples" / "osm_all-parameters.csv"

# Built-in fixture aliases — keep keys here in sync with `data/`. New entries
# are picked up by the --config flag automatically.
FIXTURES: dict[str, Path] = {
    "examples": PROJECT_DIR / "data" / "examples" / "osm_all-parameters.csv",
    "minimal": PROJECT_DIR / "data" / "minimal" / "osm_all-parameters.csv",
    "baltic": PROJECT_DIR / "data" / "baltic" / "baltic_all-parameters.csv",
    "eec_full": PROJECT_DIR / "data" / "eec_full" / "eec_all-parameters.csv",
}


def resolve_config(arg: str | None) -> Path:
    """Resolve --config NAME-or-PATH to a fixture file path."""
    if arg is None:
        return EXAMPLES_CONFIG
    if arg in FIXTURES:
        return FIXTURES[arg]
    p = Path(arg)
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Config not found: {arg!r}. Pick a built-in fixture "
        f"({sorted(FIXTURES)}) or provide an existing path."
    )


def run_benchmark(n_years: int, seed: int, config_path: Path = EXAMPLES_CONFIG) -> dict:
    """Run the Python engine once and return timing + summary stats."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import simulate

    reader = OsmoseConfigReader()
    raw = reader.read(config_path)
    raw["simulation.time.nyear"] = str(n_years)

    cfg = EngineConfig.from_dict(raw)

    grid_file = raw.get("grid.netcdf.file", "")
    if grid_file:
        from osmose.engine.grid import Grid as G

        # Resolve grid file relative to the config's directory so each fixture
        # finds its own NetCDF (eec_full vs baltic vs examples vary here).
        grid = G.from_netcdf(
            config_path.parent / grid_file,
            mask_var=raw.get("grid.var.mask", "mask"),
        )
    else:
        ny = int(raw.get("grid.nline", "1"))
        nx = int(raw.get("grid.ncolumn", "1"))
        grid = Grid.from_dimensions(ny=ny, nx=nx)

    rng = np.random.default_rng(seed)

    start = time.perf_counter()
    outputs = simulate(cfg, grid, rng)
    elapsed = time.perf_counter() - start

    # Collect final-step biomass per species. baltic + others may have
    # background species (sp >= n_focal) whose biomass is appended; guard
    # against a name list shorter than the biomass array.
    final = outputs[-1]
    names = list(cfg.species_names)
    biomass_by_species = {}
    for i in range(len(final.biomass)):
        key = names[i] if i < len(names) else f"sp{i}"
        biomass_by_species[key] = float(final.biomass[i])

    return {
        "elapsed_s": round(elapsed, 3),
        "n_steps": len(outputs),
        "n_years": n_years,
        "seed": seed,
        "per_year_s": round(elapsed / n_years, 3),
        "final_biomass": biomass_by_species,
    }


def _warn_if_not_comparable(baseline: dict, current: dict) -> None:
    """Flag the ways two saved runs can be non-comparable.

    Comparing JSONs captured at different times is the drift trap: during the
    2026-09 performance pass a change measured 6.3% faster against a baseline
    taken earlier in the same session, and vanished entirely when the control
    was re-measured back-to-back. Timings only compare if the machine, thread
    count and workload match, and even then only loosely across time.
    """
    problems = []
    for key, label in (
        ("hostname", "host"),
        ("numba_threads", "numba threads"),
        ("config", "config"),
        ("n_years", "years"),
        ("seed", "seed"),
    ):
        b, c = baseline.get(key), current.get(key)
        if b is not None and c is not None and b != c:
            problems.append(f"{label}: {b!r} vs {c!r}")
    if baseline.get("warmup") == 0 or current.get("warmup") == 0:
        problems.append("one side ran with --warmup 0 (cold-JIT run included in timings)")
    if "hostname" not in baseline or "hostname" not in current:
        problems.append("a side predates provenance capture; machine/threads unknown")

    b_at, c_at = baseline.get("captured_at"), current.get("captured_at")
    if b_at and c_at:
        print(f"Captured: baseline {b_at}   current {c_at}")
    if problems:
        print("WARNING — these runs may not be comparable:")
        for p in problems:
            print(f"  - {p}")
        print(
            "  Timings from different machines, thread counts or sessions are not\n"
            "  an A/B. Run both variants back-to-back on one machine instead.\n"
        )


def compare_results(baseline_path: Path, current_path: Path) -> None:
    """Compare two benchmark JSON files and print speedup report."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(current_path) as f:
        current = json.load(f)

    _warn_if_not_comparable(baseline, current)

    print(f"{'Metric':<25} {'Baseline':>12} {'Current':>12} {'Change':>10}")
    print("-" * 62)

    b_med = baseline["median_s"]
    c_med = current["median_s"]
    speedup = b_med / c_med if c_med > 0 else float("inf")
    print(f"{'Median time (s)':<25} {b_med:>12.3f} {c_med:>12.3f} {speedup:>9.1f}x")

    # Baselines written before min_s existed simply omit it.
    def _fmt(v):
        return f"{v:>12.3f}" if isinstance(v, (int, float)) else f"{'n/a':>12}"

    print(f"{'Min time (s)':<25} {_fmt(baseline.get('min_s'))} {_fmt(current.get('min_s'))}")

    # A delta smaller than either run's own spread is not a result.
    delta_pct = abs(b_med - c_med) / b_med * 100 if b_med > 0 else 0.0
    worst_spread = max(baseline.get("spread_pct", 0.0), current.get("spread_pct", 0.0))
    print(f"{'Delta / worst spread':<25} {delta_pct:>11.1f}% {worst_spread:>11.1f}%")
    if worst_spread and delta_pct <= worst_spread:
        print(
            f"  ^ INCONCLUSIVE: the {delta_pct:.1f}% difference is within the "
            f"{worst_spread:.1f}% run-to-run spread. Re-measure both sides "
            "back-to-back before believing it."
        )

    b_per = baseline["per_year_s"]
    c_per = current["per_year_s"]
    print(f"{'Per year (s)':<25} {b_per:>12.3f} {c_per:>12.3f} {b_per / c_per:>9.1f}x")

    # Check biomass parity
    b_bio = baseline["final_biomass"]
    c_bio = current["final_biomass"]
    print()
    print(f"{'Species':<25} {'Baseline':>15} {'Current':>15} {'Ratio':>10}")
    print("-" * 68)
    all_match = True
    for sp in b_bio:
        bv = b_bio[sp]
        cv = c_bio.get(sp, 0.0)
        if bv > 0 and cv > 0:
            ratio = cv / bv
            match = "OK" if abs(ratio - 1.0) < 1e-10 else "DIFF"
        elif bv == 0 and cv == 0:
            ratio = 1.0
            match = "OK"
        else:
            ratio = float("inf")
            match = "DIFF"
        if match != "OK":
            all_match = False
        print(f"{sp:<25} {bv:>15.2f} {cv:>15.2f} {ratio:>9.6f} {match}")

    print()
    if all_match:
        print("PARITY: EXACT — all species biomass identical")
    else:
        print("PARITY: DIFFERS — check optimization correctness!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Python OSMOSE engine")
    parser.add_argument("--years", type=int, default=1, help="Simulation years (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--repeats", type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Discarded warm-up runs before timing (default: 1; 0 = old behaviour)",
    )
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Fixture name (one of "
            + ", ".join(sorted(FIXTURES))
            + ") or path to an all-parameters.csv. Default: examples."
        ),
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("BASELINE", "CURRENT"), help="Compare two result files"
    )
    args = parser.parse_args()

    if args.compare:
        compare_results(Path(args.compare[0]), Path(args.compare[1]))
        return

    config_path = resolve_config(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)

    print(
        f"Benchmarking Python engine: config={config_path.parent.name}, "
        f"{args.years}yr, seed={args.seed}, repeats={args.repeats}"
    )
    print()

    n_threads = apply_single_run_threads()
    print(f"  Numba threads: {n_threads or 'default (numba absent)'}\n")

    result = None
    # Warm-up runs are DISCARDED. With a cold numba cache the first run pays
    # JIT: measured 45.96s against ~5.6s warm on EEC 5yr, an 8x outlier that
    # lands straight in median/min/max and has already produced one phantom
    # 10% "regression" in this repo. One warm-up is enough; --warmup 0 restores
    # the old behaviour.
    for i in range(args.warmup):
        w = run_benchmark(args.years, args.seed, config_path=config_path)
        print(f"  Warm-up {i + 1}/{args.warmup}: {w['elapsed_s']:.3f}s (discarded)")

    timings = []
    for i in range(args.repeats):
        result = run_benchmark(args.years, args.seed, config_path=config_path)
        timings.append(result["elapsed_s"])
        print(f"  Run {i + 1}/{args.repeats}: {result['elapsed_s']:.3f}s")

    timings.sort()
    median = timings[len(timings) // 2]
    spread_pct = (timings[-1] - timings[0]) / timings[0] * 100 if timings[0] > 0 else 0.0

    summary = {
        "config": config_path.parent.name,
        "n_years": args.years,
        "seed": args.seed,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "timings_s": timings,
        "median_s": round(median, 3),
        "min_s": round(timings[0], 3),
        "max_s": round(timings[-1], 3),
        "spread_pct": round(spread_pct, 1),
        "per_year_s": round(median / args.years, 3),
        # Provenance -- compare_results uses these to refuse misleading diffs.
        "captured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "numba_threads": n_threads,
        "final_biomass": result["final_biomass"],
    }

    print()
    print(f"  Median: {median:.3f}s  ({median / args.years:.3f}s/year)")
    print(f"  Min:    {timings[0]:.3f}s  Max: {timings[-1]:.3f}s  Spread: {spread_pct:.1f}%")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
