#!/usr/bin/env python3
"""Save a parity baseline for the Python OSMOSE engine.

Runs the engine with a fixed seed and saves per-step biomass, abundance,
and mortality arrays to a compressed .npz file. Used by test_engine_parity.py
to verify that optimizations produce identical outputs.

Usage:
    .venv/bin/python scripts/save_parity_baseline.py [--years N] [--seed S]
    .venv/bin/python scripts/save_parity_baseline.py --statistical [--years N] [--seeds N]

Output:
    tests/baselines/parity_baseline_bob_<years>yr_seed<seed>.npz
    tests/baselines/statistical_baseline_bob_<years>yr_<n>seeds.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
EXAMPLES_CONFIG = PROJECT_DIR / "data" / "examples" / "osm_all-parameters.csv"
BASELINE_DIR = PROJECT_DIR / "tests" / "baselines"


def save_baseline(n_years: int, seed: int) -> Path:
    """Run engine and save outputs as baseline .npz file."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import simulate

    reader = OsmoseConfigReader()
    raw = reader.read(EXAMPLES_CONFIG)
    raw["simulation.time.nyear"] = str(n_years)

    cfg = EngineConfig.from_dict(raw)

    grid_file = raw.get("grid.netcdf.file", "")
    if grid_file:
        grid = Grid.from_netcdf(
            PROJECT_DIR / "data" / "examples" / grid_file,
            mask_var=raw.get("grid.var.mask", "mask"),
        )
    else:
        ny = int(raw.get("grid.nline", "1"))
        nx = int(raw.get("grid.ncolumn", "1"))
        grid = Grid.from_dimensions(ny=ny, nx=nx)

    rng = np.random.default_rng(seed)

    print(f"Running engine: {n_years}yr, seed={seed}...")
    outputs = simulate(cfg, grid, rng)

    n_steps = len(outputs)
    n_species = len(outputs[0].biomass)
    n_causes = outputs[0].mortality_by_cause.shape[1]

    # Collect per-step arrays
    biomass = np.zeros((n_steps, n_species), dtype=np.float64)
    abundance = np.zeros((n_steps, n_species), dtype=np.float64)
    mortality = np.zeros((n_steps, n_species, n_causes), dtype=np.float64)

    for i, out in enumerate(outputs):
        biomass[i] = out.biomass
        abundance[i] = out.abundance
        mortality[i] = out.mortality_by_cause

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"parity_baseline_bob_{n_years}yr_seed{seed}.npz"
    out_path = BASELINE_DIR / filename

    np.savez_compressed(
        out_path,
        biomass=biomass,
        abundance=abundance,
        mortality=mortality,
        species_names=np.array(cfg.species_names),
        n_years=np.array(n_years),
        seed=np.array(seed),
        n_steps=np.array(n_steps),
    )

    print(f"Baseline saved: {out_path}")
    print(f"  Steps: {n_steps}, Species: {n_species}, Causes: {n_causes}")
    print(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")

    return out_path


STATISTICAL_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def save_statistical_baseline(n_years: int, n_seeds: int) -> Path:
    """Run engine with multiple seeds and save mean/std biomass as statistical baseline."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import simulate

    reader = OsmoseConfigReader()
    raw = reader.read(EXAMPLES_CONFIG)
    raw["simulation.time.nyear"] = str(n_years)

    cfg = EngineConfig.from_dict(raw)

    grid_file = raw.get("grid.netcdf.file", "")
    if grid_file:
        grid = Grid.from_netcdf(
            PROJECT_DIR / "data" / "examples" / grid_file,
            mask_var=raw.get("grid.var.mask", "mask"),
        )
    else:
        ny = int(raw.get("grid.nline", "1"))
        nx = int(raw.get("grid.ncolumn", "1"))
        grid = Grid.from_dimensions(ny=ny, nx=nx)

    seeds = STATISTICAL_SEEDS[:n_seeds]
    final_biomasses = []

    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i + 1}/{n_seeds})...")
        rng = np.random.default_rng(seed)
        outputs = simulate(cfg, grid, rng)
        final_biomasses.append(outputs[-1].biomass)

    all_bio = np.array(final_biomasses)  # (n_seeds, n_species)
    mean_bio = np.mean(all_bio, axis=0)
    std_bio = np.std(all_bio, axis=0)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"statistical_baseline_bob_{n_years}yr_{n_seeds}seeds.npz"
    out_path = BASELINE_DIR / filename

    np.savez_compressed(
        out_path,
        mean_biomass=mean_bio,
        std_biomass=std_bio,
        seeds=np.array(seeds),
        species_names=np.array(cfg.species_names),
        n_years=np.array(n_years),
    )

    print(f"Statistical baseline saved: {out_path}")
    print(f"  Seeds: {n_seeds}, Species: {len(mean_bio)}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Save parity baseline for OSMOSE engine")
    parser.add_argument("--years", type=int, default=1, help="Simulation years (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--statistical", action="store_true", help="Save multi-seed statistical baseline")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds for statistical baseline")
    args = parser.parse_args()

    if not EXAMPLES_CONFIG.exists():
        print(f"ERROR: Config not found at {EXAMPLES_CONFIG}")
        sys.exit(1)

    if args.statistical:
        save_statistical_baseline(args.years, args.seeds)
    else:
        save_baseline(args.years, args.seed)


if __name__ == "__main__":
    main()
