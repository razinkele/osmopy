#!/usr/bin/env python3
"""Save a parity baseline for the Python OSMOSE engine.

Runs the engine with a fixed seed and saves per-step biomass, abundance,
and mortality arrays to a compressed .npz file. Used by test_engine_parity.py
to verify that optimizations produce identical outputs.

Usage:
    .venv/bin/python scripts/save_parity_baseline.py [--config bob|eec] [--years N] [--seed S]
    .venv/bin/python scripts/save_parity_baseline.py --statistical [--years N] [--seeds N]
    .venv/bin/python scripts/save_parity_baseline.py --outputs [--config bob|eec]

Output:
    tests/baselines/parity_baseline_bob_<years>yr_seed<seed>.npz
    tests/baselines/parity_baseline_eec_<years>yr_seed<seed>.npz
    tests/baselines/statistical_baseline_bob_<years>yr_<n>seeds.npz
    tests/baselines/parity_outputs_{bob,eec}_<years>yr_seed<seed>.npz (--outputs: every per-step
        output the core baseline omits -- by-stage counts and rates, yield, distributions, mean
        TL, mean size, SSB; tests/test_engine_output_parity.py)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
EXAMPLES_CONFIG = PROJECT_DIR / "data" / "examples" / "osm_all-parameters.csv"
EEC_CONFIG = PROJECT_DIR / "data" / "eec_full" / "eec_all-parameters.csv"
BASELINE_DIR = PROJECT_DIR / "tests" / "baselines"

# Per-config metadata: (config_csv, base_dir, baseline_name_prefix)
_CONFIG_META = {
    "bob": (EXAMPLES_CONFIG, PROJECT_DIR / "data" / "examples", "bob"),
    "eec": (EEC_CONFIG, PROJECT_DIR / "data" / "eec_full", "eec"),
}


def run_outputs(n_years: int, seed: int, config: str = "bob"):
    """Run the engine on a named config; return ``(step_outputs, engine_config)``.

    Args:
        n_years: Number of simulation years.
        seed: RNG seed.
        config: Which config to run — "bob" (Bay of Biscay) or "eec" (Eastern English Channel).
    """
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import simulate

    if config not in _CONFIG_META:
        raise ValueError(f"Unknown config {config!r}; choose from {list(_CONFIG_META)}")

    config_path, base_dir, _ = _CONFIG_META[config]

    reader = OsmoseConfigReader()
    raw = reader.read(config_path)
    raw["simulation.time.nyear"] = str(n_years)

    cfg = EngineConfig.from_dict(raw)

    grid_file = raw.get("grid.netcdf.file", "")
    if grid_file:
        grid = Grid.from_netcdf(
            base_dir / grid_file,
            mask_var=raw.get("grid.var.mask", "mask"),
        )
    else:
        ny = int(raw.get("grid.nline", "1"))
        nx = int(raw.get("grid.ncolumn", "1"))
        grid = Grid.from_dimensions(ny=ny, nx=nx)

    rng = np.random.default_rng(seed)
    return simulate(cfg, grid, rng), cfg


# Fields the original parity baseline already covers; the outputs baseline holds the rest.
_CORE_FIELDS = ("step", "biomass", "abundance", "mortality_by_cause")


def flatten_outputs(outputs) -> dict[str, np.ndarray]:
    """Every per-step output field not in the core baseline, as ``name -> (n_steps, ...)`` arrays.

    Array fields are stacked over steps. Per-species dict fields (distributions, mean TL, mean
    size) become one array per species, ``"<field>__sp<k>"``; a step on which the species is
    absent from the dict (e.g. mean TL of a species with no biomass) is filled with NaN, so a
    change in WHICH steps carry it is a mismatch too. Fields that are None on every step are
    omitted, so a field that stops being produced shows up as a missing key.
    """
    import dataclasses

    flat: dict[str, np.ndarray] = {}
    for field in dataclasses.fields(outputs[0]):
        name = field.name
        if name in _CORE_FIELDS:
            continue
        values = [getattr(out, name) for out in outputs]
        if all(v is None for v in values):
            continue
        if any(v is None for v in values):
            raise ValueError(f"{name} is None on some steps only")
        if isinstance(values[0], dict):
            keys = sorted(set().union(*(v.keys() for v in values)))
            for k in keys:
                present = [np.asarray(v[k], dtype=np.float64) for v in values if k in v]
                gap = np.full_like(present[0], np.nan)
                flat[f"{name}__sp{k}"] = np.stack(
                    [np.asarray(v[k], dtype=np.float64) if k in v else gap for v in values]
                )
        else:
            flat[name] = np.stack([np.asarray(v, dtype=np.float64) for v in values])
    return flat


def save_outputs_baseline(n_years: int, seed: int, config: str = "bob") -> Path:
    """Run engine and save every non-core per-step output as a baseline .npz file."""
    _, _, name_prefix = _CONFIG_META[config]
    print(f"Running engine ({config}): {n_years}yr, seed={seed}...")
    outputs, _ = run_outputs(n_years, seed, config)
    flat = flatten_outputs(outputs)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BASELINE_DIR / f"parity_outputs_{name_prefix}_{n_years}yr_seed{seed}.npz"
    # numpy's stub types **kwds as the allow_pickle flag; any array keyword is valid at runtime.
    np.savez_compressed(out_path, **flat)  # type: ignore[arg-type]
    print(f"Outputs baseline saved: {out_path}")
    print(f"  Fields: {len(flat)}, File size: {out_path.stat().st_size / 1024:.1f} KB")
    return out_path


def save_baseline(n_years: int, seed: int, config: str = "bob") -> Path:
    """Run engine and save outputs as baseline .npz file.

    Args:
        n_years: Number of simulation years.
        seed: RNG seed.
        config: Which config to run — "bob" (Bay of Biscay) or "eec" (Eastern English Channel).
    """
    if config not in _CONFIG_META:
        raise ValueError(f"Unknown config {config!r}; choose from {list(_CONFIG_META)}")
    _, _, name_prefix = _CONFIG_META[config]

    print(f"Running engine ({config}): {n_years}yr, seed={seed}...")
    outputs, cfg = run_outputs(n_years, seed, config)

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
    filename = f"parity_baseline_{name_prefix}_{n_years}yr_seed{seed}.npz"
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
    parser.add_argument(
        "--config",
        choices=list(_CONFIG_META),
        default="bob",
        help="Which config to baseline: 'bob' (Bay of Biscay) or 'eec' (E. English Channel)",
    )
    parser.add_argument(
        "--statistical", action="store_true", help="Save multi-seed statistical baseline"
    )
    parser.add_argument(
        "--outputs",
        action="store_true",
        help="Save the outputs baseline: every per-step output the core baseline omits",
    )
    parser.add_argument(
        "--seeds", type=int, default=10, help="Number of seeds for statistical baseline"
    )
    args = parser.parse_args()

    config_path, _, _ = _CONFIG_META[args.config]
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)

    if args.statistical:
        save_statistical_baseline(args.years, args.seeds)
    elif args.outputs:
        save_outputs_baseline(args.years, args.seed, config=args.config)
    else:
        save_baseline(args.years, args.seed, config=args.config)


if __name__ == "__main__":
    main()
