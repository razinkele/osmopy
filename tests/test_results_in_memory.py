"""Unit tests for OsmoseResults.from_outputs() and in-memory dispatch.

Spec: docs/superpowers/specs/2026-04-19-calibration-python-engine-design.md
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osmose.config import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.output import write_outputs
from osmose.engine.rng import build_rng
from osmose.engine.simulate import simulate
from osmose.results import (
    OsmoseResults,
    _build_dataframes_from_outputs,
)

EXAMPLE_CONFIG = Path(__file__).parent.parent / "data" / "examples" / "osm_all-parameters.csv"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort for comparison: by any ("Time"/"time", "species") columns that exist."""
    sort_cols = [c for c in ("Time", "time", "species", "bin") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable")
    return df.reset_index(drop=True)


def _run_short_simulation(seed: int = 42):
    """Run a 1-year BoB simulation; return (outputs, config, grid)."""
    raw = OsmoseConfigReader().read(EXAMPLE_CONFIG)
    raw["simulation.time.nyear"] = "1"
    config = EngineConfig.from_dict(raw)

    grid_file = raw.get("grid.netcdf.file", "")
    if grid_file:
        grid_path = EXAMPLE_CONFIG.parent / grid_file
        grid = Grid.from_netcdf(grid_path, mask_var=raw.get("grid.var.mask", "mask"))
    else:
        ny = int(raw.get("grid.nlat", raw.get("grid.nrow", "10")))
        nx = int(raw.get("grid.nlon", raw.get("grid.ncol", "10")))
        grid = Grid.from_dimensions(ny=ny, nx=nx)

    rng = np.random.default_rng(seed)
    movement_rngs = build_rng(seed, config.n_species, config.movement_seed_fixed)
    mortality_rngs = build_rng(seed + 1, config.n_species, config.mortality_seed_fixed)

    outputs = simulate(
        config,
        grid,
        rng,
        movement_rngs=movement_rngs,
        mortality_rngs=mortality_rngs,
        output_dir=None,
    )
    return outputs, config, grid


@pytest.fixture(scope="module")
def _disk_and_memory_results():
    """Shared fixture: same simulation, two results — one disk, one memory."""
    outputs, config, grid = _run_short_simulation(seed=42)
    tmpdir = tempfile.TemporaryDirectory()
    write_outputs(outputs, Path(tmpdir.name), config, grid=grid)
    disk_results = OsmoseResults(Path(tmpdir.name))
    # Force-cache everything before tempdir is cleaned up.
    for getter in ("biomass", "abundance", "yield_biomass", "mortality", "diet_matrix"):
        try:
            getattr(disk_results, getter)()
        except FileNotFoundError:
            pass
    memory_results = OsmoseResults.from_outputs(outputs, config, grid)
    yield disk_results, memory_results
    tmpdir.cleanup()


def test_biomass_getter_matches_disk_path(_disk_and_memory_results):
    disk, memory = _disk_and_memory_results
    pd.testing.assert_frame_equal(_normalize(disk.biomass()), _normalize(memory.biomass()))


def test_abundance_getter_matches_disk_path(_disk_and_memory_results):
    disk, memory = _disk_and_memory_results
    pd.testing.assert_frame_equal(_normalize(disk.abundance()), _normalize(memory.abundance()))


def test_yield_biomass_getter_matches_disk_path(_disk_and_memory_results):
    disk, memory = _disk_and_memory_results
    try:
        disk_df = disk.yield_biomass()
    except FileNotFoundError:
        pytest.skip("yield output not produced by this fixture config")
    pd.testing.assert_frame_equal(_normalize(disk_df), _normalize(memory.yield_biomass()))


def test_mortality_getter_matches_disk_path(_disk_and_memory_results):
    disk, memory = _disk_and_memory_results
    try:
        disk_df = disk.mortality()
    except FileNotFoundError:
        pytest.skip("mortality output not produced by this fixture config")
    pd.testing.assert_frame_equal(_normalize(disk_df), _normalize(memory.mortality()))


def test_diet_matrix_getter_matches_disk_path(_disk_and_memory_results):
    disk, memory = _disk_and_memory_results
    try:
        disk_df = disk.diet_matrix()
    except FileNotFoundError:
        pytest.skip("diet_matrix output not produced by this fixture config")
    pd.testing.assert_frame_equal(_normalize(disk_df), _normalize(memory.diet_matrix()))


def test_spatial_biomass_raises_FileNotFoundError_in_memory_mode():
    outputs, config, grid = _run_short_simulation(seed=1)
    r = OsmoseResults.from_outputs(outputs, config, grid)
    with pytest.raises(FileNotFoundError, match="does not support NetCDF"):
        r.spatial_biomass("anything.nc")


def test_from_outputs_idempotent():
    """Calling the same getter twice returns the same DataFrame (cached)."""
    outputs, config, grid = _run_short_simulation(seed=7)
    r = OsmoseResults.from_outputs(outputs, config, grid)
    first = r.biomass()
    second = r.biomass()
    pd.testing.assert_frame_equal(first, second)


def test_from_outputs_populates_all_written_keys():
    """Cache keys from ``_build_dataframes_from_outputs`` equal keys that
    ``write_outputs`` writes to disk. Single source of truth for the
    supported-output-type set.
    """
    outputs, config, grid = _run_short_simulation(seed=42)
    prefix = "osm"

    memory_keys = set(_build_dataframes_from_outputs(outputs, config, grid).keys())

    with tempfile.TemporaryDirectory() as d:
        write_outputs(outputs, Path(d), config, grid=grid)
        disk_files = list(Path(d).rglob(f"{prefix}_*.csv"))
        per_species_keys: set[str] = set()
        cross_species_keys: set[str] = set()
        for f in disk_files:
            stem = f.stem
            # Strip prefix_ and _SimuN suffix to get <output_type>[{sep}<species>]
            assert stem.startswith(prefix + "_")
            inner = stem[len(prefix) + 1 :]
            inner = inner.rsplit("_Simu", 1)[0]
            # Cross-species outputs have no species-name suffix
            if inner in {"biomass", "abundance", "yield", "dietMatrix"}:
                cross_species_keys.add(inner)
                continue
            # Per-species: first separator ('_' or '-') splits output_type and species.
            # Mortality files use dash: mortalityRate-<species>
            # Other per-species files use underscore: abundanceBySize_<species>
            for sep in ("-", "_"):
                if sep in inner:
                    ot, sp = inner.split(sep, 1)
                    per_species_keys.add(f"{ot}_{sp}")
                    break
            else:
                per_species_keys.add(inner)

        # Aggregate disk keys into the same cross/per-species shape the
        # in-memory helper produces.
        disk_keys_aggregated: set[str] = set(cross_species_keys)
        # Per-species keys come in as e.g. "mortalityRate_Anchovy";
        # the memory helper also keys per-species as "mortalityRate_Anchovy".
        disk_keys_aggregated.update(per_species_keys)

        # The memory helper aggregates per-species entries under the bare
        # output_type key (e.g., all mortalityRate_* entries become a single
        # "mortalityRate" cache entry). Build the comparable disk-side set.
        disk_output_types: set[str] = set(cross_species_keys)
        for k in per_species_keys:
            disk_output_types.add(k.split("_", 1)[0])

        assert memory_keys == disk_output_types, (
            f"In-memory and disk paths disagree on output types.\n"
            f"In memory only: {memory_keys - disk_output_types}\n"
            f"On disk only:   {disk_output_types - memory_keys}"
        )
