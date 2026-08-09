# tests/test_baltic_oxygen_forcing.py
"""Bottom-O2 forcing file: grid, frames, units, plausibility (spec Phase 2a)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

NC = Path(__file__).resolve().parents[1] / "data" / "baltic" / "baltic_oxygen_bottom.nc"

pytestmark = pytest.mark.skipif(not NC.exists(), reason="oxygen forcing not generated yet")


def test_dims_and_frames():
    ds = xr.open_dataset(NC)
    o2 = ds["o2b"]
    assert o2.dims[0] in ("time", "t")
    # 24 frames == simulation.time.ndtperyear — PhysicalData.get_value indexes step % nframes,
    # so 12 monthly frames would silently misalign from step 13 onward (plan Global Constraints).
    assert o2.shape[0] == 24
    assert o2.shape[1:] == (40, 50) or o2.shape[1:] == (50, 40)
    ds.close()


def test_month_duplication():
    ds = xr.open_dataset(NC)
    v = ds["o2b"].values
    for m in range(12):
        np.testing.assert_array_equal(v[2 * m], v[2 * m + 1])
    ds.close()


def test_values_plausible_mmol_m3():
    ds = xr.open_dataset(NC)
    v = ds["o2b"].values
    wet = v[~np.isnan(v)]
    wet = wet[wet != 0.0]
    # Baltic bottom O2 spans anoxic deeps (~0, occasionally negative-as-H2S-proxy in ERGOM,
    # clipped to >= 0 by the writer) to well-oxygenated coasts (~300-400 mmol/m3)
    assert wet.min() >= 0.0
    # Upper bound is a plausibility guard, not a physical limit: observed max 616.7 mmol/m3
    # (Bothnian Bay coastal, April spring-bloom supersaturation, verified spatially coherent
    # -- not a download/regrid artifact). 650 keeps headroom above that real value without
    # switching the x2 month duplication (load-bearing for step % nframes alignment, see
    # test_month_duplication) for resample_to_24's smoothed interpolation just to fit 600.
    assert 150.0 <= wet.max() <= 650.0
    # hypoxia must actually exist in the domain or the coupling is vacuous
    assert (wet < 90.0).mean() > 0.02
    ds.close()


def test_no_artifact_zero_at_wet_cells():
    # f_o2_hill(0) == 0 exactly, so an artifact zero silently zeroes benthos K in real habitat.
    # Genuine near-anoxia registers as ~1e-18, never exact 0.0 — so exact zeros at wet cells
    # can only be nearest-neighbour holes (65 such cells before the masked-source fix).
    from osmose.engine.grid import Grid

    grid_nc = Path(__file__).resolve().parents[1] / "data" / "baltic" / "baltic_grid.nc"
    grid = Grid.from_netcdf(grid_nc)
    wet_mask = grid.ocean_mask  # (nlat, nlon) bool, True = ocean

    ds = xr.open_dataset(NC)
    v = ds["o2b"].values  # (time, nlat, nlon)
    assert wet_mask.shape == v.shape[1:]
    wet_values = v[:, wet_mask]  # (time, n_wet_cells)
    assert not np.any(wet_values == 0.0)
    ds.close()
