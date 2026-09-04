"""Tests for scripts/build_baltic_temperature_forcing.py (C3 bioen Stage-1 Task 10).

Synthetic-grid unit tests only -- no CMEMS files or network access required. The full
build() pipeline is exercised by running the script's CLI against the cached year-files
under data/cmems_cache/cmems_downloads (see task-10-report.md for that manual check).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "btf", ROOT / "scripts" / "build_baltic_temperature_forcing.py"
)
btf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(btf)


def test_layer0_is_nan_aware_over_depth():
    a = np.full((2, 3, 2, 2), np.nan)
    a[:, 0] = 10.0
    a[:, 1, 0, 0] = 12.0  # pixel (0,0) has 2 levels, others 1
    out = btf.layer0_from_thetao(a)
    assert out.shape == (2, 2, 2) and out[0, 0, 0] == 11.0 and out[0, 1, 1] == 10.0


def test_duplicate_months_convention():
    clim = np.arange(12, dtype=float)[:, None, None] * np.ones((1, 2, 2))
    d = btf.duplicate_months(clim)
    assert d.shape == (24, 2, 2) and d[0, 0, 0] == 0 and d[1, 0, 0] == 0 and d[16, 0, 0] == 8
    assert d[17, 0, 0] == 8


def test_bottom_depth_from_so_takes_deepest_finite_level():
    so = np.full((1, 3, 2, 2), np.nan)
    so[0, 0] = 7
    so[0, 1, 0, 0] = 8
    so[0, 2, 0, 0] = 9
    so[0, 1, 1, 1] = 8
    depth = np.array([1.0, 20.0, 60.0])
    bd = btf.bottom_depth_from_so(so, depth)
    assert bd[0, 0] == 60.0 and bd[1, 1] == 20.0 and bd[0, 1] == 1.0


def test_validate_fires_on_swapped_layers_and_passes_on_correct():
    wet = np.ones((2, 2), dtype=bool)
    temp = np.zeros((24, 2, 2, 2), dtype=np.float32)
    temp[:, 0] = 15.0
    temp[:, 1] = 5.0  # surface warm, bottom cold
    bd = np.full((2, 2), 80.0)
    ds = xr.Dataset(
        {
            "temperature": (["time", "layer", "latitude", "longitude"], temp),
            "bottom_depth": (["latitude", "longitude"], bd),
        }
    )
    btf.validate(ds, wet)
    swapped = ds.copy()
    swapped["temperature"].values[:, [0, 1]] = swapped["temperature"].values[:, [1, 0]]
    with pytest.raises(AssertionError, match="layer-order"):
        btf.validate(swapped, wet)
    bad = ds.copy()
    bad["temperature"].values[3, 0, 0, 0] = 45.0
    with pytest.raises(AssertionError, match="range"):
        btf.validate(bad, wet)


def test_masked_regrid_is_the_oxygen_builders_function():
    """Task 10 must reuse the wet-aware regrid, not reinvent (or fall back to) an
    unmasked nearest-neighbour -- the review that drove this requirement (spec 3.3)
    measured 66/616 wet cells snapping to a dry native pixel with grid.regrid.

    Identity (`is`) across two independent importlib.util.module_from_spec loads of
    make_baltic_oxygen_forcing.py doesn't hold (each load mints its own function
    object, same as the C4 harness's own importlib idiom) -- so this checks
    provenance (__module__/__name__) instead of object identity, plus a behavioural
    check against a case _masked_regrid is documented to get right and a plain
    nearest-neighbour would not: a wet cell whose geometrically nearest native pixel
    is dry must draw from the nearest *valid* pixel instead.
    """
    assert btf.masked_regrid.__module__ == "make_baltic_oxygen_forcing"
    assert btf.masked_regrid.__name__ == "_masked_regrid"

    # The wet target cell (tlon=0.4) is geometrically nearest to native pixel lon=0.0,
    # which is dry (NaN); the only valid native pixel is at lon=5.0, value 3.0. A plain
    # unmasked argmin nearest-neighbour would return NaN; the masked regrid must skip
    # the dry pixel and return 3.0.
    raw = np.array([[[np.nan, 3.0]]])  # (t=1, src_lat=1, src_lon=2)
    src_lat = np.array([0.0])
    src_lon = np.array([0.0, 5.0])
    tlat = np.array([0.0])
    tlon = np.array([0.4])
    wet = np.array([[True]])
    out = btf.masked_regrid(raw, src_lat, src_lon, tlat, tlon, wet)
    assert out[0, 0, 0] == 3.0


def test_shipped_forcing_file_passes_its_own_validate():
    """The COMMITTED .nc must satisfy every pin `validate()` enforces.

    Task 10's review (MINOR-1) noted that the unit tests above exercise only the pure
    helpers on synthetic arrays -- nothing loaded the shipped artifact. A refactor of
    the build pipeline, or a re-run of the builder against different inputs, could
    therefore have silently corrupted the committed file's axis order, frame count,
    land mask or physical range with every test still green. That is the failure mode
    this branch has hit repeatedly: a gate green over the thing it exists to protect.

    This test closes it by running the real `validate()` against the real file, so the
    artifact and its checker cannot drift apart. It is a genuine gate, not a smoke
    check: swapping the layer axis, truncating to 12 frames, or filling land with 0.0
    instead of NaN each trips one of `validate()`'s asserts.
    """
    nc = ROOT / "data" / "baltic" / "forcing" / "baltic_temperature_2layer_climatology.nc"
    assert nc.exists(), f"shipped forcing file missing: {nc}"

    # Use the builder's OWN grid constant, not a hand-written path: a hard-coded path
    # that drifts would make this test skip silently, which is the exact failure mode
    # it exists to prevent.
    grid = btf.DEFAULT_GRID_NC
    assert grid.exists(), f"baltic grid missing at {grid}"

    with xr.open_dataset(nc) as ds, xr.open_dataset(grid) as g:
        wet = np.asarray(g["mask"].values).astype(bool)
        btf.validate(ds, wet)

        # Pin the two facts a reader of the results doc would rely on, beyond validate():
        # the frame count the engine's `step % n_frames` indexing depends on, and that
        # land really is NaN rather than 0.0 (this repo's oxygen forcing uses 0.0 for
        # land, so the two conventions are genuinely different and easy to confuse).
        t = ds["temperature"].values
        assert t.shape[0] == 24, f"engine indexes step % n_frames; got {t.shape[0]} frames"
        assert np.isnan(t[:, :, ~wet]).all(), "land cells must be NaN, not 0.0"
