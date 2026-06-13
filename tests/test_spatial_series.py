"""Tests for osmose.spatial_series.cell_timeseries (per-cell NetCDF extraction)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from osmose.spatial_series import cell_timeseries, cell_timeseries_from_dataset


def _make_spatial_nc(path, *, n_time=6, species=("cod", "sprat"), ny=4, nx=5, land_cell=(0, 0)):
    """Build a synthetic spatial NetCDF matching the engine's
    (time, species, lat, lon) layout (verified against the real output).

    value[t, s, y, x] = (s + 1) * 100 + t * 10 + y + x * 0.1 — identifiable per
    cell/species/time. `land_cell` (lat_idx, lon_idx) is NaN across all time and
    species, mirroring the engine's "NaN == land" convention.
    """
    times = np.arange(n_time) / 12.0
    lat = np.linspace(54.0, 55.0, ny)
    lon = np.linspace(10.0, 12.0, nx)
    data = np.empty((n_time, len(species), ny, nx), dtype=float)
    for t in range(n_time):
        for s in range(len(species)):
            for y in range(ny):
                for x in range(nx):
                    data[t, s, y, x] = (s + 1) * 100 + t * 10 + y + x * 0.1
    ly, lx = land_cell
    data[:, :, ly, lx] = np.nan
    ds = xr.Dataset(
        {
            "spatial_biomass": (("time", "species", "lat", "lon"), data),
            # a variable without lat/lon dims, for the negative test
            "totals": (("time", "species"), np.zeros((n_time, len(species)))),
        },
        coords={"time": times, "species": list(species), "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)
    ds.close()
    return path


def test_known_trajectory_sum_over_species(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc")
    times, vals = cell_timeseries(p, "spatial_biomass", lat_index=1, lon_index=2, reduce="sum")
    assert times.shape == (6,)
    assert vals.shape == (6,)
    # per-species at (y=1, x=2): (s+1)*100 + t*10 + 1 + 0.2; sum over s in {0,1}
    expected = np.array([300 + 20 * t + 2.4 for t in range(6)])
    np.testing.assert_allclose(vals, expected)
    np.testing.assert_allclose(times, np.arange(6) / 12.0)


def test_single_species_selection(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc")
    _, vals = cell_timeseries(p, "spatial_biomass", lat_index=1, lon_index=2, species="cod")
    expected = np.array([100 + 10 * t + 1 + 0.2 for t in range(6)])  # cod is s=0
    np.testing.assert_allclose(vals, expected)


def test_mean_reduce_over_species(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc")
    _, vals = cell_timeseries(p, "spatial_biomass", lat_index=0, lon_index=1, reduce="mean")
    # mean over s of (s+1)*100 + t*10 + 0 + 0.1 = 150 + 10t + 0.1
    expected = np.array([150 + 10 * t + 0.1 for t in range(6)])
    np.testing.assert_allclose(vals, expected)


def test_land_cell_is_all_nan(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc", land_cell=(0, 0))
    _, vals = cell_timeseries(p, "spatial_biomass", lat_index=0, lon_index=0, reduce="sum")
    assert np.isnan(vals).all()


def test_land_cell_single_species_is_nan(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc", land_cell=(0, 0))
    _, vals = cell_timeseries(p, "spatial_biomass", lat_index=0, lon_index=0, species="cod")
    assert np.isnan(vals).all()


def test_out_of_range_index_raises(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc")
    with pytest.raises(IndexError):
        cell_timeseries(p, "spatial_biomass", lat_index=99, lon_index=0)
    with pytest.raises(IndexError):
        cell_timeseries(p, "spatial_biomass", lat_index=0, lon_index=-1)


def test_non_spatial_variable_raises(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc")
    with pytest.raises(ValueError):
        cell_timeseries(p, "totals", lat_index=0, lon_index=0)


def test_missing_variable_raises(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc")
    with pytest.raises(KeyError):
        cell_timeseries(p, "does_not_exist", lat_index=0, lon_index=0)


def test_invalid_reduce_raises(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc")
    with pytest.raises(ValueError):
        cell_timeseries(p, "spatial_biomass", lat_index=0, lon_index=0, reduce="median")


def test_result_is_single_cell_vector(tmp_path):
    # confirms cell selection happened (1-D over time), not a whole-cube read
    p = _make_spatial_nc(tmp_path / "s.nc")
    _, vals = cell_timeseries(p, "spatial_biomass", lat_index=2, lon_index=3, reduce="sum")
    assert vals.ndim == 1


def test_from_dataset_matches_path_variant(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc")
    t_path, v_path = cell_timeseries(p, "spatial_biomass", lat_index=1, lon_index=2, reduce="sum")
    with xr.open_dataset(p) as ds:
        t_ds, v_ds = cell_timeseries_from_dataset(
            ds, "spatial_biomass", lat_index=1, lon_index=2, reduce="sum"
        )
    np.testing.assert_allclose(v_ds, v_path)
    np.testing.assert_allclose(t_ds, t_path)


def test_from_dataset_works_on_already_open_handle(tmp_path):
    # The UI keeps one open handle; the dataset variant must read from it without
    # opening a second handle (which can raise an HDF5 locking error).
    p = _make_spatial_nc(tmp_path / "s.nc")
    with xr.open_dataset(p) as held_open:  # simulates the page's _spatial_ds
        _, vals = cell_timeseries_from_dataset(
            held_open, "spatial_biomass", lat_index=2, lon_index=2, species="cod"
        )
    expected = np.array([100 + 10 * t + 2 + 0.2 for t in range(6)])  # cod=s0 at (y=2,x=2)
    np.testing.assert_allclose(vals, expected)
