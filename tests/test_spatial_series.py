"""Tests for osmose.spatial_series.cell_timeseries (per-cell NetCDF extraction)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from osmose.spatial_series import (
    cell_timeseries,
    cell_timeseries_from_dataset,
    spatial_slice_2d,
)


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


def test_spatial_slice_2d_sum_over_species(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc", n_time=6, ny=4, nx=5, land_cell=(0, 0))
    with xr.open_dataset(p) as ds:
        sl = spatial_slice_2d(ds, "spatial_biomass", time_index=0)
    assert sl.shape == (4, 5)
    # at t=0, sum over 2 species of (s+1)*100 + 0 + y + x*0.1 = 300 + 2y + 0.2x
    assert sl[1, 2] == pytest.approx(300 + 2 * 1 + 0.2 * 2)
    assert np.isnan(sl[0, 0])  # land stays NaN


def test_spatial_slice_2d_single_species(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc", n_time=6, ny=4, nx=5)
    with xr.open_dataset(p) as ds:
        sl = spatial_slice_2d(ds, "spatial_biomass", time_index=2, species="cod")
    # cod (s=0) at t=2: 100 + 20 + y + 0.1x
    assert sl[3, 4] == pytest.approx(100 + 20 + 3 + 0.1 * 4)


def test_spatial_slice_2d_time_clamped(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc", n_time=6, ny=3, nx=3)
    with xr.open_dataset(p) as ds:
        last = spatial_slice_2d(ds, "spatial_biomass", time_index=999)
        explicit = spatial_slice_2d(ds, "spatial_biomass", time_index=5)
    np.testing.assert_allclose(last, explicit, equal_nan=True)


def test_spatial_slice_2d_non_spatial_raises(tmp_path):
    p = _make_spatial_nc(tmp_path / "s.nc")
    with xr.open_dataset(p) as ds:
        with pytest.raises(ValueError):
            spatial_slice_2d(ds, "totals", time_index=0)


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


# --- spatial_diff_2d -------------------------------------------------------

from osmose.spatial_series import grid_latlon, spatial_diff_2d  # noqa: E402


def _diff_ds(*, ny=3, nx=4, n_time=2, species=("cod", "sprat"), base=0.0, lat=None, land=None):
    """In-memory (time, species, lat, lon) dataset; cell value is identifiable."""
    lat = np.linspace(54.0, 55.0, ny) if lat is None else np.asarray(lat, dtype=float)
    lon = np.linspace(10.0, 12.0, nx)
    ns = len(species)
    data = np.fromfunction(
        lambda t, s, y, x: base + s * 1000.0 + t * 100.0 + y * 10.0 + x,
        (n_time, ns, ny, nx),
        dtype=float,
    )
    if land is not None:
        ly, lx = land
        data[:, :, ly, lx] = np.nan
    return xr.Dataset(
        {"spatial_biomass": (("time", "species", "lat", "lon"), data)},
        coords={
            "time": np.arange(n_time) / 12.0,
            "species": list(species),
            "lat": lat,
            "lon": lon,
        },
    )


def test_spatial_diff_2d_sum_over_species():
    a = _diff_ds(base=0.0)
    b = _diff_ds(base=7.0)  # every cell of B is A + 7 per species; 2 species -> +14 summed
    diff = spatial_diff_2d(a, b, "spatial_biomass", time_a=0, time_b=0)
    assert diff.shape == (3, 4)
    np.testing.assert_allclose(diff, np.full((3, 4), 14.0))


def test_spatial_diff_2d_land_nan_propagates():
    a = _diff_ds(land=(0, 0))
    b = _diff_ds(base=7.0)  # B has no land at (0,0); A does -> result NaN there
    diff = spatial_diff_2d(a, b, "spatial_biomass")
    assert np.isnan(diff[0, 0])
    assert np.isfinite(diff[1, 1])


def test_spatial_diff_2d_single_species_by_name():
    a = _diff_ds(base=0.0)
    b = _diff_ds(base=7.0)
    diff = spatial_diff_2d(a, b, "spatial_biomass", species="sprat")
    np.testing.assert_allclose(diff, np.full((3, 4), 7.0))  # one species -> +7


def test_spatial_diff_2d_time_indices_independent():
    a = _diff_ds(base=0.0)  # value includes t*100
    b = _diff_ds(base=0.0)
    # B at t=1 minus A at t=0, summed over 2 species: each species differs by +100 -> +200
    diff = spatial_diff_2d(a, b, "spatial_biomass", time_a=0, time_b=1)
    np.testing.assert_allclose(diff, np.full((3, 4), 200.0))


def test_spatial_diff_2d_identical_runs_all_zero():
    a = _diff_ds(land=(0, 0))
    diff = spatial_diff_2d(a, a, "spatial_biomass")
    assert np.isnan(diff[0, 0])
    finite = diff[np.isfinite(diff)]
    np.testing.assert_allclose(finite, 0.0)


def test_spatial_diff_2d_shape_mismatch_raises():
    a = _diff_ds(nx=4)
    b = _diff_ds(nx=5)
    with pytest.raises(ValueError, match="shape"):
        spatial_diff_2d(a, b, "spatial_biomass")


def test_spatial_diff_2d_coord_mismatch_raises():
    a = _diff_ds(lat=[54.0, 54.5, 55.0])
    b = _diff_ds(lat=[60.0, 60.5, 61.0])  # same shape, different coords
    with pytest.raises(ValueError, match="coordinate"):
        spatial_diff_2d(a, b, "spatial_biomass")


def test_spatial_diff_2d_int_species_rejected():
    a = _diff_ds()
    with pytest.raises(TypeError, match="name"):
        spatial_diff_2d(a, a, "spatial_biomass", species=1)


def test_grid_latlon_returns_coord_arrays():
    a = _diff_ds()
    lat, lon = grid_latlon(a, "spatial_biomass")
    np.testing.assert_allclose(lat, np.linspace(54.0, 55.0, 3))
    np.testing.assert_allclose(lon, np.linspace(10.0, 12.0, 4))
