# tests/test_forcing_grid.py
import numpy as np

from osmose.forcing.grid import (
    apply_land_mask,
    cell_volume_m3,
    get_var,
    regrid,
    resample_to_24,
    target_coords,
)
from osmose.maps.builder import GridSpec

BALTIC = GridSpec(nlon=50, nlat=40, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)
SMALL = GridSpec(nlon=10, nlat=8, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)


def test_target_coords_descending_lat_ascending_lon():
    lat, lon = target_coords(BALTIC)
    assert lat.shape == (40,) and lon.shape == (50,)
    assert lat[0] > lat[-1]  # north -> south
    assert lon[0] < lon[-1]  # west -> east
    # cell centers, not edges
    assert lat[0] < 66 and lat[-1] > 54


def test_regrid_picks_nearest_src_cell():
    # NON-constant field so the test distinguishes correct nearest-neighbour
    # selection (and lat/lon orientation) from a degenerate/transposed impl.
    src_lat = np.array([66.0, 60.0, 54.0])  # descending
    src_lon = np.array([10.0, 20.0, 30.0])  # ascending
    # value encodes (lat_index, lon_index): 10*li + ci
    data = np.array([[[10 * li + ci for ci in range(3)] for li in range(3)]], dtype=float)
    out = regrid(data, src_lat, src_lon, SMALL)
    assert out.shape == (1, 8, 10)
    # northern-most target row (high lat) must map to src lat index 0; southern to 2
    assert out[0, 0, 0] == 0.0  # NW corner -> src (lat0, lon0)
    assert out[0, -1, -1] == 22.0  # SE corner -> src (lat2, lon2)
    assert out[0, 0, -1] == 2.0  # NE corner -> src (lat0, lon2)


def test_regrid_warns_when_target_exceeds_source_extent(caplog):
    import logging

    # source covers only a small central patch; SMALL grid spans the full Baltic box
    src_lat = np.array([60.0, 59.0])
    src_lon = np.array([19.0, 20.0])
    data = np.ones((1, 2, 2)) * 5.0
    with caplog.at_level(logging.WARNING):
        out = regrid(data, src_lat, src_lon, SMALL)
    assert out.shape == (1, 8, 10)
    assert any("beyond source coverage" in r.message for r in caplog.records)
    # out-of-coverage cells are nearest-edge filled (still the source value here)
    assert np.allclose(out, 5.0)


def test_resample_to_24_identity_and_interp():
    data24 = np.ones((24, 2, 2))
    assert resample_to_24(data24) is data24 or np.allclose(resample_to_24(data24), data24)
    data12 = np.arange(12)[:, None, None] * np.ones((12, 2, 2))
    out = resample_to_24(data12)
    assert out.shape == (24, 2, 2)
    assert out.min() >= 0 and out.max() <= 11


def test_cell_volume_positive_and_scales_with_depth():
    v10 = cell_volume_m3(BALTIC, 10.0)
    v50 = cell_volume_m3(BALTIC, 50.0)
    assert v10 > 0
    assert np.isclose(v50, 5 * v10)


def test_apply_land_mask_nans_land_and_noops_on_mismatch():
    groups = {"A": np.ones((24, 8, 10))}
    mask = np.ones((8, 10), dtype=bool)
    mask[0, 0] = False  # one land cell
    apply_land_mask(groups, mask)
    assert np.isnan(groups["A"][0, 0, 0])
    assert not np.isnan(groups["A"][0, 1, 1])
    # shape mismatch -> no-op (no raise)
    g2 = {"B": np.ones((24, 4, 4))}
    apply_land_mask(g2, mask)
    assert not np.any(np.isnan(g2["B"]))


def test_get_var_promotes_2d_and_fills_nan():
    import xarray as xr

    ds = xr.Dataset({"x": (["latitude", "longitude"], np.array([[1.0, np.nan], [3.0, 4.0]]))})
    arr = get_var(ds, "x")
    assert arr.shape == (1, 2, 2)
    assert arr[0, 0, 1] == 0.0  # NaN -> 0
    assert get_var(ds, "missing") is None
