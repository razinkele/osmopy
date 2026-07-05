import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_baltic_salinity_forcing as bld  # noqa: E402


def test_bottom_extract_deepest_valid():
    # 1 time, 3 depths, 2x2. depth ascending. NaN = below seafloor / land.
    nan = np.nan
    arr = np.array(
        [
            [  # time 0
                [[10.0, 20.0], [nan, 5.0]],  # depth 0 (shallow)
                [[11.0, 21.0], [nan, 6.0]],  # depth 1
                [[12.0, nan], [nan, 7.0]],  # depth 2 (deep)
            ]
        ]
    )  # shape (1,3,2,2)
    out = bld.bottom_extract(arr)  # (1,2,2)
    assert out.shape == (1, 2, 2)
    assert out[0, 0, 0] == 12.0  # deepest valid = depth2
    assert out[0, 0, 1] == 21.0  # depth2 is NaN -> deepest valid = depth1
    assert np.isnan(out[0, 1, 0])  # all-NaN column -> NaN
    assert out[0, 1, 1] == 7.0


def test_fill_ocean_nan_nearest():
    field = np.array([[[1.0, np.nan, 3.0]]])  # (1,1,3), middle ocean cell NaN
    ocean = np.array([[True, True, True]])
    out = bld.fill_ocean_nan(field, ocean)
    assert np.isfinite(out[0, 0, 1])  # filled
    assert out[0, 0, 1] in (1.0, 3.0)  # nearest finite neighbor
    assert out[0, 0, 0] == 1.0 and out[0, 0, 2] == 3.0  # existing values untouched


def test_fill_ocean_nan_leaves_land():
    field = np.array([[[np.nan, 2.0]]])  # (1,1,2)
    ocean = np.array([[False, True]])  # cell 0 is land
    out = bld.fill_ocean_nan(field, ocean)
    assert np.isnan(out[0, 0, 0])  # land NaN untouched


def _month_times(path, months):
    # distinct year per file inferred from the filename's 4-digit year
    import re

    yr = int(re.search(r"(\d{4})", Path(path).name).group(1))
    return [np.datetime64(f"{yr}-{m:02d}-15") for m in months]


def _write_so_file(path, months, depth, lat, lon, fill):
    # so shape (time, depth, lat, lon); `fill(m)` returns a (depth,lat,lon) array
    data = np.stack([fill(m) for m in months], axis=0)
    ds = xr.Dataset(
        {"so": (["time", "depth", "latitude", "longitude"], data)},
        coords={
            "time": _month_times(path, months),
            "depth": depth,
            "latitude": lat,
            "longitude": lon,
        },
    )
    ds.to_netcdf(path)


def test_accumulate_climatology_two_years(tmp_path):
    depth = np.array([0.5, 5.0])  # ascending
    lat = np.array([60.0, 59.0])  # descending
    lon = np.array([15.0, 16.0])
    # year A: bottom (depth1) salinity = 10 in Jan; year B: 20 in Jan -> clim Jan = 15
    fA = tmp_path / "so_2001.nc"
    fB = tmp_path / "so_2002.nc"
    _write_so_file(fA, [1], depth, lat, lon, lambda m: np.full((2, 2, 2), 10.0))
    _write_so_file(fB, [1], depth, lat, lon, lambda m: np.full((2, 2, 2), 20.0))
    clim, slat, slon = bld.accumulate_climatology([str(fA), str(fB)])
    assert clim.shape == (12, 2, 2)
    assert np.allclose(clim[0], 15.0)  # Jan mean across the two years
    assert np.all(np.isnan(clim[1]))  # Feb had no data
    np.testing.assert_array_equal(slat, lat)


def test_artifact_shape_and_orientation():
    p = Path("data/baltic/baltic_salinity_bottom_climatology.nc")
    if not p.exists():
        pytest.skip("run scripts/build_baltic_salinity_forcing.py first")
    with xr.open_dataset(p) as ds:
        s = ds["salinity"].values  # (24, 40, 50)
    assert s.shape == (24, 40, 50)
    # every finite value is a plausible Baltic salinity
    fin = s[np.isfinite(s)]
    assert fin.min() >= 0.0 and fin.max() <= 40.0
    # ORIENTATION: mean salinity of the northern rows (low cell_y) must be LOWER
    # than the southern rows (high cell_y). North Baltic (Bothnian Bay) is ~2-3
    # psu; south-west (Arkona/Kattegat) is ~15-25 psu. If flipped, this fails.
    t0 = s[0]
    north = np.nanmean(t0[:10, :])  # cell_y 0-9 (≈ 66-63 N)
    south = np.nanmean(t0[30:, :])  # cell_y 30-39 (≈ 57-54 N)
    assert south > north, f"orientation wrong: north={north:.2f} !< south={south:.2f}"


def test_gate_loads_real_field_and_grades():
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import _load_salinity_gate
    from osmose.engine.processes.movement import _movement_salinity_weight

    cfg = OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv")
    if "movement.salinity.field.file" not in cfg:
        pytest.skip("config not wired yet")
    cfg["movement.salinity.gate.enabled"] = "true"
    cfg["movement.salinity.gate.species.enabled.sp0"] = "true"
    n_sp = int(float(cfg["simulation.nspecies"]))
    enabled, mask, lo, hi, field = _load_salinity_gate(cfg, n_sp)
    assert enabled and field is not None and not field.is_constant
    assert field.get_grid(0).shape == (40, 50)  # (ny, nx)
    ecfg = SimpleNamespace(
        salinity_gate_enabled=True,
        salinity_field=field,
        salinity_gate_s_low=lo,
        salinity_gate_s_high=hi,
    )
    w = _movement_salinity_weight(ecfg, SimpleNamespace(ny=40, nx=50), 0)
    finite = w[np.isfinite(w)]
    # GRADED: not all-0, not all-1 — a real spread of weights across [0,1]
    assert 0.0 < finite.mean() < 1.0
    assert (finite > 0).any() and (finite < 1).any()


def test_accumulate_climatology_grid_mismatch(tmp_path):
    depth = np.array([0.5, 5.0])  # ascending
    lat_a = np.array([60.0, 59.0])  # nlat=2
    lat_b = np.array([60.0, 59.0, 58.0])  # nlat=3, different
    lon = np.array([15.0, 16.0])
    fA = tmp_path / "so_2001.nc"
    fB = tmp_path / "so_2002.nc"
    # File A with 2 lat points
    _write_so_file(fA, [1], depth, lat_a, lon, lambda m: np.full((2, 2, 2), 10.0))
    # File B with 3 lat points (mismatch)
    _write_so_file(fB, [1], depth, lat_b, lon, lambda m: np.full((2, 3, 2), 20.0))
    # Should raise ValueError on grid mismatch
    with pytest.raises(ValueError, match="grid mismatch"):
        bld.accumulate_climatology([str(fA), str(fB)])


import baltic_salinity_gate_diagnostic as abdiag  # noqa: E402


def test_late_mean_basic():
    series = np.array([100.0, 200.0, 300.0, 400.0])
    # late third = last ~1 element (400) ; helper uses last third by default
    assert abdiag.late_mean(series, frac=0.5) == pytest.approx(350.0)  # mean of last 2
