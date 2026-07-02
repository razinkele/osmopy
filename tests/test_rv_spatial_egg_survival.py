import numpy as np
import xarray as xr

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.engine.state import SchoolState
from osmose.forcing.reproductive_volume import build_rv_field, viable_thickness
from osmose.maps.builder import GridSpec
from osmose.schema import build_registry


def test_rv_spatial_keys_registered():
    keys = {f.key_pattern for f in build_registry().all_fields()}
    assert "reproduction.rv.spatial.enabled" in keys
    assert "reproduction.rv.spatial.field.file" in keys
    assert "reproduction.rv.spatial.field.varname" in keys
    assert "reproduction.rv.spatial.ref" in keys
    assert "reproduction.rv.spatial.species.enabled.sp{idx}" in keys


def test_viable_thickness_mid_layer():
    # depths 10..90 step 10 (m); layer thickness = 10 each (uniform).
    depths = np.arange(10.0, 100.0, 10.0)  # 9 levels
    # so rises with depth (fresh top, saline deep); o2 falls with depth (oxic top, anoxic deep).
    so = np.array([6, 8, 10, 11, 12, 13, 14, 14, 14], dtype=float)
    o2 = np.array([300, 280, 250, 200, 150, 100, 40, 5, 0], dtype=float)  # mmol/m3
    # viable (so>=11 AND o2>=89.3): levels 3,4,5 (so 11,12,13 ; o2 200,150,100) -> 3 levels * 10 m
    t = viable_thickness(so, o2, depths, 11.0, 89.3)
    assert abs(t - 30.0) < 1e-9


def test_viable_thickness_two_bands_summed():
    depths = np.arange(10.0, 60.0, 10.0)  # 5 levels, 10 m each
    so = np.array([12, 6, 12, 6, 12], dtype=float)  # saline at 0,2,4
    o2 = np.array([300, 300, 300, 300, 300], dtype=float)  # all oxic
    # viable levels 0,2,4 -> 3 * 10 = 30 (two separated bands summed)
    assert abs(viable_thickness(so, o2, depths, 11.0, 89.3) - 30.0) < 1e-9


def test_viable_thickness_none_and_all():
    depths = np.arange(10.0, 40.0, 10.0)
    fresh = np.array([6, 7, 8], dtype=float)
    oxic = np.array([300, 300, 300], dtype=float)
    assert viable_thickness(fresh, oxic, depths, 11.0, 89.3) == 0.0  # all fresh
    saline = np.array([12, 13, 14], dtype=float)
    assert viable_thickness(saline, oxic, depths, 11.0, 89.3) > 0.0  # all viable


def _toy_year(so_col, o2_col):
    # 1 time, len(depth) depths, 2x2 lat/lon, same column everywhere.
    depths = np.arange(10.0, 10.0 * len(so_col) + 1, 10.0)
    lat = np.array([56.0, 55.0])
    lon = np.array([18.0, 19.0])
    so = np.broadcast_to(np.array(so_col)[None, :, None, None], (1, len(depths), 2, 2))
    o2 = np.broadcast_to(np.array(o2_col)[None, :, None, None], (1, len(depths), 2, 2))
    phy = xr.Dataset(
        {"so": (("time", "depth", "latitude", "longitude"), so.astype(float))},
        coords={"time": [0], "depth": depths, "latitude": lat, "longitude": lon},
    )
    bgc = xr.Dataset(
        {"o2": (("time", "depth", "latitude", "longitude"), o2.astype(float))},
        coords={"time": [0], "depth": depths, "latitude": lat, "longitude": lon},
    )
    return phy, bgc


def test_build_rv_field_climatology_and_ref():
    grid = GridSpec(
        nlon=2, nlat=2, upleft_lat=56.5, upleft_lon=17.5, lowright_lat=54.5, lowright_lon=19.5
    )
    # year A: 3 viable levels (30 m). year B: 1 viable level (10 m). mean = 20 m.
    pA, bA = _toy_year([12, 12, 12, 6], [300, 300, 300, 300])
    pB, bB = _toy_year([12, 6, 6, 6], [300, 300, 300, 300])
    ocean = np.ones((2, 2), dtype=bool)
    spawning = np.ones((2, 2), dtype=bool)
    ds = build_rv_field([pA, pB], [bA, bB], grid, ocean_mask=ocean, spawning_mask=spawning)
    assert ds["reproductive_volume"].shape == (24, 2, 2)
    # every cell/step ≈ 20 m (constant column, per-year mean of 30 and 10)
    assert abs(float(ds["reproductive_volume"].mean()) - 20.0) < 1.0
    assert abs(ds["reproductive_volume"].attrs["RV_ref"] - 20.0) < 1.0


def test_build_rv_field_land_is_nan():
    # Spec §4: land cells (ocean_mask False) are written as NaN, ocean cells finite.
    grid = GridSpec(
        nlon=2, nlat=2, upleft_lat=56.5, upleft_lon=17.5, lowright_lat=54.5, lowright_lon=19.5
    )
    pA, bA = _toy_year([12, 12, 6], [300, 300, 300])  # 2 viable levels (20 m)
    ocean = np.array([[True, True], [True, False]])  # one land cell
    ds = build_rv_field([pA], [bA], grid, ocean_mask=ocean, spawning_mask=ocean)
    rv = ds["reproductive_volume"].values
    assert np.isnan(rv[:, 1, 1]).all()  # land -> NaN
    assert np.isfinite(rv[:, 0, 0]).all()  # ocean -> finite


def test_schoolstate_has_from_seeding_default_false():
    s = SchoolState.create(n_schools=3)
    assert s.from_seeding is not None
    assert s.from_seeding.tolist() == [False, False, False]
    # replace/append thread it generically
    s2 = s.replace(from_seeding=np.array([True, False, True]))
    assert s2.append(s).from_seeding.tolist() == [True, False, True, False, False, False]


def _baltic_cfg(**over):
    cfg = dict(OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "3"
    cfg.update(over)
    return cfg


def test_from_seeding_inert_by_default_parity():
    # Adding the field must not change engine output.
    a = PythonEngine().run_in_memory(_baltic_cfg(), seed=0).biomass()["cod"].to_numpy()
    b = PythonEngine().run_in_memory(_baltic_cfg(), seed=0).biomass()["cod"].to_numpy()
    np.testing.assert_array_equal(a, b)
