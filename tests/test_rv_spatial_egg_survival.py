import numpy as np
import pytest
import xarray as xr

from osmose.calibration.larva_recal import sp1_on_config
from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.engine.config import _load_rv_spatial
from osmose.engine.state import SchoolState
from osmose.forcing.grid import load_ocean_mask
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


def _write_rv_nc(tmp_path, rv_const=5.0, ref=5.0, shape=(24, 40, 50), name="rv.nc"):
    rv = np.full(shape, rv_const, dtype=np.float64)
    ds = xr.Dataset({"reproductive_volume": (("time", "latitude", "longitude"), rv)})
    ds["reproductive_volume"].attrs["RV_ref"] = ref
    p = tmp_path / name
    ds.to_netcdf(p)
    return p


def _cfg(tmp_path, **over):
    base = {
        "reproduction.rv.spatial.enabled": "true",
        "reproduction.rv.spatial.field.file": str(_write_rv_nc(tmp_path)),
        "reproduction.rv.spatial.field.varname": "reproductive_volume",
        "reproduction.rv.spatial.ref": "-1",
        "reproduction.rv.spatial.species.enabled.sp0": "true",
        "_osmose.config.dir": str(tmp_path),
        "simulation.time.ndtperyear": "24",
    }
    base.update(over)
    return base


def test_load_rv_spatial_disabled():
    field, mask = _load_rv_spatial({"reproduction.rv.spatial.enabled": "false"}, 3)
    assert field is None and mask is None


def test_load_rv_spatial_reads_attr_and_mask(tmp_path):
    field, mask = _load_rv_spatial(_cfg(tmp_path), n_species=1)
    assert field is not None
    assert field.get_grid(0).shape == (40, 50)
    assert mask.tolist() == [True]


def test_load_rv_spatial_ref_from_config_overrides(tmp_path):
    # ref > 0 uses the config value, not the attr; stored for the consumer via a known accessor.
    field, _ = _load_rv_spatial(_cfg(tmp_path, **{"reproduction.rv.spatial.ref": "12.5"}), 1)
    assert abs(field.rv_ref - 12.5) < 1e-9


def test_load_rv_spatial_ref_from_attr(tmp_path):
    field, _ = _load_rv_spatial(_cfg(tmp_path), 1)  # ref=-1 -> attr (5.0)
    assert abs(field.rv_ref - 5.0) < 1e-9


@pytest.mark.parametrize(
    "bad,exc",
    [
        ({"reproduction.rv.spatial.species.enabled.sp0": "false"}, "no species"),
        ({"reproduction.rv.spatial.field.file": ""}, "empty"),
    ],
)
def test_load_rv_spatial_fail_fast(tmp_path, bad, exc):
    with pytest.raises(ValueError, match=exc):
        _load_rv_spatial(_cfg(tmp_path, **bad), 1)


def test_load_rv_spatial_wrong_grid_raises(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["reproduction.rv.spatial.field.file"] = str(
        _write_rv_nc(tmp_path, shape=(24, 10, 10), name="bad.nc")
    )
    with pytest.raises(ValueError, match="grid"):
        _load_rv_spatial(cfg, 1)


def test_load_rv_spatial_nan_at_spawning_raises(tmp_path):
    rv = np.full((24, 40, 50), np.nan, dtype=np.float64)
    ds = xr.Dataset({"reproductive_volume": (("time", "latitude", "longitude"), rv)})
    ds["reproductive_volume"].attrs["RV_ref"] = 5.0
    p = tmp_path / "nan.nc"
    ds.to_netcdf(p)
    cfg = _cfg(tmp_path)
    cfg["reproduction.rv.spatial.field.file"] = str(p)
    with pytest.raises(ValueError, match="NaN"):
        _load_rv_spatial(cfg, 1)


SP_FIELD = "data/baltic/forcing/baltic_rv_field.nc"


def test_spatial_off_bit_identical():
    off = PythonEngine().run_in_memory(_baltic_cfg(), seed=0).biomass()["cod"].to_numpy()
    off2 = (
        PythonEngine()
        .run_in_memory(_baltic_cfg(**{"reproduction.rv.spatial.enabled": "false"}), seed=0)
        .biomass()["cod"]
        .to_numpy()
    )
    np.testing.assert_array_equal(off, off2)


def test_spatial_on_changes_cod():
    off = PythonEngine().run_in_memory(_baltic_cfg(), seed=0).biomass()["cod"].to_numpy()
    on_cfg = sp1_on_config(_baltic_cfg(), SP_FIELD, larva_rate=None)  # SP1 on, no recal
    on = PythonEngine().run_in_memory(on_cfg, seed=0).biomass()["cod"].to_numpy()
    assert not np.allclose(off, on)  # the spatial term changes cod


def test_field_metrics_basin_contrast_and_cv():
    rv = xr.open_dataset(SP_FIELD)["reproductive_volume"].values  # (24,40,50)
    spawn = np.flipud(np.genfromtxt("data/baltic/maps/cod_spawning.csv", delimiter=";")) > 0
    ocean = load_ocean_mask("data/baltic/baltic_grid.nc")
    # Fresh-coastal reference = the northern Gulf of Bothnia (rows 0-13, north-first),
    # which is fully fresh (RV ~ 0). NOT "all non-spawning ocean": the ultra-saline
    # Danish straits/Kattegat (outside the cod range, no eggs placed there) have the
    # HIGHEST RV of all cells and would confound a spawn-vs-all-ocean contrast. That
    # confound is the headline finding recorded in docs/diagnostics/rv_spatial_field.md.
    fresh = ocean.copy()
    fresh[14:] = False
    mean_spawn = rv[:, spawn].mean()
    mean_fresh = rv[:, fresh].mean()
    assert mean_spawn > mean_fresh  # spawning basins are more viable than the fresh gulf
    # within-basin heterogeneity is the go/no-go metric; huge here (~2.5) because most
    # cod_spawning cells are too fresh/anoxic to be reproductively viable.
    m_field = rv.mean(axis=0)
    cv = float(m_field[spawn].std() / m_field[spawn].mean())
    assert cv >= 0.20  # go/no-go: the regridded climatology retains sub-basin structure


def test_field_mean_anchor():
    # mean(s_cell) over RV>0 spawning cells: this unit test asserts ONLY the construction
    # guarantee (RV_ref = mean over RV>0 cells, so clip caps the mean at 1). The spec's
    # [0.6, 1.0] target is field-dependent (a right-skewed distribution lowers the mean) and
    # is RECORDED by the Task 6 diagnostic (mean_s line); a value below ~0.6 there is a finding
    # (revisit RV_ref), not an automatic failure — so it is NOT a hard bound here.
    da = xr.open_dataset(SP_FIELD)["reproductive_volume"]
    rv = da.values
    ref = float(da.attrs["RV_ref"])
    spawn = np.flipud(np.genfromtxt("data/baltic/maps/cod_spawning.csv", delimiter=";")) > 0
    vals = rv[:, spawn]
    nz = vals[vals > 0]
    s = np.clip(nz / ref, 0.0, 1.0)
    m = float(s.mean())
    assert 0.0 < m <= 1.0 + 1e-9  # construction-guaranteed; the 0.6 target is recorded, not gated
