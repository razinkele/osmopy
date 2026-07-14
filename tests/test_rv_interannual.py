from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from osmose.forcing.reproductive_volume import build_rv_field, build_rv_field_interannual
from osmose.maps.builder import GridSpec

# Repo root, resolved from this test file's location — do NOT hardcode an absolute path
# (CI checks out under a different prefix, e.g. /home/runner/work/...).
REPO_ROOT = Path(__file__).resolve().parents[1]

# Real GridSpec (dx/dy are @property, computed from the corners — a plain stub lacks them and
# regrid()/target_coords() would AttributeError).
GRID = GridSpec(
    nlon=5, nlat=4, upleft_lat=65.5, upleft_lon=10.5, lowright_lat=54.5, lowright_lon=29.5
)


def _fake_year(seed):
    # minimal (time=12, depth=3, lat=4, lon=5) so/o2 datasets on a source grid
    rng = np.random.default_rng(seed)
    depth = np.array([5.0, 20.0, 40.0])
    lat = np.linspace(54.5, 65.5, 4)
    lon = np.linspace(10.5, 29.5, 5)
    so = 6.0 + 8.0 * rng.random((12, 3, 4, 5))  # spans the 11-psu threshold
    o2 = 50.0 + 100.0 * rng.random((12, 3, 4, 5))  # spans the 89.3 threshold
    coords = {"time": np.arange(12), "depth": depth, "latitude": lat, "longitude": lon}
    phy = xr.Dataset({"so": (("time", "depth", "latitude", "longitude"), so)}, coords=coords)
    bgc = xr.Dataset({"o2": (("time", "depth", "latitude", "longitude"), o2)}, coords=coords)
    return phy, bgc


def _masks():
    ocean = np.ones((4, 5), dtype=bool)
    spawning = np.zeros((4, 5), dtype=bool)
    spawning[1:3, 1:4] = True
    return ocean, spawning


def test_interannual_shape_and_chronological():
    ph = [_fake_year(i) for i in range(3)]
    phy_years = [p for p, _ in ph]
    bgc_years = [b for _, b in ph]
    ocean, spawning = _masks()
    ds = build_rv_field_interannual(
        phy_years, bgc_years, GRID, ocean_mask=ocean, spawning_mask=spawning, start_year=1993
    )
    rv = ds["reproductive_volume"].values
    assert rv.shape == (3 * 24, 4, 5)  # concatenate, not average
    assert ds["reproductive_volume"].attrs["start_year"] == 1993
    # Year k's 24-step block equals that single year's standalone climatology-of-one build.
    for k in range(3):
        one = build_rv_field(
            [phy_years[k]], [bgc_years[k]], GRID, ocean_mask=ocean, spawning_mask=spawning
        )["reproductive_volume"].values
        block = rv[k * 24 : (k + 1) * 24]
        np.testing.assert_allclose(np.nan_to_num(block), np.nan_to_num(one), rtol=1e-6)


def test_interannual_differs_from_climatology_mean():
    ph = [_fake_year(i) for i in range(3)]
    phy_years, bgc_years = [p for p, _ in ph], [b for _, b in ph]
    ocean, spawning = _masks()
    inter = build_rv_field_interannual(
        phy_years, bgc_years, GRID, ocean_mask=ocean, spawning_mask=spawning, start_year=1993
    )["reproductive_volume"].values
    clim = build_rv_field(phy_years, bgc_years, GRID, ocean_mask=ocean, spawning_mask=spawning)[
        "reproductive_volume"
    ].values
    # interannual is 72 steps; its per-year blocks are NOT all equal to the 24-step climatology
    assert inter.shape[0] == 72 and clim.shape[0] == 24
    assert not np.allclose(np.nan_to_num(inter[:24]), np.nan_to_num(inter[24:48]))


def test_climatology_builder_deterministic():
    # build_rv_field is not edited by this task; same inputs -> identical output (sanity).
    phy, bgc = _fake_year(7)
    ocean, spawning = _masks()
    a = build_rv_field([phy], [bgc], GRID, ocean_mask=ocean, spawning_mask=spawning)
    b = build_rv_field([phy], [bgc], GRID, ocean_mask=ocean, spawning_mask=spawning)
    np.testing.assert_array_equal(
        np.nan_to_num(a["reproductive_volume"].values),
        np.nan_to_num(b["reproductive_volume"].values),
    )


def _write_rv_nc(path, n_steps):
    lat = np.linspace(54.5, 65.5, 40)
    lon = np.linspace(10.5, 29.5, 50)
    rv = np.ones((n_steps, 40, 50), dtype=np.float32)
    ds = xr.Dataset(
        {"reproductive_volume": (("time", "latitude", "longitude"), rv)},
        coords={"time": np.arange(n_steps), "latitude": lat, "longitude": lon},
    )
    ds["reproductive_volume"].attrs["RV_ref"] = 10.0
    ds.to_netcdf(path)


def _rv_cfg(tmp_path, n_steps, nyear):
    from osmose.engine.config import _load_rv_spatial  # noqa: F401 (import path check)

    p = tmp_path / "rv.nc"
    _write_rv_nc(p, n_steps)
    return {
        "reproduction.rv.spatial.enabled": "true",
        "reproduction.rv.spatial.field.file": str(p),
        "reproduction.rv.spatial.species.enabled.sp0": "true",
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": str(nyear),
        "_osmose.config.dir": str(tmp_path),
    }


def test_wrap_guard_raises_when_run_exceeds_forcing(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # cod_spawning.csv fallback path
    from osmose.engine.config import _load_rv_spatial

    cfg = _rv_cfg(tmp_path, n_steps=696, nyear=30)  # 30*24=720 > 696 -> would wrap
    with pytest.raises(ValueError, match="wrap|exceed|forcing"):
        _load_rv_spatial(cfg, n_species=1)


def test_wrap_guard_ok_when_exact_and_climatology(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    from osmose.engine.config import _load_rv_spatial

    field, en = _load_rv_spatial(_rv_cfg(tmp_path, n_steps=696, nyear=29), 1)  # 696==696 ok
    assert field is not None and en[0]
    field2, _ = _load_rv_spatial(_rv_cfg(tmp_path, n_steps=24, nyear=50), 1)  # climatology cycles
    assert field2 is not None


def test_lagged_correlations_recovers_known_lag():
    from scripts.baltic_rv_cod_offline import lagged_correlations

    rng = np.random.default_rng(0)
    rv = rng.random(29)
    cod = np.empty(29)
    cod[2:] = rv[:-2]  # cod lags rv by 2 yr
    cod[:2] = rng.random(2)
    lc = lagged_correlations(rv, cod, max_lag=4)  # dict lag->corr
    best = max(lc, key=lambda k: lc[k])
    assert best == 2 and lc[2] > 0.9


def test_arm_overrides_shared_ref_and_files():
    from scripts.baltic_rv_hindcast import arm_overrides

    off = arm_overrides("off", rv_ref=12.0, inter_path="i.nc", clim_path="c.nc")
    clim = arm_overrides("clim", rv_ref=12.0, inter_path="i.nc", clim_path="c.nc")
    inter = arm_overrides("inter", rv_ref=12.0, inter_path="i.nc", clim_path="c.nc")
    assert off.get("reproduction.rv.spatial.enabled", "false") == "false"
    # both enabled arms share the forced RV_ref, differ only in the field file
    assert clim["reproduction.rv.spatial.ref"] == "12.0"
    assert inter["reproduction.rv.spatial.ref"] == "12.0"
    assert clim["reproduction.rv.spatial.field.file"].endswith("c.nc")
    assert inter["reproduction.rv.spatial.field.file"].endswith("i.nc")
    assert clim["reproduction.rv.spatial.species.enabled.sp0"] == "true"
    # SSB enabled on ALL arms (incl. off) so .ssb() works uniformly
    assert off["output.ssb.enabled"] == "true" and inter["output.ssb.enabled"] == "true"


def test_skill_delta_positive_when_b_tracks_observed():
    from scripts.baltic_rv_hindcast import skill_delta

    obs = np.array([1.0, 2, 3, 2, 1, 2, 3], float)
    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98], float)  # nonzero var, ~uncorrelated
    b = obs * 0.5 + 0.1  # tracks observed
    assert skill_delta(a, b, obs) > 0.5


def test_skill_delta_nan_safe_on_collapsed_arm():
    from scripts.baltic_rv_hindcast import skill_delta

    obs = np.array([1.0, 2, 3, 2, 1, 2, 3], float)
    flat = np.ones(7)  # a collapsed arm (zero variance) -> nan, NOT a crash / not 0
    assert np.isnan(skill_delta(flat, obs, obs))
