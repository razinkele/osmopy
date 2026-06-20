# tests/test_forcing_physics.py
import numpy as np
import xarray as xr

from osmose.forcing.physics import phy_to_physics
from osmose.maps.builder import GridSpec

SMALL = GridSpec(nlon=10, nlat=8, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)


def _src(vars_):
    lat = np.linspace(66, 54, 4)
    lon = np.linspace(10, 30, 5)
    data = {
        k: (["time", "latitude", "longitude"], np.ones((12, 4, 5)) * v) for k, v in vars_.items()
    }
    return xr.Dataset(data, coords={"time": np.arange(12), "latitude": lat, "longitude": lon})


def test_both_vars_present():
    out = phy_to_physics(_src({"thetao": 8.0, "so": 7.5}), SMALL)
    assert set(out) == {"temperature", "salinity"}
    assert out["temperature"]["temperature"].shape == (24, 8, 10)
    assert np.allclose(out["temperature"]["temperature"].values, 8.0)
    assert np.allclose(out["salinity"]["salinity"].values, 7.5)


def test_missing_salinity_omitted():
    out = phy_to_physics(_src({"thetao": 8.0}), SMALL)
    assert set(out) == {"temperature"}


def test_missing_both_returns_empty():
    out = phy_to_physics(_src({"o2": 200.0}), SMALL)
    assert out == {}
