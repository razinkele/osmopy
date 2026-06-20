# tests/test_forcing_io.py
import numpy as np
import pytest
import xarray as xr

from osmose.forcing import bgc_to_ltl, phy_to_physics, write_ltl, write_physics
from osmose.maps.builder import GridSpec

SMALL = GridSpec(nlon=10, nlat=8, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)


def _bgc():
    lat = np.linspace(66, 54, 4)
    lon = np.linspace(10, 30, 5)
    return xr.Dataset(
        {
            "phyc": (["time", "latitude", "longitude"], np.ones((12, 4, 5)) * 10.0),
            "zooc": (["time", "latitude", "longitude"], np.ones((12, 4, 5)) * 5.0),
        },
        coords={"time": np.arange(12), "latitude": lat, "longitude": lon},
    )


def test_write_ltl_roundtrip(tmp_path):
    ds = bgc_to_ltl(_bgc(), SMALL)
    path = write_ltl(ds, tmp_path / "ltl.nc")
    assert path.exists()
    reopened = xr.open_dataset(path)
    assert "Diatoms" in reopened.data_vars
    assert reopened["Diatoms"].shape == (24, 8, 10)
    assert float(reopened.latitude[0]) > float(reopened.latitude[-1])  # descending
    reopened.close()


def test_write_ltl_refuses_clobber(tmp_path):
    ds = bgc_to_ltl(_bgc(), SMALL)
    path = tmp_path / "ltl.nc"
    write_ltl(ds, path)
    with pytest.raises(FileExistsError):
        write_ltl(ds, path)  # default overwrite=False
    write_ltl(ds, path, overwrite=True)  # explicit overwrite OK


def test_write_physics_roundtrip(tmp_path):
    src = xr.Dataset(
        {"thetao": (["time", "latitude", "longitude"], np.ones((12, 4, 5)) * 8.0)},
        coords={
            "time": np.arange(12),
            "latitude": np.linspace(66, 54, 4),
            "longitude": np.linspace(10, 30, 5),
        },
    )
    dsets = phy_to_physics(src, SMALL)
    paths = write_physics(dsets, tmp_path, prefix="test")
    assert (tmp_path / "test_temperature.nc").exists()
    # write_physics returns a {name: path} mapping
    assert paths["temperature"].name == "test_temperature.nc"
