# tests/test_forcing_ltl.py
import numpy as np
import pytest
import xarray as xr

from osmose.forcing.ltl import bgc_to_ltl
from osmose.maps.builder import GridSpec

SMALL = GridSpec(nlon=10, nlat=8, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)
GROUPS = [
    "Diatoms",
    "Dinoflagellates",
    "Microzooplankton",
    "Mesozooplankton",
    "Macrozooplankton",
    "Benthos",
]


def _src(vars_):
    lat = np.linspace(66, 54, 4)
    lon = np.linspace(10, 30, 5)
    nt = 12
    data = {
        k: (["time", "latitude", "longitude"], np.abs(np.ones((nt, 4, 5)) * v))
        for k, v in vars_.items()
    }
    return xr.Dataset(data, coords={"time": np.arange(nt), "latitude": lat, "longitude": lon})


def test_mode_a_direct_biomass():
    ds = _src({"phyc": 10.0, "zooc": 5.0, "chl": 2.0, "nppv": 100.0})
    out = bgc_to_ltl(ds, SMALL)
    assert set(GROUPS).issubset(out.data_vars)
    for g in GROUPS:
        assert out[g].shape == (24, 8, 10)
        vals = out[g].values
        assert np.all(np.nan_to_num(vals) >= 0)
        assert np.all(np.isfinite(np.nan_to_num(vals)))
    assert "direct" in out.attrs["mode"].lower()


def test_mode_b_chl_derived():
    ds = _src({"chl": 2.0, "nppv": 100.0})
    out = bgc_to_ltl(ds, SMALL)
    assert set(GROUPS).issubset(out.data_vars)
    assert "chl" in out.attrs["mode"].lower() or "b" in out.attrs["mode"].lower()


def test_missing_all_inputs_raises():
    ds = _src({"o2": 200.0})
    with pytest.raises(ValueError, match="phyc|chl"):
        bgc_to_ltl(ds, SMALL)


def test_land_mask_applied():
    ds = _src({"phyc": 10.0, "zooc": 5.0})
    mask = np.ones((8, 10), dtype=bool)
    mask[0, 0] = False
    out = bgc_to_ltl(ds, SMALL, ocean_mask=mask)
    assert np.isnan(out["Diatoms"].values[0, 0, 0])


@pytest.mark.parametrize(
    "vars_", [{"phyc": 10.0, "zooc": 5.0, "nppv": 100.0}, {"chl": 2.0, "nppv": 100.0}]
)
def test_ocean_mask_fills_wet_cells_whose_nearest_source_pixel_is_land(vars_):
    """A coastal model cell whose nearest native pixel is land (NaN) must not come out as an
    all-zero ocean cell: with `ocean_mask`, it takes the nearest valid pixel instead. This is
    the S3 regrid hole (65 of 616 Baltic ocean cells were all-zero). Without `ocean_mask` the
    old behaviour stands. Covers both modes, so every regrid call carries the fallback."""
    ds = _src(vars_)
    for name in vars_:
        ds[name][:, 0, 0] = np.nan  # the NW native pixel is land
    wet = np.ones((8, 10), dtype=bool)
    masked = bgc_to_ltl(ds, SMALL, ocean_mask=wet)
    plain = bgc_to_ltl(ds, SMALL)
    for g in GROUPS:
        assert (masked[g].values[:, 0, 0] > 0).all(), g
        assert (plain[g].values[:, 0, 0] == 0).all(), g


def test_depth_slice_empty_raises():
    # all source depth levels are deeper than depth_integrate_m -> empty slice -> raise
    # (not a silent all-zero forcing). Also covers a descending depth axis via sortby.
    lat = np.linspace(66, 54, 4)
    lon = np.linspace(10, 30, 5)
    ds = xr.Dataset(
        {
            "phyc": (["time", "depth", "latitude", "longitude"], np.ones((12, 1, 4, 5)) * 10.0),
            "zooc": (["time", "depth", "latitude", "longitude"], np.ones((12, 1, 4, 5)) * 5.0),
        },
        coords={"time": np.arange(12), "depth": [100.0], "latitude": lat, "longitude": lon},
    )
    with pytest.raises(ValueError, match="depth"):
        bgc_to_ltl(ds, SMALL, depth_integrate_m=50.0)
