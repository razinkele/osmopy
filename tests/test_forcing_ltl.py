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


def _profiled(profile, shallow_nw: bool = False):
    """Mode-A source with 4 depth levels (1, 5, 20, 40 m) whose 0-50 m layer thicknesses are
    exactly 3, 9.5, 17.5 and 20 m (edges at level midpoints, the last capped at 50 m). Every
    pixel carries `profile` (phyc = zooc). With shallow_nw, the NW pixel's water ends after the
    5 m level (NaN below), the way a shallow coastal pixel looks in the CMEMS product."""
    lat = np.linspace(66, 54, 4)
    lon = np.linspace(10, 30, 5)
    arr = np.broadcast_to(np.asarray(profile, float)[None, :, None, None], (12, 4, 4, 5)).copy()
    if shallow_nw:
        arr[:, 2:, 0, 0] = np.nan
    dims = ["time", "depth", "latitude", "longitude"]
    return xr.Dataset(
        {"phyc": (dims, arr), "zooc": (dims, arr.copy())},
        coords={
            "time": np.arange(12),
            "depth": [1.0, 5.0, 20.0, 40.0],
            "latitude": lat,
            "longitude": lon,
        },
    )


def test_column_integral_counts_only_the_water_a_shallow_pixel_has():
    """Issue #229: the level mean x a full 50 m column credited a shallow pixel with water it does
    not have. A pixel whose water ends after 5 m holds (3 + 9.5) / 50 = 0.25 of a deep pixel's
    biomass at the same concentration; the old integration gave it 1.0."""
    out = bgc_to_ltl(_profiled([1.0, 1.0, 1.0, 1.0], shallow_nw=True), SMALL)
    for g in ("Diatoms", "Mesozooplankton"):
        vals = out[g].values[0]
        assert vals[0, 0] / vals[-1, -1] == pytest.approx(0.25), g


def test_column_integral_weights_levels_by_their_thickness():
    """The levels crowd near the surface, so an unweighted mean over-weights surface plankton.
    For the profile 4, 2, 1, 1 the thickness-weighted 0-50 m mean is (12 + 19 + 17.5 + 20) / 50
    = 1.37, not the level mean 2.0. A uniform profile is unchanged by the fix."""
    surf = bgc_to_ltl(_profiled([4.0, 2.0, 1.0, 1.0]), SMALL)
    unif = bgc_to_ltl(_profiled([1.0, 1.0, 1.0, 1.0]), SMALL)
    for g in ("Diatoms", "Mesozooplankton"):
        assert surf[g].values[0, 3, 4] / unif[g].values[0, 3, 4] == pytest.approx(1.37), g


def test_column_integral_keeps_land_pixels_invalid_for_the_masked_fallback():
    """A pixel with no valid level at all must stay NaN after integration, or the masked
    nearest-valid fallback would treat land as a valid zero and the coastal holes would return."""
    ds = _profiled([1.0, 1.0, 1.0, 1.0])
    ds["phyc"][:, :, 0, 0] = np.nan
    ds["zooc"][:, :, 0, 0] = np.nan
    wet = np.ones((8, 10), dtype=bool)
    out = bgc_to_ltl(ds, SMALL, ocean_mask=wet)
    assert (out["Diatoms"].values[:, 0, 0] > 0).all()
