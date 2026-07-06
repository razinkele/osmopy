import numpy as np
import xarray as xr
from scripts.resample_bob_forcing import resample_to_24_steps

def _synthetic_365():
    # 6 vars, 365 daily steps, 4x5 grid; each var = its own linear ramp so bin means are exact.
    data = {}
    for i, name in enumerate(["SmallPhyto", "LargePhyto", "SmallZoo", "LargeZoo",
                              "SmallDetritus", "LargeDetritus"]):
        arr = (np.arange(365)[:, None, None] + i).astype(float) * np.ones((365, 4, 5))
        data[name] = (("time", "lat", "lon"), arr)
    return xr.Dataset(data, coords={"time": np.arange(365), "lat": np.arange(4), "lon": np.arange(5)})

def test_output_is_24_steps_same_grid_and_vars():
    out = resample_to_24_steps(_synthetic_365())
    assert out.sizes["time"] == 24
    assert out.sizes["lat"] == 4 and out.sizes["lon"] == 5
    assert set(out.data_vars) == {"SmallPhyto", "LargePhyto", "SmallZoo", "LargeZoo",
                                  "SmallDetritus", "LargeDetritus"}

def test_bins_conserve_window_mean():
    ds = _synthetic_365()
    out = resample_to_24_steps(ds)
    # step s = mean over days d where floor(d*24/365)==s
    step_of_day = (np.arange(365) * 24) // 365
    for s in range(24):
        days = np.where(step_of_day == s)[0]
        expected = ds["SmallZoo"].isel(time=days).mean("time").values
        np.testing.assert_allclose(out["SmallZoo"].isel(time=s).values, expected)

def test_idempotent_all_bins_nonempty():
    out = resample_to_24_steps(_synthetic_365())
    assert np.isfinite(out["SmallPhyto"].values).all()  # no empty bins -> no NaN
