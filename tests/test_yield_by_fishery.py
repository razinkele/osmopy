"""Yield by fishery (Java FisheryOutputYield): landings and discards by species x fishery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osmose.engine.fisheries import catch_shares

ROOT = Path(__file__).resolve().parents[1]


def test_catch_shares_follow_the_engine_booking():
    """Landed deaths are D(1-frac), frac the rate-weighted discard rate: fishery f's share of
    them is term_f(1-d_f) / sum, of the discards term_f d_f / sum."""

    class M:  # the two fields catch_shares reads
        catchability = np.ones((1, 2))
        discard = np.array([[0.2, 0.5]])

    terms = np.array([[3.0, 1.0], [0.0, 0.0]])
    land, disc = catch_shares(M, np.array([0, 0], dtype=np.int32), terms)  # type: ignore[arg-type]
    np.testing.assert_allclose(land[0], [3 * 0.8, 1 * 0.5] / np.float64(3 * 0.8 + 0.5))
    np.testing.assert_allclose(disc[0], [3 * 0.2, 1 * 0.5] / np.float64(0.6 + 0.5))
    assert (land[1] == 0).all() and (disc[1] == 0).all()  # an unfished school: no share
    # conservation with the engines' booking of a school's deaths D
    d_all = 10.0
    frac = (terms[0] * M.discard[0]).sum() / terms[0].sum()
    assert (d_all * (1 - frac) * land[0]).sum() == pytest.approx(d_all * (1 - frac))


def test_baltic_landings_equal_the_yield_output_and_the_netcdf_has_javas_layout(tmp_path):
    import pandas as pd
    import xarray as xr

    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    cfg = OsmoseConfigReader().read(ROOT / "data/baltic/baltic_all-parameters.csv")
    cfg |= {"simulation.time.nyear": "3", "output.fisheries.yield.biomass.enabled": "true"}
    PythonEngine().run(cfg, tmp_path, seed=5)
    ds = xr.open_dataset(next(tmp_path.rglob("*_yieldByFishery_Simu0.nc")))
    assert ds["landings"].dims == ("time", "species", "fishery")
    assert ds["landings"].attrs["units"] == "ton"
    assert list(ds["fishery"].values)[:2] == ["trawlcod", "pelagicherring"]
    y = pd.read_csv(next(tmp_path.rglob("*_yield_Simu0.csv")), skiprows=1)
    want = y[list(ds["species"].values)].to_numpy()
    got = ds["landings"].sum("fishery").values
    assert want.sum() > 0
    np.testing.assert_allclose(got, want, rtol=1e-12)  # reduced path: all to the one fishery
    # each species' catch sits in exactly its own fishery (Baltic: fishery i catches species i)
    last = ds["landings"].isel(time=-1).values
    assert np.count_nonzero(last - np.diag(np.diag(last))) == 0


def test_the_output_is_absent_unless_switched_on(tmp_path):
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    cfg = OsmoseConfigReader().read(ROOT / "data/baltic/baltic_all-parameters.csv")
    cfg |= {"simulation.time.nyear": "1"}
    PythonEngine().run(cfg, tmp_path, seed=5)
    assert not list(tmp_path.rglob("*_yieldByFishery*"))
