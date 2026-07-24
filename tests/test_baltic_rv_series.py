"""The eastern-Baltic-cod reproductive-volume series drives the RV recruitment gate."""

from pathlib import Path

import numpy as np
import pandas as pd

RV = Path("data/baltic/reference/baltic_cod_reproductive_volume.csv")


def test_baltic_rv_series_loads_and_is_valid():
    assert RV.exists(), "RV series file missing"
    df = pd.read_csv(RV)
    assert list(df.columns[:2]) == ["year", "spawning_rv"]
    years = df["year"].to_numpy()
    assert np.array_equal(years, np.arange(years[0], years[0] + len(years)))  # contiguous ascending
    rv = df["spawning_rv"].to_numpy(dtype=float)
    assert np.all(np.isfinite(rv)) and np.all(rv >= 0)
    # documents the historical decline: high/variable early, low late (post-2003 collapse)
    assert rv[:5].mean() > 2.0 * rv[-5:].mean()
