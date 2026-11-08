"""Every Baltic ocean cell carries plankton and benthos forcing.

The 2026-04 mask rebuild (912 -> 616 cells, d61ea1c1) added coastal cells without regenerating
data/baltic/baltic_ltl_biomass.nc, so 65 of them were exactly 0 in all six groups and all 24
frames (review S3, docs/reviews/2026-11-07-temporal-spatial-dynamics-review.md). Fish mapped
there found no plankton. This guards the file against the next mask change.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from osmose.forcing import load_ocean_mask

LTL = Path("data/baltic/baltic_ltl_biomass.nc")
GRID = Path("data/baltic/baltic_grid.nc")


def _stack() -> np.ndarray:
    with xr.open_dataset(LTL) as ds:
        return np.stack([ds[g].values for g in ds.data_vars])  # (group, time, lat, lon)


def test_every_ocean_cell_has_ltl_forcing():
    ocean = load_ocean_mask(GRID)
    assert ocean is not None
    arr = _stack()
    empty = ocean & (arr == 0).all(axis=(0, 1))
    assert not empty.any(), f"{int(empty.sum())} ocean cells have no LTL in any group or frame"


def test_ltl_land_is_zero_and_file_has_no_nan():
    """Land is encoded 0.0, not NaN (the convention the oxygen forcing matches)."""
    ocean = load_ocean_mask(GRID)
    arr = _stack()
    assert not np.isnan(arr).any()
    assert (arr[..., ~ocean] == 0).all()
