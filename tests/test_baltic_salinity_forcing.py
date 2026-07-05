import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_baltic_salinity_forcing as bld  # noqa: E402


def test_bottom_extract_deepest_valid():
    # 1 time, 3 depths, 2x2. depth ascending. NaN = below seafloor / land.
    nan = np.nan
    arr = np.array(
        [
            [  # time 0
                [[10.0, 20.0], [nan, 5.0]],  # depth 0 (shallow)
                [[11.0, 21.0], [nan, 6.0]],  # depth 1
                [[12.0, nan], [nan, 7.0]],  # depth 2 (deep)
            ]
        ]
    )  # shape (1,3,2,2)
    out = bld.bottom_extract(arr)  # (1,2,2)
    assert out.shape == (1, 2, 2)
    assert out[0, 0, 0] == 12.0  # deepest valid = depth2
    assert out[0, 0, 1] == 21.0  # depth2 is NaN -> deepest valid = depth1
    assert np.isnan(out[0, 1, 0])  # all-NaN column -> NaN
    assert out[0, 1, 1] == 7.0


def test_fill_ocean_nan_nearest():
    field = np.array([[[1.0, np.nan, 3.0]]])  # (1,1,3), middle ocean cell NaN
    ocean = np.array([[True, True, True]])
    out = bld.fill_ocean_nan(field, ocean)
    assert np.isfinite(out[0, 0, 1])  # filled
    assert out[0, 0, 1] in (1.0, 3.0)  # nearest finite neighbor
    assert out[0, 0, 0] == 1.0 and out[0, 0, 2] == 3.0  # existing values untouched


def test_fill_ocean_nan_leaves_land():
    field = np.array([[[np.nan, 2.0]]])  # (1,1,2)
    ocean = np.array([[False, True]])  # cell 0 is land
    out = bld.fill_ocean_nan(field, ocean)
    assert np.isnan(out[0, 0, 0])  # land NaN untouched
