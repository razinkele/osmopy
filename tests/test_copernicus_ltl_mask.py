"""Tests for the Baltic LTL generator land-mask application.

Covers the post-processing step that stamps land cells with NaN so the
OSMOSE UI overlay renders only ocean cells. Does not exercise the CMEMS
download path or credential-guarded entry points.
"""

import numpy as np
import pytest


def test_apply_land_mask_sets_land_cells_to_nan():
    from mcp_servers.copernicus.server import _apply_land_mask

    groups = {
        "Diatoms": np.ones((4, 3, 3), dtype=np.float64),
        "Benthos": np.full((4, 3, 3), 42.0),
    }
    # Middle row is ocean, outer rows are land
    ocean = np.array(
        [
            [False, False, False],
            [True, True, True],
            [False, False, False],
        ]
    )
    _apply_land_mask(groups, ocean)

    for arr in groups.values():
        # Land rows are NaN across every time step
        assert np.all(np.isnan(arr[:, 0, :]))
        assert np.all(np.isnan(arr[:, 2, :]))
        # Ocean row retained its original value
        assert not np.any(np.isnan(arr[:, 1, :]))


def test_apply_land_mask_shape_mismatch_is_noop():
    """A wrong-shape mask must leave the arrays untouched rather than crash."""
    from mcp_servers.copernicus.server import _apply_land_mask

    original = np.ones((4, 3, 3), dtype=np.float64)
    groups = {"x": original.copy()}
    bad_mask = np.ones((5, 5), dtype=bool)

    _apply_land_mask(groups, bad_mask)
    assert np.array_equal(groups["x"], original)


def test_load_baltic_ocean_mask_returns_expected_shape():
    """If the committed grid.nc is present, the mask should load as (40, 50)."""
    from mcp_servers.copernicus.server import _load_baltic_ocean_mask

    mask = _load_baltic_ocean_mask()
    if mask is None:
        pytest.skip("baltic_grid.nc not available in this checkout")
    assert mask.shape == (40, 50)
    assert mask.dtype == bool
    # The Baltic grid has 616 ocean cells (confirmed by existing grid fixtures)
    assert mask.sum() == 616
