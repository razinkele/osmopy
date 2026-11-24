"""Fishing maps are normalised as Java's FisheryMapSet.normalize_map does (G5, 2026-11-24).

Mean over VALID ocean cells (not land, value > 0, not NaN -- Java's isValueOk) rescaled to 1.
Zero cells are excluded from the mean, so a 0/1 mask is unchanged: every shipped config fishes
exactly as before. Maps are normalised from the RAW array every time (no drift across runs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.engine.config import EngineConfig
from osmose.engine.fisheries import attach_ocean_mask, fishing_map, normalize_fishing_map

ROOT = Path(__file__).resolve().parent.parent


def test_mean_over_positive_ocean_cells_becomes_one():
    m = np.array([[2.0, 4.0, 0.0], [np.nan, -1.0, 6.0]])
    ocean = np.array([[True, True, True], [True, True, False]])  # the 6.0 sits on land
    out = normalize_fishing_map(m, ocean)
    assert out[0, 0] == pytest.approx(2 / 3) and out[0, 1] == pytest.approx(4 / 3)  # k = 3
    assert out[0, 2] == 0.0 and out[1, 1] == -1.0 and np.isnan(out[1, 0])  # meaning kept
    assert out[1, 2] == 6.0  # land cell: neither counted nor rescaled
    assert np.isclose(out[ocean & (m > 0)].mean(), 1.0)


def test_a_binary_mask_is_returned_untouched():
    m = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert normalize_fishing_map(m, np.ones((2, 2), dtype=bool)) is m  # k == 1: same array


def test_shape_mismatch_or_no_valid_cell_is_unchanged():
    m = np.array([[2.0, 3.0]])
    assert normalize_fishing_map(m, np.ones((2, 2), dtype=bool)) is m
    z = np.zeros((2, 2))
    assert normalize_fishing_map(z, np.ones((2, 2), dtype=bool)) is z


@pytest.mark.parametrize("demo", ["baltic", "eec_full"])
def test_shipped_maps_are_bit_identical_after_normalisation(demo):
    cfg = OsmoseConfigReader().read(next((ROOT / "data" / demo).glob("*all-parameters*.csv")))
    ec = EngineConfig.from_dict(cfg)
    grid = PythonEngine()._resolve_grid(cfg)
    attach_ocean_mask(ec, grid)
    raw = [m for m in ec.fishing_spatial_maps if m is not None]
    assert raw, "the demo has fishing maps (the check is not vacuous)"
    for m in raw:
        assert fishing_map(ec, m) is m  # mean over positive ocean cells is exactly 1


def test_normalised_once_from_raw_no_drift_across_calls():
    class C:
        pass

    c = C()
    c.fishing_ocean_mask = np.ones((1, 2), dtype=bool)
    m = np.array([[1.0, 3.0]])
    a = fishing_map(c, m)
    b = fishing_map(c, m)
    assert a is b and m[0, 1] == 3.0  # cached, raw untouched
    assert a[0, 0] == pytest.approx(0.5) and a[0, 1] == pytest.approx(1.5)


def test_without_an_ocean_mask_the_raw_map_is_used():
    class C:
        pass

    m = np.array([[1.0, 3.0]])
    assert fishing_map(C(), m) is m
