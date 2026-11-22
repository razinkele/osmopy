"""Bin edges / colours / legend for the Spatial Results map."""

import numpy as np

from ui.pages.spatial_colors import PALETTES, bin_colors, bin_edges, legend_entries


def test_quantile_spreads_a_skewed_field_linear_does_not():
    vals = np.r_[np.arange(1.0, 100.0), 1e6]  # 99 ordinary cells and one hotspot
    q_idx, _, _ = bin_colors(vals, scale="quantile")
    l_idx, _, _ = bin_colors(vals, scale="linear")
    assert len(set(q_idx[:-1])) >= 6  # ordinary cells use most of the palette
    assert set(l_idx[:-1]) == {0}  # equal-width: the hotspot flattens everything else
    assert q_idx[-1] == l_idx[-1] == max(q_idx)  # hotspot in the top bin either way


def test_edges_monotone_and_ties_collapse():
    e = bin_edges(np.array([5.0] * 50 + [6.0, 7.0]), "quantile")
    assert np.all(np.diff(e) > 0)
    assert bin_edges(np.array([3.0, 3.0]), "linear").tolist() == [3.0, 4.0]
    assert bin_edges(np.array([np.nan]), "quantile").tolist() == [0.0, 1.0]


def test_every_value_lands_in_a_bin_including_max_and_nan_is_minus_one():
    vals = np.array([0.1, 0.5, 2.0, 9.0, np.nan])
    idx, colors, edges = bin_colors(vals, scale="linear", n_bins=4)
    assert idx[-1] == -1
    assert idx[3] == len(colors) - 1  # last edge inclusive
    assert len(colors) == len(edges) - 1
    assert all(len(c) == 4 for c in colors)


def test_palettes_differ_and_legend_labels_ranges():
    vals = np.linspace(0, 1, 50)
    _, c_v, e = bin_colors(vals, palette="viridis")
    _, c_o, _ = bin_colors(vals, palette="ocean")
    assert c_v != c_o
    rows = legend_entries("spatial-data", c_v, e)
    assert len(rows) == len(c_v)
    assert rows[0]["label"].startswith("0 ")
    assert set(PALETTES) == {"viridis", "ocean", "chlorophyll", "thermal"}
