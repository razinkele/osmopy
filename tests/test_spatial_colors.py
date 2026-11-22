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


def test_series_edges_span_the_whole_run_so_frames_differ():
    from ui.pages.spatial_colors import series_edges

    frames = [np.full((3, 3), 1.0 + t) for t in range(10)]  # field grows over time
    edges = series_edges((lambda t: frames[t], len(frames)), "linear")
    first, _, _ = bin_colors(frames[0].ravel(), scale="linear", edges=edges)
    last, _, _ = bin_colors(frames[-1].ravel(), scale="linear", edges=edges)
    assert first.max() < last.min()  # early frame in low bins, late frame in high bins
    # per-frame edges would have put both frames in the same (single) bin
    assert (
        bin_colors(frames[0].ravel(), scale="linear")[0].tolist()
        == bin_colors(frames[-1].ravel(), scale="linear")[0].tolist()
    )


def test_series_edges_subsample_and_clamp():
    from ui.pages.spatial_colors import series_edges

    calls = []

    def fn(t):
        calls.append(t)
        return np.array([[float(t + 1)]])

    edges = series_edges((fn, 1000), "linear", max_steps=50)
    assert len(calls) <= 50 and calls[0] == 0 and calls[-1] == 999
    idx, colors, _ = bin_colors(np.array([-5.0, 1e9]), scale="linear", edges=edges)
    assert idx.tolist() == [0, len(colors) - 1]  # out-of-range values clamp to end bins
