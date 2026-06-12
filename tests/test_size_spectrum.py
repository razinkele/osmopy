from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from osmose.size_spectrum import (
    SizeSpectrum,
    _community_long,
    _infer_bin_width,
    _large_fish_indicator,
    _mean_size,
    _read_community_by_size,
    _window_by_time,
    compute_size_spectrum,
    format_size_spectrum_report,
    size_spectrum_timeseries,
    spectrum_plot_df,
)
from tests._data_guards import require_eec_output


def _write_community_csv(path, rows):
    """rows: list of (Time, Size, sp1, sp2). Clean header, NO preamble."""
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(path, index=False)


def test_read_community_by_size_finds_and_reads(tmp_path):
    d = tmp_path / "output" / "Indicators"
    d.mkdir(parents=True)
    _write_community_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", [(1.0, 0.0, 2.0, 3.0), (1.0, 10.0, 1.0, 1.0)]
    )
    wide = _read_community_by_size(tmp_path / "output", "biomassDistribBySize", "osm")
    assert list(wide.columns) == ["Time", "Size", "sp1", "sp2"]
    assert len(wide) == 2


def test_read_community_by_size_missing_raises(tmp_path):
    (tmp_path / "output").mkdir()
    with pytest.raises(FileNotFoundError):
        _read_community_by_size(tmp_path / "output", "biomassDistribBySize", "osm")


def test_read_community_by_size_empty_file_raises(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    (d / "osm_biomassDistribBySize_Simu0.csv").write_text("")  # 0-content
    with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
        _read_community_by_size(d, "biomassDistribBySize", "osm")


def test_community_long_sums_species():
    wide = pd.DataFrame(
        {"Time": [1.0, 1.0], "Size": [0.0, 10.0], "sp1": [2.0, 1.0], "sp2": [3.0, 4.0]}
    )
    long = _community_long(wide)
    assert list(long.columns) == ["time", "size", "value"]
    assert long.loc[long["size"] == 0.0, "value"].iloc[0] == 5.0
    assert long.loc[long["size"] == 10.0, "value"].iloc[0] == 5.0
    assert long["size"].dtype == float


def test_window_by_time_selects_years_not_rows():
    df = pd.DataFrame({"time": [1.0, 1.0, 2.0, 3.0], "size": [0, 10, 0, 0], "value": [1, 1, 1, 1]})
    w = _window_by_time(df, "time", 1)
    assert sorted(w["time"].unique()) == [3.0]


def test_window_by_time_rejects_nonpositive():
    df = pd.DataFrame({"time": [1.0], "size": [0.0], "value": [1.0]})
    with pytest.raises(ValueError):
        _window_by_time(df, "time", 0)


def test_infer_bin_width():
    assert _infer_bin_width([0.0, 10.0, 20.0, 30.0]) == 10.0


def test_large_fish_indicator():
    # edges 0,10,40,50 ; values 1,1,1,1 ; threshold 40 -> bins 40,50 -> 2/4
    assert _large_fish_indicator([0.0, 10.0, 40.0, 50.0], [1.0, 1.0, 1.0, 1.0], 40.0) == 0.5
    assert _large_fish_indicator([0.0], [0.0], 40.0) == 0.0  # zero total


def test_mean_size():
    # midpoints 5,15 ; values 1,3 -> (5*1+15*3)/4 = 12.5
    assert _mean_size([5.0, 15.0], [1.0, 3.0]) == 12.5


def test_compute_size_spectrum_known_powerlaw(tmp_path):
    # community value = 1e6 * midpoint^-2  -> log-log slope == -2 exactly
    d = tmp_path / "output"
    d.mkdir()
    rows = []
    for edge in (0.0, 10.0, 20.0, 30.0, 40.0):
        mid = edge + 5.0
        v = 1.0e6 * mid**-2.0
        rows.append((1.0, edge, v / 2, v / 2))  # split across 2 species
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", index=False
    )
    spec = compute_size_spectrum(d, metric="biomass", prefix="osm", window_years=1)
    assert isinstance(spec, SizeSpectrum)
    assert spec.metric == "biomass"
    assert spec.bin_edges == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert spec.slope is not None and abs(spec.slope - (-2.0)) < 1e-6
    assert spec.r_squared is not None and spec.r_squared > 0.999
    assert spec.n_bins_fit == 5
    assert spec.peak_size_cm == 5.0  # smallest midpoint has the largest value


def test_compute_size_spectrum_lfi_and_min_size(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    # equal value in every bin -> LFI@40 = (#bins edge>=40)/(#bins)
    rows = [(1.0, e, 5.0, 5.0) for e in (0.0, 10.0, 20.0, 30.0, 40.0)]
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", index=False
    )
    spec = compute_size_spectrum(d, prefix="osm", window_years=1, lfi_threshold_cm=40.0)
    assert spec.lfi == pytest.approx(1 / 5)
    # min_size_cm filter (compared to bin MIDPOINTS) drops bins below cutoff from the fit
    spec2 = compute_size_spectrum(d, prefix="osm", window_years=1, min_size_cm=20.0)
    assert spec2.min_size_cm == 20.0
    assert spec2.n_bins_fit == 3  # midpoints 25,35,45 survive (edges 20,30,40)


def test_compute_size_spectrum_single_bin_slope_none(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    pd.DataFrame([(1.0, 0.0, 1.0, 1.0)], columns=["Time", "Size", "sp1", "sp2"]).to_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", index=False
    )
    spec = compute_size_spectrum(d, prefix="osm", window_years=1)
    assert spec.slope is None and spec.intercept is None and spec.r_squared is None
    assert spec.n_bins_fit < 2


def test_compute_size_spectrum_eec_real():
    require_eec_output("eec_biomassDistribBySize*")
    require_eec_output("eec_abundanceDistribBySize*")
    spec = compute_size_spectrum(
        Path("data/eec_full/output"), metric="biomass", prefix="eec", window_years=10
    )
    assert spec.slope is not None and spec.slope < 0
    assert 0.0 <= spec.lfi <= 1.0
    assert 0.0 < spec.mean_size_cm < 210.0
    assert spec.peak_size_cm < 50.0  # peak in a small bin
    ab = compute_size_spectrum(
        Path("data/eec_full/output"), metric="abundance", prefix="eec", window_years=10
    )
    assert ab.values != spec.values  # biomass vs abundance differ


def test_spectrum_plot_df_shape(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    rows = [(1.0, e, 1.0, 1.0) for e in (0.0, 10.0, 20.0)]
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", index=False
    )
    spec = compute_size_spectrum(d, prefix="osm", window_years=1)
    pdf = spectrum_plot_df(spec)
    assert list(pdf.columns) == ["size", "abundance"]
    assert len(pdf) == 3


def test_size_spectrum_timeseries_columns(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    rows = []
    for t in (1.0, 2.0):
        for e in (0.0, 10.0, 20.0, 40.0):
            rows.append((t, e, 1.0, 1.0))
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", index=False
    )
    ts = size_spectrum_timeseries(d, prefix="osm", lfi_threshold_cm=40.0)
    assert list(ts.columns) == ["time", "slope", "lfi", "mean_size_cm"]
    assert sorted(ts["time"].unique()) == [1.0, 2.0]
    assert ts["lfi"].tolist() == pytest.approx([1 / 4, 1 / 4])


def test_format_report_contains_key_fields(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    rows = [(1.0, e, 1.0, 1.0) for e in (0.0, 10.0, 20.0, 40.0)]
    pd.DataFrame(rows, columns=["Time", "Size", "sp1", "sp2"]).to_csv(
        d / "osm_biomassDistribBySize_Simu0.csv", index=False
    )
    spec = compute_size_spectrum(d, prefix="osm", window_years=1)
    md = format_size_spectrum_report(spec)
    assert "size spectrum" in md.lower()
    assert "Large-Fish Indicator" in md
    assert "trend/comparison" in md  # the honesty caveat


def test_plotting_reuse_and_new_chart():
    import plotly.graph_objects as go

    from osmose.plotting import make_size_indicator_timeseries, make_size_spectrum_plot

    pdf = pd.DataFrame({"size": [5.0, 15.0, 25.0], "abundance": [100.0, 30.0, 5.0]})
    fig = make_size_spectrum_plot(pdf)  # reused, unchanged
    assert isinstance(fig, go.Figure)

    ts = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "slope": [-2.0, -2.1],
            "lfi": [0.1, 0.12],
            "mean_size_cm": [20.0, 21.0],
        }
    )
    fig2 = make_size_indicator_timeseries(ts)
    assert isinstance(fig2, go.Figure)
    assert len(fig2.data) == 3  # slope, lfi, mean_size traces


def test_size_indicator_timeseries_empty():
    import plotly.graph_objects as go

    from osmose.plotting import make_size_indicator_timeseries

    fig = make_size_indicator_timeseries(
        pd.DataFrame(columns=["time", "slope", "lfi", "mean_size_cm"])
    )
    assert isinstance(fig, go.Figure)
