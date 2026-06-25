import plotly.graph_objects as go

from osmose.plotting import make_kobe_plot, make_ratio_timeseries
from osmose.validation.stock_status import StockStatus


def _status():
    return [
        StockStatus(
            species="cod",
            years=[0, 1],
            b_over_bmsy=[1.2, 0.8],
            f_over_fmsy=[0.5, 1.5],
            b_ref_label="Bmsy [user]",
            latest_quadrant="red",
        )
    ]


def test_kobe_plot_builds_with_soft_quadrants_and_indicative_note():
    fig = make_kobe_plot(_status())
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) >= 4  # four quadrant rectangles
    txt = " ".join(a.text for a in fig.layout.annotations if a.text)
    assert "ndicative" in txt  # the indicative annotation


def test_kobe_skips_partial_reference_species():
    partial = [
        StockStatus(
            species="x", years=[0], b_over_bmsy=[None], f_over_fmsy=[1.0], b_ref_label="Bmsy [user]"
        )
    ]
    fig = make_kobe_plot(partial)
    pts = sum(len(t.x or []) for t in fig.data if isinstance(t, go.Scatter))
    assert pts == 0  # no plottable point (missing B axis)


def test_ratio_timeseries_builds():
    assert isinstance(make_ratio_timeseries(_status(), "f"), go.Figure)
