"""Tests for the pure build_fisheries_view helper (Task 7)."""

from osmose.validation.stock_status import StockStatus
from ui.pages.fisheries import build_fisheries_view


def test_empty_state_when_no_run():
    view = build_fisheries_view(None, None, "baltic")
    assert view["kobe_ready"] is False  # no run → CTA, not a blank plot
    assert "Enter a Bmsy" in view["kobe_cta"]
    assert view["lead"] == "fm_bars"  # never leads with an empty Kobe


def test_kobe_gated_until_a_species_has_both_axes(monkeypatch):
    import ui.pages.fisheries as page

    monkeypatch.setattr(page, "load_reference_points", lambda *a, **k: ({}, []))
    cfg = type("C", (), {"species_names": ["cod"]})()
    res = object()
    # F-only status → no quadrant → Kobe NOT ready
    monkeypatch.setattr(
        page,
        "compute_stock_status",
        lambda *a, **k: [
            StockStatus("cod", [0], [None], [0.5], "Bmsy [user]", latest_quadrant=None)
        ],
    )
    assert build_fisheries_view(res, cfg, "baltic")["kobe_ready"] is False
    # both-axis status → quadrant → Kobe ready, save target shown
    monkeypatch.setattr(
        page,
        "compute_stock_status",
        lambda *a, **k: [
            StockStatus("cod", [0], [1.2], [0.5], "Bmsy [user]", latest_quadrant="green")
        ],
    )
    v = build_fisheries_view(res, cfg, "baltic")
    assert v["kobe_ready"] is True
    assert v["save_target"].endswith("baltic/reference")


def test_build_fisheries_view_forwards_ices_snapshot_dir(monkeypatch):
    """ices_snapshot_dir kwarg must reach load_reference_points."""
    import ui.pages.fisheries as page
    from pathlib import Path

    captured = {}

    def _mock_load(ref_dir, species, *, ices_snapshot_dir=None):
        captured["ices_snapshot_dir"] = ices_snapshot_dir
        return ({}, [])

    monkeypatch.setattr(page, "load_reference_points", _mock_load)
    monkeypatch.setattr(
        page,
        "compute_stock_status",
        lambda *a, **k: [],
    )

    cfg = type("C", (), {"species_names": ["cod"]})()
    res = object()
    fake_dir = Path("/tmp/fake_ices_snapshots")
    build_fisheries_view(res, cfg, "baltic", ices_snapshot_dir=fake_dir)

    assert captured["ices_snapshot_dir"] == fake_dir
