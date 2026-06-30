"""Tests for grid page deck.gl layer builder."""

import numpy as np

from ui.pages.grid_helpers import build_grid_layers as _build_grid_layers


def test_grid_layers_with_coords():
    layers = _build_grid_layers(
        ul_lat=48.0,
        ul_lon=-5.0,
        lr_lat=43.0,
        lr_lon=0.0,
        nx=0,
        ny=0,
    )
    assert len(layers) == 1
    assert layers[0]["id"] == "grid-extent"


def test_grid_layers_zero_coords():
    layers = _build_grid_layers(ul_lat=0, ul_lon=0, lr_lat=0, lr_lon=0, nx=0, ny=0)
    assert layers == []


def test_grid_layers_with_cells():
    layers = _build_grid_layers(
        ul_lat=48.0,
        ul_lon=-5.0,
        lr_lat=43.0,
        lr_lon=0.0,
        nx=5,
        ny=4,
    )
    # extent + ocean PolygonLayer (no mask = all ocean)
    assert len(layers) == 2
    assert layers[1]["id"] == "grid-ocean"
    assert len(layers[1]["data"]) == 20


def test_grid_layers_with_mask():
    mask = np.ones((4, 5), dtype=int)
    mask[0, 3:] = -1
    mask[1, 4] = -1
    mask[3, 4] = -1
    layers = _build_grid_layers(
        ul_lat=48.0,
        ul_lon=-5.0,
        lr_lat=43.0,
        lr_lon=0.0,
        nx=5,
        ny=4,
        mask=mask,
    )
    # extent + ocean + land
    assert len(layers) == 3
    assert layers[1]["id"] == "grid-ocean"
    assert layers[2]["id"] == "grid-land"
    assert len(layers[1]["data"]) == 16
    assert len(layers[2]["data"]) == 4


def test_grid_cell_polygons_cover_grid():
    """Each cell polygon should tile the grid without gaps."""
    layers = _build_grid_layers(
        ul_lat=48.0,
        ul_lon=-6.0,
        lr_lat=43.0,
        lr_lon=-1.0,
        nx=30,
        ny=20,
    )
    ocean = layers[1]
    assert len(ocean["data"]) == 600
    # Check first cell covers expected area
    cell = ocean["data"][0]
    poly = cell["polygon"]
    # Upper-left cell: lon from -6 to -5.833..., lat from 48 to 47.75
    assert abs(poly[0][0] - (-6.0)) < 1e-10
    assert abs(poly[0][1] - 48.0) < 1e-10


def test_grid_layers_dark_mode():
    layers = _build_grid_layers(
        ul_lat=48.0,
        ul_lon=-5.0,
        lr_lat=43.0,
        lr_lon=0.0,
        nx=2,
        ny=2,
        is_dark=True,
    )
    assert len(layers) == 2


def test_model_info_modal_lists_all_models_with_engine_facts():
    from ui.pages.grid import _model_info_modal

    html = str(_model_info_modal())
    # all five model titles present
    for title in [
        "Bay of Biscay",
        "Eastern English Channel",
        "Eastern English Channel (full)",
        "Baltic Sea",
        "Minimal",
    ]:
        assert title in html
    # engine facts present
    assert "Python only" in html and "Java + Python" in html
    assert "Available models" in html  # modal title


def test_grid_loader_rename_and_labels():
    # The headline user-facing change: assert the rename + registry-driven labels on the rendered
    # loader (grid_ui() is a zero-arg module-level function that renders to HTML).
    from ui.pages.grid import grid_ui

    html = str(grid_ui())
    assert "Model selection" in html  # was "Example configuration"
    assert "— Select model —" in html  # was "— Select example —"
    assert "Eastern English Channel" in html  # eec registry title (auto-title "Eec" gone)
    assert "btn_example_info" in html  # the info button is present


def test_model_info_modal_single_model_when_selected():
    from ui.pages.grid import _model_info_modal

    html = str(_model_info_modal("baltic"))
    assert "Model: Baltic Sea" in html  # single-model modal title
    assert "Baltic Sea" in html
    assert "Bay of Biscay" not in html  # NOT the all-models overview


def test_model_info_modal_none_shows_overview():
    from ui.pages.grid import _model_info_modal

    html = str(_model_info_modal(None))
    assert "Available models" in html
    for title in ["Bay of Biscay", "Baltic Sea", "Minimal"]:
        assert title in html


def test_model_info_modal_unknown_falls_back_to_overview():
    from ui.pages.grid import _model_info_modal

    html = str(_model_info_modal("bogus"))
    assert "Available models" in html
    assert "Baltic Sea" in html
