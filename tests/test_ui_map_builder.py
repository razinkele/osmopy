"""Tests for the Map Builder Shiny page (ui/pages/map_builder.py)."""


def test_map_builder_imports_and_registered():
    import ui.pages.map_builder as mb

    assert hasattr(mb, "map_builder_ui") and hasattr(mb, "map_builder_server")
    from pathlib import Path

    app_src = (Path(__file__).resolve().parent.parent / "app.py").read_text()
    assert "map_builder_ui" in app_src and 'value="map_builder"' in app_src
    assert "map_builder_server" in app_src and "'map_builder'" in app_src  # nav-order array


def test_species_choices_from_config():
    from ui.pages.map_builder import _species_choices

    cfg = {"simulation.nspecies": "2", "species.name.sp0": "cod", "species.name.sp1": "herring"}
    assert _species_choices(cfg) == ["cod", "herring"]


def test_existing_maps_discovers_distribution_and_mask():
    from ui.pages.map_builder import _existing_maps

    cfg = {
        "movement.file.map0": "maps/cod.csv",
        "movement.species.map0": "cod",
        "movement.file.map1": "maps/herring.csv",
        "movement.species.map1": "herring",
        "grid.mask.file": "grid/mask.csv",
    }
    out = _existing_maps(cfg)
    paths = [p for _, p in out]
    assert "maps/cod.csv" in paths and "maps/herring.csv" in paths and "grid/mask.csv" in paths


def test_existing_maps_empty_when_none():
    from ui.pages.map_builder import _existing_maps

    assert _existing_maps({}) == []


def test_polygon_paint_value_mask_writes_land():
    """Applying a polygon in mask mode writes land (-99), not the paint value.

    Regression: the apply handler wrote the paint value even in mask mode, so the
    polygon mask tool corrupted cells instead of masking them.
    """
    from ui.pages.map_builder import _polygon_paint_value

    assert _polygon_paint_value("mask", 5.0) == -99.0
    assert _polygon_paint_value("brush", 5.0) == 5.0
    assert _polygon_paint_value("polygon", 1.0) == 1.0
