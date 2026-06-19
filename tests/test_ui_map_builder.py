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
