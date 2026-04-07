"""Tests for the Map Viewer page module."""

import pathlib

import pytest

_EEC_FULL_DIR = pathlib.Path(__file__).parent.parent / "data" / "eec_full"
_EEC_FULL_AVAILABLE = (_EEC_FULL_DIR / "eec_all-parameters.csv").exists()


class TestMapViewerUi:
    def test_map_viewer_ui_returns_div(self):
        from ui.pages.map_viewer import map_viewer_ui

        result = map_viewer_ui()
        html = str(result)
        assert "map_viewer_map" in html, "MapWidget ID must be present"
        assert "osm-split-layout" in html, "Must use split layout"

    def test_map_viewer_ui_has_file_list_output(self):
        from ui.pages.map_viewer import map_viewer_ui

        html = str(map_viewer_ui())
        assert "map_viewer_file_list" in html, "File list output_ui must be present"

    def test_map_viewer_ui_has_hint(self):
        from ui.pages.map_viewer import map_viewer_ui

        html = str(map_viewer_ui())
        assert "map_viewer_hint" in html, "Hint output_ui must be present"


@pytest.mark.skipif(not _EEC_FULL_AVAILABLE, reason="EEC Full not found")
class TestMapViewerFileList:
    """Test the file list builder used by the Map Viewer."""

    def test_build_file_list_choices_has_movement(self):
        from osmose.config.reader import OsmoseConfigReader
        from ui.pages.grid_helpers import discover_spatial_files

        reader = OsmoseConfigReader()
        cfg = reader.read(_EEC_FULL_DIR / "eec_all-parameters.csv")
        catalog = discover_spatial_files(cfg, _EEC_FULL_DIR)

        choices = {}
        for species, entries in sorted(catalog["movement"].items()):
            for e in entries:
                choices[str(e["path"])] = f"{species}: {e['label']}"
        assert len(choices) >= 20, f"Expected >=20 movement entries, got {len(choices)}"

    def test_build_file_list_choices_has_fishing(self):
        from osmose.config.reader import OsmoseConfigReader
        from ui.pages.grid_helpers import discover_spatial_files

        reader = OsmoseConfigReader()
        cfg = reader.read(_EEC_FULL_DIR / "eec_all-parameters.csv")
        catalog = discover_spatial_files(cfg, _EEC_FULL_DIR)
        assert len(catalog["fishing"]) >= 1
