"""Tests for discover_spatial_files and _overlay_label in grid_helpers."""

import pathlib

import pytest


_EEC_FULL_DIR = pathlib.Path(__file__).parent.parent / "data" / "eec_full"
_EEC_FULL_AVAILABLE = (_EEC_FULL_DIR / "eec_all-parameters.csv").exists()


class TestOverlayLabelMoved:
    """Verify _overlay_label is importable from grid_helpers."""

    def test_import_from_grid_helpers(self):
        from ui.pages.grid_helpers import _overlay_label

        assert callable(_overlay_label)

    def test_ltl_label(self):
        from ui.pages.grid_helpers import _overlay_label

        assert _overlay_label("eec_ltlbiomassTons.nc") == "LTL Biomass"

    def test_fishing_label(self):
        from ui.pages.grid_helpers import _overlay_label

        label = _overlay_label("fishing/fishing-distrib.csv")
        assert "ishing" in label


@pytest.mark.skipif(not _EEC_FULL_AVAILABLE, reason="EEC Full not found")
class TestDiscoverSpatialFiles:
    @pytest.fixture(autouse=True)
    def _load(self):
        from osmose.config.reader import OsmoseConfigReader

        reader = OsmoseConfigReader()
        self.cfg = reader.read(_EEC_FULL_DIR / "eec_all-parameters.csv")
        self.cfg_dir = _EEC_FULL_DIR

    def test_returns_three_categories(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files(self.cfg, self.cfg_dir)
        assert "movement" in result
        assert "fishing" in result
        assert "other" in result

    def test_movement_grouped_by_species(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files(self.cfg, self.cfg_dir)
        movement = result["movement"]
        assert isinstance(movement, dict)
        assert "cod" in movement
        assert len(movement["cod"]) >= 3

    def test_movement_has_14_species(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files(self.cfg, self.cfg_dir)
        assert len(result["movement"]) == 14

    def test_fishing_has_entries(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files(self.cfg, self.cfg_dir)
        assert len(result["fishing"]) >= 1

    def test_other_has_ltl(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files(self.cfg, self.cfg_dir)
        labels = [e["label"] for e in result["other"]]
        assert any("LTL" in l for l in labels)

    def test_all_paths_exist(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files(self.cfg, self.cfg_dir)
        for entries in result["fishing"]:
            assert entries["path"].exists(), f"Missing: {entries['path']}"
        for entries in result["other"]:
            assert entries["path"].exists(), f"Missing: {entries['path']}"

    def test_empty_config_returns_empty(self):
        from ui.pages.grid_helpers import discover_spatial_files

        result = discover_spatial_files({}, None)
        assert result == {"movement": {}, "fishing": [], "other": []}
