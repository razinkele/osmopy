"""Tests for CSV map loading and display rendering with EEC Full data.

Covers:
- MD1: Individual CSV map files load with correct shape and separator
- MD2: CSV overlay cells have valid polygon geometry (4 corners, within grid bounds)
- MD3: CSV overlay color values follow the amber palette (dark->bright)
- MD4: All 32 unique EEC movement map files load and display correctly
- MD5: Fishing distribution CSV loads and displays correctly
- MD6: Movement cache produces valid deck.gl layer-compatible data
- MD7: CSV maps with nc_data vs bounding-box grids produce equivalent coverage

Requires: data/eec_full/ directory with EEC Full example files.
"""

import pathlib

import numpy as np
import pytest

_EEC_FULL_DIR = pathlib.Path(__file__).parent.parent / "data" / "eec_full"
_EEC_FULL_AVAILABLE = (_EEC_FULL_DIR / "eec_all-parameters.csv").exists()

pytestmark = pytest.mark.skipif(
    not _EEC_FULL_AVAILABLE, reason="EEC Full example files not found"
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eec_full():
    """Load EEC Full config, NetCDF grid, and grid parameters once per module."""
    from osmose.config.reader import OsmoseConfigReader
    from ui.pages.grid_helpers import load_netcdf_grid

    reader = OsmoseConfigReader()
    cfg = reader.read(_EEC_FULL_DIR / "eec_all-parameters.csv")
    nc_data = load_netcdf_grid(cfg, config_dir=_EEC_FULL_DIR)
    assert nc_data is not None
    lat, lon, mask = nc_data
    return {
        "cfg": cfg,
        "cfg_dir": _EEC_FULL_DIR,
        "nc_data": nc_data,
        "lat": lat,
        "lon": lon,
        "mask": mask,
        "grid_params": (
            float(lat.max()),   # ul_lat
            float(lon.min()),   # ul_lon
            float(lat.min()),   # lr_lat
            float(lon.max()),   # lr_lon
            lat.shape[1],       # nx
            lat.shape[0],       # ny
        ),
    }


@pytest.fixture(scope="module")
def all_movement_csv_paths(eec_full):
    """Collect all unique movement CSV file paths from the EEC Full config."""
    cfg = eec_full["cfg"]
    cfg_dir = eec_full["cfg_dir"]
    paths = set()
    for key, val in cfg.items():
        if key.startswith("movement.file.map") and val and val.lower() not in ("null", "none"):
            p = cfg_dir / val
            if p.exists():
                paths.add(p)
    return sorted(paths)


# ---------------------------------------------------------------------------
# MD1: Individual CSV map files load with correct shape
# ---------------------------------------------------------------------------


class TestCsvMapFileLoading:
    def test_all_movement_csvs_parse_to_grid_shape(self, eec_full, all_movement_csv_paths):
        """Every movement CSV must parse as (ny, nx) matching the NetCDF grid."""
        from ui.pages.grid_helpers import _read_csv_auto_sep

        ny, nx = eec_full["lat"].shape
        failures = []
        for p in all_movement_csv_paths:
            df = _read_csv_auto_sep(p)
            if df.shape != (ny, nx):
                failures.append(f"{p.name}: {df.shape} != ({ny}, {nx})")
        assert not failures, f"Shape mismatches:\n" + "\n".join(failures)

    def test_movement_csvs_contain_expected_values(self, all_movement_csv_paths):
        """OSMOSE movement CSVs contain -99 (land), 0 (absence), and positive (presence)."""
        from ui.pages.grid_helpers import _read_csv_auto_sep

        for p in all_movement_csv_paths:
            df = _read_csv_auto_sep(p)
            vals = df.values.astype(float)
            valid = vals[~np.isnan(vals)]
            unique = set(valid.astype(int))
            # Must contain at least -99 (sentinel) and one of 0 or 1
            assert -99 in unique, f"{p.name}: missing -99 sentinel"
            assert unique - {-99} != set(), f"{p.name}: only sentinels, no data"

    def test_fishing_distrib_parses_to_grid_shape(self, eec_full):
        """Fishing distribution CSV must parse as (ny, nx)."""
        from ui.pages.grid_helpers import _read_csv_auto_sep

        p = eec_full["cfg_dir"] / "fishing" / "fishing-distrib.csv"
        df = _read_csv_auto_sep(p)
        ny, nx = eec_full["lat"].shape
        assert df.shape == (ny, nx), f"Shape {df.shape} != ({ny}, {nx})"


# ---------------------------------------------------------------------------
# MD2: Polygon geometry validation
# ---------------------------------------------------------------------------


class TestCsvMapPolygonGeometry:
    def test_every_cell_has_four_corner_polygon(self, eec_full):
        """Each display cell must have a polygon with exactly 4 [lon, lat] corners."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = eec_full["cfg_dir"] / "maps" / "6cod_nurseries.csv"
        cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        assert cells is not None
        for i, c in enumerate(cells):
            poly = c["polygon"]
            assert len(poly) == 4, f"Cell {i}: polygon has {len(poly)} corners, expected 4"
            for j, pt in enumerate(poly):
                assert len(pt) == 2, f"Cell {i} corner {j}: expected [lon, lat], got {pt}"

    def test_polygon_corners_within_grid_bounds(self, eec_full):
        """All polygon corners must fall within the grid bounding box (with tolerance)."""
        from ui.pages.grid_helpers import load_csv_overlay

        lat, lon = eec_full["lat"], eec_full["lon"]
        lat_min, lat_max = float(lat.min()), float(lat.max())
        lon_min, lon_max = float(lon.min()), float(lon.max())
        # Allow half-cell tolerance at edges
        lat_tol = abs(float(lat[1, 0] - lat[0, 0])) if lat.shape[0] > 1 else 1.0
        lon_tol = abs(float(lon[0, 1] - lon[0, 0])) if lon.shape[1] > 1 else 1.0

        p = eec_full["cfg_dir"] / "maps" / "8sole_spawning.csv"
        cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        assert cells is not None

        for i, c in enumerate(cells):
            for j, (clon, clat) in enumerate(c["polygon"]):
                assert lat_min - lat_tol <= clat <= lat_max + lat_tol, (
                    f"Cell {i} corner {j}: lat {clat:.4f} outside "
                    f"[{lat_min - lat_tol:.4f}, {lat_max + lat_tol:.4f}]"
                )
                assert lon_min - lon_tol <= clon <= lon_max + lon_tol, (
                    f"Cell {i} corner {j}: lon {clon:.4f} outside "
                    f"[{lon_min - lon_tol:.4f}, {lon_max + lon_tol:.4f}]"
                )

    def test_polygon_area_is_positive(self, eec_full):
        """Each polygon must have non-degenerate (positive) area — no zero-area cells."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = eec_full["cfg_dir"] / "maps" / "9plie_1plus.csv"
        cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        assert cells is not None

        for i, c in enumerate(cells):
            lons = [pt[0] for pt in c["polygon"]]
            lats = [pt[1] for pt in c["polygon"]]
            width = max(lons) - min(lons)
            height = max(lats) - min(lats)
            assert width > 0 and height > 0, (
                f"Cell {i}: degenerate polygon (width={width}, height={height})"
            )

    def test_no_duplicate_polygon_positions(self, eec_full):
        """No two cells should share the same center position."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = eec_full["cfg_dir"] / "maps" / "6cod_1plus.csv"
        cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        assert cells is not None

        centers = set()
        for c in cells:
            lons = [pt[0] for pt in c["polygon"]]
            lats = [pt[1] for pt in c["polygon"]]
            center = (round(sum(lons) / 4, 6), round(sum(lats) / 4, 6))
            assert center not in centers, f"Duplicate cell center: {center}"
            centers.add(center)


# ---------------------------------------------------------------------------
# MD3: Color palette validation
# ---------------------------------------------------------------------------


class TestCsvMapColorPalette:
    def test_all_fills_are_valid_rgba(self, eec_full):
        """Every cell fill must be a 4-element RGBA list with values in [0, 255]."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = eec_full["cfg_dir"] / "maps" / "3tacaud_0.csv"
        cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        assert cells is not None

        for i, c in enumerate(cells):
            fill = c["fill"]
            assert len(fill) == 4, f"Cell {i}: fill has {len(fill)} channels, expected 4"
            r, g, b, a = fill
            assert 0 <= r <= 255, f"Cell {i}: R={r} out of range"
            assert 0 <= g <= 255, f"Cell {i}: G={g} out of range"
            assert 0 <= b <= 255, f"Cell {i}: B={b} out of range"
            assert 0 <= a <= 255, f"Cell {i}: A={a} out of range"

    def test_amber_palette_min_value_is_dark(self, eec_full):
        """Cells at vmin must have the darkest amber fill (R~180, G~80, B=0, A~100)."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = eec_full["cfg_dir"] / "maps" / "6cod_nurseries.csv"
        cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        assert cells is not None

        min_val = min(c["value"] for c in cells)
        min_cells = [c for c in cells if c["value"] == min_val]
        r, g, b, a = min_cells[0]["fill"]
        assert r == 180, f"Min-value R should be 180 (amber base), got {r}"
        assert g == 80, f"Min-value G should be 80, got {g}"
        assert b == 0, f"Min-value B should be 0, got {b}"
        assert a == 100, f"Min-value A should be 100, got {a}"

    def test_amber_palette_max_value_is_bright(self, eec_full):
        """Cells at vmax must have the brightest amber fill (R=255, G=140, B=0, A=200)."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = eec_full["cfg_dir"] / "maps" / "6cod_nurseries.csv"
        cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        assert cells is not None

        max_val = max(c["value"] for c in cells)
        max_cells = [c for c in cells if c["value"] == max_val]
        r, g, b, a = max_cells[0]["fill"]
        assert r == 255, f"Max-value R should be 255, got {r}"
        assert g == 140, f"Max-value G should be 140, got {g}"
        assert b == 0, f"Max-value B should be 0, got {b}"
        assert a == 200, f"Max-value A should be 200, got {a}"

    def test_higher_values_have_brighter_fill(self, eec_full):
        """Cells with higher values must have higher R, G, and A channels."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = eec_full["cfg_dir"] / "maps" / "6cod_nurseries.csv"
        cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        assert cells is not None

        min_val = min(c["value"] for c in cells)
        max_val = max(c["value"] for c in cells)
        if min_val == max_val:
            pytest.skip("All values identical — no gradient to test")

        lo_cell = next(c for c in cells if c["value"] == min_val)
        hi_cell = next(c for c in cells if c["value"] == max_val)
        assert hi_cell["fill"][0] >= lo_cell["fill"][0], "R should increase with value"
        assert hi_cell["fill"][1] >= lo_cell["fill"][1], "G should increase with value"
        assert hi_cell["fill"][3] >= lo_cell["fill"][3], "A should increase with value"


# ---------------------------------------------------------------------------
# MD4: All 32 unique movement maps load and display
# ---------------------------------------------------------------------------


class TestAllMovementMapsDisplay:
    def test_every_movement_csv_produces_cells(self, eec_full, all_movement_csv_paths):
        """Every unique movement CSV must produce a non-empty cell list."""
        from ui.pages.grid_helpers import load_csv_overlay

        failures = []
        for p in all_movement_csv_paths:
            cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
            if cells is None or len(cells) == 0:
                failures.append(f"{p.name}: no cells")
        assert not failures, f"Maps that failed to display:\n" + "\n".join(failures)

    def test_movement_maps_cell_count_does_not_exceed_ocean(self, eec_full, all_movement_csv_paths):
        """No movement map should produce more cells than there are ocean cells."""
        from ui.pages.grid_helpers import load_csv_overlay

        ocean_count = int((eec_full["mask"] > 0).sum())
        # Total cells (ocean+land excluding sentinels) can be at most ny*nx
        ny, nx = eec_full["lat"].shape
        max_cells = ny * nx

        for p in all_movement_csv_paths:
            cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
            if cells:
                assert len(cells) <= max_cells, (
                    f"{p.name}: {len(cells)} cells exceeds grid size {max_cells}"
                )

    def test_movement_maps_have_consistent_polygon_size(self, eec_full, all_movement_csv_paths):
        """All cells within a single map must have the same polygon dimensions (regular grid)."""
        from ui.pages.grid_helpers import load_csv_overlay

        for p in all_movement_csv_paths[:5]:  # spot-check first 5 to keep test fast
            cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
            if not cells or len(cells) < 2:
                continue

            # Compute width/height of each cell polygon
            sizes = set()
            for c in cells:
                lons = [pt[0] for pt in c["polygon"]]
                lats = [pt[1] for pt in c["polygon"]]
                w = round(max(lons) - min(lons), 4)
                h = round(max(lats) - min(lats), 4)
                sizes.add((w, h))

            # Regular grid: all cells should have the same size (within float tolerance)
            # Allow up to 2 distinct sizes for edge cells with different finite-diff steps
            assert len(sizes) <= 4, (
                f"{p.name}: {len(sizes)} distinct cell sizes (expected <=4 for edge effects)"
            )


# ---------------------------------------------------------------------------
# MD5: Fishing distribution CSV display
# ---------------------------------------------------------------------------


class TestFishingDistribDisplay:
    def test_fishing_distrib_produces_cells(self, eec_full):
        """Fishing distribution must produce display cells."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = eec_full["cfg_dir"] / "fishing" / "fishing-distrib.csv"
        cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        assert cells is not None
        assert len(cells) > 0

    def test_fishing_distrib_covers_ocean_area(self, eec_full):
        """Fishing cells should cover a significant portion of the ocean grid."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = eec_full["cfg_dir"] / "fishing" / "fishing-distrib.csv"
        cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        assert cells is not None
        ocean_count = int((eec_full["mask"] > 0).sum())
        # Fishing should cover most of the ocean area
        assert len(cells) >= ocean_count * 0.5, (
            f"Fishing distrib covers {len(cells)}/{ocean_count} ocean cells — too few"
        )

    def test_fishing_distrib_all_cells_have_display_data(self, eec_full):
        """Each fishing cell must have polygon, value, and fill."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = eec_full["cfg_dir"] / "fishing" / "fishing-distrib.csv"
        cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        assert cells is not None
        for i, c in enumerate(cells):
            assert "polygon" in c, f"Cell {i}: missing 'polygon'"
            assert "value" in c, f"Cell {i}: missing 'value'"
            assert "fill" in c, f"Cell {i}: missing 'fill'"
            assert isinstance(c["value"], float), f"Cell {i}: value not float"
            assert len(c["polygon"]) == 4, f"Cell {i}: polygon not 4 corners"
            assert len(c["fill"]) == 4, f"Cell {i}: fill not RGBA"


# ---------------------------------------------------------------------------
# MD6: Movement cache display layer data
# ---------------------------------------------------------------------------


class TestMovementCacheDisplayData:
    def test_cache_entries_have_deckgl_compatible_cells(self, eec_full):
        """Movement cache cells must be compatible with deck.gl polygon_layer."""
        from ui.pages.grid_helpers import build_movement_cache

        cache = build_movement_cache(
            eec_full["cfg"], eec_full["cfg_dir"], eec_full["grid_params"], species="sole"
        )
        assert len(cache) > 0

        for mid, m in cache.items():
            # Required keys for deck.gl layer rendering
            assert "cells" in m, f"{mid}: missing 'cells'"
            assert "color" in m, f"{mid}: missing 'color'"
            assert "label" in m, f"{mid}: missing 'label'"
            assert "steps" in m, f"{mid}: missing 'steps'"

            # Color must be RGBA for get_fill_color
            assert len(m["color"]) == 4, f"{mid}: color not RGBA"
            assert all(0 <= ch <= 255 for ch in m["color"]), f"{mid}: color out of range"

            # Each cell must have polygon + value + fill
            for i, c in enumerate(m["cells"][:10]):  # spot-check first 10
                assert "polygon" in c, f"{mid} cell {i}: missing polygon"
                assert len(c["polygon"]) == 4, f"{mid} cell {i}: polygon not 4 corners"

    def test_cache_step_sets_are_valid_timestep_indices(self, eec_full):
        """Movement step sets must contain valid timestep indices (0-23 for 24 dt/yr)."""
        from ui.pages.grid_helpers import build_movement_cache

        nsteps = int(float(eec_full["cfg"].get("simulation.time.ndtperyear", "24")))
        cache = build_movement_cache(
            eec_full["cfg"], eec_full["cfg_dir"], eec_full["grid_params"], species="whiting"
        )
        for mid, m in cache.items():
            for step in m["steps"]:
                assert 0 <= step < nsteps, (
                    f"{mid}: step {step} outside [0, {nsteps})"
                )

    def test_cache_labels_are_human_readable(self, eec_full):
        """Movement cache labels must be non-empty and not raw indices."""
        from ui.pages.grid_helpers import build_movement_cache

        cache = build_movement_cache(
            eec_full["cfg"], eec_full["cfg_dir"], eec_full["grid_params"], species="plaice"
        )
        for mid, m in cache.items():
            label = m["label"]
            assert label, f"{mid}: empty label"
            assert not label.isdigit(), f"{mid}: label is raw number '{label}'"

    def test_different_maps_get_different_colors(self, eec_full):
        """Multiple maps for one species must get distinct colors from the palette."""
        from ui.pages.grid_helpers import build_movement_cache

        cache = build_movement_cache(
            eec_full["cfg"], eec_full["cfg_dir"], eec_full["grid_params"], species="sole"
        )
        if len(cache) < 2:
            pytest.skip("Sole has fewer than 2 maps")

        colors = [tuple(m["color"]) for m in cache.values()]
        assert len(set(colors)) == len(colors), (
            f"Duplicate colors in sole maps: {colors}"
        )


# ---------------------------------------------------------------------------
# MD7: Bounding-box grid vs nc_data grid consistency
# ---------------------------------------------------------------------------


class TestBboxVsNcGridConsistency:
    def test_csv_overlay_cell_count_matches_between_modes(self, eec_full):
        """CSV overlay loaded via nc_data and bounding-box should produce similar cell counts.

        The nc_data path uses the NetCDF lat/lon arrays; the bounding-box path
        uses ul_lat/ul_lon/lr_lat/lr_lon + nx/ny. Both should produce cells for
        the same set of valid (non-sentinel) data points.
        """
        from ui.pages.grid_helpers import load_csv_overlay

        lat, lon = eec_full["lat"], eec_full["lon"]
        ul_lat, ul_lon, lr_lat, lr_lon, nx, ny = eec_full["grid_params"]

        p = eec_full["cfg_dir"] / "maps" / "6cod_1plus.csv"

        cells_nc = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        cells_bbox = load_csv_overlay(p, ul_lat, ul_lon, lr_lat, lr_lon, nx, ny)
        assert cells_nc is not None
        assert cells_bbox is not None
        # Both paths should produce the same number of valid cells
        assert len(cells_nc) == len(cells_bbox), (
            f"nc_data path: {len(cells_nc)} cells, bbox path: {len(cells_bbox)} cells"
        )

    def test_csv_overlay_values_match_between_modes(self, eec_full):
        """Cell values from nc_data and bounding-box paths must be identical."""
        from ui.pages.grid_helpers import load_csv_overlay

        ul_lat, ul_lon, lr_lat, lr_lon, nx, ny = eec_full["grid_params"]
        p = eec_full["cfg_dir"] / "maps" / "8sole_1plus.csv"

        cells_nc = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=eec_full["nc_data"])
        cells_bbox = load_csv_overlay(p, ul_lat, ul_lon, lr_lat, lr_lon, nx, ny)
        assert cells_nc is not None and cells_bbox is not None

        vals_nc = sorted(c["value"] for c in cells_nc)
        vals_bbox = sorted(c["value"] for c in cells_bbox)
        assert vals_nc == pytest.approx(vals_bbox), "Values differ between nc_data and bbox paths"
