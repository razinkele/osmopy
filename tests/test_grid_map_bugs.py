"""Regression tests for map display bugs (C1, C2, C3, H1)."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# C1: Zero-height/width polygons for single-row/col NetCDF grids
# ---------------------------------------------------------------------------


def _polygon_height(polygon: list) -> float:
    """Return the latitude span of a polygon."""
    lats = [pt[1] for pt in polygon]
    return max(lats) - min(lats)


def _polygon_width(polygon: list) -> float:
    """Return the longitude span of a polygon."""
    lons = [pt[0] for pt in polygon]
    return max(lons) - min(lons)


def _all_cell_polygons(layers: list) -> list:
    """Extract all polygon coordinates from ocean/land layers."""
    polys = []
    for lyr in layers:
        if lyr.get("id") in ("grid-ocean", "grid-land"):
            for d in lyr.get("data", []):
                polys.append(d["polygon"])
    return polys


class TestSingleRowNetcdf:
    def test_single_row_cell_height_nonzero(self):
        """ny=1 NetCDF grid: cell polygons must have non-zero height."""
        from ui.pages.grid_helpers import build_netcdf_grid_layers

        lat = np.array([[47.0, 47.0, 47.0]])   # shape (1, 3)
        lon = np.array([[1.0, 2.0, 3.0]])
        mask = np.ones((1, 3))
        layers, _ = build_netcdf_grid_layers(lat, lon, mask)
        polys = _all_cell_polygons(layers)
        assert polys, "No cell polygons produced"
        for poly in polys:
            assert _polygon_height(poly) > 0, f"Zero-height polygon: {poly}"

    def test_single_col_cell_width_nonzero(self):
        """nx=1 NetCDF grid: cell polygons must have non-zero width."""
        from ui.pages.grid_helpers import build_netcdf_grid_layers

        lat = np.array([[45.0], [46.0], [47.0]])   # shape (3, 1)
        lon = np.array([[2.0], [2.0], [2.0]])
        mask = np.ones((3, 1))
        layers, _ = build_netcdf_grid_layers(lat, lon, mask)
        polys = _all_cell_polygons(layers)
        assert polys, "No cell polygons produced"
        for poly in polys:
            assert _polygon_width(poly) > 0, f"Zero-width polygon: {poly}"

    def test_single_cell_nonzero(self):
        """1×1 NetCDF grid: single cell must have non-zero height and width."""
        from ui.pages.grid_helpers import build_netcdf_grid_layers

        lat = np.array([[48.0]])
        lon = np.array([[2.0]])
        mask = np.ones((1, 1))
        layers, _ = build_netcdf_grid_layers(lat, lon, mask)
        polys = _all_cell_polygons(layers)
        assert polys, "No cell polygons produced for 1x1 grid"
        for poly in polys:
            assert _polygon_height(poly) > 0
            assert _polygon_width(poly) > 0

    def test_single_row_boundary_nonzero(self):
        """ny=1: boundary polygon must also have non-zero height."""
        from ui.pages.grid_helpers import build_netcdf_grid_layers

        lat = np.array([[47.0, 47.0, 47.0]])
        lon = np.array([[1.0, 2.0, 3.0]])
        mask = np.ones((1, 3))
        layers, _ = build_netcdf_grid_layers(lat, lon, mask)
        boundary_layers = [l for l in layers if l.get("id") == "grid-extent"]
        assert boundary_layers
        poly = boundary_layers[0]["data"][0]["polygon"]
        assert _polygon_height(poly) > 0

    def test_single_row_overlay_nonzero(self):
        """ny=1 NetCDF overlay: cell polygons must have non-zero height."""
        from ui.pages.grid_helpers import load_netcdf_overlay
        import tempfile, pathlib, xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "overlay.nc"
            lat = np.array([[47.0, 47.0, 47.0]])
            lon = np.array([[1.0, 2.0, 3.0]])
            data = np.array([[[1.0, 2.0, 3.0]]])
            ds = xr.Dataset(
                {"value": (["time", "y", "x"], data)},
                coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
            )
            ds.to_netcdf(p)
            cells = load_netcdf_overlay(p)
            assert cells, "No cells returned for single-row overlay"
            for cell in cells:
                assert _polygon_height(cell["polygon"]) > 0


# ---------------------------------------------------------------------------
# C2: Animation self-cancellation (movement_step read without isolate)
# ---------------------------------------------------------------------------


class TestMovementStepIsolation:
    def test_movement_step_read_uses_isolate(self):
        """movement_controls must read input.movement_step inside reactive.isolate()."""
        import ast, pathlib
        src = (pathlib.Path(__file__).parent.parent / "ui" / "pages" / "grid.py").read_text()
        tree = ast.parse(src)

        # Find the movement_controls function
        mc_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == "movement_controls":
                mc_func = node
                break
        assert mc_func is not None, "movement_controls not found"

        func_src = ast.get_source_segment(src, mc_func)
        assert func_src is not None

        step_pos = func_src.find("movement_step()")
        assert step_pos != -1, "movement_step() not found in movement_controls"

        # The reactive.isolate() wrapping movement_step must appear within the
        # 200 characters immediately before the call (i.e., in the same try block).
        nearby = func_src[max(0, step_pos - 200):step_pos]
        assert "reactive.isolate()" in nearby, (
            "input.movement_step() must be read inside a reactive.isolate() block. "
            f"Context before call:\n{nearby}"
        )


# ---------------------------------------------------------------------------
# C3: Hardcoded lat/lon var names in load_netcdf_overlay
# ---------------------------------------------------------------------------


class TestNetcdfOverlayCustomVarNames:
    def test_overlay_custom_lat_lon_names(self):
        """load_netcdf_overlay should find 'latitude'/'longitude' automatically
        and also respect explicit var_lat/var_lon params."""
        from ui.pages.grid_helpers import load_netcdf_overlay
        import tempfile, pathlib, xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "overlay.nc"
            latitude = np.array([[45.0, 45.0], [46.0, 46.0]])
            longitude = np.array([[1.0, 2.0], [1.0, 2.0]])
            data = np.array([[[1.0, 2.0], [3.0, 4.0]]])
            ds = xr.Dataset(
                {"value": (["time", "y", "x"], data)},
                coords={
                    "latitude": (["y", "x"], latitude),
                    "longitude": (["y", "x"], longitude),
                },
            )
            ds.to_netcdf(p)
            # "latitude"/"longitude" are now auto-detected as fallback names
            cells_default = load_netcdf_overlay(p)
            assert cells_default is not None and len(cells_default) > 0, (
                "load_netcdf_overlay should auto-detect 'latitude'/'longitude' coords"
            )
            # Explicit var_lat/var_lon should also work
            cells_explicit = load_netcdf_overlay(p, var_lat="latitude", var_lon="longitude")
            assert cells_explicit is not None and len(cells_explicit) > 0
            # Both should produce the same number of cells
            assert len(cells_default) == len(cells_explicit)


# ---------------------------------------------------------------------------
# H1: Theme change during animation skips polygon/legend update
# ---------------------------------------------------------------------------


class TestThemeInAnimationHash:
    def test_prev_active_maps_stores_theme(self):
        """_prev_active_maps reactive.Value must store (frozenset, bool) not just frozenset."""
        import ast, pathlib
        src = (pathlib.Path(__file__).parent.parent / "ui" / "pages" / "grid.py").read_text()
        # The initialisation must be tuple, not bare frozenset
        # Look for _prev_active_maps.set( and the initial value
        assert "_prev_active_maps: reactive.Value[frozenset[str]]" not in src, (
            "_prev_active_maps must store (frozenset, bool) tuple to include theme"
        )
        # And the early return must check is_dark too — check all _prev_active_maps.get() calls
        # (the early-return guard is typically 2000+ chars after the declaration)
        assert "is_dark" in src[src.find("_prev_active_maps"):src.rfind("_prev_active_maps") + 200], (
            "is_dark not included in _prev_active_maps early-return guard"
        )
