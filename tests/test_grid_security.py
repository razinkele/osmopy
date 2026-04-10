"""Security tests: path traversal prevention in grid helper functions."""

import numpy as np


class TestLoadMaskPathTraversal:
    def test_rejects_traversal(self, tmp_path):
        """load_mask must reject paths that escape the search root."""
        from ui.pages.grid_helpers import load_mask

        # Create a file outside tmp_path that traversal would reach
        outside = tmp_path.parent / "secret.csv"
        outside.write_text("1,1\n1,1\n")
        try:
            cfg = {"grid.mask.file": "../secret.csv"}
            result = load_mask(cfg, config_dir=tmp_path)
            assert result is None, "Path traversal should be rejected"
        finally:
            outside.unlink(missing_ok=True)

    def test_rejects_absolute_path(self, tmp_path):
        """load_mask must reject absolute paths outside the search root."""
        from ui.pages.grid_helpers import load_mask

        cfg = {"grid.mask.file": "/etc/passwd"}
        result = load_mask(cfg, config_dir=tmp_path)
        assert result is None

    def test_allows_valid_file(self, tmp_path):
        """load_mask must still load valid files within the search root."""
        from ui.pages.grid_helpers import load_mask

        (tmp_path / "mask.csv").write_text("1,0\n0,1\n")
        cfg = {"grid.mask.file": "mask.csv"}
        result = load_mask(cfg, config_dir=tmp_path)
        assert result is not None
        assert result.shape == (2, 2)

    def test_allows_subdir_file(self, tmp_path):
        """load_mask must allow files in subdirectories of the search root."""
        from ui.pages.grid_helpers import load_mask

        (tmp_path / "grids").mkdir()
        (tmp_path / "grids" / "mask.csv").write_text("1,1\n1,1\n")
        cfg = {"grid.mask.file": "grids/mask.csv"}
        result = load_mask(cfg, config_dir=tmp_path)
        assert result is not None


class TestLoadNetcdfGridPathTraversal:
    def test_rejects_traversal(self, tmp_path):
        """load_netcdf_grid must reject paths that escape the search root."""
        from ui.pages.grid_helpers import load_netcdf_grid
        import xarray as xr

        # Create a NetCDF file outside tmp_path
        outside = tmp_path.parent / "outside.nc"
        lat = np.array([45.0, 46.0])
        lon = np.array([1.0, 2.0])
        mask = np.ones((2, 2))
        ds = xr.Dataset(
            {"mask": (["y", "x"], mask)},
            coords={"lat": (["y"], lat), "lon": (["x"], lon)},
        )
        ds.to_netcdf(outside)
        try:
            cfg = {
                "grid.java.classname": "NcGrid",
                "grid.netcdf.file": "../outside.nc",
                "grid.var.lat": "lat",
                "grid.var.lon": "lon",
                "grid.var.mask": "mask",
            }
            result = load_netcdf_grid(cfg, config_dir=tmp_path)
            assert result is None, "Path traversal must be rejected"
        finally:
            outside.unlink(missing_ok=True)

    def test_allows_valid_file(self, tmp_path):
        """load_netcdf_grid must load valid files within the search root."""
        from ui.pages.grid_helpers import load_netcdf_grid
        import xarray as xr

        lat = np.array([45.0, 46.0])
        lon = np.array([1.0, 2.0])
        mask = np.ones((2, 2))
        ds = xr.Dataset(
            {"mask": (["y", "x"], mask)},
            coords={"lat": (["y"], lat), "lon": (["x"], lon)},
        )
        (tmp_path / "grid.nc").parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(tmp_path / "grid.nc")
        cfg = {
            "grid.netcdf.file": "grid.nc",
            "grid.var.lat": "lat",
            "grid.var.lon": "lon",
            "grid.var.mask": "mask",
        }
        result = load_netcdf_grid(cfg, config_dir=tmp_path)
        assert result is not None
        lat_out, lon_out, mask_out = result
        assert lat_out.shape[0] == 2


class TestBuildMovementCachePathTraversal:
    """build_movement_cache must reject movement map paths outside config dir."""

    def _make_cfg(self, file_val: str) -> dict:
        return {
            "movement.file.map0": file_val,
            "movement.steps.map0": "0",
            "movement.species.map0": "0",
        }

    def test_rejects_traversal(self, tmp_path):
        """movement map path with .. escape is skipped silently."""
        from ui.pages.grid_helpers import build_movement_cache

        outside = tmp_path.parent / "evil.csv"
        outside.write_text("col,row,val\n0,0,1.0\n")
        try:
            cfg = self._make_cfg("../evil.csv")
            cache = build_movement_cache(cfg, tmp_path, (0, 0, 0, 0, 2, 2), "sp0")
            assert cache == {}, "Traversal path must be skipped"
        finally:
            outside.unlink(missing_ok=True)

    def test_rejects_absolute_path(self, tmp_path):
        """movement map with absolute path outside config dir is skipped."""
        from ui.pages.grid_helpers import build_movement_cache

        cfg = self._make_cfg("/etc/passwd")
        cache = build_movement_cache(cfg, tmp_path, (0, 0, 0, 0, 2, 2), "sp0")
        assert cache == {}

    def test_accepts_valid_csv(self, tmp_path):
        """movement map CSV inside config dir is loaded."""
        from ui.pages.grid_helpers import build_movement_cache

        # Minimal 2-row CSV (col, row, value)
        csv_content = "col,row,val\n0,0,1.0\n1,0,0.5\n"
        (tmp_path / "map0.csv").write_text(csv_content)
        cfg = self._make_cfg("map0.csv")
        # grid_params: ul_lat, ul_lon, lr_lat, lr_lon, nx, ny
        cache = build_movement_cache(cfg, tmp_path, (50, -5, 48, -3, 2, 2), "sp0")
        # File existed and was within config dir — verify no security error was raised
        assert isinstance(cache, dict)


class TestValidateOverlayPath:
    """_validate_overlay_path rejects paths outside config dir."""

    def test_rejects_traversal(self, tmp_path):
        from ui.pages.grid import _validate_overlay_path

        outside = tmp_path.parent / "secret.nc"
        outside.touch()
        try:
            result = _validate_overlay_path(str(outside), tmp_path)
            assert result is None, "Path outside config dir must be rejected"
        finally:
            outside.unlink(missing_ok=True)

    def test_rejects_etc_passwd(self, tmp_path):
        from ui.pages.grid import _validate_overlay_path

        result = _validate_overlay_path("/etc/passwd", tmp_path)
        assert result is None

    def test_accepts_path_inside_config_dir(self, tmp_path):
        from ui.pages.grid import _validate_overlay_path

        inside = tmp_path / "overlay.nc"
        inside.touch()
        result = _validate_overlay_path(str(inside), tmp_path)
        assert result == inside.resolve()

    def test_returns_none_for_special_values(self, tmp_path):
        from ui.pages.grid import _validate_overlay_path

        assert _validate_overlay_path("grid_extent", tmp_path) is None
        assert _validate_overlay_path("__movement_animation__", tmp_path) is None

    def test_returns_none_when_no_cfg_dir(self):
        from ui.pages.grid import _validate_overlay_path

        assert _validate_overlay_path("/some/path.nc", None) is None
