"""Tests for EEC example map display features.

Covers:
- CSV overlay: -99 sentinel filtering (OD1)
- load_netcdf_overlay: variable selection, time slicing, colormap, vmin/vmax (OD2-OD5)
- list_nc_overlay_variables: metadata extraction, caching (OD6)
- _overlay_label: human-readable labels (OD7)
- Overlay catalog: deduplication, movement-key exclusion, MPA scan (OD8)
- EEC integration: real files if present (OD9)
"""

import pathlib

import numpy as np
import pytest
import xarray as xr

_EEC_DIR = (
    pathlib.Path(__file__).parent.parent.parent / "osmose-master/java/src/test/resources/osmose-eec"
)
_EEC_LTL = _EEC_DIR / "eec_ltlbiomassTons.nc"
_EEC_BG = _EEC_DIR / "eec_backgroundspecies_biomass.nc"
_EEC_AVAILABLE = _EEC_LTL.exists() and _EEC_BG.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ltl_nc(path: pathlib.Path, n_time=4, ny=4, nx=5) -> None:
    """Write a minimal LTL-style NetCDF (time × y × x, latitude/longitude as data vars)."""
    rng = np.random.default_rng(42)
    lat_2d = np.linspace(49.0, 51.0, ny)[:, None] * np.ones((1, nx))
    lon_2d = np.linspace(-2.0, 2.0, nx)[None, :] * np.ones((ny, 1))
    ds = xr.Dataset(
        {
            "Diatoms": (
                ["time", "y", "x"],
                rng.uniform(0, 100, (n_time, ny, nx)),
            ),
            "Dinoflagellates": (
                ["time", "y", "x"],
                rng.uniform(0, 200, (n_time, ny, nx)),
            ),
            "latitude": (["y", "x"], lat_2d),
            "longitude": (["y", "x"], lon_2d),
        },
        coords={"x": np.arange(nx), "y": np.arange(ny), "time": np.arange(n_time)},
    )
    ds.to_netcdf(path)


def _make_static_nc(path: pathlib.Path, ny=3, nx=3) -> None:
    """Write a static (no time dim) 2-D NetCDF."""
    lat = np.linspace(45.0, 47.0, ny)[:, None] * np.ones((1, nx))
    lon = np.ones((ny, 1)) * np.linspace(1.0, 3.0, nx)[None, :]
    data = np.arange(1.0, float(ny * nx + 1)).reshape(ny, nx)
    ds = xr.Dataset(
        {"temperature": (["y", "x"], data)},
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    ds.to_netcdf(path)


def _make_csv_grid(path: pathlib.Path, ny=4, nx=5, *, value=1.0, sentinel_row=0) -> None:
    """Write a comma-separated CSV with one row containing -99 sentinels."""
    import pandas as pd

    data = np.full((ny, nx), value)
    data[sentinel_row, :] = -99.0
    pd.DataFrame(data).to_csv(path, sep=",", header=False, index=False)


def _make_semicolon_csv(path: pathlib.Path, ny=4, nx=5, *, value=1.0, sentinel_value=-99.0) -> None:
    """Write a semicolon-separated CSV (OSMOSE standard format)."""
    import pandas as pd

    data = np.full((ny, nx), value)
    data[0, :] = sentinel_value  # first row is land/sentinel
    pd.DataFrame(data).to_csv(path, sep=";", header=False, index=False)


# ---------------------------------------------------------------------------
# OD1: CSV sentinel filtering
# ---------------------------------------------------------------------------


class TestCsvSentinelFilter:
    def test_sentinel_cells_excluded_from_output(self, tmp_path):
        """Cells with -99 must not appear in the returned cell list."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = tmp_path / "grid.csv"
        _make_csv_grid(p, ny=3, nx=3, value=0.5, sentinel_row=0)
        # Grid is 3×3; row 0 is -99 (3 cells), rows 1-2 are 0.5 (6 cells)
        cells = load_csv_overlay(p, ul_lat=47.0, ul_lon=1.0, lr_lat=45.0, lr_lon=3.0, nx=3, ny=3)
        assert cells is not None
        values = [c["value"] for c in cells]
        assert all(v >= -9.0 for v in values), "Sentinel (-99) cells must be filtered out"
        # Only 6 valid cells remain
        assert len(cells) == 6, f"Expected 6 valid cells, got {len(cells)}"

    def test_sentinel_excluded_from_color_range(self, tmp_path):
        """The -99 sentinel must not distort the vmin/vmax used for color scaling."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = tmp_path / "grid.csv"
        # All valid values are 0.5 except the sentinel row
        _make_csv_grid(p, ny=3, nx=3, value=0.5, sentinel_row=2)
        cells = load_csv_overlay(p, ul_lat=47.0, ul_lon=1.0, lr_lat=45.0, lr_lon=3.0, nx=3, ny=3)
        assert cells is not None
        # If sentinel had been included in vmin, colours would be wrong; vmin should be 0.5
        # All remaining cells have the same value → same colour
        fills = [tuple(c["fill"]) for c in cells]
        assert len(set(fills)) == 1, "All cells with the same value must have the same colour"

    def test_nan_cells_still_excluded(self, tmp_path):
        """NaN entries in the CSV must still be excluded (pre-existing behaviour)."""
        from ui.pages.grid_helpers import load_csv_overlay
        import pandas as pd

        p = tmp_path / "nan_grid.csv"
        data = np.ones((3, 3))
        data[1, 1] = np.nan
        pd.DataFrame(data).to_csv(p, sep=",", header=False, index=False)
        cells = load_csv_overlay(p, ul_lat=47.0, ul_lon=1.0, lr_lat=45.0, lr_lon=3.0, nx=3, ny=3)
        assert cells is not None
        assert len(cells) == 8, "NaN cell must be excluded (8 of 9 valid)"

    def test_small_negative_values_included(self, tmp_path):
        """Values between -9 and 0 are legitimate data and must NOT be excluded."""
        from ui.pages.grid_helpers import load_csv_overlay
        import pandas as pd

        p = tmp_path / "neg_grid.csv"
        data = np.array([[-0.5, -1.0, -5.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        pd.DataFrame(data).to_csv(p, sep=",", header=False, index=False)
        cells = load_csv_overlay(p, ul_lat=47.0, ul_lon=1.0, lr_lat=45.0, lr_lon=3.0, nx=3, ny=3)
        assert cells is not None
        # 0.0 is excluded (OSMOSE absent marker); negatives above -9 are kept
        assert len(cells) == 8, "Values > -9.0 and != 0.0 must be included"

    def test_all_sentinel_returns_none(self, tmp_path):
        """A CSV consisting entirely of -99 must return None."""
        from ui.pages.grid_helpers import load_csv_overlay
        import pandas as pd

        p = tmp_path / "all_sentinel.csv"
        pd.DataFrame(np.full((3, 3), -99.0)).to_csv(p, sep=",", header=False, index=False)
        cells = load_csv_overlay(p, ul_lat=47.0, ul_lon=1.0, lr_lat=45.0, lr_lon=3.0, nx=3, ny=3)
        assert cells is None

    def test_all_zeros_returns_none(self, tmp_path):
        """A CSV with only 0.0 values (after sentinel exclusion) must return None."""
        from ui.pages.grid_helpers import load_csv_overlay
        import pandas as pd

        p = tmp_path / "all_zeros.csv"
        data = np.array([[-99.0, -99.0, -99.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        pd.DataFrame(data).to_csv(p, sep=",", header=False, index=False)
        cells = load_csv_overlay(p, ul_lat=47.0, ul_lon=1.0, lr_lat=45.0, lr_lon=3.0, nx=3, ny=3)
        assert cells is None

    def test_flipud_applied_to_csv_overlay(self, tmp_path):
        """CSV row 0 (southernmost) must map to the lowest latitude after flipud."""
        from ui.pages.grid_helpers import load_csv_overlay
        import pandas as pd

        # Row 0 = value 1, row 1 = value 2, row 2 = value 3
        p = tmp_path / "ordered.csv"
        data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        pd.DataFrame(data).to_csv(p, sep=",", header=False, index=False)
        cells = load_csv_overlay(p, ul_lat=50.0, ul_lon=0.0, lr_lat=47.0, lr_lon=2.0, nx=2, ny=3)
        assert cells is not None
        # After flipud: row 0 (value 1, southernmost) → last row → lowest lat
        # Row 2 (value 3, northernmost) → first row → highest lat
        # Find cell with highest latitude (clat + hlat)
        northmost = max(cells, key=lambda c: c["polygon"][0][1])
        southmost = min(cells, key=lambda c: c["polygon"][2][1])
        # CSV row 0 = south → low lat; CSV row 2 = north → high lat
        assert northmost["value"] == 3.0, "CSV row 2 (north in file) should map to high lat"
        assert southmost["value"] == 1.0, "CSV row 0 (south in file) should map to low lat"

    def test_viridis_gradient_produces_distinct_colors(self, tmp_path):
        """Multi-value CSVs must produce distinct colors via the viridis ramp."""
        from ui.pages.grid_helpers import load_csv_overlay
        import pandas as pd

        p = tmp_path / "gradient.csv"
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pd.DataFrame(data).to_csv(p, sep=",", header=False, index=False)
        cells = load_csv_overlay(p, ul_lat=48.0, ul_lon=0.0, lr_lat=46.0, lr_lon=3.0, nx=3, ny=2)
        assert cells is not None
        assert len(cells) == 6
        fills = [tuple(c["fill"]) for c in cells]
        # Must have more than 1 distinct color for 6 distinct values
        assert len(set(fills)) > 1, "Multi-value map must produce distinct colors"
        # All RGBA channels must be in valid range
        for r, g, b, a in fills:
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255
            assert 0 <= a <= 255
        # Min value cell should have viridis start (purple-ish: low R, low G, high B)
        min_cell = min(cells, key=lambda c: c["value"])
        max_cell = max(cells, key=lambda c: c["value"])
        assert min_cell["fill"][0] < max_cell["fill"][0], "R should increase with value"
        assert min_cell["fill"][1] < max_cell["fill"][1], "G should increase with value"


# ---------------------------------------------------------------------------
# OD1b: Semicolon-separated CSV support (OSMOSE standard format)
# ---------------------------------------------------------------------------


class TestCsvSemicolonSeparator:
    """OSMOSE spatial CSVs use semicolons.  Verify auto-detection works."""

    def test_semicolon_csv_loads_correct_shape(self, tmp_path):
        """A semicolon-separated 4x5 CSV must produce 4x5 cells (minus sentinels)."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = tmp_path / "semi.csv"
        _make_semicolon_csv(p, ny=4, nx=5, value=1.0)
        # Row 0 is -99 sentinel → 5 cells excluded, 15 remain
        cells = load_csv_overlay(p, ul_lat=47.0, ul_lon=1.0, lr_lat=45.0, lr_lon=3.0, nx=5, ny=4)
        assert cells is not None
        assert len(cells) == 15, f"Expected 15 valid cells, got {len(cells)}"

    def test_semicolon_csv_values_parsed_as_numbers(self, tmp_path):
        """Cell values from semicolon CSVs must be numeric, not strings."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = tmp_path / "nums.csv"
        _make_semicolon_csv(p, ny=3, nx=3, value=42.5, sentinel_value=-99.0)
        cells = load_csv_overlay(p, ul_lat=47.0, ul_lon=1.0, lr_lat=45.0, lr_lon=3.0, nx=3, ny=3)
        assert cells is not None
        for c in cells:
            assert isinstance(c["value"], float)
            assert c["value"] == pytest.approx(42.5)

    def test_comma_csv_still_works(self, tmp_path):
        """Comma-separated CSVs must continue to work after the auto-detect change."""
        from ui.pages.grid_helpers import load_csv_overlay

        p = tmp_path / "comma.csv"
        _make_csv_grid(p, ny=3, nx=3, value=2.0, sentinel_row=0)
        cells = load_csv_overlay(p, ul_lat=47.0, ul_lon=1.0, lr_lat=45.0, lr_lon=3.0, nx=3, ny=3)
        assert cells is not None
        assert len(cells) == 6  # row 0 is sentinel, 6 remain

    def test_semicolon_mask_loads_correct_shape(self, tmp_path):
        """load_mask must handle semicolon-separated mask CSVs."""
        from ui.pages.grid_helpers import load_mask
        import pandas as pd

        p = tmp_path / "grid" / "mask.csv"
        p.parent.mkdir()
        data = np.array([[-99, -99, 0], [0, 1, 1], [1, 1, -99]])
        pd.DataFrame(data).to_csv(p, sep=";", header=False, index=False)

        mask = load_mask({"grid.mask.file": "grid/mask.csv"}, config_dir=tmp_path)
        assert mask is not None
        assert mask.shape == (3, 3)
        assert mask[1, 1] == 1
        # CSV row 0 (southernmost) becomes array row 2 after flipud
        assert mask[2, 0] == -99
        assert mask[0, 0] == 1

    def test_comma_mask_still_works(self, tmp_path):
        """load_mask must still handle comma-separated CSVs."""
        from ui.pages.grid_helpers import load_mask
        import pandas as pd

        p = tmp_path / "grid" / "mask.csv"
        p.parent.mkdir()
        data = np.array([[0, 0, 1], [1, 1, 1]])
        pd.DataFrame(data).to_csv(p, sep=",", header=False, index=False)

        mask = load_mask({"grid.mask.file": "grid/mask.csv"}, config_dir=tmp_path)
        assert mask is not None
        assert mask.shape == (2, 3)

    def test_semicolon_csv_overlay_with_nc_grid(self, tmp_path):
        """Semicolon CSV overlay mapped onto a NetCDF grid must produce cells."""
        from ui.pages.grid_helpers import load_csv_overlay
        import pandas as pd

        # Create a semicolon CSV matching a 3x4 grid
        p = tmp_path / "overlay.csv"
        data = np.ones((3, 4)) * 0.5
        data[0, :] = -99.0  # land row
        pd.DataFrame(data).to_csv(p, sep=";", header=False, index=False)

        # Fake nc_data tuple
        lat = np.array(
            [[45.0, 45.0, 45.0, 45.0], [46.0, 46.0, 46.0, 46.0], [47.0, 47.0, 47.0, 47.0]]
        )
        lon = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        mask = np.ones((3, 4))
        nc_data = (lat, lon, mask)

        cells = load_csv_overlay(p, 0, 0, 0, 0, 0, 0, nc_data=nc_data)
        assert cells is not None
        assert len(cells) == 8  # 3x4=12 minus row 0 (4 sentinel cells) = 8


# ---------------------------------------------------------------------------
# OD1c: _read_csv_auto_sep unit tests
# ---------------------------------------------------------------------------


class TestReadCsvAutoSep:
    """Direct tests of the separator auto-detection function."""

    def test_semicolon_detected(self, tmp_path):
        from ui.pages.grid_helpers import _read_csv_auto_sep
        import pandas as pd

        p = tmp_path / "semi.csv"
        pd.DataFrame([[1, 2, 3], [4, 5, 6]]).to_csv(p, sep=";", header=False, index=False)
        df = _read_csv_auto_sep(p)
        assert df.shape == (2, 3)

    def test_comma_detected(self, tmp_path):
        from ui.pages.grid_helpers import _read_csv_auto_sep
        import pandas as pd

        p = tmp_path / "comma.csv"
        pd.DataFrame([[1, 2, 3], [4, 5, 6]]).to_csv(p, sep=",", header=False, index=False)
        df = _read_csv_auto_sep(p)
        assert df.shape == (2, 3)

    def test_single_column_file_stays_single(self, tmp_path):
        """A genuine single-column file should not be widened by fallback."""
        from ui.pages.grid_helpers import _read_csv_auto_sep
        import pandas as pd

        p = tmp_path / "single.csv"
        pd.DataFrame([10, 20, 30]).to_csv(p, sep=";", header=False, index=False)
        df = _read_csv_auto_sep(p)
        assert df.shape == (3, 1)

    def test_na_strings_handled(self, tmp_path):
        """OSMOSE CSVs use 'NA' for missing data — must parse as NaN."""
        from ui.pages.grid_helpers import _read_csv_auto_sep

        p = tmp_path / "na.csv"
        p.write_text("1;NA;3\nNA;5;6\n")
        df = _read_csv_auto_sep(p)
        assert df.shape == (2, 3)
        assert np.isnan(df.iloc[0, 1])
        assert np.isnan(df.iloc[1, 0])


# ---------------------------------------------------------------------------
# OD2: load_netcdf_overlay variable selection
# ---------------------------------------------------------------------------


class TestNetcdfOverlayVariableSelection:
    def test_first_var_used_when_no_name(self, tmp_path):
        """Without var_name, the first 2D+ variable must be selected."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "multi.nc"
        _make_ltl_nc(p, n_time=2, ny=2, nx=2)
        cells = load_netcdf_overlay(p)
        assert cells is not None and len(cells) > 0

    def test_specific_var_selected(self, tmp_path):
        """var_name must select the matching variable (not necessarily the first)."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "multi.nc"
        rng = np.random.default_rng(0)
        lat = np.array([[45.0, 45.0], [46.0, 46.0]])
        lon = np.array([[1.0, 2.0], [1.0, 2.0]])
        ds = xr.Dataset(
            {
                "alpha": (["time", "y", "x"], rng.uniform(0, 1, (2, 2, 2))),
                "beta": (["time", "y", "x"], rng.uniform(100, 200, (2, 2, 2))),
                "latitude": (["y", "x"], lat),
                "longitude": (["y", "x"], lon),
            },
            coords={"x": [0, 1], "y": [0, 1], "time": [0, 1]},
        )
        ds.to_netcdf(p)

        # beta has values 100-200; alpha has values 0-1
        # Cells for beta should have value > 50; alpha should have value ≤ 1
        cells_alpha = load_netcdf_overlay(p, var_name="alpha", time_step=0)
        cells_beta = load_netcdf_overlay(p, var_name="beta", time_step=0)
        assert cells_alpha is not None and cells_beta is not None
        assert max(c["value"] for c in cells_alpha) <= 1.0 + 1e-9
        assert min(c["value"] for c in cells_beta) >= 100.0 - 1e-9

    def test_unknown_var_falls_back_to_first(self, tmp_path):
        """An unrecognised var_name must gracefully fall back to the first variable."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "multi.nc"
        _make_ltl_nc(p, n_time=1, ny=2, nx=2)
        cells = load_netcdf_overlay(p, var_name="NonExistentVariable")
        assert cells is not None and len(cells) > 0


# ---------------------------------------------------------------------------
# OD3: Time step selection
# ---------------------------------------------------------------------------


class TestNetcdfOverlayTimeStep:
    def test_time_step_selects_correct_slice(self, tmp_path):
        """Different time steps must produce different values."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "time.nc"
        lat = np.array([[45.0, 45.0], [46.0, 46.0]])
        lon = np.array([[1.0, 2.0], [1.0, 2.0]])
        # Time slices with clearly distinct values
        data = np.array(
            [
                [[10.0, 10.0], [10.0, 10.0]],  # t=0: all 10
                [[999.0, 999.0], [999.0, 999.0]],  # t=1: all 999
            ]
        )
        ds = xr.Dataset(
            {"value": (["time", "y", "x"], data)},
            coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon), "time": [0, 1]},
        )
        ds.to_netcdf(p)

        cells_t0 = load_netcdf_overlay(p, time_step=0)
        cells_t1 = load_netcdf_overlay(p, time_step=1)
        assert cells_t0 is not None and cells_t1 is not None
        assert all(c["value"] == pytest.approx(10.0) for c in cells_t0)
        assert all(c["value"] == pytest.approx(999.0) for c in cells_t1)

    def test_time_step_clamped_to_valid_range(self, tmp_path):
        """A time_step beyond the last index must be clamped, not error."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "time.nc"
        _make_ltl_nc(p, n_time=3, ny=2, nx=2)
        # time_step=999 must silently clamp to 2 (last step)
        cells = load_netcdf_overlay(p, time_step=999)
        assert cells is not None and len(cells) > 0

    def test_static_nc_ignores_time_step(self, tmp_path):
        """Static (no time dim) NetCDF must work with any time_step value."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "static.nc"
        _make_static_nc(p, ny=2, nx=2)
        cells_t0 = load_netcdf_overlay(p, time_step=0)
        cells_t99 = load_netcdf_overlay(p, time_step=99)
        assert cells_t0 is not None
        assert cells_t99 is not None
        assert len(cells_t0) == len(cells_t99)


# ---------------------------------------------------------------------------
# OD4: Colormap correctness
# ---------------------------------------------------------------------------


class TestNetcdfOverlayColormap:
    def _make_gradient_nc(self, path: pathlib.Path) -> None:
        """Write a 1×3 NC file with values [0, 50, 100]."""
        lat = np.array([[45.0, 45.0, 45.0]])
        lon = np.array([[1.0, 2.0, 3.0]])
        data = np.array([[[0.0, 50.0, 100.0]]])
        ds = xr.Dataset(
            {"value": (["time", "y", "x"], data)},
            coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
        )
        ds.to_netcdf(path)

    def test_all_cells_have_fill(self, tmp_path):
        """Every returned cell must contain a 4-element 'fill' list."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "color.nc"
        _make_ltl_nc(p, n_time=1, ny=3, nx=3)
        cells = load_netcdf_overlay(p)
        assert cells is not None
        for cell in cells:
            assert "fill" in cell, "Cell missing 'fill' key"
            assert len(cell["fill"]) == 4, "fill must be [R, G, B, A]"
            r, g, b, a = cell["fill"]
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255
            assert 0 <= a <= 255

    def test_min_value_is_blue(self, tmp_path):
        """Cell at vmin must be rendered in the blue region (R≈0, B≥160)."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "gradient.nc"
        self._make_gradient_nc(p)
        cells = load_netcdf_overlay(p, vmin=0.0, vmax=100.0)
        assert cells is not None
        # Sort by value to get the cell with value=0
        cells_sorted = sorted(cells, key=lambda c: c["value"])
        r, g, b, a = cells_sorted[0]["fill"]
        assert r < 30, f"Min-value cell should be blue (low R), got R={r}"
        assert b > 160, f"Min-value cell should be blue (high B), got B={b}"

    def test_max_value_is_yellow(self, tmp_path):
        """Cell at vmax must be rendered in the yellow region (R≈255, B≈0)."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "gradient.nc"
        self._make_gradient_nc(p)
        cells = load_netcdf_overlay(p, vmin=0.0, vmax=100.0)
        assert cells is not None
        cells_sorted = sorted(cells, key=lambda c: c["value"])
        r, g, b, a = cells_sorted[-1]["fill"]
        assert r > 200, f"Max-value cell should be yellow (high R), got R={r}"
        assert b < 30, f"Max-value cell should be yellow (low B), got B={b}"

    def test_consistent_colors_with_global_vmin_vmax(self, tmp_path):
        """The same cell value must produce the same colour regardless of time step."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "stable.nc"
        lat = np.array([[45.0, 45.0], [46.0, 46.0]])
        lon = np.array([[1.0, 2.0], [1.0, 2.0]])
        # Both time steps share the same fixed values
        data = np.stack(
            [
                np.array([[10.0, 20.0], [30.0, 40.0]]),
                np.array([[10.0, 20.0], [30.0, 40.0]]),
            ]
        )
        ds = xr.Dataset(
            {"value": (["time", "y", "x"], data)},
            coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
        )
        ds.to_netcdf(p)

        cells_t0 = load_netcdf_overlay(p, time_step=0, vmin=10.0, vmax=40.0)
        cells_t1 = load_netcdf_overlay(p, time_step=1, vmin=10.0, vmax=40.0)
        assert cells_t0 is not None and cells_t1 is not None
        fills_t0 = {round(c["value"]): tuple(c["fill"]) for c in cells_t0}
        fills_t1 = {round(c["value"]): tuple(c["fill"]) for c in cells_t1}
        assert fills_t0 == fills_t1, (
            "Same value with same vmin/vmax must produce identical colours across time steps"
        )


# ---------------------------------------------------------------------------
# OD5: Fallback lat/lon resolution (1-D and 2-D)
# ---------------------------------------------------------------------------


class TestNetcdfOverlayLatLonResolution:
    def test_1d_latlon_coords_resolved(self, tmp_path):
        """1-D lat/lon coordinates must be expanded to 2-D correctly."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "1d.nc"
        lat_1d = np.array([45.0, 46.0, 47.0])
        lon_1d = np.array([1.0, 2.0, 3.0, 4.0])
        data = np.ones((3, 4))
        ds = xr.Dataset(
            {"val": (["y", "x"], data)},
            coords={"lat": ("y", lat_1d), "lon": ("x", lon_1d)},
        )
        ds.to_netcdf(p)
        cells = load_netcdf_overlay(p)
        assert cells is not None
        assert len(cells) == 12, "All 3×4 cells must be returned"

    def test_2d_latlon_data_vars_resolved(self, tmp_path):
        """latitude/longitude stored as data variables (not coords) must be detected."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "2d_datavar.nc"
        lat_2d = np.array([[45.0, 45.0], [46.0, 46.0]])
        lon_2d = np.array([[1.0, 2.0], [1.0, 2.0]])
        ds = xr.Dataset(
            {
                "biomass": (["y", "x"], np.array([[1.0, 2.0], [3.0, 4.0]])),
                "latitude": (["y", "x"], lat_2d),
                "longitude": (["y", "x"], lon_2d),
            }
        )
        ds.to_netcdf(p)
        cells = load_netcdf_overlay(p)
        assert cells is not None and len(cells) == 4

    def test_missing_latlon_with_fallback(self, tmp_path):
        """When the file has no lat/lon, the fallback arrays must be used."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "no_latlon.nc"
        ds = xr.Dataset({"value": (["y", "x"], np.ones((2, 3)))})
        ds.to_netcdf(p)

        fb_lat = np.array([[45.0, 45.0, 45.0], [46.0, 46.0, 46.0]])
        fb_lon = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        cells = load_netcdf_overlay(p, fallback_lat=fb_lat, fallback_lon=fb_lon)
        assert cells is not None and len(cells) == 6

    def test_missing_latlon_no_fallback_returns_none(self, tmp_path):
        """Without lat/lon and no fallback, None must be returned."""
        from ui.pages.grid_helpers import load_netcdf_overlay

        p = tmp_path / "no_latlon.nc"
        ds = xr.Dataset({"value": (["y", "x"], np.ones((2, 3)))})
        ds.to_netcdf(p)
        cells = load_netcdf_overlay(p)
        assert cells is None


# ---------------------------------------------------------------------------
# OD6: list_nc_overlay_variables
# ---------------------------------------------------------------------------


class TestListNcOverlayVariables:
    def test_returns_expected_variables(self, tmp_path):
        """All non-coordinate data vars must appear in the result."""
        from ui.pages.grid_helpers import list_nc_overlay_variables

        p = tmp_path / "multi.nc"
        _make_ltl_nc(p, n_time=3, ny=2, nx=2)
        meta = list_nc_overlay_variables(str(p))
        assert meta is not None
        assert "Diatoms" in meta
        assert "Dinoflagellates" in meta
        # latitude and longitude are coordinate-like and must be excluded
        assert "latitude" not in meta
        assert "longitude" not in meta

    def test_n_time_is_correct(self, tmp_path):
        """n_time must match the file's time dimension length."""
        from ui.pages.grid_helpers import list_nc_overlay_variables

        p = tmp_path / "time.nc"
        _make_ltl_nc(p, n_time=12, ny=2, nx=2)
        meta = list_nc_overlay_variables(str(p))
        assert meta is not None
        for var_meta in meta.values():
            assert var_meta["n_time"] == 12
            assert var_meta["has_time"] is True

    def test_static_var_has_n_time_1(self, tmp_path):
        """A variable without a time dimension must report n_time=1, has_time=False."""
        from ui.pages.grid_helpers import list_nc_overlay_variables

        p = tmp_path / "static.nc"
        _make_static_nc(p, ny=3, nx=3)
        meta = list_nc_overlay_variables(str(p))
        assert meta is not None
        assert "temperature" in meta
        t_meta = meta["temperature"]
        assert t_meta["n_time"] == 1
        assert t_meta["has_time"] is False

    def test_vmin_vmax_cover_full_time_range(self, tmp_path):
        """vmin/vmax must be computed across ALL time steps, not just t=0."""
        from ui.pages.grid_helpers import list_nc_overlay_variables

        p = tmp_path / "range.nc"
        lat = np.array([[45.0, 45.0], [46.0, 46.0]])
        lon = np.array([[1.0, 2.0], [1.0, 2.0]])
        # t=0: values 0–1; t=1: values 500–501 — vmax must be ≥ 500
        data = np.array([[[0.0, 1.0], [0.5, 0.75]], [[500.0, 500.5], [500.25, 501.0]]])
        ds = xr.Dataset(
            {"v": (["time", "y", "x"], data)},
            coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
        )
        ds.to_netcdf(p)
        meta = list_nc_overlay_variables(str(p))
        assert meta is not None
        assert "v" in meta
        assert meta["v"]["vmin"] == pytest.approx(0.0, abs=1e-6)
        assert meta["v"]["vmax"] == pytest.approx(501.0, abs=1e-6)

    def test_returns_none_for_nonexistent_file(self):
        """A missing file path must return None without raising."""
        from ui.pages.grid_helpers import list_nc_overlay_variables

        meta = list_nc_overlay_variables("/nonexistent/path/to/file.nc")
        assert meta is None

    def test_returns_none_when_no_spatial_vars(self, tmp_path):
        """A file with only 1-D variables must return None."""
        from ui.pages.grid_helpers import list_nc_overlay_variables

        p = tmp_path / "scalar.nc"
        ds = xr.Dataset({"total": (["time"], [1.0, 2.0, 3.0])})
        ds.to_netcdf(p)
        meta = list_nc_overlay_variables(str(p))
        assert meta is None


# ---------------------------------------------------------------------------
# OD7: _overlay_label
# ---------------------------------------------------------------------------


class TestOverlayLabel:
    def test_ltl_biomass_label(self):
        from ui.pages.grid_helpers import _overlay_label

        assert _overlay_label("ltl/eec_ltlbiomassTons.nc") == "LTL Biomass"

    def test_background_species_label(self):
        from ui.pages.grid_helpers import _overlay_label

        assert _overlay_label("eec_backgroundspecies_biomass.nc") == "Background Species"

    def test_fishing_distrib_label(self):
        from ui.pages.grid_helpers import _overlay_label

        label = _overlay_label("fishing/fishing-distrib.csv")
        assert "ishing" in label  # "Fishing Distribution" or similar

    def test_mpa_pattern_label(self):
        from ui.pages.grid_helpers import _overlay_label

        label = _overlay_label("mpa/full_mpa.csv")
        # Must mention MPA or the filename content
        assert label != ""

    def test_generic_fallback_is_titlecase(self):
        from ui.pages.grid_helpers import _overlay_label

        label = _overlay_label("data/some_custom_file.nc")
        # Falls back to titlecase stem
        assert label == label.title() or label[0].isupper()

    def test_no_crash_on_bare_filename(self):
        from ui.pages.grid_helpers import _overlay_label

        label = _overlay_label("plain.nc")
        assert isinstance(label, str) and len(label) > 0


# ---------------------------------------------------------------------------
# OD8: Overlay catalog — deduplication and exclusions (code structure tests)
# ---------------------------------------------------------------------------


class TestOverlayCatalogStructure:
    """Inspect grid_overlay_selector source to verify catalog-building rules."""

    @pytest.fixture(autouse=True)
    def _src(self):
        import ast

        src_path = pathlib.Path(__file__).parent.parent / "ui" / "pages" / "grid.py"
        self._text = src_path.read_text()
        self._tree = ast.parse(self._text)

    def _selector_source(self) -> str:
        import ast

        for node in ast.walk(self._tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "grid_overlay_selector":
                    seg = ast.get_source_segment(self._text, node)
                    assert seg is not None
                    return seg
        pytest.fail("grid_overlay_selector not found in grid.py")

    def test_deduplication_uses_seen_paths(self):
        """Selector must use a 'seen_paths' dict keyed by canonical path."""
        src = self._selector_source()
        assert "seen_paths" in src, "Selector must deduplicate via seen_paths"

    def test_movement_map_keys_excluded(self):
        """movement.file.map keys must be in skip_prefixes."""
        src = self._selector_source()
        assert "movement.file.map" in src, "movement.file.mapN keys must appear in skip_prefixes"

    def test_resolved_path_is_overlay_id(self):
        """The overlay id passed to input_select must be the resolved path, not config key."""
        src = self._selector_source()
        # The path ID stored in seen_paths must come from resolved()
        assert "resolve()" in src or ".resolve()" in src, (
            "Selector must use .resolve() to create canonical path IDs"
        )

    def test_mpa_dir_scanned(self):
        """Selector must scan a 'mpa' directory for additional CSV overlays."""
        src = self._selector_source()
        assert "mpa" in src, "mpa/ directory scan must be present in selector"

    def test_overlay_nc_controls_present_in_grid_ui(self):
        """grid_ui must include output_ui('overlay_nc_controls')."""
        assert "overlay_nc_controls" in self._text, "grid_ui must render overlay_nc_controls"


# ---------------------------------------------------------------------------
# OD9: EEC integration tests (skipped if files not present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _EEC_AVAILABLE, reason="EEC example files not found")
class TestEecIntegration:
    def test_ltl_variables_detected(self):
        """EEC LTL file must expose 10 plankton variables."""
        from ui.pages.grid_helpers import list_nc_overlay_variables

        meta = list_nc_overlay_variables(str(_EEC_LTL))
        assert meta is not None
        assert len(meta) == 10, f"Expected 10 LTL vars, got {len(meta)}: {list(meta)}"

    def test_ltl_expected_variable_names(self):
        """Known LTL variable names must all be present."""
        from ui.pages.grid_helpers import list_nc_overlay_variables

        meta = list_nc_overlay_variables(str(_EEC_LTL))
        assert meta is not None
        for name in ("Dinoflagellates", "Diatoms", "Microzoo", "Mesozoo", "Macrozoo"):
            assert name in meta, f"Expected variable {name!r} in LTL metadata"

    def test_ltl_has_24_time_steps(self):
        """EEC LTL must have 24 monthly time steps."""
        from ui.pages.grid_helpers import list_nc_overlay_variables

        meta = list_nc_overlay_variables(str(_EEC_LTL))
        assert meta is not None
        for vn, vm in meta.items():
            assert vm["n_time"] == 24, f"{vn}: expected n_time=24, got {vm['n_time']}"
            assert vm["has_time"] is True

    def test_ltl_cell_count_matches_ocean_cells(self):
        """EEC LTL overlay must return 464 ocean cells (22×45 minus land)."""
        from ui.pages.grid_helpers import list_nc_overlay_variables, load_netcdf_overlay

        meta = list_nc_overlay_variables(str(_EEC_LTL))
        assert meta is not None
        var = "Dinoflagellates"
        vm = meta[var]
        cells = load_netcdf_overlay(
            _EEC_LTL, var_name=var, time_step=0, vmin=vm["vmin"], vmax=vm["vmax"]
        )
        assert cells is not None
        assert len(cells) == 464, f"Expected 464 ocean cells, got {len(cells)}"

    def test_ltl_vmin_vmax_span_full_range(self):
        """vmin/vmax must cover the full range across all 24 time steps."""
        from ui.pages.grid_helpers import list_nc_overlay_variables
        import xarray as xr

        meta = list_nc_overlay_variables(str(_EEC_LTL))
        assert meta is not None
        with xr.open_dataset(_EEC_LTL) as ds:
            for vn, vm in meta.items():
                arr = ds[vn].values.astype(float)
                actual_min = float(np.nanmin(arr))
                actual_max = float(np.nanmax(arr))
                assert vm["vmin"] == pytest.approx(actual_min, rel=1e-5)
                assert vm["vmax"] == pytest.approx(actual_max, rel=1e-5)

    def test_ltl_colors_differ_between_time_steps(self):
        """With seasonal variation, different time steps must yield different cell fills."""
        from ui.pages.grid_helpers import list_nc_overlay_variables, load_netcdf_overlay

        meta = list_nc_overlay_variables(str(_EEC_LTL))
        assert meta is not None
        vm = meta["Diatoms"]
        cells_t0 = load_netcdf_overlay(
            _EEC_LTL, var_name="Diatoms", time_step=0, vmin=vm["vmin"], vmax=vm["vmax"]
        )
        cells_t6 = load_netcdf_overlay(
            _EEC_LTL, var_name="Diatoms", time_step=6, vmin=vm["vmin"], vmax=vm["vmax"]
        )
        assert cells_t0 is not None and cells_t6 is not None
        fills_t0 = {tuple(c["fill"]) for c in cells_t0}
        fills_t6 = {tuple(c["fill"]) for c in cells_t6}
        # Should have some different colours due to seasonal biomass changes
        assert fills_t0 != fills_t6, (
            "Diatoms at t=0 and t=6 must have different fills (seasonal variation)"
        )

    def test_background_species_detected(self):
        """EEC background species file must expose the 'backgroundSpecies' variable."""
        from ui.pages.grid_helpers import list_nc_overlay_variables

        meta = list_nc_overlay_variables(str(_EEC_BG))
        assert meta is not None
        assert "backgroundSpecies" in meta

    def test_background_species_cell_count(self):
        """EEC background species overlay must return 464 cells."""
        from ui.pages.grid_helpers import list_nc_overlay_variables, load_netcdf_overlay

        meta = list_nc_overlay_variables(str(_EEC_BG))
        assert meta is not None
        vm = meta["backgroundSpecies"]
        cells = load_netcdf_overlay(
            _EEC_BG, var_name="backgroundSpecies", time_step=0, vmin=vm["vmin"], vmax=vm["vmax"]
        )
        assert cells is not None
        assert len(cells) == 464

    def test_ltl_all_cells_have_valid_fill(self):
        """Every cell from an EEC overlay must have a valid RGBA fill."""
        from ui.pages.grid_helpers import list_nc_overlay_variables, load_netcdf_overlay

        meta = list_nc_overlay_variables(str(_EEC_LTL))
        assert meta is not None
        vm = meta["Dinoflagellates"]
        cells = load_netcdf_overlay(
            _EEC_LTL, var_name="Dinoflagellates", time_step=3, vmin=vm["vmin"], vmax=vm["vmax"]
        )
        assert cells is not None
        for cell in cells:
            assert "fill" in cell
            assert len(cell["fill"]) == 4
            assert all(0 <= ch <= 255 for ch in cell["fill"]), (
                f"RGBA values out of range: {cell['fill']}"
            )


# ---------------------------------------------------------------------------
# OD10: EEC Full CSV overlay + movement cache (semicolon CSVs)
# ---------------------------------------------------------------------------

_EEC_FULL_DIR = pathlib.Path(__file__).parent.parent / "data" / "eec_full"
_EEC_FULL_AVAILABLE = (_EEC_FULL_DIR / "eec_all-parameters.csv").exists()


@pytest.mark.skipif(not _EEC_FULL_AVAILABLE, reason="EEC Full example files not found")
class TestEecFullCsvIntegration:
    """Tests that semicolon-separated OSMOSE CSVs load correctly with EEC Full."""

    @pytest.fixture(autouse=True)
    def _load_eec_full(self):
        from osmose.config.reader import OsmoseConfigReader
        from ui.pages.grid_helpers import load_netcdf_grid

        reader = OsmoseConfigReader()
        self.cfg = reader.read(_EEC_FULL_DIR / "eec_all-parameters.csv")
        self.cfg_dir = _EEC_FULL_DIR
        self.nc_data = load_netcdf_grid(self.cfg, config_dir=self.cfg_dir)
        assert self.nc_data is not None
        self.lat, self.lon, self.mask = self.nc_data

    def test_fishing_distrib_csv_loads(self):
        """EEC fishing-distrib.csv (semicolon) must produce 460 ocean cells."""
        from ui.pages.grid_helpers import load_csv_overlay

        fish_path = self.cfg_dir / "fishing" / "fishing-distrib.csv"
        cells = load_csv_overlay(fish_path, 0, 0, 0, 0, 0, 0, nc_data=self.nc_data)
        assert cells is not None, "Fishing distrib CSV failed to load"
        # EEC has 460 ocean cells; fishing distrib has NA for land cells
        assert len(cells) > 400, f"Expected >400 cells, got {len(cells)}"

    def test_fishing_distrib_values_are_numeric(self):
        """Fishing distrib values must be numeric floats, not strings."""
        from ui.pages.grid_helpers import load_csv_overlay

        fish_path = self.cfg_dir / "fishing" / "fishing-distrib.csv"
        cells = load_csv_overlay(fish_path, 0, 0, 0, 0, 0, 0, nc_data=self.nc_data)
        assert cells is not None
        for c in cells:
            assert isinstance(c["value"], float), f"Expected float, got {type(c['value'])}"

    def test_movement_cache_cod_has_maps(self):
        """build_movement_cache for 'cod' must return 4 movement maps."""
        from ui.pages.grid_helpers import build_movement_cache

        ul_lat, lr_lat = float(self.lat.max()), float(self.lat.min())
        ul_lon, lr_lon = float(self.lon.min()), float(self.lon.max())
        ny, nx = self.lat.shape
        cache = build_movement_cache(
            self.cfg, self.cfg_dir, (ul_lat, ul_lon, lr_lat, lr_lon, nx, ny), species="cod"
        )
        assert len(cache) == 4, f"Expected 4 cod maps, got {len(cache)}: {list(cache)}"

    def test_movement_cache_maps_have_cells(self):
        """Each cached movement map must contain valid cell data."""
        from ui.pages.grid_helpers import build_movement_cache

        ul_lat, lr_lat = float(self.lat.max()), float(self.lat.min())
        ul_lon, lr_lon = float(self.lon.min()), float(self.lon.max())
        ny, nx = self.lat.shape
        cache = build_movement_cache(
            self.cfg, self.cfg_dir, (ul_lat, ul_lon, lr_lat, lr_lon, nx, ny), species="cod"
        )
        for mid, m in cache.items():
            assert len(m["cells"]) > 0, f"Map {mid} has no cells"
            assert m["steps"], f"Map {mid} has no time steps"
            assert m["label"], f"Map {mid} has no label"
            assert m["color"] and len(m["color"]) == 4, f"Map {mid} missing RGBA color"

    def test_movement_species_list_complete(self):
        """list_movement_species must find all 14 EEC species with movement maps."""
        from ui.pages.grid_helpers import list_movement_species

        species = list_movement_species(self.cfg)
        assert len(species) == 14, f"Expected 14 species, got {len(species)}: {species}"

    def test_all_movement_species_produce_caches(self):
        """Every species with movement maps must produce a non-empty cache."""
        from ui.pages.grid_helpers import build_movement_cache, list_movement_species

        ul_lat, lr_lat = float(self.lat.max()), float(self.lat.min())
        ul_lon, lr_lon = float(self.lon.min()), float(self.lon.max())
        ny, nx = self.lat.shape
        grid_params = (ul_lat, ul_lon, lr_lat, lr_lon, nx, ny)

        species = list_movement_species(self.cfg)
        for sp in species:
            cache = build_movement_cache(self.cfg, self.cfg_dir, grid_params, species=sp)
            assert len(cache) > 0, f"Species '{sp}' produced empty movement cache"

    def test_cod_nurseries_csv_shape_matches_grid(self):
        """A specific EEC movement map must parse to the correct grid dimensions."""
        from ui.pages.grid_helpers import _read_csv_auto_sep

        p = self.cfg_dir / "maps" / "6cod_nurseries.csv"
        df = _read_csv_auto_sep(p)
        ny, nx = self.lat.shape
        assert df.shape == (ny, nx), (
            f"6cod_nurseries.csv shape {df.shape} doesn't match grid ({ny}, {nx})"
        )


def test_csv_overlay_output_stability(tmp_path):
    """Cell-by-cell output must be identical before and after vectorization."""
    import pandas as pd

    from ui.pages.grid_helpers import load_csv_overlay

    p = tmp_path / "stability.csv"
    np.random.seed(42)
    data = np.random.rand(5, 6) * 10
    data[0, :3] = -99  # some sentinels
    data[2, 2] = 0.0  # zero (filtered)
    data[3, 0] = -9.0  # boundary value (should be KEPT)
    pd.DataFrame(data).to_csv(p, sep=";", header=False, index=False)

    cells = load_csv_overlay(p, ul_lat=50.0, ul_lon=-5.0, lr_lat=45.0, lr_lon=5.0, nx=6, ny=5)
    assert cells is not None

    vals = [c["value"] for c in cells]
    assert -9.0 in vals, "-9.0 must be kept (not a sentinel)"

    snapshot = sorted((c["value"], tuple(c["polygon"][0])) for c in cells)
    cells2 = load_csv_overlay(p, ul_lat=50.0, ul_lon=-5.0, lr_lat=45.0, lr_lon=5.0, nx=6, ny=5)
    snapshot2 = sorted((c["value"], tuple(c["polygon"][0])) for c in cells2)
    assert snapshot == snapshot2, "Output must be deterministic"


def test_csv_overlay_rgba_range(tmp_path):
    """All RGBA channels must be in [0, 255] after vectorization (np.clip fix)."""
    import pandas as pd

    from ui.pages.grid_helpers import load_csv_overlay

    p = tmp_path / "range_check.csv"
    np.random.seed(99)
    data = np.random.rand(10, 12) * 20 - 5
    data[0, :3] = -99
    pd.DataFrame(data).to_csv(p, sep=";", header=False, index=False)

    cells = load_csv_overlay(p, ul_lat=50.0, ul_lon=-5.0, lr_lat=45.0, lr_lon=5.0, nx=12, ny=10)
    assert cells is not None
    for c in cells:
        r, g, b, a = c["fill"]
        assert 0 <= r <= 255, f"red {r} out of range"
        assert 0 <= g <= 255, f"green {g} out of range"
        assert 0 <= b <= 255, f"blue {b} out of range"
        assert 0 <= a <= 255, f"alpha {a} out of range"
