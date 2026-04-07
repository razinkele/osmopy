"""Pure helper functions for the Grid configuration page.

These functions are free of Shiny reactive context (no ``input``, ``state``,
``ui.notification_show``, or ``reactive.*``) and can be imported/tested
independently of the Shiny runtime.
"""

import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from osmose.logging import setup_logging

_log = setup_logging("osmose.grid.helpers")


def _safe_resolve(base: Path, rel: str) -> Path | None:
    """Resolve a relative path against base, rejecting traversal outside base."""
    candidate = (base / rel).resolve()
    if not candidate.is_relative_to(base.resolve()):
        return None
    return candidate


def _read_csv_auto_sep(path: Path) -> pd.DataFrame:
    """Read a CSV file, auto-detecting semicolon vs comma separator.

    OSMOSE spatial CSV files use semicolons; test fixtures and third-party
    files may use commas.  Try semicolon first (OSMOSE convention), fall back
    to comma if the result has only one column.
    """
    df = pd.read_csv(path, header=None, sep=";")
    if df.shape[1] == 1:
        # Single column — likely comma-separated (or truly 1-column data).
        # Re-read with comma; if still 1 column, it really is 1-column data.
        df_comma = pd.read_csv(path, header=None, sep=",")
        if df_comma.shape[1] > 1:
            return df_comma
    return df


def load_mask(config: dict[str, str], config_dir: Path | None = None) -> np.ndarray | None:
    """Load a grid mask CSV from the config if available."""
    mask_path = config.get("grid.mask.file", "")
    if not mask_path:
        return None

    # Try config_dir first, fall back to examples directory
    search_dirs = []
    if config_dir and config_dir.is_dir():
        search_dirs.append(config_dir)
    search_dirs.append(Path(__file__).parent.parent.parent / "data" / "examples")

    full_path = None
    for d in search_dirs:
        candidate = _safe_resolve(d, mask_path)
        if candidate is None:
            _log.warning("Path traversal rejected for root %s: %s", d, mask_path)
            continue
        if candidate.exists():
            full_path = candidate
            break

    if full_path is None:
        _log.warning("Mask file not found: %s", mask_path)
        return None

    try:
        return _read_csv_auto_sep(full_path).values
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        _log.warning("Mask file %s could not be parsed: %s", full_path, exc)
        return None
    except FileNotFoundError:
        _log.debug("Mask file not found: %s", full_path)
        return None
    except (OSError, ValueError) as exc:
        _log.warning("Failed to load mask %s: %s", full_path, exc)
        return None


def _zoom_for_span(span: float, default: float = 5.0) -> float:
    """Compute deck.gl zoom level to fit a geographic span in degrees."""
    if span <= 0:
        return default
    return max(1.0, min(15.0, math.log2(360 / span) - 0.5))


def build_grid_layers(
    ul_lat: float,
    ul_lon: float,
    lr_lat: float,
    lr_lon: float,
    nx: int,
    ny: int,
    is_dark: bool = False,
    mask: np.ndarray | None = None,
) -> list[dict]:
    """Build deck.gl layers for the grid preview.

    Uses PolygonLayer with exact lat/lon rectangles per cell —
    handles non-square grids correctly in geographic coordinates.
    """
    from shiny_deckgl import polygon_layer  # type: ignore[import-untyped]

    layers = []

    if ul_lat == 0 and ul_lon == 0 and lr_lat == 0 and lr_lon == 0:
        return layers

    # Grid boundary outline
    boundary = [
        {
            "polygon": [
                [ul_lon, ul_lat],
                [lr_lon, ul_lat],
                [lr_lon, lr_lat],
                [ul_lon, lr_lat],
            ],
        }
    ]
    layers.append(
        polygon_layer(
            "grid-extent",
            data=boundary,
            get_polygon="@@=d.polygon",
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[232, 168, 56, 220],
            get_line_width=2,
            line_width_min_pixels=2,
            filled=False,
            stroked=True,
            pickable=False,
        )
    )

    # Individual grid cells as lat/lon rectangles — vectorised with NumPy meshgrid
    if nx > 0 and ny > 0:
        dx = (lr_lon - ul_lon) / nx
        dy = (ul_lat - lr_lat) / ny

        col_arr = np.arange(nx)
        row_arr = np.arange(ny)
        lo0 = ul_lon + col_arr * dx  # (nx,)
        la0 = ul_lat - row_arr * dy  # (ny,)
        lo0g, la0g = np.meshgrid(lo0, la0)  # (ny, nx)
        lo1g = lo0g + dx
        la1g = la0g - dy

        if mask is not None:
            mny = min(mask.shape[0], ny)
            mnx = min(mask.shape[1], nx)
            land_g = np.zeros((ny, nx), dtype=bool)
            land_g[:mny, :mnx] = mask[:mny, :mnx] <= 0
        else:
            land_g = np.zeros((ny, nx), dtype=bool)

        lo0f = lo0g.ravel()
        lo1f = lo1g.ravel()
        la0f = la0g.ravel()
        la1f = la1g.ravel()
        rows_f = np.repeat(row_arr, nx)
        cols_f = np.tile(col_arr, ny)
        land_f = land_g.ravel()

        ocean_cells = [
            {
                "polygon": [
                    [float(lo0f[i]), float(la0f[i])],
                    [float(lo1f[i]), float(la0f[i])],
                    [float(lo1f[i]), float(la1f[i])],
                    [float(lo0f[i]), float(la1f[i])],
                ],
                "row": int(rows_f[i]),
                "col": int(cols_f[i]),
                "type": "ocean",
            }
            for i in range(len(land_f))
            if not land_f[i]
        ]
        land_cells = [
            {
                "polygon": [
                    [float(lo0f[i]), float(la0f[i])],
                    [float(lo1f[i]), float(la0f[i])],
                    [float(lo1f[i]), float(la1f[i])],
                    [float(lo0f[i]), float(la1f[i])],
                ],
                "row": int(rows_f[i]),
                "col": int(cols_f[i]),
                "type": "land",
            }
            for i in range(len(land_f))
            if land_f[i]
        ]

        # Ocean cells
        if ocean_cells:
            fill = [30, 120, 180, 90] if is_dark else [20, 100, 180, 70]
            stroke = [56, 201, 177, 140] if is_dark else [43, 168, 158, 120]
            layers.append(
                polygon_layer(
                    "grid-ocean",
                    data=ocean_cells,
                    get_polygon="@@=d.polygon",
                    get_fill_color=fill,
                    get_line_color=stroke,
                    get_line_width=1,
                    line_width_min_pixels=1,
                    filled=True,
                    stroked=True,
                    pickable=True,
                )
            )

        # Land cells
        if land_cells:
            fill = [80, 65, 45, 100] if is_dark else [190, 170, 140, 90]
            stroke = [100, 80, 60, 100] if is_dark else [160, 140, 110, 80]
            layers.append(
                polygon_layer(
                    "grid-land",
                    data=land_cells,
                    get_polygon="@@=d.polygon",
                    get_fill_color=fill,
                    get_line_color=stroke,
                    get_line_width=1,
                    line_width_min_pixels=1,
                    filled=True,
                    stroked=True,
                    pickable=True,
                )
            )

    return layers


def load_netcdf_grid(
    config: dict[str, str],
    config_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load lat, lon, mask arrays from a NetCDF grid file.

    Returns (lat, lon, mask) 2D arrays or None if unavailable.
    """
    nc_path = config.get("grid.netcdf.file", "")
    if not nc_path:
        return None

    search_dirs = []
    if config_dir and config_dir.is_dir():
        search_dirs.append(config_dir)
    search_dirs.append(Path(__file__).parent.parent.parent / "data" / "examples")

    full_path = None
    for d in search_dirs:
        candidate = _safe_resolve(d, nc_path)
        if candidate is None:
            _log.warning("Path traversal rejected for root %s: %s", d, nc_path)
            continue
        if candidate.exists():
            full_path = candidate
            break

    if full_path is None:
        _log.debug("NetCDF grid file not found: %s", nc_path)
        return None

    try:
        var_lat = config.get("grid.var.lat", "lat")
        var_lon = config.get("grid.var.lon", "lon")
        var_mask = config.get("grid.var.mask", "mask")
        with xr.open_dataset(full_path) as ds:
            lat = ds[var_lat].values
            lon = ds[var_lon].values
            mask = ds[var_mask].values
        return lat, lon, mask
    except (OSError, KeyError) as exc:
        _log.warning("Failed to load NetCDF grid %s: %s", full_path, exc)
        return None


def build_netcdf_grid_layers(
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
    is_dark: bool = False,
) -> tuple[list[dict], dict]:
    """Build deck.gl layers from 2D NetCDF grid arrays.

    Returns (layers, view_state) tuple.
    """
    from shiny_deckgl import polygon_layer  # type: ignore[import-untyped]

    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    ny, nx = lat.shape
    layers = []

    # Precompute cell step sizes (handle single-row/col degenerate grids)
    lat_step = abs(float(lat[0, 0] - lat[min(1, ny - 1), 0]))
    lon_step = abs(float(lon[0, 0] - lon[0, min(1, nx - 1)]))
    if lat_step == 0:
        lat_step = lon_step if lon_step > 0 else 1.0
    if lon_step == 0:
        lon_step = lat_step if lat_step > 0 else 1.0

    # Compute cell boundaries from center points (half-step between neighbors)
    def _cell_polygon(row: int, col: int) -> list[list[float]]:
        """Build polygon corners for a cell from neighboring center points."""
        clat = float(lat[row, col])
        clon = float(lon[row, col])
        # Half-step to neighbors (or mirror at edges)
        dlat = float(lat[min(row + 1, ny - 1), col] - lat[max(row - 1, 0), col]) / 2
        dlon = float(lon[row, min(col + 1, nx - 1)] - lon[row, max(col - 1, 0)]) / 2
        if row == 0 or row == ny - 1:
            dlat *= 2
        if col == 0 or col == nx - 1:
            dlon *= 2
        # Fallback for degenerate single-row/col grids
        if dlat == 0:
            dlat = lon_step * (1 if lat[0, 0] >= lat[min(1, ny - 1), 0] else -1)
        if dlon == 0:
            dlon = lat_step
        hlat = abs(dlat) / 2
        hlon = abs(dlon) / 2
        return [
            [clon - hlon, clat + hlat],
            [clon + hlon, clat + hlat],
            [clon + hlon, clat - hlat],
            [clon - hlon, clat - hlat],
        ]

    # Grid boundary from outer edges (uses precomputed steps — safe for single-row/col)
    ul_lat = float(lat.max()) + lat_step / 2
    ul_lon = float(lon.min()) - lon_step / 2
    lr_lat = float(lat.min()) - lat_step / 2
    lr_lon = float(lon.max()) + lon_step / 2

    boundary = [
        {
            "polygon": [
                [ul_lon, ul_lat],
                [lr_lon, ul_lat],
                [lr_lon, lr_lat],
                [ul_lon, lr_lat],
            ]
        }
    ]
    layers.append(
        polygon_layer(
            "grid-extent",
            data=boundary,
            get_polygon="@@=d.polygon",
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[232, 168, 56, 220],
            get_line_width=2,
            line_width_min_pixels=2,
            filled=False,
            stroked=True,
            pickable=False,
        )
    )

    ocean_cells = []
    land_cells = []

    # Vectorized polygon computation — avoids O(ny*nx) Python loop for large grids.
    lat_f = lat.astype(float)
    lon_f = lon.astype(float)

    # Compute per-cell half-steps using finite differences (edge cells mirror).
    dlat = np.empty_like(lat_f)
    if ny == 1:
        dlat[:] = lat_step
    else:
        dlat[0, :] = lat_f[1, :] - lat_f[0, :]
        dlat[-1, :] = lat_f[-1, :] - lat_f[-2, :]
        if ny > 2:
            dlat[1:-1, :] = (lat_f[2:, :] - lat_f[:-2, :]) / 2

    dlon = np.empty_like(lon_f)
    if nx == 1:
        dlon[:] = lon_step
    else:
        dlon[:, 0] = lon_f[:, 1] - lon_f[:, 0]
        dlon[:, -1] = lon_f[:, -1] - lon_f[:, -2]
        if nx > 2:
            dlon[:, 1:-1] = (lon_f[:, 2:] - lon_f[:, :-2]) / 2

    # Fallback for degenerate zero-step cells
    lat_sign = 1.0 if lat_f[0, 0] >= lat_f[min(1, ny - 1), 0] else -1.0
    fallback_dlat = float(lon_step * lat_sign) or lat_step
    dlat = np.where(dlat == 0, fallback_dlat, dlat)
    dlon = np.where(dlon == 0, lat_step, dlon)

    hlat = np.abs(dlat) / 2
    hlon = np.abs(dlon) / 2

    # Build polygon corner arrays: shape (ny, nx, 4, 2) in [lon, lat] order
    polys = np.stack(
        [
            np.stack([lon_f - hlon, lat_f + hlat], axis=-1),  # top-left
            np.stack([lon_f + hlon, lat_f + hlat], axis=-1),  # top-right
            np.stack([lon_f + hlon, lat_f - hlat], axis=-1),  # bottom-right
            np.stack([lon_f - hlon, lat_f - hlat], axis=-1),  # bottom-left
        ],
        axis=2,
    )
    polys_list = polys.tolist()  # nested Python lists for JSON serialization

    ocean_mask = mask > 0
    for r, c in zip(*np.where(ocean_mask)):
        ocean_cells.append(
            {"polygon": polys_list[r][c], "row": int(r), "col": int(c), "type": "ocean"}
        )
    for r, c in zip(*np.where(~ocean_mask)):
        land_cells.append(
            {"polygon": polys_list[r][c], "row": int(r), "col": int(c), "type": "land"}
        )

    if ocean_cells:
        fill = [30, 120, 180, 90] if is_dark else [20, 100, 180, 70]
        stroke = [56, 201, 177, 140] if is_dark else [43, 168, 158, 120]
        layers.append(
            polygon_layer(
                "grid-ocean",
                data=ocean_cells,
                get_polygon="@@=d.polygon",
                get_fill_color=fill,
                get_line_color=stroke,
                get_line_width=1,
                line_width_min_pixels=1,
                filled=True,
                stroked=True,
                pickable=True,
            )
        )

    if land_cells:
        fill = [80, 65, 45, 100] if is_dark else [190, 170, 140, 90]
        stroke = [100, 80, 60, 100] if is_dark else [160, 140, 110, 80]
        layers.append(
            polygon_layer(
                "grid-land",
                data=land_cells,
                get_polygon="@@=d.polygon",
                get_fill_color=fill,
                get_line_color=stroke,
                get_line_width=1,
                line_width_min_pixels=1,
                filled=True,
                stroked=True,
                pickable=True,
            )
        )

    center_lat = (ul_lat + lr_lat) / 2
    center_lon = (ul_lon + lr_lon) / 2
    span = max(abs(ul_lat - lr_lat), abs(lr_lon - ul_lon))
    zoom = _zoom_for_span(span)
    view_state = {"latitude": center_lat, "longitude": center_lon, "zoom": zoom}

    return layers, view_state


def load_csv_overlay(
    file_path: Path,
    ul_lat: float,
    ul_lon: float,
    lr_lat: float,
    lr_lon: float,
    nx: int,
    ny: int,
    nc_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[dict] | None:
    """Load a CSV spatial file and map values onto grid cells for deck.gl.

    OSMOSE CSV spatial files are typically 2D matrices (ny x nx) with one value
    per grid cell, or 1D column vectors with one value per cell in row-major order.
    Values are mapped onto the regular grid defined by the bounding box.
    """
    try:
        df = _read_csv_auto_sep(file_path)
        data = df.values

        # Determine grid dimensions to use
        if nc_data is not None:
            lat, lon = nc_data[0], nc_data[1]
            g_ny, g_nx = lat.shape
        elif nx > 0 and ny > 0:
            lat, lon = None, None
            g_ny, g_nx = ny, nx
        else:
            return None

        # Reshape 1D data to 2D if needed
        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
            flat = data.flatten()
            if len(flat) == g_ny * g_nx:
                data = flat.reshape(g_ny, g_nx)
            else:
                _log.warning(
                    "CSV %s has %d values but grid is %dx%d",
                    file_path,
                    len(flat),
                    g_ny,
                    g_nx,
                )
                return None
        elif data.shape != (g_ny, g_nx):
            # Try transposing
            if data.shape == (g_nx, g_ny):
                data = data.T
            else:
                _log.warning(
                    "CSV %s shape %s doesn't match grid %dx%d",
                    file_path,
                    data.shape,
                    g_ny,
                    g_nx,
                )
                return None

        # Compute value range for color scaling
        # -99 (and values < -9) are OSMOSE sentinel values meaning "outside grid/land" — skip them.
        numeric = data.astype(float)
        valid = numeric[(~np.isnan(numeric)) & (numeric > -9.0)]
        if len(valid) == 0:
            return None
        vmin, vmax = float(valid.min()), float(valid.max())
        vrange = vmax - vmin if vmax != vmin else 1.0

        cells = []
        for r in range(g_ny):
            for c in range(g_nx):
                v = float(numeric[r, c])
                if np.isnan(v) or v < -9.0:  # skip NaN and -99 sentinel (outside grid)
                    continue
                # Compute cell polygon from grid coordinates
                if nc_data is not None and lat is not None and lon is not None:
                    clat = float(lat[r, c] if lat.ndim == 2 else lat[r])
                    clon = float(lon[r, c] if lon.ndim == 2 else lon[c])
                    if lat.ndim == 2:
                        dlat = abs(float(lat[min(r + 1, g_ny - 1), c] - lat[max(r - 1, 0), c])) / 2
                        dlon = abs(float(lon[r, min(c + 1, g_nx - 1)] - lon[r, max(c - 1, 0)])) / 2
                    else:
                        dlat = abs(float(lat[min(r + 1, len(lat) - 1)] - lat[max(r - 1, 0)])) / 2
                        dlon = abs(float(lon[min(c + 1, len(lon) - 1)] - lon[max(c - 1, 0)])) / 2
                    if r == 0 or r == g_ny - 1:
                        dlat *= 2
                    if c == 0 or c == g_nx - 1:
                        dlon *= 2
                    hlat, hlon = dlat / 2, dlon / 2
                else:
                    # Regular grid from bounding box
                    dx = (lr_lon - ul_lon) / g_nx
                    dy = (ul_lat - lr_lat) / g_ny
                    clon = ul_lon + (c + 0.5) * dx
                    clat = ul_lat - (r + 0.5) * dy
                    hlon, hlat = dx / 2, dy / 2

                # Color: scale 0→1 to amber palette (dark→bright)
                t = (v - vmin) / vrange
                red = int(180 + 75 * t)
                green = int(80 + 60 * t)
                blue = 0
                alpha = int(100 + 100 * t)

                cells.append(
                    {
                        "polygon": [
                            [clon - hlon, clat + hlat],
                            [clon + hlon, clat + hlat],
                            [clon + hlon, clat - hlat],
                            [clon - hlon, clat - hlat],
                        ],
                        "value": v,
                        "fill": [red, green, blue, alpha],
                    }
                )
        return cells if cells else None
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        _log.warning("Failed to load CSV overlay %s: %s", file_path, exc)
        return None


def load_netcdf_overlay(
    file_path: Path,
    fallback_lat: np.ndarray | None = None,
    fallback_lon: np.ndarray | None = None,
    var_lat: str = "lat",
    var_lon: str = "lon",
    var_name: str | None = None,
    time_step: int = 0,
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[dict] | None:
    """Load a NetCDF overlay file and return colored cell data for deck.gl.

    Applies a blue→cyan→yellow colormap scaled against ``vmin``/``vmax``
    (which should be the global range across all time steps for consistent
    colour across animation).  If not provided, they are computed from the
    selected slice.

    Parameters
    ----------
    file_path:
        Path to the NetCDF file.
    fallback_lat/fallback_lon:
        Coordinate arrays to use when the file does not contain lat/lon vars
        (e.g. a 2-D NcGrid mask that shares coordinates with the base grid).
    var_lat/var_lon:
        Name of the latitude/longitude variables inside the file.
    var_name:
        Data variable to display.  If *None*, the first 2-D+ variable is used.
    time_step:
        Index along the leading time dimension (clamped to valid range).
    vmin/vmax:
        Fixed colour-scale bounds.  Pass the per-variable bounds from
        ``list_nc_overlay_variables`` for stable colours across time steps.
    """
    _TIME_DIM_NAMES = {"time", "t", "year", "month", "step", "date", "ntime"}

    try:
        with xr.open_dataset(file_path) as ds:
            # --- Select variable ---
            if var_name and var_name in ds.data_vars:
                sel_var = var_name
            else:
                sel_var = next(
                    (vn for vn in ds.data_vars if len(ds[vn].dims) >= 2),
                    None,
                )
            if not sel_var:
                return None

            da = ds[sel_var]

            # --- Slice time dimension if present ---
            has_time = len(da.dims) > 2 and da.dims[0].lower() in _TIME_DIM_NAMES
            if has_time:
                t_idx = max(0, min(time_step, da.shape[0] - 1))
                data_vals = da.values[t_idx]
            elif da.values.ndim > 2:
                data_vals = da.values[0]
            else:
                data_vals = da.values

            # --- Resolve lat/lon arrays (try several common names) ---
            _LAT_NAMES = (var_lat, "lat", "latitude", "nav_lat", "LAT", "Latitude")
            _LON_NAMES = (var_lon, "lon", "longitude", "nav_lon", "LON", "Longitude")
            olat = next((ds[n].values for n in _LAT_NAMES if n in ds), fallback_lat)
            olon = next((ds[n].values for n in _LON_NAMES if n in ds), fallback_lon)

        if olat is None or olon is None:
            return None

        data_f = data_vals.astype(float)
        ony, onx = data_f.shape

        # --- Expand 1-D lat/lon to 2-D ---
        if olat.ndim == 1:
            lat_2d = np.broadcast_to(olat[:, np.newaxis], (ony, onx)).copy()
            lon_2d = np.broadcast_to(olon[np.newaxis, :], (ony, onx)).copy()
        else:
            lat_2d = olat[:ony, :onx].astype(float)
            lon_2d = olon[:ony, :onx].astype(float)

        lat_step = abs(float(lat_2d[min(1, ony - 1), 0] - lat_2d[0, 0])) if ony > 1 else 1.0
        lon_step = abs(float(lon_2d[0, min(1, onx - 1)] - lon_2d[0, 0])) if onx > 1 else 1.0

        # Vectorised half-extents using finite differences (edge cells: single-sided)
        dlat = np.empty((ony, onx))
        if ony == 1:
            dlat[:] = lat_step
        else:
            dlat[0, :] = abs(lat_2d[1, :] - lat_2d[0, :])
            dlat[-1, :] = abs(lat_2d[-1, :] - lat_2d[-2, :])
            if ony > 2:
                dlat[1:-1, :] = abs(lat_2d[2:, :] - lat_2d[:-2, :]) / 2

        dlon = np.empty((ony, onx))
        if onx == 1:
            dlon[:] = lon_step
        else:
            dlon[:, 0] = abs(lon_2d[:, 1] - lon_2d[:, 0])
            dlon[:, -1] = abs(lon_2d[:, -1] - lon_2d[:, -2])
            if onx > 2:
                dlon[:, 1:-1] = abs(lon_2d[:, 2:] - lon_2d[:, :-2]) / 2

        dlat = np.where(dlat == 0, lon_step if lon_step > 0 else 1.0, dlat)
        dlon = np.where(dlon == 0, lat_step if lat_step > 0 else 1.0, dlon)
        hlat = dlat / 2
        hlon = dlon / 2

        # Build polygon corners: shape (ony, onx, 4, 2) in [lon, lat] order
        polys = np.stack(
            [
                np.stack([lon_2d - hlon, lat_2d + hlat], axis=-1),
                np.stack([lon_2d + hlon, lat_2d + hlat], axis=-1),
                np.stack([lon_2d + hlon, lat_2d - hlat], axis=-1),
                np.stack([lon_2d - hlon, lat_2d - hlat], axis=-1),
            ],
            axis=2,
        )
        polys_list = polys.tolist()

        # --- Colour scaling (computed only over valid cells to avoid NaN cast warnings) ---
        valid_mask = ~np.isnan(data_f)
        valid_vals = data_f[valid_mask]
        if len(valid_vals) == 0:
            return None
        vmin_eff = vmin if vmin is not None else float(valid_vals.min())
        vmax_eff = vmax if vmax is not None else float(valid_vals.max())
        vrange = vmax_eff - vmin_eff if vmax_eff != vmin_eff else 1.0

        # Blue→cyan→yellow colormap applied only to the valid subset
        t_valid = np.clip((valid_vals - vmin_eff) / vrange, 0.0, 1.0)
        lo = t_valid < 0.5
        s_lo = t_valid * 2
        s_hi = (t_valid - 0.5) * 2
        r_ch = np.where(lo, 0.0, s_hi * 255)
        g_ch = np.where(lo, 80.0 + s_lo * 130, 210.0 + s_hi * 20)
        b_ch = np.where(lo, 200.0 - s_lo * 20, 180.0 - s_hi * 180)
        a_ch = np.where(lo, 120.0 + s_lo * 40, 160.0 + s_hi * 40)
        rgba_valid = np.stack([r_ch, g_ch, b_ch, a_ch], axis=-1).astype(np.uint8)

        cells = []
        for i, (r_idx, c_idx) in enumerate(zip(*np.where(valid_mask))):
            cells.append(
                {
                    "polygon": polys_list[r_idx][c_idx],
                    "value": float(valid_vals[i]),
                    "fill": rgba_valid[i].tolist(),
                }
            )
        return cells if cells else None
    except (OSError, KeyError, ValueError, StopIteration) as exc:
        _log.warning("Failed to load overlay %s: %s", file_path, exc)
        return None


# Coordinate and time-like dimension names (excluded from overlay variable list)
_NC_COORD_NAMES: frozenset[str] = frozenset(
    {
        "lat",
        "lon",
        "latitude",
        "longitude",
        "nav_lat",
        "nav_lon",
        "x",
        "y",
        "i",
        "j",
        "time",
        "t",
        "year",
        "month",
        "depth",
        "z",
        "level",
        "row",
        "col",
    }
)
_NC_TIME_DIM_NAMES: frozenset[str] = frozenset(
    {"time", "t", "year", "month", "step", "date", "ntime"}
)


@lru_cache(maxsize=16)
def list_nc_overlay_variables(file_path_str: str) -> dict[str, dict] | None:
    """Return per-variable metadata for a NetCDF overlay file.

    Results are cached by file path string so repeated reactive evaluations
    do not incur disk I/O.  Invalidate by restarting the app after file changes.

    Returns
    -------
    dict mapping variable name to metadata::

        {
            "Dinoflagellates": {
                "n_time": 24,
                "has_time": True,
                "vmin": 0.0,
                "vmax": 4500.3,
            },
            ...
        }

    Returns *None* on error or when no suitable 2-D+ data variable is found.
    """
    try:
        with xr.open_dataset(Path(file_path_str)) as ds:
            result: dict[str, dict] = {}
            for vn in ds.data_vars:
                if vn.lower() in _NC_COORD_NAMES:
                    continue
                da = ds[vn]
                if len(da.dims) < 2:
                    continue
                has_time = len(da.dims) > 2 and da.dims[0].lower() in _NC_TIME_DIM_NAMES
                n_time = int(da.shape[0]) if has_time else 1
                vals = da.values.astype(float)
                valid = vals[~np.isnan(vals)]
                if len(valid) == 0:
                    continue
                result[vn] = {
                    "n_time": n_time,
                    "has_time": has_time,
                    "vmin": float(valid.min()),
                    "vmax": float(valid.max()),
                }
        return result or None
    except (OSError, KeyError, ValueError) as exc:
        _log.warning("Failed to inspect NC file %s: %s", file_path_str, exc)
        return None


_LABEL_RE = re.compile(r"^\d+[A-Za-z]+_(.+)$")


def derive_map_label(filename: str, map_index: int) -> str:
    """Derive a human-readable label from an OSMOSE movement map filename."""
    stem = Path(filename).stem
    m = _LABEL_RE.match(stem)
    if not m:
        return f"Map {map_index}"
    label_part = m.group(1).replace("_", " ").title()
    if label_part.strip().isdigit():
        return f"Map {map_index}"
    return label_part


def parse_movement_steps(raw: str | None) -> set[int]:
    """Parse a semicolon-separated list of time step indices into a set."""
    if not raw:
        return set()
    steps = set()
    for part in raw.split(";"):
        part = part.strip()
        if part:
            try:
                steps.add(int(part))
            except ValueError:
                _log.debug("Skipping non-integer step value: %r", part)
    return steps


MOVEMENT_PALETTE: list[list[int]] = [
    [30, 120, 200, 140],  # Blue
    [220, 60, 60, 140],  # Red
    [40, 180, 80, 140],  # Green
    [240, 150, 30, 140],  # Orange
    [160, 60, 200, 140],  # Purple
    [30, 190, 200, 140],  # Cyan
    [220, 100, 160, 140],  # Pink
    [200, 200, 40, 140],  # Yellow
]


def _format_age_range(min_age: str | None, max_age: str | None) -> str:
    """Format age range for legend labels."""
    try:
        lo = float(min_age) if min_age else 0
        hi = float(max_age) if max_age else None
    except (ValueError, TypeError):
        return ""
    if hi is None:
        return f"{lo:.0f}+ yr"
    return f"{lo:.0f}-{hi:.0f} yr"


def build_movement_cache(
    cfg: dict[str, str],
    config_dir: Path | None,
    grid_params: tuple[float, float, float, float, int, int],
    species: str,
    nc_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, dict]:
    """Pre-read all movement maps for a species and return a cache dict.

    Parameters
    ----------
    cfg
        Raw config dict (key -> value strings).
    config_dir
        Directory containing the config files (for resolving relative paths).
    grid_params
        Tuple of (ul_lat, ul_lon, lr_lat, lr_lon, nx, ny) for grid bounds.
    species
        Species name to filter maps for.
    nc_data
        Optional (lat, lon, mask) arrays from a NetCDF grid.  When provided,
        CSV overlays are mapped onto the NetCDF grid coordinates instead of
        the regular bounding-box grid (required for NcGrid configs where
        grid.nlon/nlat may not match the NetCDF dimensions).

    Returns
    -------
    dict
        Map ID -> {"label", "steps", "age_range", "color", "cells"} for each valid map.
    """
    ul_lat, ul_lon, lr_lat, lr_lon, nx, ny = grid_params

    map_indices: list[str] = []
    for key, val in cfg.items():
        if key.startswith("movement.species.map") and val == species:
            idx = key[len("movement.species.map") :]
            map_indices.append(idx)

    if map_indices and len(map_indices) > len(MOVEMENT_PALETTE):
        _log.warning(
            "Species %s has %d maps but palette has %d colors; colors will cycle",
            species,
            len(map_indices),
            len(MOVEMENT_PALETTE),
        )

    cache: dict[str, dict] = {}
    color_idx = 0
    for idx in sorted(map_indices, key=lambda x: int(x) if x.isdigit() else 0):
        file_val = cfg.get(f"movement.file.map{idx}", "")
        if not file_val or file_val.lower() in ("null", "none"):
            continue

        if config_dir:
            file_path = _safe_resolve(config_dir, file_val)
            if file_path is None:
                _log.warning("Path traversal in movement map: %s", file_val)
                continue
            if not file_path.exists():
                _log.warning("Movement map file not found: %s", file_val)
                continue
        else:
            continue

        steps = parse_movement_steps(cfg.get(f"movement.steps.map{idx}"))
        if not steps:
            _log.debug("Movement map%s has no valid time steps, skipping", idx)
            continue

        cells = load_csv_overlay(file_path, ul_lat, ul_lon, lr_lat, lr_lon, nx, ny, nc_data=nc_data)
        if not cells:
            continue

        label = derive_map_label(file_val, int(idx) if idx.isdigit() else 0)
        age_range = _format_age_range(
            cfg.get(f"movement.initialAge.map{idx}"),
            cfg.get(f"movement.lastAge.map{idx}"),
        )

        cache[f"map{idx}"] = {
            "label": label,
            "steps": steps,
            "age_range": age_range,
            "color": list(MOVEMENT_PALETTE[color_idx % len(MOVEMENT_PALETTE)]),
            "cells": cells,
        }
        color_idx += 1

    return cache


def list_movement_species(cfg: dict[str, str]) -> list[str]:
    """Return sorted list of unique species names that have movement maps defined."""
    species: set[str] = set()
    for key, val in cfg.items():
        if key.startswith("movement.species.map") and val:
            species.add(val)
    return sorted(species)


def make_legend(entries: list[dict], **kwargs) -> dict:
    """Create a deck.gl legend control, adapting to shiny_deckgl version.

    The dev venv (v1.9.1 fork) exports ``deck_legend_control``, while the
    server (stock PyPI) exports ``layer_legend_widget``.  Accepts
    ``placement`` as an alias for ``position`` for convenience.
    """
    import shiny_deckgl as _sdgl  # type: ignore[import-untyped]

    if hasattr(_sdgl, "layer_legend_widget"):
        return _sdgl.layer_legend_widget(entries=entries, **kwargs)
    if hasattr(_sdgl, "deck_legend_control"):
        kw = dict(kwargs)
        if "placement" in kw:
            kw["position"] = kw.pop("placement")
        return _sdgl.deck_legend_control(entries=entries, **kw)
    return {}


def make_spatial_map(
    ds,
    var_name: str,
    time_idx: int = 0,
    title: str | None = None,
    template: str = "osmose",
):
    """Create a Plotly imshow heatmap from a spatial xarray Dataset."""
    import plotly.express as px

    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=time_idx)
    data = da.values
    lat = ds["lat"].values
    lon = ds["lon"].values
    fig = px.imshow(
        data,
        x=lon,
        y=lat,
        origin="lower",
        color_continuous_scale="Viridis",
        labels={"x": "Longitude", "y": "Latitude", "color": var_name},
        title=title or f"{var_name} (t={time_idx})",
    )
    fig.update_layout(template=template)
    return fig
