"""Grid configuration page."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from shiny import ui, reactive, render

from shiny_deckgl import (  # type: ignore[import-untyped]
    MapWidget,
    polygon_layer,
    CARTO_POSITRON,
    CARTO_DARK,
    zoom_widget,
    compass_widget,
    fullscreen_widget,
    scale_widget,
    deck_legend_control,
)

from osmose.logging import setup_logging
from osmose.schema.grid import GRID_FIELDS
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.components.param_form import render_field
from ui.state import get_theme_mode, sync_inputs

_log = setup_logging("osmose.grid.ui")

GRID_GLOBAL_KEYS: list[str] = [f.key_pattern for f in GRID_FIELDS if not f.indexed]


def _load_mask(config: dict[str, str], config_dir: Path | None = None) -> np.ndarray | None:
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
        candidate = d / mask_path
        if candidate.exists():
            full_path = candidate
            break

    if full_path is None:
        _log.debug("Mask file not found: %s", mask_path)
        return None

    try:
        return pd.read_csv(full_path, header=None).values
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        _log.warning("Mask file %s could not be parsed: %s", full_path, exc)
        return None
    except FileNotFoundError:
        _log.debug("Mask file not found: %s", full_path)
        return None
    except (OSError, ValueError) as exc:
        _log.warning("Failed to load mask %s: %s", full_path, exc)
        return None


def _build_grid_layers(
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

    # Individual grid cells as lat/lon rectangles
    if nx > 0 and ny > 0:
        dx = (lr_lon - ul_lon) / nx
        dy = (ul_lat - lr_lat) / ny

        ocean_cells = []
        land_cells = []

        for row in range(ny):
            for col in range(nx):
                lon0 = ul_lon + col * dx
                lon1 = lon0 + dx
                lat0 = ul_lat - row * dy
                lat1 = lat0 - dy
                is_land = (
                    mask is not None
                    and row < mask.shape[0]
                    and col < mask.shape[1]
                    and mask[row, col] <= 0
                )
                cell = {
                    "polygon": [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]],
                    "row": row,
                    "col": col,
                    "type": "land" if is_land else "ocean",
                }
                if is_land:
                    land_cells.append(cell)
                else:
                    ocean_cells.append(cell)

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


def _load_netcdf_grid(
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
        candidate = d / nc_path
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
        ds = xr.open_dataset(full_path)
        lat = ds[var_lat].values
        lon = ds[var_lon].values
        mask = ds[var_mask].values
        ds.close()
        return lat, lon, mask
    except (OSError, KeyError, ValueError) as exc:
        _log.warning("Failed to load NetCDF grid %s: %s", full_path, exc)
        return None


def _build_netcdf_grid_layers(
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
    is_dark: bool = False,
) -> tuple[list[dict], dict]:
    """Build deck.gl layers from 2D NetCDF grid arrays.

    Returns (layers, view_state) tuple.
    """
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    ny, nx = lat.shape
    layers = []

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
        hlat = abs(dlat) / 2
        hlon = abs(dlon) / 2
        return [
            [clon - hlon, clat + hlat],
            [clon + hlon, clat + hlat],
            [clon + hlon, clat - hlat],
            [clon - hlon, clat - hlat],
        ]

    # Grid boundary from outer edges
    ul_lat = float(lat.max()) + abs(float(lat[0, 0] - lat[min(1, ny - 1), 0])) / 2
    ul_lon = float(lon.min()) - abs(float(lon[0, 0] - lon[0, min(1, nx - 1)])) / 2
    lr_lat = float(lat.min()) - abs(float(lat[0, 0] - lat[min(1, ny - 1), 0])) / 2
    lr_lon = float(lon.max()) + abs(float(lon[0, 0] - lon[0, min(1, nx - 1)])) / 2

    boundary = [{"polygon": [
        [ul_lon, ul_lat], [lr_lon, ul_lat], [lr_lon, lr_lat], [ul_lon, lr_lat],
    ]}]
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

    for row in range(ny):
        for col in range(nx):
            is_ocean = mask[row, col] > 0
            cell = {
                "polygon": _cell_polygon(row, col),
                "row": row,
                "col": col,
                "type": "ocean" if is_ocean else "land",
            }
            if is_ocean:
                ocean_cells.append(cell)
            else:
                land_cells.append(cell)

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
    zoom = max(1, min(15, math.log2(360 / span) - 0.5)) if span > 0 else 5
    view_state = {"latitude": center_lat, "longitude": center_lon, "zoom": zoom}

    return layers, view_state


def _load_csv_overlay(
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
        df = pd.read_csv(file_path, header=None)
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
                    file_path, len(flat), g_ny, g_nx,
                )
                return None
        elif data.shape != (g_ny, g_nx):
            # Try transposing
            if data.shape == (g_nx, g_ny):
                data = data.T
            else:
                _log.warning(
                    "CSV %s shape %s doesn't match grid %dx%d",
                    file_path, data.shape, g_ny, g_nx,
                )
                return None

        # Compute value range for color scaling
        numeric = data.astype(float)
        valid = numeric[~np.isnan(numeric)]
        if len(valid) == 0:
            return None
        vmin, vmax = float(valid.min()), float(valid.max())
        vrange = vmax - vmin if vmax != vmin else 1.0

        cells = []
        for r in range(g_ny):
            for c in range(g_nx):
                v = float(numeric[r, c])
                if np.isnan(v):
                    continue
                # Compute cell polygon from grid coordinates
                if nc_data is not None and lat is not None and lon is not None:
                    clat = float(lat[r, c] if lat.ndim == 2 else lat[r])
                    clon = float(lon[r, c] if lon.ndim == 2 else lon[c])
                    if lat.ndim == 2:
                        dlat = abs(float(
                            lat[min(r + 1, g_ny - 1), c] - lat[max(r - 1, 0), c]
                        )) / 2
                        dlon = abs(float(
                            lon[r, min(c + 1, g_nx - 1)] - lon[r, max(c - 1, 0)]
                        )) / 2
                    else:
                        dlat = abs(float(
                            lat[min(r + 1, len(lat) - 1)] - lat[max(r - 1, 0)]
                        )) / 2
                        dlon = abs(float(
                            lon[min(c + 1, len(lon) - 1)] - lon[max(c - 1, 0)]
                        )) / 2
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

                cells.append({
                    "polygon": [
                        [clon - hlon, clat + hlat],
                        [clon + hlon, clat + hlat],
                        [clon + hlon, clat - hlat],
                        [clon - hlon, clat - hlat],
                    ],
                    "value": v,
                    "fill": [red, green, blue, alpha],
                })
        return cells if cells else None
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        _log.warning("Failed to load CSV overlay %s: %s", file_path, exc)
        return None


def _load_netcdf_overlay(
    file_path: Path,
    fallback_lat: np.ndarray | None = None,
    fallback_lon: np.ndarray | None = None,
) -> list[dict] | None:
    """Load a NetCDF file and return overlay cell data for deck.gl."""
    try:
        ds = xr.open_dataset(file_path)
        var_name = None
        for vn in ds.data_vars:
            if len(ds[vn].dims) >= 2:
                var_name = vn
                break
        if not var_name:
            ds.close()
            return None

        data_vals = ds[var_name].values
        if len(data_vals.shape) > 2:
            data_vals = data_vals[0]  # first time step

        olat = ds["lat"].values if "lat" in ds else fallback_lat
        olon = ds["lon"].values if "lon" in ds else fallback_lon
        ds.close()

        if olat is None or olon is None:
            return None

        ony, onx = data_vals.shape
        cells = []
        for r in range(min(ony, olat.shape[0] if olat.ndim > 1 else ony)):
            for c in range(min(onx, olon.shape[1] if olon.ndim > 1 else onx)):
                v = float(data_vals[r, c])
                if np.isnan(v):
                    continue
                cell_lat = float(olat[r, c] if olat.ndim == 2 else olat[r])
                cell_lon = float(olon[r, c] if olon.ndim == 2 else olon[c])
                if olat.ndim == 2:
                    dlat = abs(float(olat[min(r + 1, ony - 1), c] - olat[max(r - 1, 0), c])) / 2
                    dlon = abs(float(olon[r, min(c + 1, onx - 1)] - olon[r, max(c - 1, 0)])) / 2
                else:
                    dlat = abs(float(olat[min(r + 1, len(olat) - 1)] - olat[max(r - 1, 0)])) / 2
                    dlon = abs(float(olon[min(c + 1, len(olon) - 1)] - olon[max(c - 1, 0)])) / 2
                if r == 0 or r == ony - 1:
                    dlat *= 2
                if c == 0 or c == onx - 1:
                    dlon *= 2
                hlat, hlon = dlat / 2, dlon / 2
                cells.append({
                    "polygon": [
                        [cell_lon - hlon, cell_lat + hlat],
                        [cell_lon + hlon, cell_lat + hlat],
                        [cell_lon + hlon, cell_lat - hlat],
                        [cell_lon - hlon, cell_lat - hlat],
                    ],
                    "value": v,
                })
        return cells if cells else None
    except (OSError, KeyError, ValueError) as exc:
        _log.warning("Failed to load overlay %s: %s", file_path, exc)
        return None


def grid_ui():
    grid_map = MapWidget(
        "grid_map",
        view_state={
            "latitude": 46.0,
            "longitude": -4.5,
            "zoom": 5,
            "pitch": 0,
            "bearing": 0,
        },
        style=CARTO_POSITRON,
        tooltip={
            "html": "Cell ({properties.row}, {properties.col}) — {properties.type}",
            "style": {"fontSize": "12px"},
        },
        controls=[],
    )

    return ui.div(
        expand_tab("Grid Type", "grid"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Grid Type", "grid"),
                ui.output_ui("grid_fields"),
            ),
            ui.card(
                ui.card_header("Grid Preview"),
                ui.output_ui("grid_overlay_selector"),
                ui.output_ui("grid_hint"),
                grid_map.ui(height="500px"),
            ),
            col_widths=[6, 6],
        ),
        class_="osm-split-layout",
        id="split_grid",
    )


def grid_server(input, output, session, state):
    grid_type_field = next((f for f in GRID_FIELDS if "classname" in f.key_pattern), None)
    regular_fields = [
        f
        for f in GRID_FIELDS
        if (
            f.key_pattern.startswith("grid.n")
            or f.key_pattern.startswith("grid.up")
            or f.key_pattern.startswith("grid.low")
        )
        and "netcdf" not in f.key_pattern
    ]
    netcdf_fields = [
        f for f in GRID_FIELDS if "netcdf" in f.key_pattern or f.key_pattern.startswith("grid.var")
    ]

    @render.ui
    def grid_fields():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        return ui.div(
            render_field(grid_type_field, config=cfg) if grid_type_field else ui.div(),
            ui.hr(),
            ui.h5("Regular Grid Settings"),
            *[render_field(f, config=cfg) for f in regular_fields],
            ui.hr(),
            ui.h5("NetCDF Grid Settings"),
            *[render_field(f, config=cfg) for f in netcdf_fields if not f.advanced],
        )

    @render.ui
    def grid_overlay_selector():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        choices: dict[str, str] = {"grid_extent": "Grid extent"}
        # Keys that define the base grid itself — not useful as overlays
        skip_prefixes = ("grid.", "osmose.configuration.", "simulation.restart")

        # Scan config directly for any key whose value is a .nc or .csv file
        for key, val in sorted(cfg.items()):
            if not val or not isinstance(val, str):
                continue
            if not (val.endswith(".nc") or val.endswith(".csv")):
                continue
            if any(key.startswith(p) for p in skip_prefixes):
                continue
            # Build a readable label from the config key
            label = key.replace(".", " ").replace("_", " ").title()
            choices[key] = label

        if len(choices) <= 1:
            return ui.div()
        return ui.input_select(
            "grid_overlay", "Overlay data", choices=choices, selected="grid_extent"
        )

    # Lightweight handle to the widget rendered in grid_ui().
    # shiny_deckgl routes .update() messages by widget ID ("grid_map").
    _map = MapWidget(
        "grid_map",
        view_state={"latitude": 46.0, "longitude": -4.5, "zoom": 5},
        style=CARTO_POSITRON,
    )

    def _read_grid_values() -> tuple[float, float, float, float, int, int]:
        """Read grid bounds from inputs, falling back to config.

        After example loading, grid_fields re-renders with new inputs, but
        the new values may not have round-tripped through the client yet.
        Fall back to config so the preview works immediately.
        """
        state.load_trigger.get()  # re-run when example loads
        try:
            ul_lat = float(input.grid_upleft_lat() or 0)
            ul_lon = float(input.grid_upleft_lon() or 0)
            lr_lat = float(input.grid_lowright_lat() or 0)
            lr_lon = float(input.grid_lowright_lon() or 0)
            nx = int(input.grid_nlon() or 0)
            ny = int(input.grid_nlat() or 0)
        except (AttributeError, TypeError):
            ul_lat = ul_lon = lr_lat = lr_lon = 0.0
            nx = ny = 0

        # Fall back to config if inputs haven't populated yet
        if ul_lat == 0 and ul_lon == 0 and lr_lat == 0 and lr_lon == 0:
            with reactive.isolate():
                cfg = state.config.get()
            ul_lat = float(cfg.get("grid.upleft.lat", 0))
            ul_lon = float(cfg.get("grid.upleft.lon", 0))
            lr_lat = float(cfg.get("grid.lowright.lat", 0))
            lr_lon = float(cfg.get("grid.lowright.lon", 0))
            nx = int(float(cfg.get("grid.nlon", 0)))
            ny = int(float(cfg.get("grid.nlat", 0)))

        return ul_lat, ul_lon, lr_lat, lr_lon, nx, ny

    @render.ui
    def grid_hint():
        ul_lat, ul_lon, lr_lat, lr_lon, _, _ = _read_grid_values()
        with reactive.isolate():
            cfg = state.config.get()
        is_ncgrid = "NcGrid" in cfg.get("grid.java.classname", "")
        if not is_ncgrid and ul_lat == 0 and ul_lon == 0 and lr_lat == 0 and lr_lon == 0:
            return ui.p("Configure coordinates or load an example to see a preview.")
        return ui.div()

    @reactive.effect
    @reactive.event(input.deckgl_ready)
    def _handle_deckgl_ready():
        """Bump load_trigger so update_grid_map re-sends layers after late init."""
        with reactive.isolate():
            state.load_trigger.set(state.load_trigger.get() + 1)

    @reactive.effect
    async def update_grid_map():
        ul_lat, ul_lon, lr_lat, lr_lon, nx, ny = _read_grid_values()

        is_dark = get_theme_mode(input) == "dark"

        with reactive.isolate():
            cfg = state.config.get()
            cfg_dir = state.config_dir.get()

        # Detect grid type: NcGrid uses NetCDF file, OriginalGrid uses bounds
        is_ncgrid = "NcGrid" in cfg.get("grid.java.classname", "")
        nc_data = _load_netcdf_grid(cfg, config_dir=cfg_dir) if is_ncgrid else None

        if nc_data is not None:
            nc_lat, nc_lon, nc_mask = nc_data
            layers, view_state = _build_netcdf_grid_layers(
                nc_lat, nc_lon, nc_mask, is_dark
            )
        else:
            mask = _load_mask(cfg, config_dir=cfg_dir)
            layers = _build_grid_layers(
                ul_lat, ul_lon, lr_lat, lr_lon, nx, ny, is_dark, mask
            )

            # Compute view state to fit grid bounds
            if ul_lat != 0 or ul_lon != 0 or lr_lat != 0 or lr_lon != 0:
                center_lat = (ul_lat + lr_lat) / 2
                center_lon = (ul_lon + lr_lon) / 2
                lat_span = abs(ul_lat - lr_lat)
                lon_span = abs(lr_lon - ul_lon)
                span = max(lat_span, lon_span)
                if span > 0:
                    zoom = max(1, min(15, math.log2(360 / span) - 0.5))
                else:
                    zoom = 5
                view_state = {
                    "latitude": center_lat,
                    "longitude": center_lon,
                    "zoom": zoom,
                }
            else:
                view_state = {"latitude": 46.0, "longitude": -4.5, "zoom": 5}

        # Update map style based on theme
        style = CARTO_DARK if is_dark else CARTO_POSITRON
        if style != _map.style:
            _map.style = style
            await _map.set_style(session, style)

        # Build legend entries based on which layers are present
        legend_entries = []
        for lyr in layers:
            lid = lyr.get("id", "")
            if lid == "grid-extent":
                legend_entries.append(
                    {
                        "layer_id": "grid-extent",
                        "label": "Grid Extent",
                        "color": [232, 168, 56],
                        "shape": "line",
                    }
                )
            elif lid == "grid-ocean":
                legend_entries.append(
                    {
                        "layer_id": "grid-ocean",
                        "label": "Ocean Cells",
                        "color": [30, 120, 180] if is_dark else [20, 100, 180],
                        "shape": "rect",
                    }
                )
            elif lid == "grid-land":
                legend_entries.append(
                    {
                        "layer_id": "grid-land",
                        "label": "Land Cells",
                        "color": [80, 65, 45] if is_dark else [190, 170, 140],
                        "shape": "rect",
                    }
                )

        # Load overlay data if selected
        # Read outside isolate so reactive dependency is established
        overlay = input.grid_overlay() if hasattr(input, "grid_overlay") else None
        if not overlay:
            overlay = "grid_extent"

        if overlay != "grid_extent":
            overlay_path_str = cfg.get(overlay, "")
            if overlay_path_str and cfg_dir:
                overlay_file = (cfg_dir / overlay_path_str).resolve()
                if not overlay_file.exists():
                    ui.notification_show(
                        f"File not found: {overlay_path_str}", type="warning", duration=3
                    )
                elif overlay_file.suffix == ".nc":
                    fb_lat = nc_data[0] if nc_data else None
                    fb_lon = nc_data[1] if nc_data else None
                    cells = _load_netcdf_overlay(overlay_file, fb_lat, fb_lon)
                    if cells:
                        layers.append(polygon_layer(
                            "grid-overlay",
                            data=cells,
                            get_polygon="@@=d.polygon",
                            get_fill_color=[255, 140, 0, 150],
                            get_line_color=[0, 0, 0, 0],
                            filled=True,
                            stroked=False,
                            pickable=True,
                        ))
                        legend_entries.append({
                            "layer_id": "grid-overlay",
                            "label": "Overlay Data",
                            "color": [255, 140, 0],
                            "shape": "rect",
                        })
                elif overlay_file.suffix == ".csv":
                    csv_cells = _load_csv_overlay(
                        overlay_file, ul_lat, ul_lon, lr_lat, lr_lon, nx, ny,
                        nc_data=nc_data,
                    )
                    if csv_cells:
                        layers.append(polygon_layer(
                            "grid-overlay",
                            data=csv_cells,
                            get_polygon="@@=d.polygon",
                            get_fill_color="@@=d.fill",
                            get_line_color=[0, 0, 0, 0],
                            filled=True,
                            stroked=False,
                            pickable=True,
                        ))
                        legend_entries.append({
                            "layer_id": "grid-overlay",
                            "label": "Overlay Data",
                            "color": [255, 140, 0],
                            "shape": "rect",
                        })

        widgets = [
            fullscreen_widget(placement="top-left"),
            zoom_widget(placement="top-right"),
            compass_widget(placement="top-right"),
            scale_widget(placement="bottom-right"),
        ]
        if legend_entries:
            widgets.append(
                deck_legend_control(
                    entries=legend_entries,
                    position="bottom-left",
                    show_checkbox=True,
                    title="Layers",
                )
            )

        await _map.update(
            session,
            layers=layers,
            view_state=view_state,
            transition_duration=800,
            widgets=widgets,
        )

    @reactive.effect
    def sync_grid_inputs():
        sync_inputs(input, state, GRID_GLOBAL_KEYS)
