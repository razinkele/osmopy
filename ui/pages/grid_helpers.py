"""Pure helper functions for the Grid configuration page.

These functions are free of Shiny reactive context (no ``input``, ``state``,
``ui.notification_show``, or ``reactive.*``) and can be imported/tested
independently of the Shiny runtime.
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from shiny_deckgl import polygon_layer  # type: ignore[import-untyped]

from osmose.logging import setup_logging

_log = setup_logging("osmose.grid.helpers")


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


def build_netcdf_grid_layers(
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


def load_netcdf_overlay(
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
                pass
    return steps


MOVEMENT_PALETTE: list[list[int]] = [
    [30, 120, 200, 140],   # Blue
    [220, 60, 60, 140],    # Red
    [40, 180, 80, 140],    # Green
    [240, 150, 30, 140],   # Orange
    [160, 60, 200, 140],   # Purple
    [30, 190, 200, 140],   # Cyan
    [220, 100, 160, 140],  # Pink
    [200, 200, 40, 140],   # Yellow
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

    Returns
    -------
    dict
        Map ID -> {"label", "steps", "age_range", "color", "cells"} for each valid map.
    """
    ul_lat, ul_lon, lr_lat, lr_lon, nx, ny = grid_params

    map_indices: list[str] = []
    for key, val in cfg.items():
        if key.startswith("movement.species.map") and val == species:
            idx = key[len("movement.species.map"):]
            map_indices.append(idx)

    if map_indices and len(map_indices) > len(MOVEMENT_PALETTE):
        _log.warning(
            "Species %s has %d maps but palette has %d colors; colors will cycle",
            species, len(map_indices), len(MOVEMENT_PALETTE),
        )

    cache: dict[str, dict] = {}
    color_idx = 0
    for idx in sorted(map_indices, key=lambda x: int(x) if x.isdigit() else 0):
        file_val = cfg.get(f"movement.file.map{idx}", "")
        if not file_val or file_val in ("null", "None"):
            continue

        if config_dir:
            file_path = (config_dir / file_val).resolve()
            if not file_path.is_relative_to(config_dir.resolve()):
                _log.warning("Path traversal in movement map: %s", file_val)
                continue
            if not file_path.exists():
                _log.warning("Movement map file not found: %s", file_val)
                continue
        else:
            continue

        steps = parse_movement_steps(cfg.get(f"movement.steps.map{idx}"))
        if not steps:
            continue

        cells = load_csv_overlay(file_path, ul_lat, ul_lon, lr_lat, lr_lon, nx, ny)
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
