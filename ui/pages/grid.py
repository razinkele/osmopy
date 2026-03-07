"""Grid configuration page."""

from pathlib import Path

import numpy as np
import pandas as pd
from shiny import ui, reactive, render

from shiny_deckgl import (
    MapWidget,
    polygon_layer,
    geojson_layer,
    CARTO_POSITRON,
    CARTO_DARK,
    zoom_widget,
    compass_widget,
    scale_widget,
    deck_legend_control,
)

from osmose.logging import setup_logging
from osmose.schema.grid import GRID_FIELDS
from ui.components.param_form import render_field
from ui.state import sync_inputs

_log = setup_logging("osmose.grid.ui")

GRID_GLOBAL_KEYS: list[str] = [f.key_pattern for f in GRID_FIELDS if not f.indexed]


def _load_mask(config: dict[str, str]) -> np.ndarray | None:
    """Load a grid mask CSV from the config if available."""
    mask_path = config.get("grid.mask.file", "")
    if not mask_path:
        return None

    examples_dir = Path(__file__).parent.parent.parent / "data" / "examples"
    full_path = examples_dir / mask_path
    if not full_path.exists():
        return None

    try:
        return pd.read_csv(full_path, header=None).values
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    except Exception as exc:
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
                cell = {
                    "polygon": [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]],
                    "row": row,
                    "col": col,
                }

                is_land = (
                    mask is not None
                    and row < mask.shape[0]
                    and col < mask.shape[1]
                    and mask[row, col] <= 0
                )
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


def grid_ui():
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

    return ui.layout_columns(
        ui.card(
            ui.card_header("Grid Type"),
            render_field(grid_type_field) if grid_type_field else ui.div(),
            ui.hr(),
            ui.h5("Regular Grid Settings"),
            *[render_field(f) for f in regular_fields],
            ui.hr(),
            ui.h5("NetCDF Grid Settings"),
            *[render_field(f) for f in netcdf_fields if not f.advanced],
        ),
        ui.card(
            ui.card_header("Grid Preview"),
            ui.output_ui("grid_hint"),
            grid_map.ui(height="500px"),
        ),
        col_widths=[6, 6],
    )


def grid_server(input, output, session, state):
    # Create a per-session map widget with the same ID
    _map = MapWidget(
        "grid_map",
        view_state={"latitude": 46.0, "longitude": -4.5, "zoom": 5},
        style=CARTO_POSITRON,
    )

    @render.ui
    def grid_hint():
        ul_lat = float(input.grid_upleft_lat() or 0)
        ul_lon = float(input.grid_upleft_lon() or 0)
        lr_lat = float(input.grid_lowright_lat() or 0)
        lr_lon = float(input.grid_lowright_lon() or 0)
        if ul_lat == 0 and ul_lon == 0 and lr_lat == 0 and lr_lon == 0:
            return ui.p("Configure coordinates or load an example to see a preview.")
        return ui.div()

    @reactive.effect
    async def update_grid_map():
        ul_lat = float(input.grid_upleft_lat() or 0)
        ul_lon = float(input.grid_upleft_lon() or 0)
        lr_lat = float(input.grid_lowright_lat() or 0)
        lr_lon = float(input.grid_lowright_lon() or 0)
        nx = int(input.grid_ncolumn() or 0)
        ny = int(input.grid_nline() or 0)
        print(f"[GRID DEBUG] ul=({ul_lat},{ul_lon}) lr=({lr_lat},{lr_lon}) nx={nx} ny={ny}")

        try:
            is_dark = input.theme_mode() == "dark"
        except Exception:
            is_dark = False

        # Load land-sea mask from config if available
        with reactive.isolate():
            cfg = state.config.get()
        mask = _load_mask(cfg)

        layers = _build_grid_layers(ul_lat, ul_lon, lr_lat, lr_lon, nx, ny, is_dark, mask)
        print(f"[GRID DEBUG] layers={len(layers)}, mask={'yes' if mask is not None else 'no'}")

        # Compute view state to fit grid bounds
        if ul_lat != 0 or ul_lon != 0 or lr_lat != 0 or lr_lon != 0:
            center_lat = (ul_lat + lr_lat) / 2
            center_lon = (ul_lon + lr_lon) / 2
            lat_span = abs(ul_lat - lr_lat)
            lon_span = abs(lr_lon - ul_lon)
            span = max(lat_span, lon_span)
            if span > 0:
                import math

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
            _map.set_style(session, style)

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

        widgets = [
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
