"""Grid configuration page."""

import math

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
from ui.pages.grid_helpers import (
    build_grid_layers,
    build_netcdf_grid_layers,
    load_csv_overlay,
    load_mask,
    load_netcdf_grid,
    load_netcdf_overlay,
)
from ui.state import get_theme_mode, sync_inputs

_log = setup_logging("osmose.grid.ui")

GRID_GLOBAL_KEYS: list[str] = [f.key_pattern for f in GRID_FIELDS if not f.indexed]


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
        except (AttributeError, TypeError, ValueError):
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
        nc_data = load_netcdf_grid(cfg, config_dir=cfg_dir) if is_ncgrid else None

        if nc_data is not None:
            nc_lat, nc_lon, nc_mask = nc_data
            layers, view_state = build_netcdf_grid_layers(
                nc_lat, nc_lon, nc_mask, is_dark
            )
        else:
            mask = load_mask(cfg, config_dir=cfg_dir)
            layers = build_grid_layers(
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
                    cells = load_netcdf_overlay(overlay_file, fb_lat, fb_lon)
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
                    csv_cells = load_csv_overlay(
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
