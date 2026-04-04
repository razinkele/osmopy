"""Grid configuration page."""

import math

from shiny import ui, reactive, render
from shiny.types import SilentException

import shiny_deckgl as _sdgl  # type: ignore[import-untyped]

from shiny_deckgl import (  # type: ignore[import-untyped]
    MapWidget,
    polygon_layer,
    CARTO_POSITRON,
    CARTO_DARK,
    zoom_widget,
    compass_widget,
    fullscreen_widget,
    scale_widget,
)

from osmose.logging import setup_logging
from osmose.schema.grid import GRID_FIELDS
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.components.param_form import render_field
from ui.pages.grid_helpers import (
    build_grid_layers,
    build_movement_cache,
    build_netcdf_grid_layers,
    list_movement_species,
    load_csv_overlay,
    load_mask,
    load_netcdf_grid,
    load_netcdf_overlay,
)
from ui.state import get_theme_mode, sync_inputs

_log = setup_logging("osmose.grid.ui")

GRID_GLOBAL_KEYS: list[str] = [f.key_pattern for f in GRID_FIELDS if not f.indexed]


def _make_legend(entries: list[dict], **kwargs) -> dict:
    """Create a legend widget/control, adapting to shiny_deckgl version."""
    if hasattr(_sdgl, "layer_legend_widget"):
        return _sdgl.layer_legend_widget(entries=entries, **kwargs)
    if hasattr(_sdgl, "deck_legend_control"):
        kw = dict(kwargs)
        if "placement" in kw:
            kw["position"] = kw.pop("placement")
        return _sdgl.deck_legend_control(entries=entries, **kw)
    return {}


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
                ui.output_ui("grid_overlay_selector"),
                ui.output_ui("movement_controls"),
                ui.output_ui("grid_fields"),
            ),
            ui.div(
                ui.output_ui("grid_hint"),
                grid_map.ui(height="100%"),
                class_="osm-grid-map-container",
            ),
            col_widths=[5, 7],
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
        # Keys whose .csv/.nc values are NOT spatial grids
        skip_prefixes = (
            "grid.",
            "osmose.configuration.",
            "simulation.restart",
            "predation.accessibility",
            "fisheries.catchability",
            "fisheries.discards",
        )

        for key, val in sorted(cfg.items()):
            if not val or not isinstance(val, str):
                continue
            if not (val.endswith(".nc") or val.endswith(".csv")):
                continue
            if any(key.startswith(p) for p in skip_prefixes):
                continue
            if "season" in key:
                continue
            label = key.replace(".", " ").replace("_", " ").title()
            choices[key] = label

        # Add Movement Animation entry
        choices["__movement_animation__"] = "Movement Animation"

        if len(choices) <= 1:
            return ui.div()
        return ui.input_select(
            "grid_overlay", "Overlay data", choices=choices, selected="grid_extent"
        )

    @render.ui
    def movement_controls():
        """Show species/speed/slider when Movement Animation is selected."""
        try:
            overlay_val = input.grid_overlay()
        except SilentException:
            return ui.div()

        if overlay_val != "__movement_animation__":
            return ui.div()

        with reactive.isolate():
            cfg = state.config.get()

        species_list = list_movement_species(cfg)
        if not species_list:
            return ui.p(
                "No movement maps configured. Define maps in the Movement tab.",
                style="color: var(--osm-text-muted); font-size: 12px; margin-top: 8px;",
            )

        species_choices = {s: s for s in species_list}
        speed_choices = {"2000": "0.5x", "1000": "1x", "500": "2x", "250": "4x"}

        try:
            nsteps = int(float(cfg.get("simulation.time.ndtperyear", "24") or "24"))
        except (ValueError, TypeError):
            _log.warning(
                "Could not parse simulation.time.ndtperyear=%r, defaulting to 24",
                cfg.get("simulation.time.ndtperyear"),
            )
            nsteps = 24

        try:
            interval = int(input.movement_speed())
        except (SilentException, ValueError, TypeError):
            interval = 1000

        try:
            current_step = input.movement_step()
        except SilentException:
            current_step = 0

        return ui.div(
            ui.input_select(
                "movement_species",
                "Species",
                choices=species_choices,
                selected=species_list[0],
            ),
            ui.input_select(
                "movement_speed",
                "Speed",
                choices=speed_choices,
                selected=str(interval),
            ),
            ui.input_slider(
                "movement_step",
                "Time step",
                min=0,
                max=nsteps - 1,
                value=current_step,
                step=1,
                animate=ui.AnimationOptions(
                    interval=interval,
                    loop=True,
                    play_button="Play",
                    pause_button="Pause",
                ),
            ),
            class_="osm-movement-controls",
        )

    # Lightweight handle to the widget rendered in grid_ui().
    # shiny_deckgl routes .update() messages by widget ID ("grid_map").
    _map = MapWidget(
        "grid_map",
        view_state={"latitude": 46.0, "longitude": -4.5, "zoom": 5},
        style=CARTO_POSITRON,
    )

    # Movement animation state
    _movement_cache: reactive.Value[dict[str, dict]] = reactive.Value({})
    _prev_active_maps: reactive.Value[frozenset[str]] = reactive.Value(frozenset())

    def _read_grid_values() -> tuple[float, float, float, float, int, int]:
        """Read grid bounds from inputs, falling back to config.

        After example loading, grid_fields re-renders with new inputs, but
        the new values may not have round-tripped through the client yet.
        Fall back to config so the preview works immediately.
        """
        state.load_trigger.get()  # re-run when example loads
        inputs_available = True
        try:
            ul_lat = float(input.grid_upleft_lat() or 0)
            ul_lon = float(input.grid_upleft_lon() or 0)
            lr_lat = float(input.grid_lowright_lat() or 0)
            lr_lon = float(input.grid_lowright_lon() or 0)
            nx = int(input.grid_nlon() or 0)
            ny = int(input.grid_nlat() or 0)
        except (SilentException, AttributeError):
            inputs_available = False
            ul_lat = ul_lon = lr_lat = lr_lon = 0.0
            nx = ny = 0

        # Fall back to config if inputs haven't populated yet
        if not inputs_available:
            with reactive.isolate():
                cfg = state.config.get()
            try:
                ul_lat = float(cfg.get("grid.upleft.lat", 0))
                ul_lon = float(cfg.get("grid.upleft.lon", 0))
                lr_lat = float(cfg.get("grid.lowright.lat", 0))
                lr_lon = float(cfg.get("grid.lowright.lon", 0))
                nx = int(float(cfg.get("grid.nlon", 0)))
                ny = int(float(cfg.get("grid.nlat", 0)))
            except (ValueError, TypeError):
                _log.warning("Invalid grid coordinate values in config")
                ul_lat = ul_lon = lr_lat = lr_lon = 0.0
                nx = ny = 0

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
    def _rebuild_movement_cache():
        """Rebuild the movement map cache when species or config changes."""
        state.load_trigger.get()
        try:
            overlay = input.grid_overlay()
        except SilentException:
            return
        if overlay != "__movement_animation__":
            _movement_cache.set({})
            _prev_active_maps.set(frozenset())
            return
        try:
            species = input.movement_species()
        except SilentException:
            return
        if not species:
            return

        with reactive.isolate():
            cfg = state.config.get()
            cfg_dir = state.config_dir.get()

        with reactive.isolate():
            ul_lat, ul_lon, lr_lat, lr_lon, nx, ny = _read_grid_values()
        grid_params = (ul_lat, ul_lon, lr_lat, lr_lon, nx, ny)

        cache = build_movement_cache(cfg, cfg_dir, grid_params, species=species)
        _movement_cache.set(cache)
        _prev_active_maps.set(frozenset())

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
            layers, view_state = build_netcdf_grid_layers(nc_lat, nc_lon, nc_mask, is_dark)
        else:
            mask_path = cfg.get("grid.mask.file", "")
            mask = load_mask(cfg, config_dir=cfg_dir)
            if mask_path and mask is None:
                ui.notification_show(
                    f"Grid mask file configured but could not be loaded: {mask_path}. "
                    "Grid preview may be inaccurate.",
                    type="warning",
                    duration=10,
                )
            layers = build_grid_layers(ul_lat, ul_lon, lr_lat, lr_lon, nx, ny, is_dark, mask)

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
        overlay = input.grid_overlay() if hasattr(input, "grid_overlay") else None
        if not overlay:
            overlay = "grid_extent"

        if overlay == "__movement_animation__":
            # Movement animation mode — use cached maps
            cache = _movement_cache.get()
            if cache:
                try:
                    step = input.movement_step()
                except SilentException:
                    step = 0
                active_ids = frozenset(mid for mid, m in cache.items() if step in m["steps"])
                prev = _prev_active_maps.get()
                if active_ids == prev and prev:
                    return  # skip update — no visual change
                _prev_active_maps.set(active_ids)

                for mid in sorted(active_ids):
                    m = cache[mid]
                    layer_id = f"movement-{mid}"
                    layers.append(
                        polygon_layer(
                            layer_id,
                            data=m["cells"],
                            get_polygon="@@=d.polygon",
                            get_fill_color=m["color"],
                            get_line_color=[0, 0, 0, 0],
                            filled=True,
                            stroked=False,
                            pickable=True,
                        )
                    )
                    age_suffix = f" ({m['age_range']})" if m["age_range"] else ""
                    legend_entries.append(
                        {
                            "layer_id": layer_id,
                            "label": f"{m['label']}{age_suffix}",
                            "color": m["color"][:3],
                            "shape": "rect",
                        }
                    )

        elif overlay != "grid_extent":
            overlay_path_str = cfg.get(overlay, "")
            if overlay_path_str and cfg_dir:
                overlay_file = (cfg_dir / overlay_path_str).resolve()
                if not overlay_file.is_relative_to(cfg_dir.resolve()):
                    _log.warning("Skipping path traversal in overlay: %s", overlay_path_str)
                elif not overlay_file.exists():
                    ui.notification_show(
                        f"File not found: {overlay_path_str}",
                        type="warning",
                        duration=3,
                    )
                elif overlay_file.suffix == ".nc":
                    fb_lat = nc_data[0] if nc_data else None
                    fb_lon = nc_data[1] if nc_data else None
                    cells = load_netcdf_overlay(overlay_file, fb_lat, fb_lon)
                    if cells:
                        layers.append(
                            polygon_layer(
                                "grid-overlay",
                                data=cells,
                                get_polygon="@@=d.polygon",
                                get_fill_color=[255, 140, 0, 150],
                                get_line_color=[0, 0, 0, 0],
                                filled=True,
                                stroked=False,
                                pickable=True,
                            )
                        )
                        legend_entries.append(
                            {
                                "layer_id": "grid-overlay",
                                "label": "Overlay Data",
                                "color": [255, 140, 0],
                                "shape": "rect",
                            }
                        )
                elif overlay_file.suffix == ".csv":
                    csv_cells = load_csv_overlay(
                        overlay_file,
                        ul_lat,
                        ul_lon,
                        lr_lat,
                        lr_lon,
                        nx,
                        ny,
                        nc_data=nc_data,
                    )
                    if csv_cells:
                        layers.append(
                            polygon_layer(
                                "grid-overlay",
                                data=csv_cells,
                                get_polygon="@@=d.polygon",
                                get_fill_color="@@=d.fill",
                                get_line_color=[0, 0, 0, 0],
                                filled=True,
                                stroked=False,
                                pickable=True,
                            )
                        )
                        legend_entries.append(
                            {
                                "layer_id": "grid-overlay",
                                "label": "Overlay Data",
                                "color": [255, 140, 0],
                                "shape": "rect",
                            }
                        )

        widgets = [
            fullscreen_widget(placement="top-left"),
            zoom_widget(placement="top-right"),
            compass_widget(placement="top-right"),
            scale_widget(placement="bottom-right"),
        ]
        if legend_entries:
            widgets.append(
                _make_legend(
                    entries=legend_entries,
                    placement="bottom-left",
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
