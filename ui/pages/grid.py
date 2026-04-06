"""Grid configuration page."""

from pathlib import Path

from shiny import ui, reactive, render
from shiny.types import SilentException

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
    list_nc_overlay_variables,
    load_csv_overlay,
    load_mask,
    load_netcdf_grid,
    load_netcdf_overlay,
    make_legend,
    _zoom_for_span,
)
from ui.state import get_theme_mode, sync_inputs

_log = setup_logging("osmose.grid.ui")

GRID_GLOBAL_KEYS: list[str] = [f.key_pattern for f in GRID_FIELDS if not f.indexed]


def _validate_overlay_path(overlay_val: str, cfg_dir: Path | None) -> Path | None:
    """Resolve and validate an overlay path is within the config directory.

    The overlay selector stores canonical resolved paths as option values.
    This server-side check ensures a crafted client request cannot read
    arbitrary files outside the config directory.

    Returns the resolved Path if valid, None otherwise.
    """
    if not cfg_dir or not overlay_val or overlay_val in ("grid_extent", "__movement_animation__"):
        return None
    candidate = Path(overlay_val).resolve()
    try:
        candidate.relative_to(cfg_dir.resolve())
    except ValueError:
        _log.warning("Overlay path rejected (outside config dir): %s", overlay_val)
        return None
    return candidate


def _overlay_label(rel_path: str) -> str:
    """Generate a human-readable label from an overlay file path."""
    stem = Path(rel_path).stem.lower()
    if "ltl" in stem or ("ltlbiomass" in stem.replace("_", "").replace("-", "")):
        return "LTL Biomass"
    if "backgroundspecies" in stem.replace("_", "").replace("-", ""):
        return "Background Species"
    if "mpa" in stem or ("marine" in stem and "protected" in stem):
        return stem.replace("_", " ").replace("-", " ").title()
    if "distrib" in stem or ("fishing" in stem and "distrib" in stem):
        return "Fishing Distribution"
    if "fishing" in stem:
        return "Fishing"
    return stem.replace("_", " ").replace("-", " ").title()


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
                ui.output_ui("overlay_nc_controls"),
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
            cfg_dir = state.config_dir.get()

        choices: dict[str, str] = {"grid_extent": "Grid extent"}

        # Config keys whose file values are NOT displayable spatial grids
        skip_prefixes = (
            "grid.",
            "osmose.configuration.",
            "simulation.restart",
            "predation.accessibility",
            "fisheries.catchability",
            "fisheries.discards",
            "movement.file.map",  # movement CSVs — shown separately below
            "movement.species.map",  # movement species labels
            "movement.initialAge.",
            "movement.lastAge.",
            "movement.steps.",
            "movement.distribution.",
        )

        # Deduplicate by canonical resolved path: many config keys may point to the
        # same file (e.g. species.file.sp14–sp23 all pointing to eec_ltlbiomassTons.nc).
        seen_paths: dict[str, str] = {}  # resolved_path_str -> label

        if cfg_dir and cfg_dir.is_dir():
            cfg_dir_resolved = cfg_dir.resolve()
            for key, val in sorted(cfg.items()):
                if not val or not isinstance(val, str):
                    continue
                if not (val.endswith(".nc") or val.endswith(".csv")):
                    continue
                if key.startswith(skip_prefixes):
                    continue
                if "season" in key:
                    continue
                try:
                    resolved = (cfg_dir / val).resolve()
                except Exception:
                    continue
                # Reject path traversal
                try:
                    resolved.relative_to(cfg_dir_resolved)
                except ValueError:
                    continue
                path_id = str(resolved)
                if path_id not in seen_paths:
                    seen_paths[path_id] = _overlay_label(val)

            # Add species distribution maps (from movement.file.mapN keys)
            for key, val in sorted(cfg.items()):
                if not key.startswith("movement.file.map") or not val:
                    continue
                if not val.endswith(".csv"):
                    continue
                try:
                    resolved = (cfg_dir / val).resolve()
                    resolved.relative_to(cfg_dir_resolved)
                except (Exception, ValueError):
                    continue
                path_id = str(resolved)
                if path_id not in seen_paths:
                    # Get species name from movement.species.mapN
                    idx = key.removeprefix("movement.file.map")
                    species = cfg.get(f"movement.species.map{idx}", "")
                    label = f"{species}: {_overlay_label(val)}" if species else _overlay_label(val)
                    seen_paths[path_id] = label

            # Add fishing distribution maps (from fisheries.movement.file.mapN keys)
            for key, val in sorted(cfg.items()):
                if not key.startswith("fisheries.movement.file.map") or not val:
                    continue
                if not val.endswith(".csv"):
                    continue
                try:
                    resolved = (cfg_dir / val).resolve()
                    resolved.relative_to(cfg_dir_resolved)
                except (Exception, ValueError):
                    continue
                path_id = str(resolved)
                if path_id not in seen_paths:
                    seen_paths[path_id] = f"Fishing: {_overlay_label(val)}"

            # Scan mpa/ directory for MPA files not referenced in the config
            mpa_dir = cfg_dir / "mpa"
            if mpa_dir.is_dir():
                for mpa_file in sorted(mpa_dir.glob("*.csv")):
                    path_id = str(mpa_file.resolve())
                    if path_id not in seen_paths:
                        seen_paths[path_id] = f"MPA: {mpa_file.stem.replace('_', ' ').title()}"

        choices.update(seen_paths)
        choices["__movement_animation__"] = "Movement Animation"

        if len(choices) <= 1:
            return ui.div()
        return ui.input_select(
            "grid_overlay", "Overlay data", choices=choices, selected="grid_extent"
        )

    @render.ui
    def overlay_nc_controls():
        """Show variable selector and/or time slider for multi-var/time NetCDF overlays."""
        try:
            overlay_val = input.grid_overlay()
        except SilentException:
            return ui.div()

        if overlay_val in ("grid_extent", "__movement_animation__"):
            return ui.div()

        with reactive.isolate():
            cfg_dir = state.config_dir.get()
        overlay_path = _validate_overlay_path(overlay_val, cfg_dir)
        if overlay_path is None or overlay_path.suffix != ".nc" or not overlay_path.exists():
            return ui.div()

        meta = list_nc_overlay_variables(str(overlay_path))
        if not meta:
            return ui.div()

        controls = []

        # Variable selector (only when multiple data variables present)
        if len(meta) > 1:
            var_choices = {k: k.replace("_", " ").title() for k in meta}
            try:
                current_var = input.nc_var_select()
            except SilentException:
                current_var = next(iter(meta))
            if current_var not in meta:
                current_var = next(iter(meta))
            controls.append(
                ui.input_select(
                    "nc_var_select",
                    "Variable",
                    choices=var_choices,
                    selected=current_var,
                )
            )
            sel_var = current_var
        else:
            # Single variable — hidden select so input.nc_var_select() is always defined
            sole_var = next(iter(meta))
            controls.append(
                ui.div(
                    ui.input_select(
                        "nc_var_select",
                        "Variable",
                        choices={sole_var: sole_var},
                        selected=sole_var,
                    ),
                    style="display:none",
                )
            )
            sel_var = sole_var

        # Time step slider when the selected variable has multiple time steps
        var_meta = meta.get(sel_var, {})
        n_time = var_meta.get("n_time", 1)
        if n_time > 1:
            try:
                current_step = int(input.nc_time_step())
            except (SilentException, ValueError, TypeError):
                current_step = 0
            current_step = max(0, min(current_step, n_time - 1))
            controls.append(
                ui.input_slider(
                    "nc_time_step",
                    "Time step",
                    min=0,
                    max=n_time - 1,
                    value=current_step,
                    step=1,
                )
            )

        return ui.div(*controls) if controls else ui.div()

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

        with reactive.isolate():
            try:
                current_step = input.movement_step()
            except (SilentException, AttributeError):
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
    _prev_active_maps: reactive.Value[tuple[frozenset[str], bool]] = reactive.Value(
        (frozenset(), False)
    )

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
            _prev_active_maps.set((frozenset(), False))
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
        _prev_active_maps.set((frozenset(), False))

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
                zoom = _zoom_for_span(span)
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
        try:
            overlay = input.grid_overlay()
        except (SilentException, AttributeError):
            overlay = None
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
                prev_ids, prev_dark = _prev_active_maps.get()
                if active_ids == prev_ids and prev_ids and is_dark == prev_dark:
                    return  # skip update — same maps AND same theme
                _prev_active_maps.set((active_ids, is_dark))

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
            # Validate overlay value is within the config directory (server-side check
            # against crafted client requests; selector already dedupes by canonical path)
            overlay_file = _validate_overlay_path(overlay, cfg_dir)
            if overlay_file is None or not overlay_file.exists():
                if overlay_file is not None:
                    ui.notification_show(
                        f"Overlay file not found: {overlay_file.name}",
                        type="warning",
                        duration=3,
                    )
            elif overlay_file.suffix == ".nc":
                # Read variable and time-step controls (populated by overlay_nc_controls)
                try:
                    nc_var = input.nc_var_select() or None
                except (SilentException, AttributeError):
                    nc_var = None
                try:
                    nc_time = max(0, int(input.nc_time_step()))
                except (SilentException, ValueError, TypeError, AttributeError):
                    nc_time = 0

                # Fetch stable vmin/vmax from cached metadata for consistent colours
                meta = list_nc_overlay_variables(str(overlay_file))
                nc_vmin = nc_vmax = None
                if meta:
                    sel_meta = meta.get(nc_var or "") or next(iter(meta.values()), None)
                    if sel_meta:
                        nc_vmin = sel_meta["vmin"]
                        nc_vmax = sel_meta["vmax"]

                fb_lat = nc_data[0] if nc_data else None
                fb_lon = nc_data[1] if nc_data else None
                cells = load_netcdf_overlay(
                    overlay_file,
                    fb_lat,
                    fb_lon,
                    var_lat=cfg.get("grid.var.lat", "lat"),
                    var_lon=cfg.get("grid.var.lon", "lon"),
                    var_name=nc_var,
                    time_step=nc_time,
                    vmin=nc_vmin,
                    vmax=nc_vmax,
                )
                if cells:
                    layers.append(
                        polygon_layer(
                            "grid-overlay",
                            data=cells,
                            get_polygon="@@=d.polygon",
                            get_fill_color="@@=d.fill",
                            get_line_color=[0, 0, 0, 0],
                            filled=True,
                            stroked=False,
                            pickable=True,
                        )
                    )
                    overlay_label = nc_var.replace("_", " ").title() if nc_var else "Overlay Data"
                    legend_entries.append(
                        {
                            "layer_id": "grid-overlay",
                            "label": overlay_label,
                            "color": [0, 170, 180],  # mid-range cyan
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
                            "label": _overlay_label(overlay_file.name),
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
                make_legend(
                    entries=legend_entries,
                    placement="bottom-left",
                    show_checkbox=False,
                    collapsed=True,
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
