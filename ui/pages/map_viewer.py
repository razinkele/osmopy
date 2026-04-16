"""Map Viewer page — browse and preview CSV/NC spatial files."""

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
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.components.renderer_badge import renderer_badge
from ui.pages.grid_helpers import (
    _overlay_label,
    _zoom_for_span,
    build_grid_layers,
    build_netcdf_grid_layers,
    discover_spatial_files,
    list_nc_overlay_variables,
    load_csv_overlay,
    load_mask,
    load_netcdf_grid,
    load_netcdf_overlay,
    make_legend,
)
from ui.state import get_theme_mode

_log = setup_logging("osmose.map_viewer")

_DEFAULT_VIEW_STATE = {"latitude": 46.0, "longitude": -4.5, "zoom": 5, "pitch": 0, "bearing": 0}


def map_viewer_ui():
    viewer_map = MapWidget(
        "map_viewer_map",
        view_state=_DEFAULT_VIEW_STATE,
        style=CARTO_POSITRON,
        tooltip={"html": "Value: {properties.value}", "style": {"fontSize": "12px"}},
        controls=[],
    )

    return ui.div(
        expand_tab("Map Viewer", "map_viewer"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Map Viewer", "map_viewer"),
                ui.output_ui("map_viewer_file_list"),
            ),
            ui.div(
                ui.output_ui("map_viewer_hint"),
                ui.output_ui("map_viewer_nc_controls"),
                viewer_map.ui(height="100%"),
                ui.output_ui("map_viewer_metadata"),
                renderer_badge(),
                class_="osm-grid-map-container",
            ),
            col_widths=[5, 7],
        ),
        class_="osm-split-layout",
        id="split_map_viewer",
    )


def map_viewer_server(input, output, session, state):
    _map = MapWidget(
        "map_viewer_map",
        view_state=_DEFAULT_VIEW_STATE,
        style=CARTO_POSITRON,
    )

    @render.ui
    def map_viewer_hint():
        state.load_trigger.get()
        with reactive.isolate():
            name = state.config_name.get()
        if not name:
            return ui.p(
                "Load a configuration to browse spatial files.",
                style="color: var(--osm-text-muted); text-align: center; padding: 40px 20px;",
            )
        return ui.div()

    @render.ui
    def map_viewer_file_list():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
            cfg_dir = state.config_dir.get()

        if not cfg or not cfg_dir:
            return ui.p(
                "No configuration loaded.",
                style="color: var(--osm-text-muted); font-size: 12px; padding: 8px;",
            )

        catalog = discover_spatial_files(cfg, cfg_dir)
        choices: dict = {}

        # Movement maps grouped by species
        movement = catalog.get("movement", {})
        movement = movement if isinstance(movement, dict) else {}  # type: ignore[assignment]
        total_mvmt = sum(len(v) for v in movement.values())
        if movement:
            group_choices: dict[str, str] = {}
            for species in sorted(movement):
                for entry in movement[species]:
                    entry_dict = entry if isinstance(entry, dict) else {}  # type: ignore[assignment]
                    key = str(entry_dict.get("path", ""))
                    suffix = f" ({entry_dict.get('age')})" if entry_dict.get("age") else ""
                    group_choices[key] = f"{species}: {entry_dict.get('label', '')}{suffix}"
            choices[f"Movement Maps ({total_mvmt})"] = group_choices

        # Fishing
        fishing = catalog.get("fishing", [])
        fishing = fishing if isinstance(fishing, list) else []  # type: ignore[assignment]
        if fishing:
            fish_choices: dict[str, str] = {}
            for entry in fishing:
                entry_dict = entry if isinstance(entry, dict) else {}  # type: ignore[assignment]
                fish_choices[str(entry_dict.get("path", ""))] = str(entry_dict.get("label", ""))
            choices[f"Fishing ({len(fishing)})"] = fish_choices

        # Other
        other = catalog.get("other", [])
        other = other if isinstance(other, list) else []  # type: ignore[assignment]
        if other:
            other_choices: dict[str, str] = {}
            for entry in other:
                entry_dict = entry if isinstance(entry, dict) else {}  # type: ignore[assignment]
                other_choices[str(entry_dict.get("path", ""))] = str(entry_dict.get("label", ""))
            choices[f"Other ({len(other)})"] = other_choices

        if not choices:
            return ui.p(
                "No spatial files found in this configuration.",
                style="color: var(--osm-text-muted); font-size: 12px; padding: 8px;",
            )

        return ui.input_select(
            "map_viewer_file",
            "Select a file to preview",
            choices=choices,  # type: ignore[arg-type]
            size="20",
        )

    @render.ui
    def map_viewer_nc_controls():
        """Show time slider for NC files with time dimension."""
        try:
            file_val = input.map_viewer_file()
        except SilentException:
            return ui.div()

        if not file_val:
            return ui.div()

        file_path = Path(file_val)
        if file_path.suffix != ".nc" or not file_path.exists():
            return ui.div()

        meta = list_nc_overlay_variables(str(file_path))
        if not meta:
            return ui.div()

        controls = []

        if len(meta) > 1:
            var_choices = {k: k.replace("_", " ").title() for k in meta}
            try:
                current_var = input.mv_nc_var()
            except SilentException:
                current_var = next(iter(meta))
            if current_var not in meta:
                current_var = next(iter(meta))
            controls.append(
                ui.input_select("mv_nc_var", "Variable", choices=var_choices, selected=current_var)
            )
            sel_var = current_var
        else:
            sole_var = next(iter(meta))
            controls.append(
                ui.div(
                    ui.input_select(
                        "mv_nc_var", "Variable", choices={sole_var: sole_var}, selected=sole_var
                    ),
                    style="display:none",
                )
            )
            sel_var = sole_var

        var_meta = meta.get(sel_var, {})
        n_time = var_meta.get("n_time", 1)
        if n_time > 1:
            try:
                current_step = int(input.mv_nc_time())
            except (SilentException, ValueError, TypeError):
                current_step = 0
            current_step = max(0, min(current_step, n_time - 1))
            controls.append(
                ui.input_slider(
                    "mv_nc_time", "Time step", min=0, max=n_time - 1, value=current_step, step=1
                )
            )

        return ui.div(*controls) if controls else ui.div()

    @render.ui
    def map_viewer_metadata():
        """Show file metadata below the map."""
        try:
            file_val = input.map_viewer_file()
        except SilentException:
            return ui.div()

        if not file_val:
            return ui.div()

        file_path = Path(file_val)
        if not file_path.exists():
            return ui.div()

        parts = [
            ui.tags.span(file_path.name, style="color: var(--osm-text-muted); font-size: 11px;")
        ]
        return ui.div(
            *parts,
            style="padding: 4px 8px; font-size: 11px; color: var(--osm-text-muted);",
        )

    @reactive.effect
    async def update_map_viewer():
        try:
            file_val = input.map_viewer_file()
        except (SilentException, AttributeError):
            return

        if not file_val:
            return

        file_path = Path(file_val)
        if not file_path.exists():
            return

        is_dark = get_theme_mode(input) == "dark"

        with reactive.isolate():
            cfg = state.config.get()
            cfg_dir = state.config_dir.get()

        # Read bbox values unconditionally (needed for CSV overlay even when NcGrid)
        try:
            ul_lat = float(cfg.get("grid.upleft.lat", 0))
            ul_lon = float(cfg.get("grid.upleft.lon", 0))
            lr_lat = float(cfg.get("grid.lowright.lat", 0))
            lr_lon = float(cfg.get("grid.lowright.lon", 0))
            nx = int(float(cfg.get("grid.nlon", 0)))
            ny = int(float(cfg.get("grid.nlat", 0)))
        except (ValueError, TypeError):
            ul_lat = ul_lon = lr_lat = lr_lon = 0.0
            nx = ny = 0

        # Load grid base layers
        is_ncgrid = "NcGrid" in cfg.get("grid.java.classname", "")
        nc_data = load_netcdf_grid(cfg, config_dir=cfg_dir) if is_ncgrid else None

        if nc_data is not None:
            nc_lat, nc_lon, nc_mask = nc_data
            layers, view_state = build_netcdf_grid_layers(nc_lat, nc_lon, nc_mask, is_dark)
        else:
            mask = load_mask(cfg, config_dir=cfg_dir)
            layers = build_grid_layers(ul_lat, ul_lon, lr_lat, lr_lon, nx, ny, is_dark, mask)
            if ul_lat != 0 or ul_lon != 0 or lr_lat != 0 or lr_lon != 0:
                center_lat = (ul_lat + lr_lat) / 2
                center_lon = (ul_lon + lr_lon) / 2
                span = max(abs(ul_lat - lr_lat), abs(lr_lon - ul_lon))
                view_state = {
                    "latitude": center_lat,
                    "longitude": center_lon,
                    "zoom": _zoom_for_span(span),
                }
            else:
                view_state = {"latitude": 46.0, "longitude": -4.5, "zoom": 5}

        # Load overlay
        legend_entries = []
        if file_path.suffix == ".nc":
            try:
                nc_var = input.mv_nc_var() or None
            except (SilentException, AttributeError):
                nc_var = None
            try:
                nc_time = max(0, int(input.mv_nc_time()))
            except (SilentException, ValueError, TypeError, AttributeError):
                nc_time = 0
            meta = list_nc_overlay_variables(str(file_path))
            nc_vmin = nc_vmax = None
            if meta:
                sel_meta = meta.get(nc_var or "") or next(iter(meta.values()), None)
                if sel_meta:
                    nc_vmin, nc_vmax = sel_meta["vmin"], sel_meta["vmax"]
            fb_lat = nc_data[0] if nc_data else None
            fb_lon = nc_data[1] if nc_data else None
            cells = load_netcdf_overlay(
                file_path,
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
                        "viewer-overlay",
                        data=cells,
                        getPolygon="@@=d.polygon",
                        getFillColor="@@=d.fill",
                        getLineColor=[0, 0, 0, 0],
                        filled=True,
                        stroked=False,
                        pickable=True,
                    )
                )
                label = (nc_var or "Overlay").replace("_", " ").title()
                legend_entries.append(
                    {
                        "layer_id": "viewer-overlay",
                        "label": label,
                        "color": [0, 170, 180],
                        "shape": "rect",
                    }
                )
        elif file_path.suffix == ".csv":
            if nc_data is not None:
                cells = load_csv_overlay(file_path, 0, 0, 0, 0, 0, 0, nc_data=nc_data)
            else:
                cells = load_csv_overlay(file_path, ul_lat, ul_lon, lr_lat, lr_lon, nx, ny)
            if cells:
                layers.append(
                    polygon_layer(
                        "viewer-overlay",
                        data=cells,
                        getPolygon="@@=d.polygon",
                        getFillColor="@@=d.fill",
                        getLineColor=[0, 0, 0, 0],
                        filled=True,
                        stroked=False,
                        pickable=True,
                    )
                )
                legend_entries.append(
                    {
                        "layer_id": "viewer-overlay",
                        "label": _overlay_label(file_path.name),
                        "color": [255, 140, 0],
                        "shape": "rect",
                    }
                )

        # Widgets
        style = CARTO_DARK if is_dark else CARTO_POSITRON
        if style != _map.style:
            _map.style = style
            await _map.set_style(session, style)

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
            session, layers=layers, view_state=view_state, transition_duration=800, widgets=widgets
        )
