"""Spatial Results page — post-run spatial output visualization."""

from pathlib import Path

from shiny import ui, reactive, render
from shiny.types import SilentException
from shinywidgets import output_widget, render_plotly  # type: ignore[import-untyped]

import numpy as np
import plotly.graph_objects as go

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
from osmose.results import OsmoseResults
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.pages.grid_helpers import (
    make_legend,
    make_spatial_map,
    _zoom_for_span,
)
from ui.state import get_theme_mode

_log = setup_logging("osmose.spatial_results")

# Map NC filename patterns to human-readable labels
_NC_LABELS: dict[str, str] = {
    "ltl": "LTL",
    "biomass": "Biomass",
    "abundance": "Abundance",
    "yield": "Yield",
    "size": "Size",
    "meantl": "Trophic Level",
}


def _nc_label(filename: str) -> str:
    """Derive a human-readable label from a NetCDF output filename."""
    stem = Path(filename).stem.lower()
    for key, label in _NC_LABELS.items():
        if key.lower() in stem:
            return label
    return Path(filename).stem.replace("_", " ").title()


def _get_var_name(ds, input) -> str | None:
    """Get the selected spatial variable name from the dataset."""
    var_names = [v for v in ds.data_vars if "lat" in ds[v].dims and "lon" in ds[v].dims]
    if not var_names:
        return None
    try:
        selected = input.spatial_species()
    except (SilentException, AttributeError):
        selected = "__all__"
    if selected == "__all__" or selected not in var_names:
        return "biomass" if "biomass" in var_names else var_names[0]
    return selected


def spatial_results_ui():
    spatial_map = MapWidget(
        "spatial_map",
        view_state={
            "latitude": 46.0,
            "longitude": -4.5,
            "zoom": 5,
            "pitch": 0,
            "bearing": 0,
        },
        style=CARTO_POSITRON,
        tooltip={
            "html": "({properties.row}, {properties.col})<br>Value: {properties.value}",
            "style": {"fontSize": "12px"},
        },
        controls=[],
    )

    return ui.div(
        expand_tab("Spatial Results", "spatial_results"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Spatial Results", "spatial_results"),
                ui.output_ui("spatial_nc_selector"),
                ui.output_ui("spatial_var_selector"),
                ui.output_ui("spatial_time_controls"),
                ui.output_ui("spatial_scale_info"),
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Map View",
                    ui.div(
                        spatial_map.ui(height="100%"),
                        class_="osm-grid-map-container",
                    ),
                ),
                ui.nav_panel(
                    "Flat View",
                    output_widget("spatial_flat_chart"),
                ),
            ),
            col_widths=[5, 7],
        ),
        class_="osm-split-layout",
        id="split_spatial_results",
    )


def spatial_results_server(input, output, session, state):
    """Server logic for the Spatial Results page."""
    _spatial_ds = reactive.Value(None)  # xarray Dataset
    _spatial_nc_files = reactive.Value([])  # list of NC filenames
    _prev_output_dir = reactive.Value(None)

    # Lightweight handle for map updates
    _map = MapWidget(
        "spatial_map",
        view_state={"latitude": 46.0, "longitude": -4.5, "zoom": 5},
        style=CARTO_POSITRON,
    )

    # ── Pill disabled state ──────────────────────────────────────
    @reactive.effect
    async def _toggle_pill():
        out_dir = state.output_dir.get()
        has_results = out_dir is not None and out_dir.exists()
        action = "remove" if has_results else "add"
        await session.send_custom_message(
            "toggle-spatial-pill",
            {"action": action},
        )

    # ── Auto-load NC files when output_dir changes ───────────────
    @reactive.effect
    def _auto_load_spatial():
        out_dir = state.output_dir.get()
        if out_dir is None or not out_dir.exists():
            _spatial_ds.set(None)
            _spatial_nc_files.set([])
            return
        if out_dir == _prev_output_dir.get():
            return
        _prev_output_dir.set(out_dir)

        res = OsmoseResults(out_dir, strict=False)
        nc_files = [f for f in res.list_outputs() if f.endswith(".nc")]
        _spatial_nc_files.set(nc_files)

        if nc_files:
            _spatial_ds.set(res.read_netcdf(nc_files[0]))

    # ── NC file selector ─────────────────────────────────────────
    @render.ui
    def spatial_nc_selector():
        nc_files = _spatial_nc_files.get()
        if not nc_files:
            return ui.p(
                "No spatial outputs available. Run a simulation first.",
                style="color: var(--osm-text-muted); font-size: 12px; margin-top: 8px;",
            )
        choices = {f: _nc_label(f) for f in nc_files}
        return ui.input_select(
            "spatial_result_type", "Output type", choices=choices, selected=nc_files[0]
        )

    # ── Load dataset when NC file selection changes ──────────────
    @reactive.effect
    @reactive.event(input.spatial_result_type)
    def _load_selected_nc():
        out_dir = state.output_dir.get()
        if out_dir is None:
            return
        try:
            filename = input.spatial_result_type()
        except SilentException:
            return
        res = OsmoseResults(out_dir, strict=False)
        _spatial_ds.set(res.read_netcdf(filename))

    # ── Variable (species) selector ──────────────────────────────
    @render.ui
    def spatial_var_selector():
        ds = _spatial_ds.get()
        if ds is None:
            return ui.div()
        var_names = [v for v in ds.data_vars if "lat" in ds[v].dims and "lon" in ds[v].dims]
        if len(var_names) <= 1:
            return ui.div()
        choices = {"__all__": "All (sum)"}
        choices.update({v: v.replace("_", " ").title() for v in var_names})
        return ui.input_select(
            "spatial_species", "Variable / Species", choices=choices, selected="__all__"
        )

    # ── Time step slider ─────────────────────────────────────────
    @render.ui
    def spatial_time_controls():
        ds = _spatial_ds.get()
        if ds is None:
            return ui.div()
        n_time = ds.sizes.get("time", 1)
        if n_time <= 1:
            return ui.div()
        return ui.input_slider(
            "spatial_time",
            "Time step",
            min=0,
            max=n_time - 1,
            value=0,
            step=1,
            animate=ui.AnimationOptions(
                interval=500,
                loop=True,
                play_button="Play",
                pause_button="Pause",
            ),
        )

    # ── Color scale info ─────────────────────────────────────────
    @render.ui
    def spatial_scale_info():
        ds = _spatial_ds.get()
        if ds is None:
            return ui.div()
        var_name = _get_var_name(ds, input)
        if var_name is None:
            return ui.div()
        vals = ds[var_name].values
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        return ui.div(
            ui.p(
                f"Range: {vmin:.4g} \u2014 {vmax:.4g}",
                style="font-size: 12px; color: var(--osm-text-muted);",
            ),
        )

    # ── Flat View (Plotly heatmap) ───────────────────────────────
    @render_plotly
    def spatial_flat_chart():
        tmpl = "osmose-light" if get_theme_mode(input) == "light" else "osmose"
        ds = _spatial_ds.get()
        if ds is None:
            return go.Figure().update_layout(title="No spatial data loaded", template=tmpl)
        var_name = _get_var_name(ds, input)
        if var_name is None:
            return go.Figure().update_layout(title="No spatial variables found", template=tmpl)
        try:
            time_idx = int(input.spatial_time())
        except (SilentException, ValueError, TypeError):
            time_idx = 0
        max_t = ds.sizes.get("time", 1) - 1
        return make_spatial_map(ds, var_name, time_idx=min(time_idx, max_t), template=tmpl)

    # ── Map View (deck.gl polygon layer) ─────────────────────────
    @reactive.effect
    async def _update_spatial_map():
        ds = _spatial_ds.get()
        is_dark = get_theme_mode(input) == "dark"
        style = CARTO_DARK if is_dark else CARTO_POSITRON
        if style != _map.style:
            _map.style = style
            await _map.set_style(session, style)

        if ds is None:
            await _map.update(
                session,
                layers=[],
                view_state={"latitude": 46.0, "longitude": -4.5, "zoom": 5},
            )
            return

        var_name = _get_var_name(ds, input)
        if var_name is None:
            return

        try:
            time_idx = int(input.spatial_time())
        except (SilentException, ValueError, TypeError):
            time_idx = 0
        max_t = ds.sizes.get("time", 1) - 1
        time_idx = min(time_idx, max_t)

        data_slice = ds[var_name].isel(time=time_idx).values  # (ny, nx)
        lat_raw = ds["lat"].values
        lon_raw = ds["lon"].values

        # Collapse 2D coordinate arrays to 1D (NcGrid outputs may have lat[y,x])
        lat = lat_raw[:, 0] if lat_raw.ndim == 2 else lat_raw
        lon = lon_raw[0, :] if lon_raw.ndim == 2 else lon_raw

        vmin = float(np.nanmin(data_slice))
        vmax = float(np.nanmax(data_slice))
        val_range = vmax - vmin if vmax > vmin else 1.0

        # Cell spacing — hoisted above loop (regular grid assumption)
        ny, nx = data_slice.shape
        dlat = abs(float(lat[1] - lat[0])) if ny > 1 else 1.0
        dlon = abs(float(lon[1] - lon[0])) if nx > 1 else 1.0

        # Build polygon cells with value-mapped colors (Viridis-like)
        cells = []
        for r in range(ny):
            la = float(lat[r])
            for c in range(nx):
                val = float(data_slice[r, c])
                if np.isnan(val) or val == 0:
                    continue
                lo = float(lon[c])
                t = (val - vmin) / val_range
                red = int(min(255, max(0, 68 + t * 187)))
                green = int(min(255, max(0, 1 + t * 204)))
                blue = int(min(255, max(0, 84 - t * 84)))
                alpha = 180
                cells.append(
                    {
                        "polygon": [
                            [lo - dlon / 2, la - dlat / 2],
                            [lo + dlon / 2, la - dlat / 2],
                            [lo + dlon / 2, la + dlat / 2],
                            [lo - dlon / 2, la + dlat / 2],
                        ],
                        "fill": [red, green, blue, alpha],
                        "value": round(val, 4),
                        "row": r,
                        "col": c,
                    }
                )

        layers = []
        if cells:
            layers.append(
                polygon_layer(
                    "spatial-data",
                    data=cells,
                    get_polygon="@@=d.polygon",
                    get_fill_color="@@=d.fill",
                    get_line_color=[0, 0, 0, 0],
                    filled=True,
                    stroked=False,
                    pickable=True,
                )
            )

        legend_entries = []
        if cells:
            legend_entries.append(
                {
                    "layer_id": "spatial-data",
                    "label": f"{var_name} ({vmin:.2g}\u2013{vmax:.2g})",
                    "color": [68, 100, 84],
                    "shape": "rect",
                }
            )

        widgets = [
            fullscreen_widget(placement="top-left"),
            zoom_widget(placement="top-right"),
            compass_widget(placement="top-right"),
            scale_widget(placement="bottom-left"),
        ]
        if legend_entries:
            widgets.append(make_legend(legend_entries, placement="bottom-right"))

        # Compute view state from lat/lon bounds
        center_lat = float(np.mean(lat))
        center_lon = float(np.mean(lon))
        lat_span = float(lat.max() - lat.min()) if len(lat) > 1 else 10.0
        lon_span = float(lon.max() - lon.min()) if len(lon) > 1 else 10.0
        span = max(lat_span, lon_span)
        zoom = _zoom_for_span(span)

        view_state = {"latitude": center_lat, "longitude": center_lon, "zoom": zoom}

        await _map.update(session, layers=layers, view_state=view_state, widgets=widgets)
