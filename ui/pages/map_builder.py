"""Map Builder page — draw/paint spatial maps (distribution / mask / zone) onto the grid.

The pure map-editing core lives in ``osmose.maps.builder`` (GridSpec / MapGrid / save_map);
this module is the Shiny wiring only.
"""

from __future__ import annotations

import numpy as np
from shiny import reactive, render, ui
from shiny.types import SilentException

from shiny_deckgl import (  # type: ignore[import-untyped]
    CARTO_DARK,
    CARTO_POSITRON,
    MapWidget,
    compass_widget,
    fullscreen_widget,
    polygon_layer,
    scale_widget,
    zoom_widget,
)

from osmose.logging import setup_logging
from osmose.maps.builder import GridSpec, MapGrid, lonlat_to_cell
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.components.renderer_badge import renderer_badge
from ui.pages.grid_helpers import _zoom_for_span, build_grid_layers, load_mask
from ui.state import get_theme_mode

_log = setup_logging("osmose.map_builder")

_DEFAULT_VIEW_STATE = {"latitude": 46.0, "longitude": -4.5, "zoom": 5, "pitch": 0, "bearing": 0}

# MapWidget id — determines input names input.mb_map_map_click / input.mb_map_drawn_features.
_MAP_ID = "mb_map"


def map_builder_ui():
    builder_map = MapWidget(
        f"{_MAP_ID}",
        view_state=_DEFAULT_VIEW_STATE,
        style=CARTO_POSITRON,
        tooltip={"html": "Value: {properties.value}", "style": {"fontSize": "12px"}},
        controls=[],
    )

    tools_card = ui.card(
        collapsible_card_header("Map Builder", "map_builder"),
        ui.output_ui("mb_hint"),
        ui.input_radio_buttons(
            "map_type",
            "Map type",
            choices={
                "distribution": "Distribution (movement)",
                "mask": "Land mask",
                "zone": "Generic zone",
            },
            selected="distribution",
        ),
        ui.input_radio_buttons(
            "tool_mode",
            "Tool",
            choices={
                "polygon": "Polygon draw",
                "brush": "Brush",
                "eraser": "Eraser",
                "mask": "Mask edit",
            },
            selected="polygon",
        ),
        ui.input_numeric("paint_value", "Paint value", value=1, step=1),
        ui.div(
            ui.input_action_button("mb_new_blank", "New blank map", class_="btn-sm"),
            ui.output_ui("mb_load_existing"),
            class_="d-flex flex-column gap-2",
        ),
        ui.output_ui("mb_staged_indicator"),
        ui.input_action_button("apply_polygons", "Apply polygon(s)", class_="btn-sm btn-primary"),
        ui.output_ui("mb_applicability"),
        ui.hr(),
        ui.input_text("mb_filename", "Filename", placeholder="my_map.csv"),
        ui.input_action_button("mb_save", "Save map", class_="btn-sm btn-success"),
    )

    return ui.div(
        expand_tab("Map Builder", "map_builder"),
        ui.layout_columns(
            tools_card,
            ui.div(
                builder_map.ui(height="100%"),
                renderer_badge(),
                class_="osm-grid-map-container",
            ),
            col_widths=[5, 7],
        ),
        class_="osm-split-layout",
        id="split_map_builder",
    )


def map_builder_server(input, output, session, state):
    _map = MapWidget(
        f"{_MAP_ID}",
        view_state=_DEFAULT_VIEW_STATE,
        style=CARTO_POSITRON,
    )

    # --- editing state -----------------------------------------------------
    grid_array: reactive.Value[MapGrid | None] = reactive.Value(None)
    base_mask: reactive.Value[np.ndarray | None] = reactive.Value(None)
    dirty: reactive.Value[int] = reactive.Value(0)

    def _grid_spec() -> GridSpec | None:
        """Build a GridSpec from the active config, or None if no usable grid."""
        with reactive.isolate():
            cfg = state.config.get()
        if not cfg:
            return None
        try:
            return GridSpec.from_config(cfg)
        except (KeyError, ValueError, TypeError):
            return None

    def _paint_value() -> float:
        try:
            return float(input.paint_value() or 0)
        except (SilentException, ValueError, TypeError):
            return 0.0

    def _new_blank() -> None:
        grid = _grid_spec()
        if grid is None:
            return
        with reactive.isolate():
            cfg = state.config.get()
            cfg_dir = state.config_dir.get()
        mask = load_mask(cfg, config_dir=cfg_dir)
        base_mask.set(mask)
        grid_array.set(MapGrid.blank(grid, base_mask=mask))
        dirty.set(dirty.get() + 1)

    @render.ui
    def mb_hint():
        state.load_trigger.get()
        grid = _grid_spec()
        if grid is None:
            return ui.p(
                "Load a configuration with a regular lon/lat grid to build maps.",
                style="color: var(--osm-text-muted); padding: 8px;",
            )
        return ui.div()

    @render.ui
    def mb_load_existing():
        return ui.div()

    @render.ui
    def mb_staged_indicator():
        return ui.div()

    @render.ui
    def mb_applicability():
        return ui.div()

    # --- new-blank trigger -------------------------------------------------
    @reactive.effect
    @reactive.event(input.mb_new_blank, ignore_none=False)
    def _on_new_blank():
        _new_blank()

    # --- initial render + draw enablement, gated on deckgl_ready -----------
    @reactive.effect
    @reactive.event(input.deckgl_ready)
    async def _on_deckgl_ready():
        if grid_array.get() is None:
            _new_blank()
        await _render_full()
        try:
            mode = input.tool_mode()
        except SilentException:
            mode = "polygon"
        if mode == "polygon":
            await _map.enable_draw(session, modes=["draw_polygon"], default_mode="draw_polygon")

    def _view_state(grid: GridSpec) -> dict:
        center_lat = (grid.upleft_lat + grid.lowright_lat) / 2
        center_lon = (grid.upleft_lon + grid.lowright_lon) / 2
        span = max(
            abs(grid.upleft_lat - grid.lowright_lat),
            abs(grid.lowright_lon - grid.upleft_lon),
        )
        return {
            "latitude": center_lat,
            "longitude": center_lon,
            "zoom": _zoom_for_span(span),
        }

    def _value_cells_layer(mg: MapGrid, grid: GridSpec) -> dict | None:
        """Value-colored cells layer for painted (non-zero, non-land) cells."""
        arr = mg.array
        rs, cs = np.where((arr != 0) & (arr != -99))
        if len(rs) == 0:
            return None
        vals = arr[rs, cs]
        vmin, vmax = float(vals.min()), float(vals.max())
        vrange = vmax - vmin if vmax != vmin else 1.0
        cells = []
        for i in range(len(rs)):
            r, c = int(rs[i]), int(cs[i])
            v = float(vals[i])
            t = (v - vmin) / vrange
            r_ch = int(np.clip(68 + 187 * t * t, 0, 255))
            g_ch = int(np.clip(1 + 209 * t, 0, 255))
            b_ch = int(np.clip(84 + 86 * t - 170 * t * t, 0, 255))
            a_ch = int(np.clip(150 + 80 * t, 0, 255))
            cells.append(
                {
                    "polygon": grid.cell_polygon(r, c),
                    "value": v,
                    "fill": [r_ch, g_ch, b_ch, a_ch],
                }
            )
        return polygon_layer(
            "mb-value-cells",
            data=cells,
            getPolygon="@@=d.polygon",
            getFillColor="@@=d.fill",
            getLineColor=[0, 0, 0, 0],
            filled=True,
            stroked=False,
            pickable=True,
        )

    def _build_layers(mg: MapGrid, grid: GridSpec, is_dark: bool) -> list[dict]:
        layers = build_grid_layers(
            grid.upleft_lat,
            grid.upleft_lon,
            grid.lowright_lat,
            grid.lowright_lon,
            grid.nlon,
            grid.nlat,
            is_dark,
            base_mask.get(),
        )
        cells_layer = _value_cells_layer(mg, grid)
        if cells_layer is not None:
            layers.append(cells_layer)
        return layers

    async def _render_full() -> None:
        grid = _grid_spec()
        mg = grid_array.get()
        if grid is None or mg is None:
            return
        is_dark = get_theme_mode(input) == "dark"
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
        await _map.update(
            session,
            layers=_build_layers(mg, grid, is_dark),
            view_state=_view_state(grid),
            transition_duration=600,
            widgets=widgets,
        )

    # --- tool-mode toggle: enable polygon draw only in polygon mode --------
    @reactive.effect
    @reactive.event(input.tool_mode)
    async def _on_tool_mode():
        try:
            mode = input.tool_mode()
        except SilentException:
            return
        if mode == "polygon":
            await _map.enable_draw(session, modes=["draw_polygon"], default_mode="draw_polygon")
        else:
            await _map.disable_draw(session)

    # --- single-cell paint via map click (brush/eraser/mask) ---------------
    @reactive.effect
    @reactive.event(getattr(input, f"{_MAP_ID}_map_click"))
    def _on_map_click():
        try:
            mode = input.tool_mode()
        except SilentException:
            return
        if mode not in ("brush", "eraser", "mask"):
            return
        click = getattr(input, f"{_MAP_ID}_map_click")()
        if not isinstance(click, dict):
            return
        lon = click.get("longitude")
        lat = click.get("latitude")
        if lon is None or lat is None:
            return
        grid = _grid_spec()
        mg = grid_array.get()
        if grid is None or mg is None:
            return
        cell = lonlat_to_cell(grid, float(lon), float(lat))
        if cell is None:
            return
        r, c = cell
        # Block painting on a -99 (land) cell unless we are in mask mode.
        if mode != "mask" and mg.array[r, c] == -99:
            return
        if mode == "brush":
            mg.apply_cells([cell], _paint_value())
        elif mode == "eraser":
            mg.erase([cell])
        elif mode == "mask":
            mg.set_mask([cell], True)
        dirty.set(dirty.get() + 1)
