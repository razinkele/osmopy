"""Map Builder page — draw/paint spatial maps (distribution / mask / zone) onto the grid.

The pure map-editing core lives in ``osmose.maps.builder`` (GridSpec / MapGrid / save_map);
this module is the Shiny wiring only.
"""

from __future__ import annotations

from shiny import reactive, render, ui

from shiny_deckgl import (  # type: ignore[import-untyped]
    CARTO_POSITRON,
    MapWidget,
)

from osmose.logging import setup_logging
from osmose.maps.builder import GridSpec
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.components.renderer_badge import renderer_badge

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
