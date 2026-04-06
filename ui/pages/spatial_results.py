"""Spatial Results page — post-run spatial output visualization."""

from pathlib import Path

from shiny import ui, reactive, render
from shiny.types import SilentException
from shinywidgets import output_widget, render_plotly  # type: ignore[import-untyped]

import numpy as np
import plotly.graph_objects as go

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
from osmose.results import OsmoseResults
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.pages.grid_helpers import (
    build_grid_layers,
    build_netcdf_grid_layers,
    load_netcdf_grid,
    make_legend,
    make_spatial_map,
    _zoom_for_span,
)
from ui.state import get_theme_mode

_log = setup_logging("osmose.spatial_results")

# Map NC filename patterns to human-readable labels
_NC_LABELS: dict[str, str] = {
    "biomass": "Biomass",
    "abundance": "Abundance",
    "yield": "Yield",
    "size": "Size",
    "ltl": "LTL",
    "meanTL": "Trophic Level",
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
    pass
