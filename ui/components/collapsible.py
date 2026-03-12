"""Collapsible panel helpers for page layouts."""

from shiny import ui as _ui


def collapsible_card_header(title: str, page_id: str):
    """Card header with a collapse toggle button.

    Parameters
    ----------
    title
        Text displayed in the card header.
    page_id
        Unique identifier used for localStorage persistence and DOM targeting.
    """
    return _ui.card_header(
        _ui.tags.span(title),
        _ui.tags.button(
            "\u00ab",
            class_="osm-collapse-btn",
            onclick=f"togglePanel('{page_id}')",
            title="Collapse panel",
        ),
    )


def expand_tab(title: str, page_id: str):
    """Vertical expand tab shown when the left panel is collapsed.

    Placed as a flex sibling before the layout_columns `.row` inside
    an `osm-split-layout` wrapper div.

    Parameters
    ----------
    title
        Text displayed vertically on the tab.
    page_id
        Must match the page_id used in collapsible_card_header.
    """
    return _ui.div(
        title,
        class_="osm-expand-tab",
        id=f"expand_{page_id}",
        onclick=f"togglePanel('{page_id}')",
    )
