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
            **{"aria-label": "Collapse panel", "aria-expanded": "true"},
        ),
    )


def body_collapse_header(title: str, panel_id: str):
    """Card header with a button that collapses the card's *body* vertically.

    Unlike :func:`collapsible_card_header` (which collapses a left column
    sideways into an :func:`expand_tab`), this toggles a ``osm-body-collapsed``
    class on the enclosing ``.card``, hiding everything below the header while
    leaving the header bar visible. State is persisted in localStorage keyed by
    ``panel_id`` and restored client-side (see ``toggleCardBody`` in app.py).

    Reuses the ``osm-collapse-btn`` styling so it matches the sideways-collapse
    buttons used elsewhere.

    Parameters
    ----------
    title
        Text displayed in the card header.
    panel_id
        Unique identifier used for localStorage persistence (must be unique
        across all body-collapsible cards on the page).
    """
    return _ui.card_header(
        _ui.tags.span(title),
        _ui.tags.button(
            "«",
            class_="osm-collapse-btn",
            onclick="toggleCardBody(this)",
            title="Collapse panel",
            **{
                "data-osm-card-toggle": panel_id,
                "aria-label": "Collapse panel",
                "aria-expanded": "true",
            },
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
    return _ui.tags.button(
        title,
        class_="osm-expand-tab",
        id=f"expand_{page_id}",
        onclick=f"togglePanel('{page_id}')",
        **{"aria-label": "Expand panel", "aria-expanded": "false"},
    )
