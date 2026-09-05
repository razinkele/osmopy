"""Shared shell for "coming soon" module pages (Genetics, Economic).

Both pages render the same single-card split layout with a collapsible header
and a body that is either an engine-mode hint or a short description plus a
bullet list of planned parameters. Keeping the shell here means adding another
placeholder module is a two-function page (``*_ui`` / ``*_server``) with no
copy-pasted markup.
"""

from __future__ import annotations

from collections.abc import Sequence

from htmltools import Tag
from shiny import ui

from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.styles import STYLE_EMPTY

_BULLET_STYLE = "color: var(--osm-text-muted); font-size: 0.82rem;"


def placeholder_ui(page_id: str, title: str, output_id: str) -> Tag:
    """Single-card split layout whose body is a ``ui.output_ui(output_id)``."""
    return ui.div(
        expand_tab(title, page_id),
        ui.layout_columns(
            ui.card(
                collapsible_card_header(title, page_id),
                ui.output_ui(output_id),
            ),
            col_widths=[12],
        ),
        class_="osm-split-layout",
        id=f"split_{page_id}",
    )


def placeholder_content(
    heading: str,
    intro: str,
    note: str,
    bullets: Sequence[str],
) -> Tag:
    """Body shown when the module is reachable but not yet implemented."""
    return ui.div(
        ui.h5(heading),
        ui.p(intro),
        ui.hr(),
        ui.p(note, style=STYLE_EMPTY),
        ui.tags.ul(*(ui.tags.li(b) for b in bullets), style=_BULLET_STYLE),
    )


def engine_mode_hint(module_name: str) -> Tag:
    """Body shown when the Java engine is selected and the module is Python-only."""
    return ui.p(f"Switch to Python engine to access {module_name}.", style=STYLE_EMPTY)
