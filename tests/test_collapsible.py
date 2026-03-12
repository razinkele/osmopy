"""Tests for collapsible panel helpers."""

from ui.components.collapsible import collapsible_card_header, expand_tab


def _to_html(obj):
    """Convert a Shiny UI object to its HTML string."""
    if hasattr(obj, "resolve"):
        return str(obj.resolve())
    return str(obj)


def test_collapsible_card_header_renders():
    """collapsible_card_header returns a card header with collapse button."""
    header = collapsible_card_header("Grid Type", "grid")
    html = _to_html(header)
    assert "Grid Type" in html
    assert "osm-collapse-btn" in html
    assert "togglePanel" in html
    assert "grid" in html


def test_expand_tab_renders():
    """expand_tab returns a div with vertical expand tab."""
    tab = expand_tab("Grid Type", "grid")
    html = _to_html(tab)
    assert "Grid Type" in html
    assert "osm-expand-tab" in html
    assert "expand_grid" in html
    assert "togglePanel" in html
    assert "grid" in html


def test_collapsible_card_header_different_pages():
    """Helper generates unique IDs per page."""
    h1 = _to_html(collapsible_card_header("Forcing", "forcing"))
    h2 = _to_html(collapsible_card_header("Fishing", "fishing"))
    assert "forcing" in h1
    assert "fishing" in h2
    assert "togglePanel" in h1
    assert "togglePanel" in h2


def test_expand_tab_different_pages():
    """expand_tab generates unique IDs per page."""
    t1 = _to_html(expand_tab("Forcing", "forcing"))
    t2 = _to_html(expand_tab("Fishing", "fishing"))
    assert "expand_forcing" in t1
    assert "expand_fishing" in t2
