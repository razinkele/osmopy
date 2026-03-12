"""Test app module structure and navigation layout."""

from shiny import App


def test_app_imports():
    """App module can be imported without error."""
    from app import app

    assert isinstance(app, App)


def test_app_ui_is_page_fillable():
    """Top-level UI uses page_fillable (not page_navbar)."""
    from app import app_ui

    # page_fillable returns a Tag; check it renders without error
    html = str(app_ui)
    assert "nav-pills" in html or "pill" in html.lower()


def test_nav_sections_present():
    """All 10 nav panels are present in the rendered HTML."""
    from app import app_ui

    html = str(app_ui)
    expected_labels = [
        "Setup",
        "Grid",
        "Forcing",
        "Fishing",
        "Movement",
        "Run",
        "Results",
        "Calibration",
        "Scenarios",
        "Advanced",
    ]
    for label in expected_labels:
        assert label in html, f"Missing nav panel: {label}"


def test_section_headers_present():
    """Grouped section headers appear in the navigation."""
    from app import app_ui

    html = str(app_ui)
    for header in ["Configure", "Execute", "Optimize", "Manage"]:
        assert header in html, f"Missing section header: {header}"


def test_hamburger_toggle_present():
    """Hamburger nav toggle button is in the rendered HTML."""
    from app import app_ui

    html = str(app_ui)
    assert "osm-hamburger" in html
    assert "toggleNav()" in html
