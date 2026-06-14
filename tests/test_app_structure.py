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


def test_nav_collapse_toggle_present():
    """Nav collapse button is in the rendered HTML."""
    from app import app_ui

    html = str(app_ui)
    assert "osm-nav-collapse-btn" in html
    assert "toggleNav()" in html


def test_persist_hook_wired_in_calibration_handlers():
    """The live sensitivity run persists its result via save_sobol_result."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent / "ui" / "pages" / "calibration_handlers.py"
    ).read_text()
    assert "save_sobol_result" in src


def test_sensitivity_panel_present():
    """The Sensitivity page is wired into app_ui with its full widget set."""
    from app import app_ui

    html = str(app_ui)
    # The Calibration page already contains the substring "Sensitivity" (a sub-tab),
    # so assert on the new top-level nav panel's unique lowercase value instead.
    assert 'data-value="sensitivity"' in html
    for wid in [
        "sens_run",
        "sens_objective_ui",
        "sens_index",
        "sens_threshold",
        "sens_sort",
        "sens_tornado",
        "sens_table",
        "sens_export_csv",
        "sens_export_keys",
    ]:
        assert wid in html, f"Missing widget id: {wid}"


def test_sensitivity_server_wired():
    """app.py calls sensitivity_explorer_server."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent / "app.py").read_text()
    assert "sensitivity_explorer_server" in src
    assert "sensitivity_explorer_ui" in src
