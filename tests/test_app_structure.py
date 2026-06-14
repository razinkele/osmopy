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


def test_about_modal_renders_doc_tabs():
    from ui.components.help_modal import about_modal

    html = str(about_modal())
    # Assert the navset TAB structure (data-value=title) — discriminating vs the old modal,
    # which merely had a "### Changelog" heading substring.
    assert 'data-value="Overview"' in html
    assert 'data-value="README"' in html
    assert 'data-value="Changelog"' in html
    # Renders real CHANGELOG content (build-time read), not the old hardcoded block.
    assert "0.13.0" in html
    # The stale hardcoded-changelog marker is gone (was a list item "Initial release").
    assert "Initial release" not in html


def test_changelog_modal_present():
    from ui.components.help_modal import changelog_modal

    html = str(changelog_modal())
    assert "changelogModal" in html
    assert "What's new" in html


def test_startup_changelog_wired_in_app():
    from app import app_ui

    html = str(app_ui)
    assert "changelogModal" in html
    assert "window.OSMOSE_VERSION" in html
    assert "osmose_seen_changelog_version" in html


def test_feedback_modal_has_form_ids():
    from ui.components.feedback_modal import feedback_modal

    html = str(feedback_modal())
    assert "feedbackModal" in html
    for wid in ["feedback_type", "feedback_message", "feedback_contact", "feedback_submit"]:
        assert wid in html, f"Missing feedback widget id: {wid}"
