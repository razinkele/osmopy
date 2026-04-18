"""End-to-end tests for reactive UI behavior using Playwright.

Run these tests explicitly with:
    .venv/bin/python -m pytest tests/test_e2e_reactive.py -v

They are excluded from the default test suite (``pytest -m 'not e2e'``)
because Playwright's event loop conflicts with pytest-asyncio async tests.
"""

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")


def test_app_loads_with_navigation(page: Page, app: ShinyAppProc):
    """App should load with all navigation tabs visible."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15000)

    # Core navigation tabs should be visible in the main pill list
    for tab in ["Setup", "Run", "Results", "Scenarios", "Advanced"]:
        loc = page.locator(f".nav-pills .nav-link:has-text('{tab}')")
        expect(loc.first).to_be_visible(timeout=5000)


def test_default_tab_is_domain(page: Page, app: ShinyAppProc):
    """Domain (grid) tab should be active by default — it hosts the example loader."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15000)

    active = page.locator("#main_nav .nav-link.active[data-value='grid']")
    expect(active).to_be_visible(timeout=5000)


def test_load_example_updates_species_count(page: Page, app: ShinyAppProc):
    """Loading bay_of_biscay example should update n_species and show config header."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15000)

    # Select bay_of_biscay example and click Load
    page.select_option("#load_example", "bay_of_biscay")
    page.click("#btn_load_example")

    # Wait for notification confirming load
    page.wait_for_selector(".shiny-notification", timeout=15000)

    # n_species should be updated (bay_of_biscay has species)
    n_species_input = page.locator("#n_species")
    n_species_val = n_species_input.input_value()
    assert int(n_species_val) > 0, f"Expected n_species > 0, got {n_species_val}"


def test_config_header_shows_after_load(page: Page, app: ShinyAppProc):
    """Loading an example should populate the config header with name and species count."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15000)

    # Load example
    page.select_option("#load_example", "bay_of_biscay")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=15000)

    # Config header should contain the config name (title-cased)
    header = page.locator("#config_header")
    expect(header).to_contain_text("Bay Of Biscay", timeout=10000)
    # Should also show species count
    expect(header).to_contain_text("species", timeout=5000)


def test_species_panels_render_after_load(page: Page, app: ShinyAppProc):
    """Loading an example should render species input panels with correct values."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15000)

    # Load an example
    page.select_option("#load_example", "bay_of_biscay")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=15000)

    # Wait for species panels to render
    page.wait_for_timeout(3000)

    # Species name input for first species should be visible and populated
    species_name_input = page.locator("#spt_species_name_0")
    expect(species_name_input).to_be_visible(timeout=10000)

    # Value should be a non-empty species name (not the default empty)
    species_name = species_name_input.input_value()
    assert len(species_name) > 0, "Expected species name to be populated"
    assert species_name != "", f"Expected non-empty species name, got '{species_name}'"


def test_navigation_preserves_state(page: Page, app: ShinyAppProc):
    """Switching tabs should preserve loaded config state."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15000)

    # Load example
    page.select_option("#load_example", "bay_of_biscay")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=15000)

    # Record n_species value
    n_species_before = page.locator("#n_species").input_value()
    assert int(n_species_before) > 0

    # Navigate to Advanced tab
    page.locator(".nav-pills .nav-link:has-text('Advanced')").click()
    page.wait_for_timeout(1500)

    # Navigate back to Setup
    page.locator(".nav-pills .nav-link:has-text('Setup')").click()
    page.wait_for_timeout(1500)

    # Config header should still show Bay Of Biscay (title-cased)
    header = page.locator("#config_header")
    expect(header).to_contain_text("Bay Of Biscay", timeout=5000)

    # n_species should still be the same
    n_species_after = page.locator("#n_species").input_value()
    assert n_species_after == n_species_before, (
        f"n_species changed after navigation: {n_species_before} -> {n_species_after}"
    )


def test_theme_toggle(page: Page, app: ShinyAppProc):
    """Dark/light theme toggle should change data-theme attribute."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15000)

    # Get initial theme
    initial_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")

    # Click theme toggle button
    page.click("#themeToggle")
    page.wait_for_timeout(500)

    # Theme should have changed
    new_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
    assert new_theme != initial_theme, (
        f"Theme should toggle: was '{initial_theme}', still '{new_theme}'"
    )

    # Toggle back
    page.click("#themeToggle")
    page.wait_for_timeout(500)
    restored_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
    assert restored_theme == initial_theme, (
        f"Theme should restore: was '{initial_theme}', got '{restored_theme}'"
    )
