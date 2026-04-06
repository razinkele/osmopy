"""End-to-end tests for the Spatial Results page.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_spatial_results.py -v -m e2e

The Spatial Results pill is disabled until a simulation run completes.
These tests verify the disabled state and basic page functionality.
"""

import re

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000
_NAV_TIMEOUT = 10_000


def test_spatial_results_pill_disabled_initially(page: Page, app: ShinyAppProc):
    """Spatial Results pill should be disabled before any run."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    pill = page.locator(".nav-pills .nav-link[data-value='spatial_results']")
    expect(pill).to_be_visible(timeout=_NAV_TIMEOUT)

    # Should have osm-disabled class (use Playwright retry, not sleep)
    expect(pill).to_have_class(re.compile(r"osm-disabled"), timeout=_NAV_TIMEOUT)


def test_spatial_results_pill_exists_in_execute_section(page: Page, app: ShinyAppProc):
    """Spatial Results pill should appear in the Execute section of navigation."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    pill = page.locator(".nav-pills .nav-link[data-value='spatial_results']")
    expect(pill).to_be_visible(timeout=_NAV_TIMEOUT)
    assert pill.text_content().strip() == "Spatial Results"


def test_grid_tab_renamed_to_grid(page: Page, app: ShinyAppProc):
    """Grid tab should be labeled 'Grid' (not 'Grid & Maps')."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)

    grid_pill = page.locator(".nav-pills .nav-link[data-value='grid']")
    expect(grid_pill).to_be_visible(timeout=_NAV_TIMEOUT)
    assert grid_pill.text_content().strip() == "Grid"

    # Old name should not exist
    old_pills = page.locator(".nav-pills .nav-link:has-text('Grid & Maps')")
    assert old_pills.count() == 0, "Old 'Grid & Maps' pill should not exist"
