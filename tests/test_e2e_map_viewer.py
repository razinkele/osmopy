"""E2E Playwright tests for the Map Viewer page.

Run: .venv/bin/python -m pytest tests/test_e2e_map_viewer.py -v -m e2e
"""

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_TIMEOUT = 25_000


def _load_eec_full_and_goto_map_viewer(page: Page, app: ShinyAppProc) -> None:
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15_000)
    page.select_option("#load_example", "eec_full")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=15_000)
    page.locator(".nav-pills .nav-link[data-value='map_viewer']").click()
    page.wait_for_timeout(2_000)


def test_map_viewer_tab_accessible(page: Page, app: ShinyAppProc):
    """Map Viewer tab should be reachable after loading EEC Full."""
    _load_eec_full_and_goto_map_viewer(page, app)
    selector = page.locator("#map_viewer_file")
    expect(selector).to_be_visible(timeout=_TIMEOUT)


def test_map_viewer_file_list_populated(page: Page, app: ShinyAppProc):
    """File list should have movement and fishing entries with optgroup headers."""
    _load_eec_full_and_goto_map_viewer(page, app)
    page.wait_for_function(
        "(document.querySelector('#map_viewer_file')?.options?.length ?? 0) >= 5",
        timeout=_TIMEOUT,
    )
    count = page.evaluate(
        "document.querySelector('#map_viewer_file')?.options?.length ?? 0"
    )
    assert count >= 10, f"Expected >=10 file options, got {count}"

    # Verify optgroup headers exist (grouped select)
    optgroups = page.locator("#map_viewer_file optgroup").all()
    assert len(optgroups) >= 2, (
        f"Expected >=2 optgroup headers (Movement, Fishing/Other), got {len(optgroups)}"
    )
    labels = [og.get_attribute("label") or "" for og in optgroups]
    assert any("Movement" in lbl for lbl in labels), f"Expected 'Movement' optgroup, got: {labels}"


def test_map_viewer_renders_csv_overlay(page: Page, app: ShinyAppProc):
    """Selecting a CSV file should render a viewer-overlay layer."""
    _load_eec_full_and_goto_map_viewer(page, app)
    page.wait_for_function(
        "(document.querySelector('#map_viewer_file')?.options?.length ?? 0) >= 5",
        timeout=_TIMEOUT,
    )
    # Select the first valid option using Playwright's select_option
    first_value = page.evaluate(
        """
        (() => {
            const sel = document.querySelector('#map_viewer_file');
            for (const opt of sel.options) {
                if (opt.value && !opt.disabled) return opt.value;
            }
            return '';
        })()
    """
    )
    assert first_value, "No selectable option found"
    page.select_option("#map_viewer_file", value=first_value)
    page.wait_for_function(
        """
        (() => {
            const inst = window.__deckgl_instances?.['map_viewer_map'];
            if (!inst || !inst.lastLayers) return false;
            return inst.lastLayers.some(l => l.id === 'viewer-overlay');
        })()
        """,
        timeout=_TIMEOUT,
    )


def test_map_viewer_empty_without_config(page: Page, app: ShinyAppProc):
    """Without loading a config, Map Viewer should show a hint."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15_000)
    page.locator(".nav-pills .nav-link[data-value='map_viewer']").click()
    page.wait_for_timeout(2_000)
    hint = page.locator("text=Load a configuration")
    expect(hint).to_be_visible(timeout=_TIMEOUT)
