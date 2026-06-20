"""End-to-end test for the live movement view on the Run page.

Run explicitly: .venv/bin/python -m pytest tests/test_e2e_live_movement.py -v -m e2e
"""

from __future__ import annotations

import pathlib
import re

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_REPO = pathlib.Path(__file__).resolve().parent.parent
_LOAD_TIMEOUT = 20_000
_RUN_TIMEOUT = 60_000


def test_live_movement_renders_during_python_run(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    dismiss_changelog_modal(page)

    # Load the Baltic example config via the Grid/Domain page loader (where #load_example
    # lives — grid.py:105; e2e precedent test_e2e_grid_maps.py:39-41).
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.select_option("#load_example", "baltic")  # "baltic" is a valid value (osmose/demo.py:71)
    page.click("#btn_load_example")
    # Wait for the load to settle before navigating (mirrors test_e2e_grid_maps.py:41) —
    # otherwise the run can start before state.config holds the Baltic config + movement maps.
    page.wait_for_selector(".shiny-notification", timeout=_LOAD_TIMEOUT)

    # Navigate to the Run page.
    page.locator(".nav-pills .nav-link[data-value='run']").click()

    # The engine defaults to Java (ui/state.py:65), so clicking #engineBtnPython is
    # REQUIRED (not defensive) to enable the Python-only live view (engine toggle buttons
    # #engineBtnJava — app.py:195, #engineBtnPython — app.py:201). It also switches the
    # engine_mode to "python" (app.py engine toggle), which reveals the Python panel_conditional
    # in run.py so #py_param_overrides becomes visible — that field lives inside that panel,
    # so the fill MUST come after the engine switch or Playwright sees a hidden element.
    page.locator("#engineBtnPython").click()

    # Shorten the run: 1 year. (#py_param_overrides on the Run page — run.py:217.)
    py_overrides = page.locator("#py_param_overrides")
    expect(py_overrides).to_be_visible(timeout=_LOAD_TIMEOUT)
    py_overrides.fill("simulation.time.nyear=1")

    # Enable the live movement view, then run.
    page.locator("#live_movement_view").click()
    page.locator("#btn_run").click()

    # The live map container renders (note: #live_map is a static basemap, present as soon
    # as the Run page is active — it does NOT prove the run started).
    expect(page.locator("#live_map")).to_be_visible(timeout=_LOAD_TIMEOUT)
    # Guard against a validation-blocked run with a clear diagnostic. "Validation failed"
    # is set on #run_status (run.py:486), NOT #live_movement_status — assert the right one.
    expect(page.locator("#run_status")).not_to_contain_text(
        "Validation failed", timeout=_LOAD_TIMEOUT
    )
    # The status reads "running" as soon as the run starts (_live_status_val is set in
    # handle_run before the executor dispatch), then "done" on completion.
    expect(page.locator("#live_movement_status")).to_contain_text("running", timeout=_RUN_TIMEOUT)
    expect(page.locator("#live_movement_status")).to_contain_text("done", timeout=_RUN_TIMEOUT)

    # Toggle to dots mode (re-renders the retained final frame).
    page.locator("#live_movement_mode").get_by_text("Dots").click()
    page.screenshot(path=str(_REPO / "screenshots" / "live_movement_e2e.png"))


def test_live_movement_cancel_path(page: Page, app: ShinyAppProc):
    """Cancelling mid-run leaves the retained frame and shows a 'cancelled' status
    (covers the terminal-snapshot-direct-set cancel branch, which has no unit test)."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    dismiss_changelog_modal(page)
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.select_option("#load_example", "baltic")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_LOAD_TIMEOUT)
    page.locator(".nav-pills .nav-link[data-value='run']").click()
    # Switch to the Python engine first — this activates the Python nav_panel that holds
    # #py_param_overrides (run.py:214-218), so the field is visible before we fill it.
    page.locator("#engineBtnPython").click()
    py_overrides = page.locator("#py_param_overrides")
    expect(py_overrides).to_be_visible(timeout=_LOAD_TIMEOUT)
    py_overrides.fill("simulation.time.nyear=10")  # ~10-14s warm; long cancel window
    page.locator("#live_movement_view").click()
    page.locator("#btn_run").click()
    # Gate the cancel on a REAL emitted frame — "running step N/M" appears only after the
    # observer has pushed a snapshot (bare "running" is set pre-dispatch and would not prove
    # the run is mid-flight; #live_map is a static basemap that resolves instantly).
    expect(page.locator("#live_movement_status")).to_contain_text(
        re.compile(r"running step \d+"), timeout=_RUN_TIMEOUT
    )
    page.locator("#btn_cancel").click()
    expect(page.locator("#live_movement_status")).to_contain_text("cancelled", timeout=_RUN_TIMEOUT)
    expect(page.locator("#live_map")).to_be_visible()
