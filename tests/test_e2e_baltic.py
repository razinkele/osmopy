"""Full Baltic end-to-end: Python run (1 yr) + live movement + Results outputs.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_baltic.py -m e2e -o addopts="" -p no:cacheprovider
"""

from __future__ import annotations

import pathlib

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_REPO = pathlib.Path(__file__).resolve().parent.parent
_LOAD_TIMEOUT = 20_000
_RUN_TIMEOUT = 120_000  # 1-yr Baltic Python run + spatial I/O


def test_baltic_full_run_and_outputs(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    dismiss_changelog_modal(page)

    # 1. Load Baltic (Domain/Grid page).
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.wait_for_selector("#load_example", timeout=_LOAD_TIMEOUT)
    page.select_option("#load_example", "baltic")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_LOAD_TIMEOUT)

    # 2. Run page: Python engine (Baltic is Python-only), short run + spatial output on.
    page.locator(".nav-pills .nav-link[data-value='run']").click()
    page.locator("#engineBtnPython").click()
    overrides = page.locator("#py_param_overrides")
    expect(overrides).to_be_visible(timeout=_LOAD_TIMEOUT)
    overrides.fill("simulation.time.nyear=1\noutput.spatial.enabled=true")

    # 3. Live movement + run.
    page.locator("#live_movement_view").click()
    page.locator("#btn_run").click()
    expect(page.locator("#run_status")).not_to_contain_text(
        "Validation failed", timeout=_LOAD_TIMEOUT
    )
    expect(page.locator("#live_movement_status")).to_contain_text("running", timeout=_RUN_TIMEOUT)
    expect(page.locator("#live_movement_status")).to_contain_text("done", timeout=_RUN_TIMEOUT)
    page.locator("#live_movement_mode").get_by_text("Dots").click()
