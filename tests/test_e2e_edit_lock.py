"""While a run (or any other busy operation) is in progress, the config-editing tabs are locked.

Every page's sync drops its writes while ``state.busy`` is set (writing then raced the run's own
use of the config), so an edit made during a run was silently lost: the input showed the new value,
the config kept the old one. The tabs are now ``inert`` for the duration, with a banner giving the
reason, and come back when the run ends. The Run tab itself stays live (Cancel must work).

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_edit_lock.py -m e2e -o addopts="" -p no:cacheprovider
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")
_T = 60_000
_RUN_T = 180_000
_LOCKED = ["grid", "setup", "forcing", "fishing", "movement", "advanced"]


def _pane(page: Page, value: str):
    return page.locator(f".tab-pane[data-value='{value}']")


def test_config_tabs_are_locked_while_a_run_is_in_progress(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_T)
    dismiss_changelog_modal(page)
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.select_option("#load_example", "baltic")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_T)
    page.wait_for_timeout(2_000)
    for value in _LOCKED:  # unlocked once the load (itself a busy operation) is done
        expect(_pane(page, value)).not_to_have_attribute("inert", re.compile(".*"), timeout=_T)

    page.locator(".nav-pills .nav-link[data-value='run']").click()
    page.locator("#engineBtnPython").click()
    overrides = page.locator("#py_param_overrides")
    expect(overrides).to_be_visible(timeout=_T)
    overrides.fill("simulation.time.nyear=10")  # long enough to observe the lock mid-run
    page.locator("#btn_run").click()

    expect(page.locator(".osm-busy-pill")).to_contain_text("Running simulation", timeout=_RUN_T)
    for value in _LOCKED:
        pane = _pane(page, value)
        expect(pane).to_have_attribute("inert", re.compile(".*"), timeout=_T)
        expect(pane).to_have_attribute("data-lock-reason", re.compile("Running simulation"))
    expect(_pane(page, "run")).not_to_have_attribute("inert", re.compile(".*"))

    # The lock is visible, not just effective: the Species tab shows why it cannot be edited.
    page.locator(".nav-pills .nav-link[data-value='setup']").click()
    expect(_pane(page, "setup")).to_have_class(re.compile(r"\bosm-edit-locked\b"))

    page.locator(".nav-pills .nav-link[data-value='run']").click()
    page.locator("#btn_cancel").click()
    expect(page.locator(".osm-busy-pill")).to_have_count(0, timeout=_RUN_T)
    for value in _LOCKED:
        expect(_pane(page, value)).not_to_have_attribute("inert", re.compile(".*"), timeout=_T)
        expect(_pane(page, value)).not_to_have_class(re.compile(r"\bosm-edit-locked\b"))
