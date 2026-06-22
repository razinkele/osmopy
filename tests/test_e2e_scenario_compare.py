"""End-to-end test for the Compare Scenarios modal.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_scenario_compare.py -v -m e2e

Excluded from the default suite (`-m 'not e2e'`). The compare edge logic is
covered purely by tests/test_scenario_compare_state.py; this asserts the modal
opens and renders the shared badged diff table end to end.
"""

import shutil
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from osmose.scenarios import Scenario, ScenarioManager
from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000
_SCENARIOS_DIR = Path("data/scenarios")  # state.scenarios_dir default (ui/state.py:35)


def test_compare_modal_renders_diff(page: Page, app: ShinyAppProc):
    # Seed two scenarios differing in exactly one key (deterministic + fast).
    mgr = ScenarioManager(_SCENARIOS_DIR)
    name_a = f"e2e_cmp_a_{uuid.uuid4().hex[:8]}"
    name_b = f"e2e_cmp_b_{uuid.uuid4().hex[:8]}"
    try:
        mgr.save(Scenario(name=name_a, config={"species.linf.sp0": "50.0"}))
        mgr.save(Scenario(name=name_b, config={"species.linf.sp0": "70.0"}))

        page.goto(app.url)
        page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
        dismiss_changelog_modal(page)
        page.locator(".nav-pills .nav-link[data-value='scenarios']").click()
        page.wait_for_selector("#btn_compare_open", timeout=_LOAD_TIMEOUT)

        page.click("#btn_compare_open")
        # Scope locators to the *shown* modal (.modal.show) — `.modal` alone matches
        # all static modal elements (changelogModal, helpModal, etc.) in the DOM,
        # some of which contain hidden tables that fool the visibility check.
        modal = page.locator(".modal.show")
        modal.locator("#compare_a").wait_for(timeout=_LOAD_TIMEOUT)
        page.select_option(".modal.show #compare_a", name_a)
        page.select_option(".modal.show #compare_b", name_b)
        page.click(".modal.show #btn_compare")

        expect(modal.locator("table").first).to_be_visible(timeout=_LOAD_TIMEOUT)
        expect(modal.locator(".badge")).to_have_count(1)
        expect(modal).to_contain_text("1 differing config key")
    finally:
        shutil.rmtree(_SCENARIOS_DIR / name_a, ignore_errors=True)
        shutil.rmtree(_SCENARIOS_DIR / name_b, ignore_errors=True)
