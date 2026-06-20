"""End-to-end test for the New Scenario wizard.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_scenario_wizard.py -v -m e2e

Excluded from the default suite (`-m 'not e2e'`). The wizard's pure logic
(apply_basics override, validation, resolve) is covered by
tests/test_scenario_wizard.py; this asserts the modal stepper flow end to end
plus that the override actually reached the persisted config.
"""

import json
import shutil
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000
_SCENARIOS_DIR = Path("data/scenarios")  # state.scenarios_dir default (ui/state.py:35)


def _goto_scenarios(page: Page, app: ShinyAppProc) -> None:
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    dismiss_changelog_modal(page)
    page.locator(".nav-pills .nav-link[data-value='scenarios']").click()
    page.wait_for_selector("#btn_new_scenario", timeout=_LOAD_TIMEOUT)


def test_wizard_creates_scenario_from_demo(page: Page, app: ShinyAppProc):
    name = f"e2e_wiz_{uuid.uuid4().hex[:8]}"
    scen_dir = _SCENARIOS_DIR / name
    try:
        _goto_scenarios(page, app)
        # Open the wizard
        page.click("#btn_new_scenario")
        page.wait_for_selector("#wizard_source_sel", timeout=_LOAD_TIMEOUT)
        # Step 1: pick a bundled demo, Next
        page.select_option("#wizard_source_sel", "demo:baltic")
        page.click("#btn_wizard_next")
        # Step 2: set years, Next
        page.wait_for_selector("#wizard_nyear", timeout=_LOAD_TIMEOUT)
        page.fill("#wizard_nyear", "7")
        page.click("#btn_wizard_next")
        # Step 3: name, then Create (the same button, now labelled "Create")
        page.wait_for_selector("#wizard_name", timeout=_LOAD_TIMEOUT)
        page.fill("#wizard_name", name)
        page.click("#btn_wizard_next")
        # Success: toast + the new scenario appears in the saved-scenarios list.
        # Scope the name check to #scenario_list (the name also appears in the success
        # toast, so a page-wide get_by_text would be ambiguous).
        note = page.locator(".shiny-notification").last
        expect(note).to_be_visible(timeout=_LOAD_TIMEOUT)
        expect(page.locator("#scenario_list").get_by_text(name)).to_be_visible(
            timeout=_LOAD_TIMEOUT
        )
        # Read back the persisted config — the Years override must have reached the save path.
        scen_json = scen_dir / "scenario.json"
        assert scen_json.exists(), f"expected saved scenario at {scen_json}"
        saved = json.loads(scen_json.read_text())
        assert saved["config"]["simulation.time.nyear"] == "7"
    finally:
        shutil.rmtree(scen_dir, ignore_errors=True)
