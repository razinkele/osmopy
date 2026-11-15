"""Opening the Species tab must not flatten a multi-value (per-stage) config entry.

``sync_species_inputs`` (ui/pages/setup.py) writes every species-table cell back into the config
as soon as the table renders. A numeric cell whose config value is a per-stage array (EEC full sp5:
``predation.predprey.sizeratio.max.sp5 = "2.3;1.8"``; Benguela uses commas, ``"15, 5"``) cannot be
shown in a single numeric input. If the table renders it anyway, the schema default is written
over the array and the engine refuses the run (the size-ratio count no longer matches the stages).

The instrument is the saved config itself: a scenario saved right after the load (the baseline,
which proves the arrays are visible to this check) and one saved after the Species tab has
rendered and synced. The Advanced page's multi-value list is NOT usable here: it re-renders only
on explicit loads, so it cannot see a write made by the Species tab.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_species_tab_multivalue.py -m e2e -o addopts="" \
        -p no:cacheprovider
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Page
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")
_T = 30_000
_SCENARIOS_DIR = Path("data/scenarios")  # state.scenarios_dir default (ui/state.py)


def _save_scenario(page: Page, name: str) -> dict[str, str]:
    page.locator(".nav-pills .nav-link[data-value='scenarios']").click()
    page.locator("#scenario_name").fill(name, timeout=_T)
    page.click("#btn_save_scenario")
    path = _SCENARIOS_DIR / name / "scenario.json"
    for _ in range(40):
        if path.exists():
            break
        page.wait_for_timeout(250)
    return json.loads(path.read_text())["config"]


@pytest.mark.parametrize(
    ("example", "keys"),
    [
        (
            "eec_full",
            ["predation.predprey.sizeratio.max.sp5", "predation.predprey.sizeratio.min.sp5"],
        ),
        (
            "benguela",
            ["predation.predprey.sizeratio.max.sp0", "predation.predprey.sizeratio.min.sp0"],
        ),
    ],
)
def test_species_tab_keeps_per_stage_arrays(page: Page, app: ShinyAppProc, example, keys):
    tag = uuid.uuid4().hex[:8]
    names = [f"e2e_mv_{example}_before_{tag}", f"e2e_mv_{example}_after_{tag}"]
    try:
        page.goto(app.url)
        page.wait_for_selector(".nav-pills", timeout=_T)
        dismiss_changelog_modal(page)
        page.locator(".nav-pills .nav-link[data-value='grid']").click()
        page.wait_for_selector("#load_example", timeout=_T)
        page.select_option("#load_example", example)
        page.click("#btn_load_example")
        page.wait_for_selector(".shiny-notification", timeout=_T)
        page.wait_for_timeout(3_000)

        before = _save_scenario(page, names[0])
        for k in keys:
            assert ";" in before[k] or "," in before[k], f"{example} {k}: baseline {before[k]!r}"

        page.locator(".nav-pills .nav-link[data-value='setup']").click()
        page.wait_for_selector("#spt_species_maturity_size_0", timeout=_T)
        page.wait_for_timeout(3_000)  # let sync_species_inputs run

        after = _save_scenario(page, names[1])
        changed = {k: (before[k], after.get(k)) for k in keys if after.get(k) != before[k]}
        assert not changed, f"{example}: opening the Species tab rewrote {changed}"
    finally:
        for n in names:
            shutil.rmtree(_SCENARIOS_DIR / n, ignore_errors=True)
