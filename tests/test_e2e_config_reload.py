"""Loading a second example must give exactly that example's config, whatever tabs were viewed.

A tab hidden during a load does not re-render (Shiny suspends hidden outputs), so its inputs keep
the FIRST config's values; its sync effect still fires on the load and wrote them into the second.
Measured 2026-11-15 before the fix: load EEC full, view Setup, load Baltic -> 169 of Baltic's 700
keys were EEC full's (species names, growth, maturity, recruitment, ...); via Movement, 92 keys
(map species, ages, steps). The fix (AppState.mark_rendered / rendered_current) blocks the write
until the tab has re-rendered for the current load.

The other arm matters as much: a sync that never ran would pass the first test, so an edit made
after the reload must still be saved.

The instrument is the saved config (a scenario), compared key by key with the example's own file.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_config_reload.py -m e2e -o addopts="" \
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

from osmose.config.reader import OsmoseConfigReader
from osmose.demo import osmose_demo
from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")
_T = 30_000
_SCENARIOS_DIR = Path("data/scenarios")
_IGNORED = {"_osmose.config.dir"}  # each load extracts the demo to a new temp dir


def _start(page: Page, app: ShinyAppProc) -> None:
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_T)
    dismiss_changelog_modal(page)


def _load_example(page: Page, example: str) -> None:
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.wait_for_selector("#load_example", timeout=_T)
    page.select_option("#load_example", example)
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_T)
    page.wait_for_timeout(3_000)


def _visit(page: Page, tab: str) -> None:
    page.locator(f".nav-pills .nav-link[data-value='{tab}']").click()
    page.wait_for_timeout(4_000)  # render, then let the sync effects run


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


def _expected(example: str, tmp_path: Path) -> dict[str, str]:
    info = osmose_demo(example, tmp_path)
    return dict(OsmoseConfigReader().read(Path(info["config_file"])))


def _differences(expected: dict[str, str], saved: dict[str, str]) -> dict[str, tuple]:
    return {
        k: (v, saved.get(k)) for k, v in expected.items() if k not in _IGNORED and saved.get(k) != v
    }


@pytest.mark.parametrize("tab", ["setup", "movement", "fishing", "forcing", "grid"])
@pytest.mark.parametrize(("first", "second"), [("eec_full", "baltic"), ("baltic", "eec_full")])
def test_second_load_is_not_corrupted_by_a_viewed_tab(
    page: Page, app: ShinyAppProc, tmp_path, tab, first, second
):
    expected = _expected(second, tmp_path)
    name = f"e2e_reload_{tab}_{uuid.uuid4().hex[:8]}"
    try:
        _start(page, app)
        _load_example(page, first)
        _visit(page, tab)
        _load_example(page, second)
        _visit(page, tab)  # re-render for the second config; nothing may change
        saved = _save_scenario(page, name)
        wrong = _differences(expected, saved)
        assert not wrong, (
            f"{first} -> {tab} -> {second}: {len(wrong)} keys differ from {second}'s file, "
            f"e.g. {dict(list(wrong.items())[:4])}"
        )
    finally:
        shutil.rmtree(_SCENARIOS_DIR / name, ignore_errors=True)


@pytest.mark.parametrize(
    ("tab", "input_id", "key", "value"),
    [
        ("setup", "spt_species_linf_0", "species.linf.sp0", "77.0"),
        ("movement", "movement_initialage_map0", "movement.initialage.map0", "7"),
    ],
)
def test_an_edit_after_the_reload_is_still_saved(
    page: Page, app: ShinyAppProc, tmp_path, tab, input_id, key, value
):
    expected = _expected("baltic", tmp_path)
    name = f"e2e_reload_edit_{tab}_{uuid.uuid4().hex[:8]}"
    try:
        _start(page, app)
        _load_example(page, "eec_full")
        _visit(page, tab)
        _load_example(page, "baltic")
        _visit(page, tab)
        page.locator(f"#{input_id}").fill(value, timeout=_T)
        page.locator(f"#{input_id}").press("Tab")
        page.wait_for_timeout(2_000)
        saved = _save_scenario(page, name)
        assert float(saved[key]) == float(value), f"edit lost: {key} = {saved.get(key)!r}"
        wrong = _differences({k: v for k, v in expected.items() if k != key}, saved)
        assert not wrong, f"{len(wrong)} other keys changed, e.g. {dict(list(wrong.items())[:4])}"
    finally:
        shutil.rmtree(_SCENARIOS_DIR / name, ignore_errors=True)
