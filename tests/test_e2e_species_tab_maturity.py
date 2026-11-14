"""Opening the Species tab must not change what a run computes.

The Species table renders every visible field and ``sync_species_inputs`` (ui/pages/setup.py) writes
each rendered value back into the loaded config, so a key the config omits is materialized at its
schema default. ``species.maturity.age`` used to default to 3.0, and the Python engine requires
``length >= size AND age >= age``: after this tab rendered, every species of a UI-launched Python run
waited until age 3 to mature. On the Baltic that collapsed cod_west (19 kt -> 14 t, seed 42, 50 yr).

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_species_tab_maturity.py -m e2e -o addopts="" \
        -p no:cacheprovider
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Page
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from osmose.scenarios import Scenario, ScenarioManager
from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")
_T = 30_000


def _sp0_current_value(page: Page, search: str, key_pattern: str) -> str:
    """sp0's value of `key_pattern` from the Advanced page's "Current Value" column ("-" = absent).

    The table re-renders when the search box changes and reads the config at that moment."""
    page.locator(".nav-pills .nav-link[data-value='advanced']").click()
    box = page.locator("#adv_search")
    box.wait_for(state="visible", timeout=_T)
    box.fill("")
    page.wait_for_timeout(500)
    box.fill(search)
    row = page.locator("tr", has=page.locator("td", has_text=key_pattern)).first
    row.wait_for(state="visible", timeout=_T)
    page.wait_for_timeout(1_000)
    return row.locator("td").nth(2).inner_text().strip()


def test_species_tab_does_not_impose_a_maturity_age(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_T)
    dismiss_changelog_modal(page)
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.wait_for_selector("#load_example", timeout=_T)
    page.select_option("#load_example", "baltic")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_T)
    page.wait_for_timeout(3_000)

    key = "species.maturity.age.sp{idx}"
    assert _sp0_current_value(page, "maturity.age", key) == "-"  # the Baltic file sets no age

    page.locator(".nav-pills .nav-link[data-value='setup']").click()
    page.wait_for_selector("#spt_species_maturity_age_0", timeout=_T)
    page.wait_for_timeout(3_000)  # let sync_species_inputs run

    after = _sp0_current_value(page, "maturity.age", key)
    assert after == "-" or float(after) == 0.0, (
        f"opening the Species tab wrote species.maturity.age.sp0 = {after!r}; the engine reads any "
        "non-zero value as a minimum maturity age"
    )


def test_copy_species_0_refreshes_an_age_based_destination_cell(
    page: Page, app: ShinyAppProc, tmp_path
):
    """The table reads the config in isolate, so after "copy species 0 to all" it must re-render
    (species_table_rev in ui/pages/setup.py) or a destination cell keeps its old widget.

    The app starts from schema defaults, which give species 0-2 a maturity size, and an import
    merges keys without removing any, so an AGE-based species 0 cannot be built here. This test
    builds the mirror case through the same refresh: species 3 is imported with a maturity age and
    no size (an "age-based" cell); copying the size-based species 0 gives it a size, and its cell
    must become a size input."""
    flat = tmp_path / "age_based_sp3.csv"
    flat.write_text(
        "simulation.nspecies;4\nspecies.name.sp3;agefish\nspecies.maturity.age.sp3;1.0\n"
    )
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_T)
    dismiss_changelog_modal(page)

    page.locator(".nav-pills .nav-link[data-value='advanced']").click()
    page.set_input_files("#import_config", str(flat))
    page.locator("#confirm_import").click(timeout=_T)
    page.wait_for_timeout(2_000)

    page.locator(".nav-pills .nav-link[data-value='setup']").click()
    page.locator("#n_species").fill("4")
    page.locator("#n_species").press("Tab")
    page.wait_for_selector("#spt_species_maturity_size_2", timeout=_T)
    page.wait_for_timeout(1_500)
    assert page.locator("#spt_species_maturity_size_3").count() == 0  # age-based: no size input
    assert page.get_by_text("age-based").count() >= 1

    page.get_by_role("button", name="Copy to all").click()  # header button of species 0
    page.wait_for_selector("#spt_species_maturity_size_3", timeout=_T)  # re-rendered as an input
    src = page.locator("#spt_species_maturity_size_0").input_value()
    assert float(page.locator("#spt_species_maturity_size_3").input_value()) == float(src)


_SCENARIOS_DIR = Path("data/scenarios")  # state.scenarios_dir default (ui/state.py)


def test_copying_an_age_based_species_0_turns_a_destination_size_input_age_based(
    page: Page, app: ShinyAppProc
):
    """Species 0 matures by age (no size), species 1 by size. Copying species 0 to all removes
    species 1's size, and the re-rendered table must replace its size input with "age-based".

    Loading a scenario REPLACES the config (state.load_config), so it can start from an age-based
    species 0, which the schema defaults and the merge-only import cannot."""
    name = f"e2e_agebased_{uuid.uuid4().hex[:8]}"
    config = {
        "simulation.nspecies": "2",
        "species.name.sp0": "agefish",
        "species.maturity.age.sp0": "1.0",
        "species.name.sp1": "sizefish",
        "species.maturity.size.sp1": "38.0",
    }
    try:
        ScenarioManager(_SCENARIOS_DIR).save(Scenario(name=name, config=config))
        page.goto(app.url)
        page.wait_for_selector(".nav-pills", timeout=_T)
        dismiss_changelog_modal(page)

        page.locator(".nav-pills .nav-link[data-value='scenarios']").click()
        page.locator(f"#selected_scenario input[value='{name}']").check(timeout=_T)
        page.click("#btn_load_scenario")
        page.wait_for_timeout(2_000)

        page.locator(".nav-pills .nav-link[data-value='setup']").click()
        page.locator("#n_species").fill("2")
        page.locator("#n_species").press("Tab")
        page.wait_for_selector("#spt_species_maturity_size_1", timeout=_T)
        page.wait_for_timeout(1_500)
        assert page.locator("#spt_species_maturity_size_0").count() == 0  # age-based source
        assert float(page.locator("#spt_species_maturity_size_1").input_value()) == 38.0

        page.get_by_role("button", name="Copy to all").click()
        page.wait_for_selector("#spt_species_maturity_size_1", state="detached", timeout=_T)
        assert page.get_by_text("age-based").count() >= 2
    finally:
        shutil.rmtree(_SCENARIOS_DIR / name, ignore_errors=True)


def test_baltic_sprat_size_cell_is_age_based(page: Page, app: ShinyAppProc):
    """Baltic sprat (sp2) matures by age only (2026-11-13), so the Species table must show its
    size cell as read-only "age-based": a size input would write the 50 cm schema default back and
    stop sprat maturing in both engines."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_T)
    dismiss_changelog_modal(page)
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.wait_for_selector("#load_example", timeout=_T)
    page.select_option("#load_example", "baltic")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_T)
    page.wait_for_timeout(3_000)

    page.locator(".nav-pills .nav-link[data-value='setup']").click()
    page.wait_for_selector("#spt_species_maturity_size_1", timeout=_T)  # herring: size-based
    page.wait_for_timeout(2_000)
    assert page.locator("#spt_species_maturity_size_2").count() == 0
    assert page.get_by_text("age-based").count() >= 1
    assert float(page.locator("#spt_species_maturity_age_2").input_value()) == 1.0
