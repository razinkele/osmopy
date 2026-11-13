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

import pytest
from playwright.sync_api import Page
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

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
