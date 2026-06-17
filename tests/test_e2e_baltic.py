"""Full Baltic end-to-end: Python run (1 yr) + live movement + Results outputs.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_baltic.py -m e2e -o addopts="" -p no:cacheprovider
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
_RUN_TIMEOUT = 120_000  # 1-yr Baltic Python run + spatial I/O


def test_baltic_full_run_and_outputs(page: Page, app: ShinyAppProc):
    # Capture client-side console errors — guards the diet_chart "unexpected state"
    # OutputProgressReporter desync (caused by _get_result_data's lazy cache write
    # re-invalidating its own readers; fixed by isolating the memo cache).
    console_errors: list[str] = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
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
    # nyear=1 for speed; spatial output needs BOTH the master flag AND a sub-flag
    # (the per-quantity flag is what actually writes the NetCDF — output.py:811).
    overrides.fill(
        "simulation.time.nyear=1\noutput.spatial.enabled=true\noutput.spatial.biomass.enabled=true"
    )

    # 3. Live movement + run.
    page.locator("#live_movement_view").click()
    page.locator("#btn_run").click()
    expect(page.locator("#run_status")).not_to_contain_text(
        "Validation failed", timeout=_LOAD_TIMEOUT
    )
    expect(page.locator("#live_movement_status")).to_contain_text("running", timeout=_RUN_TIMEOUT)
    expect(page.locator("#live_movement_status")).to_contain_text("done", timeout=_RUN_TIMEOUT)
    page.locator("#live_movement_mode").get_by_text("Dots").click()

    # 4. Results: biomass time-series + diet heatmap (real data, not empty-state).
    page.locator(".nav-pills .nav-link[data-value='results']").click()

    # Biomass chart (Time Series card, always visible). A non-empty plotly plot has traces.
    expect(page.locator("#results_chart .js-plotly-plot")).to_be_visible(timeout=_RUN_TIMEOUT)
    assert page.locator("#results_chart .js-plotly-plot .trace").count() > 0, (
        "biomass chart has no traces"
    )

    # Diet Composition heatmap — click its tab, assert it rendered and is NOT the
    # "(no prey data)" empty-state (the bug a real Baltic run uniquely surfaced).
    page.get_by_role("tab", name="Diet Composition").click()
    diet = page.locator("#diet_chart")
    expect(diet.locator(".js-plotly-plot")).to_be_visible(timeout=_LOAD_TIMEOUT)
    assert diet.locator(".js-plotly-plot .heatmaplayer").count() > 0, "diet heatmap has no cells"
    assert "no prey data" not in diet.inner_text().lower(), "diet heatmap is the empty-state"

    # 5. Spatial output (enabled via the override). The Spatial Results pill is ALWAYS
    # present but carries `osm-disabled` until the server detects spatial output; wait for
    # it to ENABLE (proves the run's spatial NetCDF flowed to the UI), then render the Flat
    # View plotly heatmap (controls auto-select: result type = first nc, species = sum).
    spatial_pill = page.locator(".nav-pills .nav-link[data-value='spatial_results']")
    expect(spatial_pill).not_to_have_class(re.compile(r"osm-disabled"), timeout=_RUN_TIMEOUT)
    spatial_pill.click()
    # The nc auto-loads (#spatial_result_type = output-type/file select) and the controls
    # render (#spatial_map_species = species select, default "sum over species"), then the
    # default Map View paints the deck.gl canvas. Asserting these proves the run's spatial
    # NetCDF flowed engine→UI and rendered — without the brittle Flat-View tab + per-cell
    # heatmap assertion (the spatial page's own render is covered by test_e2e_spatial_results).
    page.wait_for_selector("#spatial_result_type", timeout=_RUN_TIMEOUT)
    expect(page.locator("#spatial_map_species")).to_be_visible(timeout=_RUN_TIMEOUT)
    # deck.gl renders multiple canvases (deck overlay + basemap) → take the first.
    expect(page.locator("#spatial_map canvas").first).to_be_visible(timeout=_RUN_TIMEOUT)

    _REPO.joinpath("screenshots").mkdir(exist_ok=True)
    page.screenshot(path=str(_REPO / "screenshots" / "baltic_full_e2e.png"), full_page=True)

    # Regression guard: no diet_chart client state-sync errors (the OutputProgressReporter
    # desync fixed by isolating _get_result_data's memo cache so it recalcs exactly once).
    diet_errors = [e for e in console_errors if "diet_chart" in e and "unexpected state" in e]
    assert not diet_errors, f"diet_chart client state errors present: {diet_errors}"
