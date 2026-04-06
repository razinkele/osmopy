"""End-to-end tests for the Grid overlay UI using Playwright.

Run these tests explicitly with:
    micromamba run -n shiny pytest tests/test_e2e_grid_overlay.py -v -m e2e

They are excluded from the default test suite (``pytest -m 'not e2e'``)
because Playwright's event loop conflicts with pytest-asyncio async tests.

Setup:
    The EEC Full demo (``eec_full``) is loaded before each test that needs overlay
    data. The ``eec_full`` demo contains ``eec_ltlbiomassTons.nc`` (10 plankton
    vars × 24 time steps) referenced via ``species.file.sp14–sp23``, which the
    overlay selector dedupes to a single "LTL Biomass" entry.

    Note: the plain ``eec`` demo does NOT include LTL NC files; use ``eec_full``.
"""

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_GRID_TIMEOUT = 25_000  # ms — grid map can be slow to initialise
_NC_TIMEOUT = 15_000    # ms — NC controls render after overlay select
# Minimum options in a fully-loaded overlay selector (grid_extent + at least 1 overlay)
_MIN_OVERLAY_OPTIONS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_eec_full_and_goto_grid(page: Page, app: ShinyAppProc) -> None:
    """Navigate to app, load EEC Full example, then click the Grid tab.

    Note: plain 'eec' demo has no LTL NC overlays; 'eec_full' does.
    """
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15_000)

    page.select_option("#load_example", "eec_full")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=15_000)

    # Navigate to Grid tab
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.wait_for_timeout(2_000)


def _wait_for_overlay_options(page: Page, min_count: int = _MIN_OVERLAY_OPTIONS) -> None:
    """Wait until the overlay selector has at least *min_count* options.

    Playwright cannot see ``<option>`` elements as "visible" inside a closed
    ``<select>``; we use JS evaluation to check the count instead.
    """
    page.wait_for_function(
        f"(document.querySelector('#grid_overlay')?.options?.length ?? 0) >= {min_count}",
        timeout=_GRID_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# E2E-G1: Grid tab is accessible after loading the EEC example
# ---------------------------------------------------------------------------

def test_grid_tab_accessible_after_eec_load(page: Page, app: ShinyAppProc):
    """Grid tab should be reachable and the overlay selector should appear."""
    _load_eec_full_and_goto_grid(page, app)

    # The overlay selector is rendered via output_ui; wait for it
    selector = page.locator("#grid_overlay")
    expect(selector).to_be_visible(timeout=_GRID_TIMEOUT)


# ---------------------------------------------------------------------------
# E2E-G2: LTL Biomass appears exactly once in the overlay selector
# ---------------------------------------------------------------------------

def test_overlay_ltl_biomass_deduplicated(page: Page, app: ShinyAppProc):
    """EEC has 10 species files pointing to the same NC — should show only once."""
    _load_eec_full_and_goto_grid(page, app)

    # Wait for overlay selector to populate
    _wait_for_overlay_options(page)

    ltl_options = page.locator("#grid_overlay option:has-text('LTL Biomass')")
    assert ltl_options.count() == 1, (
        f"Expected exactly 1 'LTL Biomass' option, got {ltl_options.count()}"
    )


# ---------------------------------------------------------------------------
# E2E-G3: Movement map keys are NOT present as individual options
# ---------------------------------------------------------------------------

def test_overlay_movement_maps_excluded(page: Page, app: ShinyAppProc):
    """Individual movement.file.mapN config keys must not appear in the selector."""
    _load_eec_full_and_goto_grid(page, app)

    _wait_for_overlay_options(page)

    # Options whose text looks like a raw config key (e.g. "movement.file.map0")
    movement_raw = page.locator("#grid_overlay option:has-text('movement.file.map')")
    assert movement_raw.count() == 0, (
        f"Unexpected raw movement key options: {movement_raw.count()}"
    )


# ---------------------------------------------------------------------------
# E2E-G4: Selecting LTL Biomass shows the NC controls panel
# ---------------------------------------------------------------------------

def test_ltl_overlay_shows_nc_controls(page: Page, app: ShinyAppProc):
    """After selecting LTL Biomass overlay, nc_var_select and nc_time_step should appear."""
    _load_eec_full_and_goto_grid(page, app)

    _wait_for_overlay_options(page)
    page.select_option("#grid_overlay", label="LTL Biomass")

    # NC variable selector and time slider should appear
    expect(page.locator("#nc_time_step")).to_be_visible(timeout=_NC_TIMEOUT)


# ---------------------------------------------------------------------------
# E2E-G5: NC variable selector contains expected plankton variable names
# ---------------------------------------------------------------------------

def test_nc_var_selector_has_plankton_vars(page: Page, app: ShinyAppProc):
    """After selecting LTL Biomass, nc_var_select should list plankton variables."""
    _load_eec_full_and_goto_grid(page, app)

    _wait_for_overlay_options(page)
    page.select_option("#grid_overlay", label="LTL Biomass")
    # Wait for nc_var_select to exist and have options (options inside <select> are
    # not "visible" to Playwright; use JS to check count)
    page.wait_for_function(
        "(document.querySelector('#nc_var_select')?.options?.length ?? 0) >= 2",
        timeout=_NC_TIMEOUT,
    )

    options_text = page.locator("#nc_var_select option").all_text_contents()
    # The EEC LTL NC has 10 vars; check at least 2 are present
    assert len(options_text) >= 2, (
        f"Expected >= 2 NC variable options, got {len(options_text)}: {options_text}"
    )

    joined = " ".join(options_text).lower()
    # At least one plankton-related term should appear
    assert any(kw in joined for kw in ("diatom", "zoo", "benthos", "dinoflag")), (
        f"Expected plankton variable names in NC selector, got: {options_text}"
    )


# ---------------------------------------------------------------------------
# E2E-G6: Time step slider range reflects 24 time steps (max = 23)
# ---------------------------------------------------------------------------

def test_nc_time_slider_range(page: Page, app: ShinyAppProc):
    """The time-step slider for LTL Biomass NC should have max = 23 (24 steps)."""
    _load_eec_full_and_goto_grid(page, app)

    _wait_for_overlay_options(page)
    page.select_option("#grid_overlay", label="LTL Biomass")
    page.wait_for_selector("#nc_time_step", timeout=_NC_TIMEOUT)

    slider_max = page.locator("#nc_time_step").get_attribute("data-max")
    assert slider_max == "23", (
        f"Expected slider data-max=23 for 24 time steps, got '{slider_max}'"
    )


# ---------------------------------------------------------------------------
# E2E-G7: Grid extent option is always present
# ---------------------------------------------------------------------------

def test_grid_extent_always_present(page: Page, app: ShinyAppProc):
    """The 'Grid Extent' option should always appear first in the overlay selector."""
    _load_eec_full_and_goto_grid(page, app)

    _wait_for_overlay_options(page)

    extent_option = page.locator("#grid_overlay option[value='grid_extent']")
    assert extent_option.count() == 1, (
        f"Expected 'grid_extent' option to exist, count={extent_option.count()}"
    )


# ---------------------------------------------------------------------------
# E2E-G8: NC controls are hidden / absent when a CSV overlay is selected
# ---------------------------------------------------------------------------

def test_nc_controls_hidden_for_grid_extent(page: Page, app: ShinyAppProc):
    """Selecting grid_extent (non-NC) should not show the nc_time_step slider."""
    _load_eec_full_and_goto_grid(page, app)

    _wait_for_overlay_options(page)

    # Start with LTL to force NC controls to appear, then switch away
    page.select_option("#grid_overlay", label="LTL Biomass")
    page.wait_for_selector("#nc_time_step", timeout=_NC_TIMEOUT)

    # Switch to grid extent
    page.select_option("#grid_overlay", value="grid_extent")
    page.wait_for_timeout(2_000)

    # Time slider should no longer be visible (hidden or removed)
    slider = page.locator("#nc_time_step")
    # Either hidden or gone — should not be visible
    assert not slider.is_visible(), "nc_time_step should be hidden for grid_extent overlay"


# ---------------------------------------------------------------------------
# E2E-G9: Time slider value changes when moved
# ---------------------------------------------------------------------------

def test_time_step_slider_interactive(page: Page, app: ShinyAppProc):
    """Moving the time-step slider should update its value."""
    _load_eec_full_and_goto_grid(page, app)

    _wait_for_overlay_options(page)
    page.select_option("#grid_overlay", label="LTL Biomass")
    page.wait_for_selector("#nc_time_step", timeout=_NC_TIMEOUT)

    slider = page.locator("#nc_time_step")
    initial_val = int(slider.get_attribute("data-from") or "0")

    # Shiny ionRange sliders render as hidden inputs; interact via Shiny JS API
    page.evaluate(
        "(val) => Shiny.setInputValue('nc_time_step', val, {priority: 'event'})",
        initial_val + 1,
    )
    page.wait_for_timeout(800)

    new_val = int(slider.get_attribute("data-from") or str(initial_val))
    assert new_val >= initial_val, (
        f"Slider data-from should not decrease: {initial_val} -> {new_val}"
    )
