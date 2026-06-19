"""End-to-end tests for the Map Builder page with the Baltic dataset.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_map_builder.py -v -m e2e

Excluded from the default suite (``pytest -m 'not e2e'``) because Playwright's
event loop conflicts with pytest-asyncio; the ``test_e2e_*`` name also hits the
conftest collect-ignore so a missing Playwright/browser never breaks collection.

Baltic is used (not EEC Full) because the Map Builder needs a regular lat/lon
bounding-box grid — ``GridSpec.from_config`` reads ``grid.nlon``/``grid.nlat``/
``grid.upleft.*``/``grid.lowright.*``, which Baltic has (50x40, 10..30E / 54..66N)
and the NcGrid-based EEC Full does not.

The deck.gl draw/pick seam (pixel-precise brush + MapboxDraw polygon Apply) is
NOT asserted here — mapping a screen pixel to a known grid cell through deck.gl's
picking is brittle. That path is covered by the pure-core unit tests
(``rasterize_polygon``/``lonlat_to_cell``/``apply_polygon`` in
``tests/test_maps_builder.py``) and is a documented manual check (see the module
docstring of ``ui/pages/map_builder.py``). These e2e tests assert the wiring the
unit tests cannot: the page mounts, the deck.gl canvas renders on a real config,
the tool/map-type controls toggle the right sub-forms, and a Save round-trips
through the engine writer to a notification.
"""

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from tests._e2e_support import dismiss_changelog_modal

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000
_MAP_TIMEOUT = 30_000


def _load_baltic(page: Page, app: ShinyAppProc) -> None:
    """Load the Baltic example dataset (regular lat/lon grid)."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    dismiss_changelog_modal(page)
    page.select_option("#load_example", "baltic")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_LOAD_TIMEOUT)


def _goto_map_builder(page: Page) -> None:
    """Navigate to the Map Builder tab."""
    page.locator(".nav-pills .nav-link[data-value='map_builder']").click()
    page.wait_for_timeout(2_000)


def test_map_builder_renders_canvas(page: Page, app: ShinyAppProc):
    """Loading Baltic and navigating to Map Builder renders the deck.gl canvas."""
    _load_baltic(page, app)
    _goto_map_builder(page)
    page.wait_for_selector("#mb_map canvas", timeout=_MAP_TIMEOUT)
    assert page.locator("#mb_map canvas").count() >= 1, "Expected a canvas in #mb_map"


def test_map_type_toggles_applicability_form(page: Page, app: ShinyAppProc):
    """Distribution shows the species applicability form; zone/mask hide it."""
    _load_baltic(page, app)
    _goto_map_builder(page)
    page.wait_for_selector("#mb_map canvas", timeout=_MAP_TIMEOUT)

    # Distribution is the default -> the species select renders.
    page.wait_for_selector("#mb_species", timeout=_LOAD_TIMEOUT)
    species = page.locator("#mb_species option").all_text_contents()
    assert len(species) >= 1, f"Expected Baltic species in the applicability form, got {species}"

    # Switch to Generic zone -> the applicability form disappears.
    page.click("#map_type input[value='zone']")
    page.wait_for_timeout(1_000)
    assert not page.locator("#mb_species").is_visible(), (
        "Species applicability form should be hidden for a zone map"
    )


def test_save_zone_map_round_trips_to_notification(page: Page, app: ShinyAppProc):
    """A zone-map save (no painting / no species needed) writes the CSV and toasts."""
    _load_baltic(page, app)
    _goto_map_builder(page)
    page.wait_for_selector("#mb_map canvas", timeout=_MAP_TIMEOUT)

    # New blank map, generic zone (no species required, empty grid is valid to save).
    page.click("#mb_new_blank")
    page.click("#map_type input[value='zone']")
    page.wait_for_timeout(500)

    page.fill("#mb_filename", "e2e_zone")
    page.click("#mb_save")

    # A save notification appears (success summary from save_map / wire_map_into_config).
    note = page.locator(".shiny-notification").last
    expect(note).to_be_visible(timeout=_LOAD_TIMEOUT)
