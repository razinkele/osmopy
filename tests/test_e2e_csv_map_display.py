"""End-to-end Playwright tests for CSV map display with EEC Full data.

Run explicitly with:
    .venv/bin/python -m pytest tests/test_e2e_csv_map_display.py -v -m e2e

Covers:
- E2E-CSV1: Fishing Distribution CSV overlay renders layers on the deck.gl map
- E2E-CSV2: Movement Animation with CSV maps renders for a selected species
- E2E-CSV3: A specific movement map CSV overlay renders when selected individually
- E2E-CSV4: Overlay selector contains movement map entries with species labels
- E2E-CSV5: Switching between CSV overlay and grid_extent removes overlay layer
- E2E-CSV6: Movement Animation controls appear and function (species, speed, slider)

Requires: data/eec_full/ directory with EEC Full example files.
"""

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_GRID_TIMEOUT = 25_000
_OVERLAY_TIMEOUT = 15_000
_MAP_RENDER_WAIT = 3_000  # ms — wait for deck.gl to render layers after update
_MIN_OVERLAY_OPTIONS = 5  # grid_extent + LTL + Fishing + movement maps + Movement Animation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_eec_full_and_goto_grid(page: Page, app: ShinyAppProc) -> None:
    """Navigate to app, load EEC Full example, then click the Grid tab."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=15_000)

    page.select_option("#load_example", "eec_full")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=15_000)

    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.wait_for_timeout(2_000)


def _wait_for_overlay_options(page: Page, min_count: int = _MIN_OVERLAY_OPTIONS) -> None:
    """Wait until the overlay selector has at least *min_count* options."""
    page.wait_for_function(
        f"(document.querySelector('#grid_overlay')?.options?.length ?? 0) >= {min_count}",
        timeout=_GRID_TIMEOUT,
    )


def _get_deckgl_layer_ids(page: Page) -> list:
    """Return the IDs of all deck.gl layers currently rendered on grid_map.

    shiny_deckgl caches raw layer props in ``instance.lastLayers`` — each entry
    has an ``id`` field set by the Python ``polygon_layer()`` call.
    """
    return page.evaluate("""
        (() => {
            const inst = window.__deckgl_instances?.['grid_map'];
            if (!inst || !inst.lastLayers) return [];
            return inst.lastLayers.map(l => l.id || '');
        })()
    """)


def _wait_for_layer_id(page: Page, layer_id: str, timeout: int = _OVERLAY_TIMEOUT) -> None:
    """Wait until a specific layer ID appears in the deck.gl instance."""
    page.wait_for_function(
        f"""
        (() => {{
            const inst = window.__deckgl_instances?.['grid_map'];
            if (!inst || !inst.lastLayers) return false;
            return inst.lastLayers.some(l => l.id === '{layer_id}');
        }})()
        """,
        timeout=timeout,
    )


def _wait_for_no_layer_id(page: Page, layer_id: str, timeout: int = _OVERLAY_TIMEOUT) -> None:
    """Wait until a specific layer ID is no longer in the deck.gl instance."""
    page.wait_for_function(
        f"""
        (() => {{
            const inst = window.__deckgl_instances?.['grid_map'];
            if (!inst || !inst.lastLayers) return true;
            return !inst.lastLayers.some(l => l.id === '{layer_id}');
        }})()
        """,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# E2E-CSV1: Fishing Distribution CSV overlay renders
# ---------------------------------------------------------------------------


def test_fishing_distrib_overlay_renders_layer(page: Page, app: ShinyAppProc):
    """Selecting 'Fishing Distribution' overlay must add a 'grid-overlay' layer."""
    _load_eec_full_and_goto_grid(page, app)
    _wait_for_overlay_options(page)

    # Find and select the Fishing Distribution option
    options = page.locator("#grid_overlay option").all()
    fishing_option = None
    for opt in options:
        text = opt.text_content()
        if text and "ishing" in text and "Movement" not in text:
            fishing_option = opt.get_attribute("value")
            break

    assert fishing_option is not None, (
        "Could not find Fishing Distribution option in overlay selector"
    )

    page.select_option("#grid_overlay", value=fishing_option)

    # Wait for the overlay layer to appear (deck.gl update is async)
    _wait_for_layer_id(page, "grid-overlay")
    layer_ids = _get_deckgl_layer_ids(page)
    assert "grid-overlay" in layer_ids, (
        f"Expected 'grid-overlay' layer after selecting Fishing Distribution, "
        f"got layers: {layer_ids}"
    )


# ---------------------------------------------------------------------------
# E2E-CSV2: Movement Animation renders CSV maps
# ---------------------------------------------------------------------------


def test_movement_animation_renders_layers(page: Page, app: ShinyAppProc):
    """Selecting Movement Animation and a species must render movement-* layers."""
    _load_eec_full_and_goto_grid(page, app)
    _wait_for_overlay_options(page)

    # Select Movement Animation
    page.select_option("#grid_overlay", value="__movement_animation__")
    page.wait_for_timeout(1_000)

    # Movement controls should appear
    species_select = page.locator("#movement_species")
    expect(species_select).to_be_visible(timeout=_OVERLAY_TIMEOUT)

    # Select a species (cod has 4 maps)
    page.select_option("#movement_species", value="cod")

    # Wait for at least one movement layer to appear
    page.wait_for_function(
        """
        (() => {
            const inst = window.__deckgl_instances?.['grid_map'];
            if (!inst || !inst.lastLayers) return false;
            return inst.lastLayers.some(l => (l.id || '').startsWith('movement-'));
        })()
        """,
        timeout=_OVERLAY_TIMEOUT,
    )
    layer_ids = _get_deckgl_layer_ids(page)
    movement_layers = [lid for lid in layer_ids if lid.startswith("movement-")]
    assert len(movement_layers) > 0, (
        f"Expected movement-* layers after selecting cod, got layers: {layer_ids}"
    )


# ---------------------------------------------------------------------------
# E2E-CSV3: Individual movement map CSV overlay renders
# ---------------------------------------------------------------------------


def test_individual_movement_map_overlay_renders(page: Page, app: ShinyAppProc):
    """Selecting a specific movement map (e.g. cod nurseries) must render a grid-overlay layer."""
    _load_eec_full_and_goto_grid(page, app)
    _wait_for_overlay_options(page)

    # Find a movement map option (contains a species name like 'cod:')
    options = page.locator("#grid_overlay option").all()
    movement_option = None
    for opt in options:
        text = opt.text_content() or ""
        if "cod:" in text.lower() or "sole:" in text.lower():
            movement_option = opt.get_attribute("value")
            break

    if movement_option is None:
        pytest.skip("No individual movement map option found in overlay selector")

    page.select_option("#grid_overlay", value=movement_option)
    _wait_for_layer_id(page, "grid-overlay")

    layer_ids = _get_deckgl_layer_ids(page)
    assert "grid-overlay" in layer_ids, (
        f"Expected 'grid-overlay' layer after selecting movement map, "
        f"got layers: {layer_ids}"
    )


# ---------------------------------------------------------------------------
# E2E-CSV4: Overlay selector contains movement map entries with species labels
# ---------------------------------------------------------------------------


def test_overlay_selector_has_species_labeled_maps(page: Page, app: ShinyAppProc):
    """Overlay selector must contain entries with species labels (e.g. 'cod: Nurseries')."""
    _load_eec_full_and_goto_grid(page, app)
    _wait_for_overlay_options(page)

    all_text = page.locator("#grid_overlay option").all_text_contents()
    species_entries = [t for t in all_text if ":" in t and "Movement Animation" not in t]
    assert len(species_entries) > 0, (
        f"Expected species-labeled overlay entries, got: {all_text}"
    )

    # Verify known species appear
    all_joined = " ".join(all_text).lower()
    known_species = ["cod", "sole", "herring", "whiting"]
    found = [sp for sp in known_species if sp in all_joined]
    assert len(found) >= 2, (
        f"Expected at least 2 known species in overlay labels, found: {found}"
    )


# ---------------------------------------------------------------------------
# E2E-CSV5: Switching away from CSV overlay removes overlay layer
# ---------------------------------------------------------------------------


def test_switching_to_grid_extent_removes_overlay(page: Page, app: ShinyAppProc):
    """After selecting a CSV overlay, switching to grid_extent must remove the overlay layer."""
    _load_eec_full_and_goto_grid(page, app)
    _wait_for_overlay_options(page)

    # First select a CSV overlay to add the layer
    options = page.locator("#grid_overlay option").all()
    csv_option = None
    for opt in options:
        text = opt.text_content() or ""
        if "ishing" in text:
            csv_option = opt.get_attribute("value")
            break

    if csv_option is None:
        pytest.skip("No CSV overlay option found")

    page.select_option("#grid_overlay", value=csv_option)
    _wait_for_layer_id(page, "grid-overlay")

    # Verify overlay layer is present
    layer_ids = _get_deckgl_layer_ids(page)
    assert "grid-overlay" in layer_ids, "Overlay layer should be present"

    # Switch to grid_extent
    page.select_option("#grid_overlay", value="grid_extent")
    _wait_for_no_layer_id(page, "grid-overlay")

    # Verify overlay layer is gone
    layer_ids_after = _get_deckgl_layer_ids(page)
    assert "grid-overlay" not in layer_ids_after, (
        f"grid-overlay layer should be removed after switching to grid_extent, "
        f"got: {layer_ids_after}"
    )


# ---------------------------------------------------------------------------
# E2E-CSV6: Movement Animation controls work
# ---------------------------------------------------------------------------


def test_movement_animation_controls_functional(page: Page, app: ShinyAppProc):
    """Movement Animation controls (species, speed, step slider) must be interactive."""
    _load_eec_full_and_goto_grid(page, app)
    _wait_for_overlay_options(page)

    page.select_option("#grid_overlay", value="__movement_animation__")
    page.wait_for_timeout(1_000)

    # All three controls should appear
    expect(page.locator("#movement_species")).to_be_visible(timeout=_OVERLAY_TIMEOUT)
    expect(page.locator("#movement_speed")).to_be_visible(timeout=_OVERLAY_TIMEOUT)
    expect(page.locator("#movement_step")).to_be_visible(timeout=_OVERLAY_TIMEOUT)

    # Species selector should have 14 species
    species_count = page.evaluate(
        "document.querySelector('#movement_species')?.options?.length ?? 0"
    )
    assert species_count == 14, (
        f"Expected 14 species in movement selector, got {species_count}"
    )

    # Speed selector should have 4 options (0.5x, 1x, 2x, 4x)
    speed_count = page.evaluate(
        "document.querySelector('#movement_speed')?.options?.length ?? 0"
    )
    assert speed_count == 4, f"Expected 4 speed options, got {speed_count}"

    # Step slider should have max=23 (24 steps, 0-indexed)
    step_max = page.locator("#movement_step").get_attribute("data-max")
    assert step_max == "23", f"Expected step slider max=23, got {step_max}"

    # Switch species and verify layers update
    page.select_option("#movement_species", value="sole")
    page.wait_for_function(
        """
        (() => {
            const inst = window.__deckgl_instances?.['grid_map'];
            if (!inst || !inst.lastLayers) return false;
            return inst.lastLayers.some(l => (l.id || '').startsWith('movement-'));
        })()
        """,
        timeout=_OVERLAY_TIMEOUT,
    )
    layer_ids = _get_deckgl_layer_ids(page)
    movement_layers = [lid for lid in layer_ids if lid.startswith("movement-")]
    assert len(movement_layers) > 0, (
        f"Expected movement layers after switching to sole, got: {layer_ids}"
    )
