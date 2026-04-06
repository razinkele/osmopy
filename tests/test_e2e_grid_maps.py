"""End-to-end tests for Grid page with the EEC Full dataset.

Run explicitly:
    .venv/bin/python -m pytest tests/test_e2e_grid_maps.py -v -m e2e

Excluded from default suite (``pytest -m 'not e2e'``) because Playwright's
event loop conflicts with pytest-asyncio.

The EEC Full dataset uses:
- NcGrid (NetCDF-based grid with mask) — not a regular lat/lon bounding-box grid
- 14 focal species with movement maps (lesserSpottedDogfish, cod, sole, etc.)
- 10 LTL resource species sharing eec_ltlbiomassTons.nc
- Background species config is commented out in EEC Full (no bg overlay)
"""

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

pytestmark = pytest.mark.e2e

app = create_app_fixture("../app.py")

_LOAD_TIMEOUT = 15_000
_GRID_TIMEOUT = 25_000
_MAP_TIMEOUT = 30_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_eec_full(page: Page, app: ShinyAppProc) -> None:
    """Load the EEC Full example dataset."""
    page.goto(app.url)
    page.wait_for_selector(".nav-pills", timeout=_LOAD_TIMEOUT)
    page.select_option("#load_example", "eec_full")
    page.click("#btn_load_example")
    page.wait_for_selector(".shiny-notification", timeout=_LOAD_TIMEOUT)


def _goto_grid(page: Page) -> None:
    """Navigate to the Grid tab."""
    page.locator(".nav-pills .nav-link[data-value='grid']").click()
    page.wait_for_timeout(2_000)


def _wait_for_overlay_options(page: Page, min_count: int = 3) -> None:
    """Wait until the overlay selector has at least *min_count* options."""
    page.wait_for_function(
        f"(document.querySelector('#grid_overlay')?.options?.length ?? 0) >= {min_count}",
        timeout=_GRID_TIMEOUT,
    )


def _wait_for_map_widget(page: Page) -> None:
    """Wait until the deck.gl map widget canvas has rendered."""
    page.wait_for_selector("#grid_map canvas", timeout=_MAP_TIMEOUT)


# ---------------------------------------------------------------------------
# GM1: Grid page loads and renders map with EEC Full NcGrid
# ---------------------------------------------------------------------------

def test_grid_page_renders_map(page: Page, app: ShinyAppProc):
    """Loading EEC Full and navigating to Grid should render the deck.gl map canvas."""
    _load_eec_full(page, app)
    _goto_grid(page)
    _wait_for_map_widget(page)

    canvas = page.locator("#grid_map canvas")
    assert canvas.count() >= 1, "Expected at least one canvas element in #grid_map"


# ---------------------------------------------------------------------------
# GM2: NcGrid settings are displayed (not regular grid inputs)
# ---------------------------------------------------------------------------

def test_ncgrid_settings_displayed(page: Page, app: ShinyAppProc):
    """EEC Full uses NcGrid — grid fields should show NetCDF file and variable settings."""
    _load_eec_full(page, app)
    _goto_grid(page)

    # Wait for grid fields to render
    page.wait_for_selector("#grid_fields", timeout=_GRID_TIMEOUT)

    # The grid type classname input should contain NcGrid
    grid_fields_html = page.locator("#grid_fields").inner_html()
    assert "NcGrid" in grid_fields_html or "ncgrid" in grid_fields_html.lower(), (
        "Expected NcGrid class reference in grid fields"
    )


# ---------------------------------------------------------------------------
# GM3: Grid coordinate inputs populate from config
# ---------------------------------------------------------------------------

def test_grid_fields_have_values(page: Page, app: ShinyAppProc):
    """Grid fields should be populated after loading EEC Full (NcGrid)."""
    _load_eec_full(page, app)
    _goto_grid(page)

    page.wait_for_selector("#grid_fields", timeout=_GRID_TIMEOUT)

    # NcGrid: the grid type selector should show NcGrid as selected
    grid_fields_html = page.locator("#grid_fields").inner_html()
    assert "NcGrid" in grid_fields_html, (
        f"Expected NcGrid in grid fields HTML, got: {grid_fields_html[:300]}"
    )

    # The NetCDF Grid Settings section should be visible
    grid_fields_text = page.locator("#grid_fields").inner_text()
    assert "NetCDF Grid Settings" in grid_fields_text, (
        "Expected 'NetCDF Grid Settings' heading in grid fields"
    )


# ---------------------------------------------------------------------------
# GM4: Overlay selector has expected entries for EEC Full
# ---------------------------------------------------------------------------

def test_overlay_selector_entries(page: Page, app: ShinyAppProc):
    """EEC Full overlay selector should contain Grid Extent, LTL Biomass,
    species distribution maps, fishing distribution, and Movement Animation."""
    _load_eec_full(page, app)
    _goto_grid(page)
    _wait_for_overlay_options(page, min_count=10)

    options = page.locator("#grid_overlay option").all_text_contents()
    options_lower = [o.lower() for o in options]

    assert any("grid extent" in o for o in options_lower), (
        f"Missing 'Grid Extent' in options: {options}"
    )
    assert any("ltl biomass" in o for o in options_lower), (
        f"Missing 'LTL Biomass' in options: {options}"
    )
    assert any("movement" in o for o in options_lower), (
        f"Missing 'Movement Animation' in options: {options}"
    )
    # Species distribution maps should appear (e.g., cod, sole)
    assert any("cod:" in o for o in options_lower), (
        f"Missing cod distribution maps in options: {options}"
    )
    # Fishing fleet distribution should appear
    assert any("fishing" in o for o in options_lower), (
        f"Missing Fishing distribution in options: {options}"
    )


# ---------------------------------------------------------------------------
# GM5: LTL Biomass NC overlay has 10 plankton variables
# ---------------------------------------------------------------------------

def test_ltl_biomass_has_plankton_vars(page: Page, app: ShinyAppProc):
    """LTL Biomass NC overlay should populate nc_var_select with plankton variables."""
    _load_eec_full(page, app)
    _goto_grid(page)
    _wait_for_overlay_options(page)

    page.select_option("#grid_overlay", label="LTL Biomass")
    page.wait_for_function(
        "(document.querySelector('#nc_var_select')?.options?.length ?? 0) >= 2",
        timeout=_GRID_TIMEOUT,
    )

    var_options = page.locator("#nc_var_select option").all_text_contents()
    assert len(var_options) >= 5, (
        f"Expected at least 5 plankton variables, got {len(var_options)}: {var_options}"
    )


# ---------------------------------------------------------------------------
# GM6: Movement Animation shows species selector with EEC species
# ---------------------------------------------------------------------------

def test_movement_animation_species_list(page: Page, app: ShinyAppProc):
    """Selecting Movement Animation should show species selector with focal species."""
    _load_eec_full(page, app)
    _goto_grid(page)
    _wait_for_overlay_options(page)

    page.select_option("#grid_overlay", label="Movement Animation")

    # Wait for movement controls to render
    page.wait_for_function(
        "(document.querySelector('#movement_species')?.options?.length ?? 0) >= 1",
        timeout=_GRID_TIMEOUT,
    )

    species_options = page.locator("#movement_species option").all_text_contents()
    assert len(species_options) >= 5, (
        f"Expected at least 5 species with movement maps, got {len(species_options)}: "
        f"{species_options}"
    )

    # Check that known EEC species appear
    joined = " ".join(species_options).lower()
    assert any(sp in joined for sp in ("cod", "sole", "whiting", "dogfish")), (
        f"Expected known EEC species in movement list: {species_options}"
    )


# ---------------------------------------------------------------------------
# GM7: Movement Animation shows speed and step controls
# ---------------------------------------------------------------------------

def test_movement_animation_controls(page: Page, app: ShinyAppProc):
    """Movement Animation should show speed selector and time step slider."""
    _load_eec_full(page, app)
    _goto_grid(page)
    _wait_for_overlay_options(page)

    page.select_option("#grid_overlay", label="Movement Animation")

    page.wait_for_function(
        "(document.querySelector('#movement_species')?.options?.length ?? 0) >= 1",
        timeout=_GRID_TIMEOUT,
    )

    # Speed selector
    speed = page.locator("#movement_speed")
    expect(speed).to_be_visible(timeout=5_000)
    speed_options = page.locator("#movement_speed option").all_text_contents()
    assert any("1x" in o for o in speed_options), (
        f"Expected '1x' speed option, got: {speed_options}"
    )

    # Time step slider
    step_slider = page.locator("#movement_step")
    expect(step_slider).to_be_visible(timeout=5_000)
    slider_max = step_slider.get_attribute("data-max")
    assert slider_max == "23", (
        f"Expected movement step slider max=23 (24 time steps), got '{slider_max}'"
    )


# ---------------------------------------------------------------------------
# GM8: Switching between overlays updates controls correctly
# ---------------------------------------------------------------------------

def test_overlay_switching_updates_controls(page: Page, app: ShinyAppProc):
    """Switching from LTL Biomass to Movement Animation to Grid Extent should
    show/hide the correct controls each time."""
    _load_eec_full(page, app)
    _goto_grid(page)
    _wait_for_overlay_options(page)

    # Step 1: Select LTL Biomass — NC controls should appear
    page.select_option("#grid_overlay", label="LTL Biomass")
    expect(page.locator("#nc_time_step")).to_be_visible(timeout=_GRID_TIMEOUT)

    # Movement controls should NOT be visible
    assert not page.locator("#movement_species").is_visible(), (
        "Movement species should not be visible during LTL overlay"
    )

    # Step 2: Switch to Movement Animation — movement controls appear, NC controls gone
    page.select_option("#grid_overlay", label="Movement Animation")
    page.wait_for_function(
        "(document.querySelector('#movement_species')?.options?.length ?? 0) >= 1",
        timeout=_GRID_TIMEOUT,
    )
    expect(page.locator("#movement_species")).to_be_visible(timeout=5_000)
    assert not page.locator("#nc_time_step").is_visible(), (
        "NC time step should not be visible during Movement Animation"
    )

    # Step 3: Switch to Grid Extent — no overlay controls
    page.select_option("#grid_overlay", value="grid_extent")
    page.wait_for_timeout(2_000)
    assert not page.locator("#nc_time_step").is_visible(), (
        "NC time step should not be visible for Grid Extent"
    )
    assert not page.locator("#movement_species").is_visible(), (
        "Movement species should not be visible for Grid Extent"
    )


# ---------------------------------------------------------------------------
# GM9: Map canvas persists across overlay switches
# ---------------------------------------------------------------------------

def test_map_persists_across_overlays(page: Page, app: ShinyAppProc):
    """The deck.gl map canvas should remain present when switching overlays."""
    _load_eec_full(page, app)
    _goto_grid(page)
    _wait_for_map_widget(page)
    _wait_for_overlay_options(page)

    for label_or_value in ["LTL Biomass", "grid_extent"]:
        if label_or_value == "grid_extent":
            page.select_option("#grid_overlay", value=label_or_value)
        else:
            page.select_option("#grid_overlay", label=label_or_value)
        page.wait_for_timeout(1_500)
        canvas = page.locator("#grid_map canvas")
        assert canvas.count() >= 1, (
            f"Map canvas missing after switching to {label_or_value}"
        )


# ---------------------------------------------------------------------------
# GM10: Grid hint is hidden when NcGrid config is loaded
# ---------------------------------------------------------------------------

def test_grid_hint_hidden_for_ncgrid(page: Page, app: ShinyAppProc):
    """EEC Full uses NcGrid — the 'configure coordinates' hint should not appear."""
    _load_eec_full(page, app)
    _goto_grid(page)

    page.wait_for_timeout(3_000)

    hint_text = page.locator("#grid_hint").inner_text()
    assert "configure coordinates" not in hint_text.lower(), (
        f"Grid hint should be hidden for NcGrid, but shows: '{hint_text}'"
    )


# ---------------------------------------------------------------------------
# GM11: Selecting a species in Movement Animation changes step slider
# ---------------------------------------------------------------------------

def test_movement_species_switch(page: Page, app: ShinyAppProc):
    """Changing the species selector in Movement Animation should keep controls stable."""
    _load_eec_full(page, app)
    _goto_grid(page)
    _wait_for_overlay_options(page)

    page.select_option("#grid_overlay", label="Movement Animation")
    page.wait_for_function(
        "(document.querySelector('#movement_species')?.options?.length ?? 0) >= 2",
        timeout=_GRID_TIMEOUT,
    )

    species_options = page.locator("#movement_species option").all_text_contents()
    if len(species_options) < 2:
        pytest.skip("Need at least 2 species for switch test")

    # Select second species
    page.select_option("#movement_species", label=species_options[1])
    page.wait_for_timeout(2_000)

    # Controls should still be present
    expect(page.locator("#movement_step")).to_be_visible(timeout=5_000)
    expect(page.locator("#movement_speed")).to_be_visible(timeout=5_000)


# ---------------------------------------------------------------------------
# GM12: Changing NC variable in LTL Biomass keeps time slider
# ---------------------------------------------------------------------------

def test_nc_variable_change_preserves_slider(page: Page, app: ShinyAppProc):
    """Switching the selected NC variable should keep the time step slider visible."""
    _load_eec_full(page, app)
    _goto_grid(page)
    _wait_for_overlay_options(page)

    page.select_option("#grid_overlay", label="LTL Biomass")
    page.wait_for_function(
        "(document.querySelector('#nc_var_select')?.options?.length ?? 0) >= 2",
        timeout=_GRID_TIMEOUT,
    )

    # Get variable options and select the second one
    var_options = page.locator("#nc_var_select option")
    if var_options.count() < 2:
        pytest.skip("Need at least 2 NC variables for switch test")

    second_var_value = var_options.nth(1).get_attribute("value")
    page.select_option("#nc_var_select", value=second_var_value)
    page.wait_for_timeout(2_000)

    # Time slider should still be visible after variable switch
    expect(page.locator("#nc_time_step")).to_be_visible(timeout=_GRID_TIMEOUT)
