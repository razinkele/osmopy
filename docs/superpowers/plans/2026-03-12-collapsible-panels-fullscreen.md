# Collapsible Panels + Fullscreen Map Toggle — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add collapsible left nav sidebar, collapsible page panels on all two-column pages, and fullscreen toggle widget on the grid preview map.

**Architecture:** Pure client-side CSS+JS approach. A hamburger button toggles `nav-collapsed` class on `<html>`. Each page wraps its `layout_columns` in a `osm-split-layout` div with collapse button and expand tab. The fullscreen widget is a one-line addition from shiny-deckgl.

**Tech Stack:** Shiny for Python, shiny-deckgl (fullscreen_widget), CSS transitions, vanilla JS, localStorage

**Spec:** `docs/superpowers/specs/2026-03-12-collapsible-panels-fullscreen-design.md`

---

## Chunk 1: Foundation (collapsible helpers + CSS + JS)

### Task 1: Create `ui/components/collapsible.py` helpers

**Files:**
- Create: `ui/components/collapsible.py`
- Test: `tests/test_collapsible.py`

- [ ] **Step 1: Write failing tests for collapsible helpers**

Create `tests/test_collapsible.py`:

```python
"""Tests for collapsible panel helpers."""

from ui.components.collapsible import collapsible_card_header, expand_tab


def test_collapsible_card_header_renders():
    """collapsible_card_header returns a card header with collapse button."""
    header = collapsible_card_header("Grid Type", "grid")
    html = str(header)
    assert "Grid Type" in html
    assert "osm-collapse-btn" in html
    assert "togglePanel('grid')" in html


def test_expand_tab_renders():
    """expand_tab returns a div with vertical expand tab."""
    tab = expand_tab("Grid Type", "grid")
    html = str(tab)
    assert "Grid Type" in html
    assert "osm-expand-tab" in html
    assert "expand_grid" in html
    assert "togglePanel('grid')" in html


def test_collapsible_card_header_different_pages():
    """Helper generates unique IDs per page."""
    h1 = str(collapsible_card_header("Forcing", "forcing"))
    h2 = str(collapsible_card_header("Fishing", "fishing"))
    assert "togglePanel('forcing')" in h1
    assert "togglePanel('fishing')" in h2


def test_expand_tab_different_pages():
    """expand_tab generates unique IDs per page."""
    t1 = str(expand_tab("Forcing", "forcing"))
    t2 = str(expand_tab("Fishing", "fishing"))
    assert "expand_forcing" in t1
    assert "expand_fishing" in t2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_collapsible.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.components.collapsible'`

- [ ] **Step 3: Implement collapsible helpers**

Create `ui/components/collapsible.py`:

```python
"""Collapsible panel helpers for page layouts."""

from shiny import ui as _ui


def collapsible_card_header(title: str, page_id: str):
    """Card header with a collapse toggle button.

    Parameters
    ----------
    title
        Text displayed in the card header.
    page_id
        Unique identifier used for localStorage persistence and DOM targeting.
    """
    return _ui.card_header(
        _ui.tags.span(title),
        _ui.tags.button(
            "\u00ab",
            class_="osm-collapse-btn",
            onclick=f"togglePanel('{page_id}')",
            title="Collapse panel",
        ),
    )


def expand_tab(title: str, page_id: str):
    """Vertical expand tab shown when the left panel is collapsed.

    Placed as a flex sibling before the layout_columns `.row` inside
    an `osm-split-layout` wrapper div.

    Parameters
    ----------
    title
        Text displayed vertically on the tab.
    page_id
        Must match the page_id used in collapsible_card_header.
    """
    return _ui.div(
        title,
        class_="osm-expand-tab",
        id=f"expand_{page_id}",
        onclick=f"togglePanel('{page_id}')",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_collapsible.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add ui/components/collapsible.py tests/test_collapsible.py
git commit -m "feat: add collapsible panel helpers (card header + expand tab)"
```

---

### Task 2: Add CSS for collapsible nav, split-layout panels, and light theme

**Files:**
- Modify: `www/osmose.css` (append before the first `@media` block at line 1205)
- Note: `ui/components/__init__.py` is empty — no update needed for the new module to be importable

- [ ] **Step 1: Add all collapsible CSS rules**

Insert the following CSS **before** the `/* ── Responsive / mobile */` section (line 1205 of `www/osmose.css`):

```css
/* ── Collapsible nav sidebar ──────────────────────────────────── */
.osm-hamburger {
    display: block;
    width: 100%;
    background: none;
    border: 1px solid var(--osm-border);
    border-radius: 6px;
    color: var(--osm-text-secondary);
    cursor: pointer;
    padding: 8px;
    margin-bottom: 8px;
    font-size: 16px;
    line-height: 1;
    text-align: center;
    transition: all var(--osm-transition);
}
.osm-hamburger:hover {
    color: var(--osm-accent);
    background: var(--osm-accent-dim);
}
.osm-hamburger-icon {
    display: inline-block;
    width: 18px;
    height: 2px;
    background: currentColor;
    position: relative;
    vertical-align: middle;
}
.osm-hamburger-icon::before,
.osm-hamburger-icon::after {
    content: '';
    display: block;
    width: 18px;
    height: 2px;
    background: currentColor;
    position: absolute;
    left: 0;
}
.osm-hamburger-icon::before { top: -6px; }
.osm-hamburger-icon::after { top: 6px; }

/* Nav column transitions */
.nav-pills {
    transition: width var(--osm-transition-slow), padding var(--osm-transition-slow);
}

/* Collapsed nav — scoped to navset_pill_list container */
html.nav-collapsed .bslib-navs-pill-list > .row > [class*="col-"]:first-child {
    width: 42px !important;
    min-width: 42px !important;
    max-width: 42px !important;
    padding-left: 4px !important;
    padding-right: 4px !important;
    transition: width var(--osm-transition-slow), min-width var(--osm-transition-slow);
}
html.nav-collapsed .bslib-navs-pill-list > .row > [class*="col-"]:last-child {
    flex: 1;
    max-width: 100% !important;
    transition: max-width var(--osm-transition-slow);
}
html.nav-collapsed .osmose-section-label {
    display: none;
}
html.nav-collapsed .nav-pills .nav-link {
    font-size: 0 !important;
    padding: 8px 6px !important;
    text-align: center;
    overflow: hidden;
    white-space: nowrap;
}
html.nav-collapsed .nav-pills .nav-link.active {
    box-shadow: none;
    border-radius: 6px !important;
}
html.nav-collapsed .osm-hamburger {
    font-size: 16px !important;
}

/* ── Collapsible page panels ──────────────────────────────────── */
.osm-split-layout {
    display: flex;
    height: 100%;
}
.osm-split-layout > .row {
    flex: 1;
    flex-wrap: nowrap;
}
.osm-split-layout > .row > div:first-child {
    transition: width var(--osm-transition-slow),
                opacity var(--osm-transition),
                padding var(--osm-transition-slow),
                min-width var(--osm-transition-slow);
}
.osm-split-layout > .row > div:first-child.collapsed {
    width: 0 !important;
    flex: 0 !important;
    min-width: 0 !important;
    overflow: hidden;
    opacity: 0;
    padding: 0 !important;
}
.osm-split-layout > .row > div:first-child.collapsed ~ div {
    flex: 1;
    max-width: 100%;
}

/* ── Expand tab ───────────────────────────────────────────────── */
.osm-expand-tab {
    display: none;
    writing-mode: vertical-rl;
    text-orientation: mixed;
    background: var(--osm-bg-card);
    border: 1px solid var(--osm-border);
    border-left: 3px solid var(--osm-accent);
    padding: 12px 6px;
    border-radius: 0 8px 8px 0;
    cursor: pointer;
    color: var(--osm-text-secondary);
    font-size: 12px;
    font-weight: 600;
    align-self: center;
    flex-shrink: 0;
    transition: all var(--osm-transition);
}
.osm-expand-tab:hover {
    color: var(--osm-accent);
    background: var(--osm-bg-card-hover);
}
.osm-expand-tab.visible {
    display: block;
}

/* ── Collapse button in card header ───────────────────────────── */
.osm-collapse-btn {
    background: none;
    border: none;
    color: var(--osm-text-muted);
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 16px;
    line-height: 1;
    margin-left: auto;
    transition: all var(--osm-transition);
}
.osm-collapse-btn:hover {
    color: var(--osm-accent);
    background: var(--osm-accent-dim);
}
.card-header:has(.osm-collapse-btn) {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* ── Light theme overrides for collapsible components ─────────── */
[data-theme="light"] .osm-expand-tab {
    background: var(--osm-bg-card);
    border-color: var(--osm-border);
    border-left-color: var(--osm-accent);
    color: var(--osm-text-secondary);
}
[data-theme="light"] .osm-expand-tab:hover {
    color: var(--osm-accent);
    background: var(--osm-bg-card-hover);
}
[data-theme="light"] .osm-collapse-btn {
    color: var(--osm-text-muted);
}
[data-theme="light"] .osm-collapse-btn:hover {
    color: var(--osm-accent);
    background: var(--osm-accent-dim);
}
[data-theme="light"] .osm-hamburger {
    color: var(--osm-text-secondary);
}
[data-theme="light"] .osm-hamburger:hover {
    color: var(--osm-accent);
}
```

**Important:** The nav-collapsed selector uses `.bslib-navs-pill-list > .row > [class*="col-"]` which is the actual class Shiny's `navset_pill_list` generates. During implementation, if the app renders differently, inspect the DOM and adjust the selector. The key constraint: do NOT use bare `.col-sm-2`.

- [ ] **Step 2: Verify CSS is valid — run the app briefly**

Run: `.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000 &` then check no CSS parse errors in browser console. Kill the server after.

- [ ] **Step 3: Commit**

```bash
git add www/osmose.css
git commit -m "feat: add CSS for collapsible nav, split-layout panels, and expand tabs"
```

---

### Task 3: Add JS functions to `app.py`

**Files:**
- Modify: `app.py:46-97` (inline script block)

- [ ] **Step 1: Add toggleNav, togglePanel, and restore logic**

In `app.py`, inside the existing `ui.tags.script("""...""")` block, add the following **after** the existing `toggleHelpMode` / restore help mode IIFE (after line 97, before the closing `"""`):

```javascript
        // ── Nav collapse ──────────────────────────────────
        function toggleNav() {
            var html = document.documentElement;
            var collapsed = html.classList.toggle('nav-collapsed');
            localStorage.setItem('osmose-nav-collapsed', collapsed ? '1' : '0');
        }
        // Restore nav state immediately (before render)
        (function() {
            if (localStorage.getItem('osmose-nav-collapsed') === '1') {
                document.documentElement.classList.add('nav-collapsed');
            }
        })();

        // ── Panel collapse ────────────────────────────────
        function togglePanel(pageId) {
            var container = document.getElementById('split_' + pageId);
            if (!container) return;
            var row = container.querySelector('.row');
            if (!row) return;
            var left = row.children[0];
            var tab = document.getElementById('expand_' + pageId);

            var collapsed = left.classList.toggle('collapsed');
            if (tab) tab.classList.toggle('visible', collapsed);
            localStorage.setItem('osmose-panel-collapsed-' + pageId, collapsed ? '1' : '0');
        }

        // ── Restore panel states on tab activation ────────
        (function() {
            var restoredPanels = {};
            function restorePanelIfNeeded(pageId) {
                if (restoredPanels[pageId]) return;
                restoredPanels[pageId] = true;
                if (localStorage.getItem('osmose-panel-collapsed-' + pageId) === '1') {
                    setTimeout(function() { togglePanel(pageId); }, 100);
                }
            }
            var pageIds = ['setup','grid','forcing','fishing','movement',
                           'run','results','calibration','scenarios','advanced'];

            document.addEventListener('DOMContentLoaded', function() {
                // Restore the initially active tab's panel
                var activeLink = document.querySelector('.nav-pills .nav-link.active');
                if (activeLink) {
                    var val = activeLink.getAttribute('data-value') || '';
                    pageIds.forEach(function(pid) {
                        if (val.indexOf(pid) !== -1) restorePanelIfNeeded(pid);
                    });
                }
                // Restore panels as tabs are activated (lazy rendering)
                document.addEventListener('shown.bs.tab', function(e) {
                    var val = e.target.getAttribute('data-value') ||
                              e.target.getAttribute('href') || '';
                    pageIds.forEach(function(pid) {
                        if (val.indexOf(pid) !== -1) restorePanelIfNeeded(pid);
                    });
                });
            });
        })();
```

- [ ] **Step 2: Add hamburger nav_control to navset_pill_list**

In `app.py`, add the hamburger button as the first item inside `ui.navset_pill_list(` (after line 148, before `_nav_section("Configure")`):

```python
        # Hamburger toggle for collapsible nav
        ui.nav_control(
            ui.tags.button(
                ui.tags.span(class_="osm-hamburger-icon"),
                class_="osm-hamburger",
                onclick="toggleNav()",
                title="Toggle navigation",
            ),
        ),
```

- [ ] **Step 3: Run existing tests to verify nothing is broken**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py -v`
Expected: 4 PASSED (app still imports, nav sections still present, hamburger is just an additional nav_control)

- [ ] **Step 4: Write a test for the hamburger button**

Add to `tests/test_app_structure.py`:

```python
def test_hamburger_toggle_present():
    """Hamburger nav toggle button is in the rendered HTML."""
    from app import app_ui

    html = str(app_ui)
    assert "osm-hamburger" in html
    assert "toggleNav()" in html
```

- [ ] **Step 5: Run all app structure tests**

Run: `.venv/bin/python -m pytest tests/test_app_structure.py -v`
Expected: 5 PASSED

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app_structure.py
git commit -m "feat: add nav collapse JS + hamburger toggle button"
```

---

## Chunk 2: Fullscreen widget + Grid page collapsible panel

### Task 4: Add fullscreen_widget to grid map

**Files:**
- Modify: `ui/pages/grid.py:11` (imports) and `ui/pages/grid.py:820` (widgets list)

- [ ] **Step 1: Add fullscreen_widget import**

In `ui/pages/grid.py`, add `fullscreen_widget` to the shiny_deckgl import block (line 11-19):

```python
from shiny_deckgl import (  # type: ignore[import-untyped]
    MapWidget,
    polygon_layer,
    CARTO_POSITRON,
    CARTO_DARK,
    zoom_widget,
    compass_widget,
    fullscreen_widget,
    scale_widget,
    deck_legend_control,
)
```

- [ ] **Step 2: Add fullscreen_widget to the widgets list**

In `ui/pages/grid.py`, modify line 820-824 to include `fullscreen_widget`:

```python
        widgets = [
            fullscreen_widget(placement="top-left"),
            zoom_widget(placement="top-right"),
            compass_widget(placement="top-right"),
            scale_widget(placement="bottom-right"),
        ]
```

- [ ] **Step 3: Write a test for fullscreen_widget**

Add to `tests/test_collapsible.py`:

```python
def test_grid_has_fullscreen_widget_import():
    """grid.py imports fullscreen_widget from shiny_deckgl."""
    from ui.pages import grid
    assert hasattr(grid, 'fullscreen_widget') or 'fullscreen_widget' in dir(grid)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ -v -k grid`
Expected: All grid tests pass

- [ ] **Step 5: Commit**

```bash
git add ui/pages/grid.py tests/test_collapsible.py
git commit -m "feat: add fullscreen toggle widget to grid preview map"
```

---

### Task 5: Make grid page panel collapsible

**Files:**
- Modify: `ui/pages/grid.py:532-562` (grid_ui function)

- [ ] **Step 1: Add collapsible imports to grid.py**

Add at the top of `ui/pages/grid.py` (after the existing imports around line 24):

```python
from ui.components.collapsible import collapsible_card_header, expand_tab
```

- [ ] **Step 2: Wrap grid_ui layout in split-layout with collapsible header**

Replace the `return ui.layout_columns(...)` block in `grid_ui()` (lines 550-562) with:

```python
    return ui.div(
        expand_tab("Grid Type", "grid"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Grid Type", "grid"),
                ui.output_ui("grid_fields"),
            ),
            ui.card(
                ui.card_header("Grid Preview"),
                ui.output_ui("grid_overlay_selector"),
                ui.output_ui("grid_hint"),
                grid_map.ui(height="500px"),
            ),
            col_widths=[6, 6],
        ),
        class_="osm-split-layout",
        id="split_grid",
    )
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/ -v -k grid`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add ui/pages/grid.py
git commit -m "feat: make grid page panel collapsible with expand tab"
```

---

## Chunk 3: Wire remaining pages (batch 1 — Configure group)

### Task 6: Make Setup page panel collapsible

**Files:**
- Modify: `ui/pages/setup.py`

- [ ] **Step 1: Read setup_ui and add collapsible imports**

Add import at top of `ui/pages/setup.py`:

```python
from ui.components.collapsible import collapsible_card_header, expand_tab
```

- [ ] **Step 2: Wrap setup_ui layout**

In `setup_ui()`, wrap the `return ui.layout_columns(...)` (lines 25-53) in a split-layout div. Replace `ui.card_header("Simulation Settings")` with `collapsible_card_header("Simulation Settings", "setup")`. Add `expand_tab("Simulation Settings", "setup")` before `ui.layout_columns`.

The pattern for every page is identical:

```python
return ui.div(
    expand_tab("TITLE", "PAGE_ID"),
    ui.layout_columns(
        ui.card(
            collapsible_card_header("TITLE", "PAGE_ID"),
            # ... existing card body content unchanged ...
        ),
        # ... remaining cards unchanged ...
        col_widths=[X, Y],
    ),
    class_="osm-split-layout",
    id="split_PAGE_ID",
)
```

- [ ] **Step 3: Commit**

```bash
git add ui/pages/setup.py
git commit -m "feat: make setup page panel collapsible"
```

---

### Task 7: Make Forcing page panel collapsible

**Files:**
- Modify: `ui/pages/forcing.py`

- [ ] **Step 1: Add imports and wrap layout**

Same pattern as Task 6. Import collapsible helpers. Wrap `forcing_ui()` (lines 19-36). Replace `ui.card_header("Lower Trophic Level (Plankton)")` with `collapsible_card_header("Lower Trophic Level (Plankton)", "forcing")`. Add expand tab.

- [ ] **Step 2: Commit**

```bash
git add ui/pages/forcing.py
git commit -m "feat: make forcing page panel collapsible"
```

---

### Task 8: Make Fishing page panel collapsible

**Files:**
- Modify: `ui/pages/fishing.py`

- [ ] **Step 1: Add imports and wrap layout**

Same pattern. Title: `"Fisheries Module"`, page_id: `"fishing"`. Lines 13-26.

- [ ] **Step 2: Commit**

```bash
git add ui/pages/fishing.py
git commit -m "feat: make fishing page panel collapsible"
```

---

### Task 9: Make Movement page panel collapsible

**Files:**
- Modify: `ui/pages/movement.py`

- [ ] **Step 1: Add imports and wrap layout**

Same pattern. Title: `"Movement Settings"`, page_id: `"movement"`. Lines 13-26.

- [ ] **Step 2: Commit**

```bash
git add ui/pages/movement.py
git commit -m "feat: make movement page panel collapsible"
```

---

## Chunk 4: Wire remaining pages (batch 2 — Execute + Optimize + Manage groups)

### Task 10: Make Run page panel collapsible

**Files:**
- Modify: `ui/pages/run.py`

- [ ] **Step 1: Add imports and wrap layout**

Same pattern. Title: `"Run Configuration"`, page_id: `"run"`. Lines 103-128.

- [ ] **Step 2: Commit**

```bash
git add ui/pages/run.py
git commit -m "feat: make run page panel collapsible"
```

---

### Task 11: Make Results page panel collapsible

**Files:**
- Modify: `ui/pages/results.py`

- [ ] **Step 1: Add imports and wrap layout**

Title: `"Output Controls"`, page_id: `"results"`. This page has a more complex structure:
- `results_ui()` returns `ui.div(ui.layout_columns(..., col_widths=[3, 9]), ui.navset_card_tab(...))`
- The outer `ui.div` contains both `layout_columns` AND a `navset_card_tab` (Diet, Spatial, Compare)
- The split-layout wrapper should wrap ONLY the `layout_columns`, not the `navset_card_tab`

So the pattern is slightly different — wrap the `layout_columns` in a split-layout div, keep the `navset_card_tab` as a sibling:

```python
def results_ui():
    return ui.div(
        ui.div(
            expand_tab("Output Controls", "results"),
            ui.layout_columns(
                ui.card(
                    collapsible_card_header("Output Controls", "results"),
                    # ... existing card body unchanged ...
                ),
                ui.card(
                    ui.card_header("Time Series"),
                    output_widget("results_chart"),
                ),
                col_widths=[3, 9],
            ),
            class_="osm-split-layout",
            id="split_results",
        ),
        ui.navset_card_tab(
            # ... unchanged ...
        ),
    )
```

Replace `ui.card_header("Output Controls")` (line 113) with `collapsible_card_header("Output Controls", "results")`.

- [ ] **Step 2: Commit**

```bash
git add ui/pages/results.py
git commit -m "feat: make results page panel collapsible"
```

---

### Task 12: Make Calibration page panel collapsible

**Files:**
- Modify: `ui/pages/calibration.py`

- [ ] **Step 1: Add imports and wrap layout**

Title: `"Calibration Setup"`, page_id: `"calibration"`. Lines 45-113.

- [ ] **Step 2: Commit**

```bash
git add ui/pages/calibration.py
git commit -m "feat: make calibration page panel collapsible"
```

---

### Task 13: Make Scenarios page panel collapsible (3-column special case)

**Files:**
- Modify: `ui/pages/scenarios.py`

- [ ] **Step 1: Add imports and wrap layout**

Title: `"Save Scenario"`, page_id: `"scenarios"`. This page has a complex structure:
- `scenarios_ui()` returns `ui.div(ui.layout_columns(..., col_widths=[3, 5, 4]), ui.card("Bulk Operations"))`
- The outer `ui.div` wraps the 3-column `layout_columns` AND a separate "Bulk Operations" card
- The split-layout wrapper should wrap the ENTIRE existing `ui.div` content

The 3-column layout has: col-3 "Save Scenario", col-5 "Saved Scenarios", col-4 "Compare Scenarios". The JS `togglePanel` targets `row.children[0]` (the "Save Scenario" col-3). The CSS `div:first-child.collapsed ~ div` expands the remaining two columns proportionally.

```python
def scenarios_ui():
    return ui.div(
        expand_tab("Save Scenario", "scenarios"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Save Scenario", "scenarios"),
                # ... existing Save Scenario card body unchanged ...
            ),
            ui.card(
                ui.card_header("Saved Scenarios"),
                # ... unchanged ...
            ),
            ui.card(
                ui.card_header("Compare Scenarios"),
                # ... unchanged ...
            ),
            col_widths=[3, 5, 4],
        ),
        ui.card(
            ui.card_header("Bulk Operations"),
            # ... unchanged ...
        ),
        class_="osm-split-layout",
        id="split_scenarios",
    )
```

Replace `ui.card_header("Save Scenario")` (line 16) with `collapsible_card_header("Save Scenario", "scenarios")`. Move the outer `ui.div(` wrapper to use `class_="osm-split-layout", id="split_scenarios"`. Add `expand_tab` before `layout_columns`.

- [ ] **Step 2: Commit**

```bash
git add ui/pages/scenarios.py
git commit -m "feat: make scenarios page panel collapsible (3-column layout)"
```

---

### Task 14: Make Advanced page panel collapsible

**Files:**
- Modify: `ui/pages/advanced.py`

- [ ] **Step 1: Add imports and wrap layout**

Title: `"Config I/O"`, page_id: `"advanced"`. Note: `advanced_ui()` returns a `ui.div()` wrapper (line 32). The `layout_columns` starts at line 33 with `col_widths=[4, 8]`.

- [ ] **Step 2: Commit**

```bash
git add ui/pages/advanced.py
git commit -m "feat: make advanced page panel collapsible"
```

---

## Chunk 5: Final tests + verification

### Task 15: Add comprehensive tests for all collapsible pages

**Files:**
- Modify: `tests/test_collapsible.py`

- [ ] **Step 1: Add tests for all pages**

Append to `tests/test_collapsible.py`:

```python
import pytest


@pytest.mark.parametrize(
    "page_mod,page_id",
    [
        ("ui.pages.setup", "setup"),
        ("ui.pages.grid", "grid"),
        ("ui.pages.forcing", "forcing"),
        ("ui.pages.fishing", "fishing"),
        ("ui.pages.movement", "movement"),
        ("ui.pages.run", "run"),
        ("ui.pages.results", "results"),
        ("ui.pages.calibration", "calibration"),
        ("ui.pages.scenarios", "scenarios"),
        ("ui.pages.advanced", "advanced"),
    ],
)
def test_page_has_split_layout(page_mod, page_id):
    """Each page has osm-split-layout wrapper, expand tab, and collapse button."""
    import importlib

    mod = importlib.import_module(page_mod)
    ui_fn = getattr(mod, f"{page_mod.split('.')[-1]}_ui")
    html = str(ui_fn())
    assert f"split_{page_id}" in html, f"Missing split_{page_id} wrapper"
    assert f"expand_{page_id}" in html, f"Missing expand_{page_id} tab"
    assert "osm-collapse-btn" in html, f"Missing collapse button on {page_id}"
```

- [ ] **Step 2: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_collapsible.py -v`
Expected: All PASSED (4 unit tests + 10 parametrized page tests)

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All 400+ tests pass, 0 failures

- [ ] **Step 4: Lint check**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`
Expected: No errors

- [ ] **Step 5: Commit test additions**

```bash
git add tests/test_collapsible.py
git commit -m "test: add comprehensive tests for all collapsible page panels"
```

---

### Task 16: Manual verification checklist

- [ ] **Step 1: Start the app**

Run: `.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000`

- [ ] **Step 2: Verify nav collapse**

Open browser. Click the hamburger icon in the nav sidebar. Verify:
- Sidebar shrinks to ~42px, showing only hamburger icon
- Click again to expand back
- Reload page — state persists

- [ ] **Step 3: Verify grid panel collapse + fullscreen**

Navigate to Grid & Maps. Verify:
- `<<` button appears in "Grid Type" card header
- Clicking it collapses the left panel, "Grid Type" expand tab appears on left edge
- Click expand tab to restore
- Fullscreen button appears top-left of map
- Click fullscreen — map goes fullscreen
- ESC exits fullscreen
- Reload — panel collapse state persists

- [ ] **Step 4: Verify all other pages**

Navigate to each page (Setup, Forcing, Fishing, Movement, Run, Results, Calibration, Scenarios, Advanced). For each:
- `<<` collapse button present in left card header
- Collapse/expand works
- State persists per-page

- [ ] **Step 5: Verify light/dark theme**

Toggle theme. Verify expand tab, collapse button, and hamburger have appropriate contrast in both themes.

- [ ] **Step 6: Verify nav collapse scoping**

With nav collapsed, navigate to different pages. Verify the page-level `layout_columns` columns are NOT affected by the nav-collapsed CSS (only the nav sidebar shrinks, not page content columns).
