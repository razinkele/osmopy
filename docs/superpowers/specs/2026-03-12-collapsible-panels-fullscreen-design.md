# Collapsible Panels + Fullscreen Map Toggle

**Date**: 2026-03-12
**Status**: Draft

## Summary

Add three UI enhancements to OSMOPY:

1. **Collapsible left nav sidebar** — hamburger toggle shrinks nav to icon strip, persists via localStorage
2. **Collapsible page panels** — all two-column pages get a collapse button on their left card, persists per-page via localStorage
3. **Fullscreen map widget** — add `fullscreen_widget("top-left")` to the grid preview map

All changes are client-side CSS+JS. No server-side reactive logic changes needed.

## 1. Collapsible Left Nav Sidebar

### Current State

`navset_pill_list` with `widths=(2, 10)` generates a Bootstrap row: col-2 sidebar + col-10 content. The sidebar contains 4 section groups (Configure, Execute, Optimize, Manage) with 10 nav pills. The nav links are `ui.nav_panel("Label", ...)` which generates `<a class="nav-link">Label</a>` with text as a direct text node (not wrapped in `<span>`).

### Design

- Add a hamburger toggle button as the first `ui.nav_control()` in the pill list
- Clicking it toggles a `nav-collapsed` class on `<html>` (same pattern as theme toggle)
- When collapsed: sidebar shrinks from ~180px to 42px, shows only hamburger icon
- Nav link text hidden via `font-size: 0; overflow: hidden` on the `.nav-link` itself (text is a direct text node, not a `<span>`, so `display: none` on children won't work)
- Section labels hidden via `display: none`
- State saved to `localStorage('osmose-nav-collapsed')`
- JS `toggleNav()` function added to the inline script block in `app.py`
- On page load, init block reads localStorage and applies class if saved

### CSS Rules

Selectors are scoped to the navset container to avoid matching `col-sm-2` elements elsewhere on the page:

```css
/* Collapsed sidebar — scoped to the navset_pill_list container */
html.nav-collapsed .tab-content-wrapper > .row > .col-sm-2,
html.nav-collapsed #main_nav ~ .row > .col-sm-2 {
  width: 42px !important;
  min-width: 42px !important;
  max-width: 42px !important;
  transition: width var(--osm-transition-slow);
}
html.nav-collapsed .tab-content-wrapper > .row > .col-sm-10,
html.nav-collapsed #main_nav ~ .row > .col-sm-10 {
  flex: 1;
  max-width: 100% !important;
}

/* Hide section labels entirely */
html.nav-collapsed .osmose-section-label {
  display: none;
}

/* Hide nav link text — text is a direct text node, use font-size: 0 */
html.nav-collapsed .nav-pills .nav-link {
  font-size: 0 !important;
  padding: 8px 10px !important;
  text-align: center;
  width: 42px;
  overflow: hidden;
}

/* Keep hamburger icon visible at normal size */
html.nav-collapsed .osm-hamburger {
  font-size: 16px !important;
}
```

**Note**: The exact scoping selector depends on the HTML structure Shiny generates for `navset_pill_list`. During implementation, inspect the generated DOM to confirm the correct parent selector. The key constraint is: do NOT use bare `.col-sm-2` which would match page-level `layout_columns` too.

### Hamburger Button

Placed as `ui.nav_control()` at the top of `navset_pill_list`:

```python
ui.nav_control(
    ui.tags.button(
        ui.tags.span(class_="osm-hamburger-icon"),
        class_="osm-hamburger",
        onclick="toggleNav()",
        title="Toggle navigation",
    ),
),
```

## 2. Collapsible Page Panels

### Current State

All pages use `ui.layout_columns(..., col_widths=[X, Y])` for two-column layouts. The left column holds configuration/control fields, the right holds output/preview.

### Affected Pages

| Page | Left Card Title | col_widths | page_id | Notes |
|------|----------------|-----------|---------|-------|
| Grid | Grid Type | 6/6 | grid | Standard 2-col |
| Forcing | Forcing Fields | 7/5 | forcing | Standard 2-col |
| Fishing | Fishing Fields | 8/4 | fishing | Standard 2-col |
| Movement | Movement Fields | 5/7 | movement | Standard 2-col |
| Setup | Configuration | 4/8 | setup | Standard 2-col |
| Advanced | Import/Export | 4/8 | advanced | Standard 2-col |
| Calibration | Parameters | 4/8 | calibration | Standard 2-col |
| Results | Filters | 3/9 | results | Standard 2-col |
| Run | Controls | 4/8 | run | Standard 2-col |
| Scenarios | Scenario List | 3/5/4 | scenarios | **3-column — see below** |

### Scenarios Page Special Handling

The Scenarios page has `col_widths=[3, 5, 4]` (3 columns: Scenario List, Comparison, Bulk Operations). For this page:

- Collapse the **first column only** (Scenario List, col-3)
- The remaining two columns share the freed space proportionally
- The expand tab still shows "Scenario List" when collapsed
- The JS `togglePanel` function works the same way (targets `row.children[0]`)

### Design

Each page's `*_ui()` function:

1. Wraps the `layout_columns` in `ui.div(class_="osm-split-layout", id="split_{page_id}")`
2. Replaces the left card's `ui.card_header()` with `collapsible_card_header(title, page_id)` — adds a `<<` collapse button
3. Adds an `expand_tab(title, page_id)` div **inside the wrapper div, before the `layout_columns`** — it sits as a flex sibling of the `.row`

When the collapse button is clicked:

- JS `togglePanel(pageId)` adds `.collapsed` class to the first `.col-*` child of `.row`
- The expand tab becomes visible
- Remaining column(s) expand to fill via CSS
- State saved to `localStorage('osmose-panel-collapsed-{pageId}')`

When the expand tab is clicked:

- Left column re-expands
- Expand tab hides
- localStorage updated

### DOM Structure

```html
<div class="osm-split-layout" id="split_grid">
  <!-- expand tab: flex sibling of the row -->
  <div class="osm-expand-tab" id="expand_grid">Grid Type</div>
  <!-- layout_columns generates this: -->
  <div class="row">
    <div class="col-sm-6"><!-- left card --></div>
    <div class="col-sm-6"><!-- right card --></div>
  </div>
</div>
```

The `.osm-split-layout` uses `display: flex` so the expand tab and `.row` are flex siblings. The `.row` gets `flex: 1` to fill remaining space.

### Python Helpers

New file `ui/components/collapsible.py`:

```python
from shiny import ui as _ui


def collapsible_card_header(title: str, page_id: str):
    """Card header with a collapse toggle button."""
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
    """Vertical expand tab shown when the left panel is collapsed."""
    return _ui.div(
        title,
        class_="osm-expand-tab",
        id=f"expand_{page_id}",
        onclick=f"togglePanel('{page_id}')",
    )
```

### CSS Rules

Only one set of selectors — targeting the Bootstrap-generated `.row > div:first-child`:

```css
/* ── Collapsible page panels ─────────────────────── */
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
              padding var(--osm-transition-slow);
}

.osm-split-layout > .row > div:first-child.collapsed {
  width: 0 !important;
  flex: 0 !important;
  min-width: 0 !important;
  overflow: hidden;
  opacity: 0;
  padding: 0 !important;
}

/* Remaining columns expand */
.osm-split-layout > .row > div:first-child.collapsed ~ div {
  flex: 1;
  max-width: 100%;
}

/* ── Expand tab ──────────────────────────────────── */
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

/* ── Collapse button in card header ──────────────── */
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

/* Card header needs flex for collapse button alignment */
.card-header:has(.osm-collapse-btn) {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
```

### Light Theme Overrides

```css
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

**Note**: Since these components use CSS variables (`var(--osm-bg-card)`, etc.) which are already overridden in the light theme, most light-mode styles cascade automatically. The explicit overrides above are for safety and any edge cases where the variable values don't provide sufficient contrast.

## 3. Fullscreen Map Widget

### Current State

`update_grid_map()` in `ui/pages/grid.py` passes widgets to `_map.update()` inside the reactive effect (around line 820):

```python
await _map.update(
    session,
    layers=layers,
    view_state=view_state,
    transition_duration=800,
    widgets=[zoom_widget(), compass_widget(), scale_widget(), deck_legend_control(...)],
)
```

### Design

Add `fullscreen_widget("top-left")` to the widgets list in the `_map.update()` call:

```python
from shiny_deckgl import fullscreen_widget

# In the update_grid_map effect:
await _map.update(
    session,
    layers=layers,
    view_state=view_state,
    transition_duration=800,
    widgets=[
        fullscreen_widget("top-left"),
        zoom_widget(),
        compass_widget(),
        scale_widget(),
        deck_legend_control(...),
    ],
)
```

This is a native deck.gl `FullscreenWidget` using the browser Fullscreen API. Handles enter/exit fullscreen, ESC key, and icon toggle automatically. No custom CSS or JS required.

Placement `"top-left"` avoids crowding the zoom/compass/scale on the right.

## 4. JS Functions

Added to the inline `<script>` block in `app.py`:

```javascript
// ── Nav collapse ──────────────────────────────────
function toggleNav() {
    var html = document.documentElement;
    var collapsed = html.classList.toggle('nav-collapsed');
    localStorage.setItem('osmose-nav-collapsed', collapsed ? '1' : '0');
}

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

// ── Restore saved states on load ──────────────────
(function() {
    // Nav — apply immediately (before render) to avoid flash
    if (localStorage.getItem('osmose-nav-collapsed') === '1') {
        document.documentElement.classList.add('nav-collapsed');
    }

    // Panels — restore when each tab is shown (lazy rendering)
    // navset_pill_list renders inactive tabs lazily, so DOMContentLoaded
    // won't find elements for non-active tabs. Instead, listen for
    // Bootstrap tab show events and restore on first activation.
    var restoredPanels = {};
    function restorePanelIfNeeded(pageId) {
        if (restoredPanels[pageId]) return;
        restoredPanels[pageId] = true;
        if (localStorage.getItem('osmose-panel-collapsed-' + pageId) === '1') {
            // Small delay to let Shiny render the tab content
            setTimeout(function() { togglePanel(pageId); }, 100);
        }
    }

    // Map page_id to nav panel value attributes
    var pageIdMap = {
        'setup': 'setup', 'grid': 'grid', 'forcing': 'forcing',
        'fishing': 'fishing', 'movement': 'movement', 'run': 'run',
        'results': 'results', 'calibration': 'calibration',
        'scenarios': 'scenarios', 'advanced': 'advanced'
    };

    // Restore active tab's panel on DOMContentLoaded
    document.addEventListener('DOMContentLoaded', function() {
        // Find the initially active tab and restore its panel
        var activeLink = document.querySelector('.nav-pills .nav-link.active');
        if (activeLink) {
            var href = activeLink.getAttribute('data-value') ||
                       activeLink.getAttribute('href');
            Object.keys(pageIdMap).forEach(function(pid) {
                if (href && href.indexOf(pageIdMap[pid]) !== -1) {
                    restorePanelIfNeeded(pid);
                }
            });
        }

        // Listen for tab switches to restore panels on activation
        document.addEventListener('shown.bs.tab', function(e) {
            var target = e.target;
            var value = target.getAttribute('data-value') ||
                        target.getAttribute('href') || '';
            Object.keys(pageIdMap).forEach(function(pid) {
                if (value.indexOf(pageIdMap[pid]) !== -1) {
                    restorePanelIfNeeded(pid);
                }
            });
        });
    });
})();
```

## 5. Files Modified

| File | Change |
|------|--------|
| `app.py` | Add hamburger `nav_control`, JS functions for toggleNav/togglePanel/restore |
| `www/osmose.css` | Add nav-collapsed rules (scoped), split-layout, expand-tab, collapse-btn, light theme overrides |
| `ui/components/collapsible.py` | **New file**: `collapsible_card_header()`, `expand_tab()` |
| `ui/pages/grid.py` | Add `fullscreen_widget` import + usage in `_map.update()`, wrap `layout_columns` in split-layout div, use collapsible header |
| `ui/pages/forcing.py` | Wrap in split-layout, use collapsible header |
| `ui/pages/fishing.py` | Wrap in split-layout, use collapsible header |
| `ui/pages/movement.py` | Wrap in split-layout, use collapsible header |
| `ui/pages/setup.py` | Wrap in split-layout, use collapsible header |
| `ui/pages/advanced.py` | Wrap in split-layout, use collapsible header |
| `ui/pages/calibration.py` | Wrap in split-layout, use collapsible header |
| `ui/pages/results.py` | Wrap in split-layout, use collapsible header |
| `ui/pages/run.py` | Wrap in split-layout, use collapsible header |
| `ui/pages/scenarios.py` | Wrap in split-layout, use collapsible header (first of 3 columns) |

## 6. Testing

- Verify nav collapse/expand toggles correctly, persists across reload
- Verify each page's panel collapse/expand works independently
- Verify fullscreen widget on grid map enters/exits fullscreen
- Verify form inputs remain functional when panels are collapsed (CSS hidden, not removed from DOM)
- Verify reactive effects still trigger when panels are re-expanded
- Verify light/dark theme works with collapsed states
- Verify localStorage states restore correctly on page load
- Verify panel restore works for lazily-rendered tabs (switch to a tab with saved collapsed state)
- Verify Scenarios page 3-column collapse works (only first column collapses)
- Verify nav-collapsed CSS doesn't affect page-level layout_columns col-sm-* elements

## 7. Non-Goals

- No responsive/mobile breakpoints (desktop-focused app)
- No animated icons on the hamburger (simple 3-line icon)
- No drag-to-resize on panels (just toggle collapse/expand)
