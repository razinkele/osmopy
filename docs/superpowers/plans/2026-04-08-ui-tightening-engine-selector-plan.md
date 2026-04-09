# UI Tightening, Layer Control & Engine Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten the UI spacing, verify the layer control widget with shiny-deckgl 1.9.2, add a global Java/Python engine selector, and create stub UI pages for Python-only features.

**Architecture:** The engine selector is a reactive `AppState.engine_mode` field driving conditional UI across the app. New Python-only pages (genetics, economic, diagnostics) follow the existing `*_ui()` / `*_server()` pattern. CSS changes are isolated to `www/osmose.css`. The Run page gains internal tabs for Java/Python configs.

**Tech Stack:** Shiny for Python, shiny-deckgl 1.9.2, CSS3, reactive state

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `www/osmose.css` | Modify | Spacing tightening + engine toggle + nav-disabled styling |
| `ui/state.py` | Modify | Add `engine_mode` reactive field to `AppState` |
| `app.py` | Modify | Header engine toggle, nav items for new pages, server wiring, engine gate JS |
| `ui/pages/grid.py` | Modify | Verify layer control with shiny-deckgl 1.9.2 |
| `ui/pages/grid_helpers.py` | Modify | Verify `make_legend()` compat wrapper |
| `ui/pages/run.py` | Modify | Add Java/Python tabbed layout |
| `ui/pages/genetics.py` | Create | Ev-OSMOSE genetics stub page |
| `ui/pages/economic.py` | Create | Economic module stub page |
| `ui/pages/diagnostics.py` | Create | Python engine diagnostics stub page |

---

### Task 1: Tighten Header Spacing

**Files:**
- Modify: `www/osmose.css:79-166`

- [ ] **Step 1: Tighten header bar padding and gap**

In `www/osmose.css`, replace the `.osmose-header` block:

```css
/* ── App Header ───────────────────────────────────────────────────── */
.osmose-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 12px;
  background: linear-gradient(
    135deg,
    rgba(22, 38, 52, 0.95) 0%,
    rgba(15, 25, 35, 0.98) 100%
  );
  border-bottom: 1px solid var(--osm-border);
  box-shadow: 0 1px 8px rgba(0, 0, 0, 0.25);
  position: relative;
  z-index: 10;
}
```

Changes: `gap: 16px` → `8px`, `padding: 8px 16px` → `4px 12px`, `box-shadow` slightly reduced.

- [ ] **Step 2: Reduce logo and badge sizes**

In `www/osmose.css`, replace `.osmose-logo`:

```css
.osmose-logo {
  font-size: 0.95rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--osm-text-primary);
  text-transform: uppercase;
  margin: 0;
  line-height: 1;
}

.osmose-logo .subtitle {
  font-size: 0.6rem;
  font-weight: 400;
  letter-spacing: 0.02em;
  color: var(--osm-text-muted);
  text-transform: none;
  margin-left: 2px;
}

.osmose-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  background: var(--osm-accent-dim);
  border: 1px solid var(--osm-border-accent);
  border-radius: 20px;
  font-size: 0.6rem;
  font-weight: 600;
  color: var(--osm-accent);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
```

Changes: logo `1.1rem` → `0.95rem`, subtitle `0.7rem` → `0.6rem`, badge gap `5px` → `4px`, padding `3px 10px` → `2px 8px`, font `0.65rem` → `0.6rem`.

- [ ] **Step 3: Reduce header button padding**

In `www/osmose.css`, replace `.osmose-header-btn`:

```css
.osmose-header-btn {
  color: var(--osm-text-muted) !important;
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.03em;
  padding: 3px 8px;
  border-radius: 6px;
  text-decoration: none !important;
  transition: all var(--osm-transition);
  cursor: pointer;
}
```

Changes: font `0.78rem` → `0.72rem`, padding `5px 12px` → `3px 8px`.

- [ ] **Step 4: Verify header renders correctly**

Run: `/opt/micromamba/envs/shiny/bin/shiny run app.py --host 0.0.0.0 --port 8000`

Check: header is visually tighter, no clipping, all elements visible.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add www/osmose.css
git -C /home/razinka/osmose/osmose-python commit -m "style: tighten header bar spacing and reduce element sizes"
```

---

### Task 2: Tighten Nav Pills and Content Gap

**Files:**
- Modify: `www/osmose.css:324-408`

- [ ] **Step 1: Tighten nav pills container and section headers**

In `www/osmose.css`, replace the nav-pills block at line 324:

```css
.nav-pills {
  padding: 6px 6px !important;
}

/* Section headers */
.nav-pills .nav-item .osmose-section-label {
  font-size: 0.55rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--osm-text-muted);
  padding: 8px 12px 2px;
  display: block;
  position: relative;
}

.nav-pills .nav-item:first-child .osmose-section-label {
  padding-top: 2px;
}
```

Changes: pills padding `12px 8px` → `6px 6px`, section label font `0.6rem` → `0.55rem`, padding `12px 14px 4px` → `8px 12px 2px`, first-child `4px` → `2px`.

- [ ] **Step 2: Tighten nav link padding**

In `www/osmose.css`, replace `.nav-pills .nav-link` at line 356:

```css
.nav-pills .nav-link {
  color: var(--osm-text-secondary) !important;
  background: transparent !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 3px 10px !important;
  font-size: 0.78rem;
  font-weight: 500;
  transition: all var(--osm-transition);
  position: relative;
  margin: 0;
}
```

Changes: padding `4px 12px` → `3px 10px`, font `0.82rem` → `0.78rem`, border-radius `8px` → `6px`, margin `1px 0` → `0`.

- [ ] **Step 3: Tighten card header and content padding**

In `www/osmose.css`, replace `.card-header` at line 400:

```css
.card-header {
  background: rgba(0, 0, 0, 0.15) !important;
  border-bottom: 1px solid var(--osm-border) !important;
  padding: 8px 14px !important;
  font-size: 0.82rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--osm-text-primary);
  position: relative;
```

Changes: padding `12px 18px` → `8px 14px`, font `0.85rem` → `0.82rem`.

- [ ] **Step 4: Tighten tab content padding**

In `www/osmose.css`, replace `.tab-content > .tab-pane` at line 790:

```css
.tab-content > .tab-pane {
  padding: 4px 2px;
}
```

Changes: padding `8px 4px` → `4px 2px`.

- [ ] **Step 5: Verify nav and content render correctly**

Run the app and check: nav pills are tighter, card headers compact, content area has less wasted space. Target: ~30-40% reduction in vertical space at the top.

- [ ] **Step 6: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add www/osmose.css
git -C /home/razinka/osmose/osmose-python commit -m "style: tighten nav pills, card headers, and content gap spacing"
```

---

### Task 3: Add engine_mode to AppState

**Files:**
- Modify: `ui/state.py:34-51`
- Test: `tests/test_state_engine.py`

- [ ] **Step 1: Write failing test for engine_mode**

Create `tests/test_state_engine.py`:

```python
"""Tests for engine_mode reactive state."""

from ui.state import AppState


def test_engine_mode_default_is_java():
    state = AppState()
    assert state.engine_mode.get() == "java"


def test_engine_mode_can_be_set_to_python():
    state = AppState()
    state.engine_mode.set("python")
    assert state.engine_mode.get() == "python"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_state_engine.py -v`

Expected: FAIL with `AttributeError: 'AppState' object has no attribute 'engine_mode'`

- [ ] **Step 3: Add engine_mode to AppState**

In `ui/state.py`, after line 51 (`self.key_case_map`), add:

```python
        self.engine_mode: reactive.Value[str] = reactive.Value("java")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_state_engine.py -v`

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add ui/state.py tests/test_state_engine.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: add engine_mode reactive field to AppState"
```

---

### Task 4: Add Engine Selector Toggle to Header

**Files:**
- Modify: `app.py:143-181` (header div)
- Modify: `www/osmose.css` (toggle styling)

- [ ] **Step 1: Add engine toggle CSS**

Append to `www/osmose.css`, before the closing comments or at the end of the header section (after `.osmose-header-btn:hover`):

```css
/* ── Engine Mode Toggle ──────────────────────────────────────────── */
.osm-engine-toggle {
  display: inline-flex;
  align-items: center;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid var(--osm-border);
  border-radius: 6px;
  overflow: hidden;
  margin: 0 4px;
}

.osm-engine-toggle .osm-engine-btn {
  padding: 2px 10px;
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--osm-text-muted);
  background: transparent;
  border: none;
  cursor: pointer;
  transition: all var(--osm-transition);
}

.osm-engine-toggle .osm-engine-btn.active {
  color: var(--osm-accent);
  background: var(--osm-accent-dim);
}

.osm-engine-toggle .osm-engine-btn:hover:not(.active) {
  color: var(--osm-text-primary);
  background: rgba(255, 255, 255, 0.05);
}
```

- [ ] **Step 2: Add engine toggle widget to header in app.py**

In `app.py`, replace the header `ui.div(...)` block (lines 144-181). Insert the engine toggle between the badge and the header-actions div:

```python
    # ── App header ──────────────────────────────────────────────
    ui.div(
        ui.tags.h4(
            "OSMOPY",
            ui.tags.span(" | Marine Ecosystem Simulator", class_="subtitle"),
            class_="osmose-logo",
        ),
        ui.tags.span(
            ui.tags.span(class_="dot"),
            f"v{__version__}",
            class_="osmose-badge",
        ),
        # Engine mode toggle
        ui.div(
            ui.tags.button(
                "Java",
                class_="osm-engine-btn active",
                id="engineBtnJava",
                onclick="setEngineMode('java')",
            ),
            ui.tags.button(
                "Python",
                class_="osm-engine-btn",
                id="engineBtnPython",
                onclick="setEngineMode('python')",
            ),
            class_="osm-engine-toggle",
        ),
        ui.div(
            ui.output_ui("config_header"),
            ui.tags.button(
                ui.tags.span("\u2600\ufe0f", class_="icon-sun"),
                ui.tags.span("\u263e", class_="icon-moon"),
                class_="osmose-theme-toggle",
                id="themeToggle",
                title="Toggle light/dark theme",
                onclick="toggleTheme()",
                **{"aria-label": "Toggle light/dark theme"},
            ),
            ui.tags.a(
                "About",
                class_="osmose-header-btn",
                href="#",
                **{"data-bs-toggle": "modal", "data-bs-target": "#aboutModal"},
            ),
            ui.tags.a(
                "Help",
                class_="osmose-header-btn",
                href="#",
                **{"data-bs-toggle": "modal", "data-bs-target": "#helpModal"},
            ),
            class_="osmose-header-actions",
        ),
        class_="osmose-header",
    ),
```

- [ ] **Step 3: Add engine mode JS + Shiny bridge**

In `app.py`, inside the `<head>` JS block (the `ui.tags.script(...)` that contains `toggleTheme`), add the `setEngineMode` function. Append this before the closing `})();`:

```javascript
        // Engine mode toggle — syncs with Shiny input and localStorage
        window.setEngineMode = function(mode) {
            localStorage.setItem('osmose-engine', mode);
            var jBtn = document.getElementById('engineBtnJava');
            var pBtn = document.getElementById('engineBtnPython');
            if (mode === 'java') {
                jBtn.classList.add('active');
                pBtn.classList.remove('active');
            } else {
                pBtn.classList.add('active');
                jBtn.classList.remove('active');
            }
            if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                Shiny.setInputValue('engine_mode', mode);
            }
            // Toggle disabled state on Python-only nav items
            document.querySelectorAll('.osm-engine-gated').forEach(function(el) {
                if (mode === 'python') {
                    el.classList.remove('osm-disabled');
                    el.removeAttribute('title');
                } else {
                    el.classList.add('osm-disabled');
                    el.setAttribute('title', 'Requires Python engine');
                }
            });
        };
        // Restore engine mode from localStorage on page load
        (function() {
            var saved = localStorage.getItem('osmose-engine') || 'java';
            // Defer until DOM is ready so buttons exist
            var poll = setInterval(function() {
                if (document.getElementById('engineBtnJava')) {
                    clearInterval(poll);
                    setEngineMode(saved);
                }
            }, 100);
        })();
```

- [ ] **Step 4: Wire engine_mode Shiny input to AppState in server function**

In `app.py`, inside the `server()` function (after `state.reset_to_defaults()`), add:

```python
    @reactive.effect
    @reactive.event(input.engine_mode)
    def _sync_engine_mode():
        mode = input.engine_mode()
        if mode in ("java", "python"):
            state.engine_mode.set(mode)
```

- [ ] **Step 5: Verify engine toggle renders and switches**

Run the app. Check: toggle appears in header between badge and actions, clicking switches active state, `input.engine_mode` updates in Shiny.

- [ ] **Step 6: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add app.py www/osmose.css
git -C /home/razinka/osmose/osmose-python commit -m "feat: add Java/Python engine selector toggle in header"
```

---

### Task 5: Verify Layer Control with shiny-deckgl 1.9.2

**Files:**
- Modify: `ui/pages/grid_helpers.py:1121-1140` (if needed)
- Modify: `ui/pages/grid.py` (if needed)

- [ ] **Step 1: Check make_legend() dispatches correctly**

Read `ui/pages/grid_helpers.py:1121-1140` and verify the `make_legend()` wrapper calls `layer_legend_widget()` from shiny-deckgl 1.9.2. The function should detect `layer_legend_widget` via `hasattr(shiny_deckgl, "layer_legend_widget")`.

Run a quick check:

```bash
/opt/micromamba/envs/shiny/bin/python -c "import shiny_deckgl; print(hasattr(shiny_deckgl, 'layer_legend_widget'))"
```

Expected: `True`

- [ ] **Step 2: Check layer_legend_widget API signature compatibility**

Run:

```bash
/opt/micromamba/envs/shiny/bin/python -c "from shiny_deckgl.widgets import layer_legend_widget; import inspect; print(inspect.signature(layer_legend_widget))"
```

Verify the signature accepts: `entries`, `placement`, `show_checkbox`, `collapsed`, `title`. These are the kwargs passed by `make_legend()`.

- [ ] **Step 3: Run app and test layer toggle**

Run the app, load a config with a grid, navigate to Grid page. Verify:
1. Legend widget appears on the map (bottom-left)
2. Checkboxes next to "Grid Extent", "Ocean Cells", "Land Cells"
3. Unchecking "Grid Extent" hides the extent layer
4. Rechecking restores it
5. Overlay layers appear when an overlay is selected

- [ ] **Step 4: Confirm no code changes needed**

The `layer_legend_widget()` API in 1.9.2 accepts exactly the parameters `make_legend()` passes: `entries`, `placement`, `show_checkbox`, `collapsed`, `title`. The `deck_legend_control` fallback branch in `make_legend()` can be kept for backward compatibility but is not exercised with 1.9.2.

If steps 1-3 all pass, no code changes or commits are needed for this task. If any step reveals an issue, fix and commit:

```bash
git -C /home/razinka/osmose/osmose-python add ui/pages/grid_helpers.py ui/pages/grid.py
git -C /home/razinka/osmose/osmose-python commit -m "fix: update layer control for shiny-deckgl 1.9.2 compatibility"
```

---

### Task 6: Create Genetics Stub Page

**Files:**
- Create: `ui/pages/genetics.py`

- [ ] **Step 1: Create genetics page module**

Create `ui/pages/genetics.py`:

```python
"""Genetics page — Ev-OSMOSE evolutionary genetics configuration (Python engine only)."""

from shiny import ui

from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.styles import STYLE_EMPTY


def genetics_ui():
    return ui.div(
        expand_tab("Genetics Configuration", "genetics"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Genetics Configuration", "genetics"),
                ui.div(
                    ui.h5("Ev-OSMOSE Genetics Module"),
                    ui.p(
                        "Configure evolutionary genetics parameters for species traits. "
                        "This module enables heritable trait variation, mutation, and "
                        "natural selection across generations.",
                    ),
                    ui.hr(),
                    ui.p(
                        "Trait heritability, mutation rates, and selection pressure "
                        "parameters will be available here once the Ev-OSMOSE engine "
                        "module is implemented.",
                        style=STYLE_EMPTY,
                    ),
                    ui.tags.ul(
                        ui.tags.li("Trait heritability coefficients per species"),
                        ui.tags.li("Mutation rate and variance"),
                        ui.tags.li("Selection pressure functions"),
                        ui.tags.li("Genetic diversity metrics"),
                        style="color: var(--osm-text-muted); font-size: 0.82rem;",
                    ),
                ),
            ),
            col_widths=[12],
        ),
        class_="osm-split-layout",
        id="split_genetics",
    )


def genetics_server(input, output, session, state):
    pass
```

- [ ] **Step 2: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add ui/pages/genetics.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: add Ev-OSMOSE genetics stub page module"
```

---

### Task 7: Create Economic Stub Page

**Files:**
- Create: `ui/pages/economic.py`

- [ ] **Step 1: Create economic page module**

Create `ui/pages/economic.py`:

```python
"""Economic page — fleet economics and market configuration (Python engine only)."""

from shiny import ui

from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.styles import STYLE_EMPTY


def economic_ui():
    return ui.div(
        expand_tab("Economic Configuration", "economic"),
        ui.layout_columns(
            ui.card(
                collapsible_card_header("Economic Configuration", "economic"),
                ui.div(
                    ui.h5("Economic Module"),
                    ui.p(
                        "Configure fleet economics, market dynamics, and quota "
                        "management. This module couples economic decision-making "
                        "with ecological simulation.",
                    ),
                    ui.hr(),
                    ui.p(
                        "Fleet cost structures, market prices, and quota parameters "
                        "will be available here once the economic engine module is "
                        "implemented.",
                        style=STYLE_EMPTY,
                    ),
                    ui.tags.ul(
                        ui.tags.li("Fleet cost structures (fuel, labour, maintenance)"),
                        ui.tags.li("Market prices and demand curves"),
                        ui.tags.li("Quota management and allocation rules"),
                        ui.tags.li("Effort dynamics and fleet behaviour"),
                        style="color: var(--osm-text-muted); font-size: 0.82rem;",
                    ),
                ),
            ),
            col_widths=[12],
        ),
        class_="osm-split-layout",
        id="split_economic",
    )


def economic_server(input, output, session, state):
    pass
```

- [ ] **Step 2: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add ui/pages/economic.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: add economic module stub page module"
```

---

### Task 8: Create Diagnostics Stub Page

**Files:**
- Create: `ui/pages/diagnostics.py`

- [ ] **Step 1: Create diagnostics page module**

Create `ui/pages/diagnostics.py`:

```python
"""Diagnostics page — Python engine performance and runtime metrics."""

from shiny import render, ui

from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.styles import STYLE_EMPTY


def diagnostics_ui():
    return ui.div(
        expand_tab("Engine Diagnostics", "diagnostics"),
        ui.layout_columns(
            # Left: Timing breakdown
            ui.card(
                collapsible_card_header("Engine Diagnostics", "diagnostics"),
                ui.div(
                    ui.h5("Performance Dashboard"),
                    ui.p(
                        "After running the Python engine, timing breakdowns, "
                        "Numba JIT status, and memory usage will appear here.",
                        style=STYLE_EMPTY,
                    ),
                    ui.hr(),
                    ui.h5("Process Timing"),
                    ui.output_ui("diag_timing"),
                    ui.hr(),
                    ui.h5("Numba JIT Status"),
                    ui.output_ui("diag_numba"),
                    ui.hr(),
                    ui.h5("Memory Profile"),
                    ui.output_ui("diag_memory"),
                ),
            ),
            # Right: Comparison
            ui.card(
                ui.card_header("Engine Comparison"),
                ui.div(
                    ui.p(
                        "Run both Java and Python engines on the same config "
                        "to see a side-by-side timing comparison.",
                        style=STYLE_EMPTY,
                    ),
                    ui.output_ui("diag_comparison"),
                ),
            ),
            col_widths=[7, 5],
        ),
        class_="osm-split-layout",
        id="split_diagnostics",
    )


def diagnostics_server(input, output, session, state):
    @render.ui
    def diag_timing():
        if state.engine_mode.get() != "python":
            return ui.p("Switch to Python engine to view diagnostics.", style=STYLE_EMPTY)
        result = state.run_result.get()
        if result is None:
            return ui.p("No run results yet. Run the Python engine first.", style=STYLE_EMPTY)
        timing = getattr(result, "timing", None)
        if timing is None:
            return ui.p("No timing data available for this run.", style=STYLE_EMPTY)
        rows = []
        for process, seconds in sorted(timing.items()):
            rows.append(ui.tags.tr(
                ui.tags.td(process, style="font-weight: 500;"),
                ui.tags.td(f"{seconds:.3f}s"),
            ))
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(ui.tags.th("Process"), ui.tags.th("Time"))),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped",
        )

    @render.ui
    def diag_numba():
        return ui.p("Numba JIT compilation info will appear after a Python engine run.",
                     style=STYLE_EMPTY)

    @render.ui
    def diag_memory():
        return ui.p("Memory usage profiling will appear after a Python engine run.",
                     style=STYLE_EMPTY)

    @render.ui
    def diag_comparison():
        return ui.p("Comparison data will appear after running both engines.",
                     style=STYLE_EMPTY)
```

- [ ] **Step 2: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add ui/pages/diagnostics.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: add Python engine diagnostics page module"
```

---

### Task 9: Wire New Pages and Engine-Gated Nav Items in app.py

**Files:**
- Modify: `app.py` (imports, nav items, server wiring)
- Modify: `www/osmose.css` (nav gating CSS)

**Depends on:** Tasks 6, 7, 8 (page modules must exist)

- [ ] **Step 1: Add imports for new pages**

In `app.py`, after the `map_viewer` import (line 27), add:

```python
from ui.pages.genetics import genetics_ui, genetics_server
from ui.pages.economic import economic_ui, economic_server
from ui.pages.diagnostics import diagnostics_ui, diagnostics_server
```

- [ ] **Step 2: Add engine-gated nav items**

In `app.py`, update the `navset_pill_list` to include the three new pages. Add them with the `osm-engine-gated osm-disabled` classes (disabled by default since engine_mode defaults to Java).

Replace the Configure, Execute, and Manage sections:

```python
        # Configure
        _nav_section("Configure"),
        ui.nav_panel("Setup", setup_ui(), value="setup"),
        ui.nav_panel("Grid", grid_ui(), value="grid"),
        ui.nav_panel("Forcing", forcing_ui(), value="forcing"),
        ui.nav_panel("Fishing", fishing_ui(), value="fishing"),
        ui.nav_panel("Movement", movement_ui(), value="movement"),
        ui.nav_panel(
            ui.span("Genetics", class_="osm-engine-gated osm-disabled"),
            genetics_ui(),
            value="genetics",
        ),
        ui.nav_panel(
            ui.span("Economic", class_="osm-engine-gated osm-disabled"),
            economic_ui(),
            value="economic",
        ),
        # Execute
        _nav_section("Execute"),
        ui.nav_panel("Run", run_ui(), value="run"),
        ui.nav_panel("Results", results_ui(), value="results"),
        ui.nav_panel("Spatial Results", spatial_results_ui(), value="spatial_results"),
        ui.nav_panel(
            ui.span("Diagnostics", class_="osm-engine-gated osm-disabled"),
            diagnostics_ui(),
            value="diagnostics",
        ),
        # Optimize
        _nav_section("Optimize"),
        ui.nav_panel("Calibration", calibration_ui(), value="calibration"),
        # Manage
        _nav_section("Manage"),
        ui.nav_panel("Scenarios", scenarios_ui(), value="scenarios"),
        ui.nav_panel("Advanced", advanced_ui(), value="advanced"),
        ui.nav_panel("Map Viewer", map_viewer_ui(), value="map_viewer"),
```

- [ ] **Step 3: Wire server functions**

In the `server()` function, after `map_viewer_server(...)`:

```python
    genetics_server(input, output, session, state)
    economic_server(input, output, session, state)
    diagnostics_server(input, output, session, state)
```

- [ ] **Step 4: Add CSS for engine-gated nav links**

The existing `.osm-disabled` class in `www/osmose.css:1871` handles the styling. Add this rule to ensure it propagates from the span to the nav-link:

```css
/* Engine-gated nav items — disabled when Java engine selected */
.nav-pills .nav-link:has(.osm-engine-gated.osm-disabled) {
  opacity: 0.4;
  pointer-events: none;
  cursor: not-allowed;
}
```

- [ ] **Step 5: Verify all three pages and engine gating**

Run the app. Check:
1. "Genetics", "Economic", "Diagnostics" appear in nav but greyed out (Java default)
2. Toggle to Python — they become active and clickable
3. Each page renders its stub content
4. Toggle back to Java — disabled again

- [ ] **Step 6: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add app.py www/osmose.css
git -C /home/razinka/osmose/osmose-python commit -m "feat: wire new pages with engine-gated nav items"
```

---

### Task 10: Add Java/Python Tabs to Run Page

**Files:**
- Modify: `ui/pages/run.py:147-185` (run_ui function)
- Modify: `ui/pages/run.py:188+` (run_server function)

- [ ] **Step 1: Refactor run_ui into tabbed layout**

In `ui/pages/run.py`, replace the `run_ui()` function:

```python
def run_ui():
    return ui.div(
        expand_tab("Run Configuration", "run"),
        ui.layout_columns(
            # Left: Run controls with engine tabs
            ui.card(
                collapsible_card_header("Run Configuration", "run"),
                ui.navset_tab(
                    ui.nav_panel(
                        "Java",
                        ui.output_ui("jar_selector"),
                        ui.input_text(
                            "java_opts", "Java options", value="-Xmx2g",
                            placeholder="-Xmx4g -Xms1g",
                        ),
                        ui.input_numeric(
                            "run_timeout", "Timeout (seconds)",
                            value=3600, min=60, max=86400,
                        ),
                        ui.input_text_area(
                            "param_overrides",
                            "Parameter overrides (key=value, one per line)",
                            rows=4,
                        ),
                        value="run_java_tab",
                    ),
                    ui.nav_panel(
                        "Python",
                        ui.input_numeric(
                            "py_threads", "Threads (Numba prange)",
                            value=1, min=1, max=32,
                        ),
                        ui.input_select(
                            "py_verbosity", "Verbosity",
                            choices={"0": "Quiet", "1": "Normal", "2": "Verbose"},
                            selected="1",
                        ),
                        ui.input_text_area(
                            "py_param_overrides",
                            "Parameter overrides (key=value, one per line)",
                            rows=4,
                        ),
                        value="run_python_tab",
                    ),
                    id="run_engine_tabs",
                ),
                ui.hr(),
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_run", "Start Run", class_="btn-success btn-lg w-100"
                    ),
                    ui.input_action_button(
                        "btn_cancel", "Cancel", class_="btn-danger btn-lg w-100"
                    ),
                    col_widths=[6, 6],
                ),
                ui.hr(),
                ui.h5("Run Status"),
                ui.output_text("run_status"),
            ),
            # Right: Console output
            ui.card(
                ui.card_header("Console Output"),
                ui.output_ui("run_console"),
            ),
            col_widths=[4, 8],
        ),
        class_="osm-split-layout",
        id="split_run",
    )
```

- [ ] **Step 2: Add engine tab sync to run_server**

In `ui/pages/run.py`, inside `run_server()`, add a reactive effect to sync the active tab with the engine mode. Add after the `sync_jar_path` effect:

```python
    @reactive.effect
    def _sync_engine_tab():
        mode = state.engine_mode.get()
        tab = "run_java_tab" if mode == "java" else "run_python_tab"
        ui.update_navs("run_engine_tabs", selected=tab, session=session)
```

- [ ] **Step 3: Verify tabs render and sync with engine toggle**

Run the app. Check:
1. Run page shows "Java" and "Python" tabs
2. Java tab has JAR selector, Java opts, timeout, overrides
3. Python tab has threads, verbosity, overrides
4. Switching engine toggle in header auto-selects matching tab
5. Start Run button works from Java tab (existing functionality preserved)

- [ ] **Step 4: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add ui/pages/run.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: add Java/Python tabbed layout to Run page"
```

---

### Task 11: Final Integration Test

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`

Expected: All existing tests pass, plus new `test_state_engine.py` tests.

- [ ] **Step 2: Run lint**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`

Expected: Clean (no lint errors)

- [ ] **Step 3: End-to-end manual verification**

Run the app and verify:
1. Header is visually tighter (~30-40% less vertical space at top)
2. Engine toggle works (Java ↔ Python)
3. Genetics, Economic, Diagnostics nav items disabled when Java selected
4. Genetics, Economic, Diagnostics nav items enabled when Python selected
5. Grid map legend widget shows with visibility checkboxes
6. Grid extent can be toggled off/on independently
7. Run page has Java/Python tabs
8. Tab auto-switches when engine toggle changes
9. Existing functionality (load config, run Java simulation) still works

- [ ] **Step 4: Commit any remaining fixes**

If any issues found in steps 1-3, fix and commit individually.
